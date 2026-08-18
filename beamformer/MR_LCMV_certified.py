"""
Manifold-Retracted LCMV with Dual Correction (MR-LCMV)
======================================================

A fast, few-shot beamformer/nullformer for phase-only arrays whose elements live
on a lossy, non-monotonic complex manifold c_n(V) (amplitude-phase coupling,
branch folding).

Why this exists
---------------
The Alternating-Projection (AP) backbone minimizes a *pattern-matching residual*.
Main-beam gain is never in its objective, so it is collateral damage: the FFT ->
clip -> IFFT -> damp cycle shrinks weight magnitudes and the per-element manifold
projection lands on low-amplitude points, giving normalized gain as low as
-20 dB. AP also cannot separate "I must keep gain" from "I must dig a null".

MR-LCMV instead solves the problem we actually care about:

        maximize   |c^T a(u_t)|^2            (coherent main-beam gain)
        subject to  c^T a(u_m) = 0   for every null u_m
                    c_n in M_n                (hardware manifold, per element)

Conventions (match beamformer.controllers.metrics):
    a(u)_n   = exp(+j*pi*n*u)                 steering vector for AF(u)=sum_n c_n a(u)_n
    s(u)_n   = conj(a(u)_n) = exp(-j*pi*n*u)  matched ("conjugate") weight; c=s(u_t) => AF=N

Algorithm (three cheap pieces)
------------------------------
1. CLOSED-FORM LCMV ANCHOR.  For a continuous (unconstrained-modulus) array the
   max-gain weight that perfectly nulls is the matched filter projected off the
   null subspace:

        c0 = P (e^{j psi} s_t),   P = I - conj(A) (A^T conj(A))^{-1} A^T

   where A = [a(u_1), ..., a(u_M)].  This is the *optimal dual* of the continuous
   problem and gives a perfect-null, near-max-gain starting point in ONE shot.

2. MANIFOLD RETRACTION.  Each element independently snaps onto its manifold by a
   pure inner-product maximization (a *linear* objective, hence the global gain
   optimum for a fixed alignment):

        c_n = argmax_{c in M_n} Re( c * conj(target_n) )

   Crucially the target is the *full-amplitude matched field*, never a shrunken
   IFFT pattern -- so retraction preserves coherent gain by construction.

3. DUAL (GAUSS-NEWTON) CORRECTION.  Retraction perturbs the nulls. We repair them
   with a few steps of the SAME small MxM solve, accounting for null-null
   coupling, while the gain term in the target keeps the beam locked:

        target = e^{j psi} s_t - conj(A) @ mu
        c      = retract(target)
        mu    += Ginv @ (A^T @ c)            # drive leakage A^T c -> 0

   M = number of nulls (typically 1-4), so each step is a 2x2-ish solve: trivial.

A coarse 1-D sweep over the global gauge psi finds the insertion-loss sweet spot
of the manifold. Total cost: a handful of vectorized argmaxes -- "few shot".
"""

import numpy as np


# --------------------------------------------------------------------------- #
# Optional cost-accounting (for fair, machine-independent benchmarking)
# --------------------------------------------------------------------------- #
# These counters let a benchmark report MR-LCMV's cost in the SAME currency as
# the iterative/derivative-free baselines: the number of *full forward-model
# evaluations* (one complete array-factor evaluation of a candidate weight
# vector at the constraint directions). This is the standard machine-independent
# optimization metric and replaces wall-clock time for publication.
#
#   _EVAL_COUNT : full array-factor evaluations (dual-correction leakage checks,
#                 gauge scoring, PTNR gauge scoring). Comparable to one call of
#                 the baselines' penalized objective J(V).
#   _GLCP_UPDATES : committed Gain-Locked Coordinate Polish coordinate moves.
#                 GLCP examines the manifold grid via O(M) rank-1 incremental
#                 updates and performs NO full array-factor evaluation, so these
#                 are reported SEPARATELY (not folded into _EVAL_COUNT) to avoid
#                 either inflating or understating the true forward-model cost.
#
# The counters are pure bookkeeping: when nobody resets/reads them they simply
# increment a module global and change no returned value, so existing callers
# and tests are unaffected.
_EVAL_COUNT = 0
_GLCP_UPDATES = 0
_GLOBAL_HIST = []          # list of dicts
_BEST_LEAK_GLOBAL = np.inf
_BEST_NULL_DB_GLOBAL = np.inf
_CURRENT_PHASE = "init"
_CURRENT_GAUGE = None

def reset_counters():
    global _EVAL_COUNT, _GLCP_UPDATES, _GLOBAL_HIST
    global _BEST_LEAK_GLOBAL, _BEST_NULL_DB_GLOBAL
    _EVAL_COUNT = 0
    _GLCP_UPDATES = 0
    _GLOBAL_HIST = []
    _BEST_LEAK_GLOBAL = np.inf
    _BEST_NULL_DB_GLOBAL = np.inf

def set_phase(phase, gauge=None):
    """Call this whenever the algorithm changes major stage."""
    global _CURRENT_PHASE, _CURRENT_GAUGE
    _CURRENT_PHASE = phase          # "dual", "score", "ptnr", "glcp", "sll", ...
    if gauge is not None:
        _CURRENT_GAUGE = float(gauge)

def _record_oracle(leak, gain_lin, accepted=False):
    global _BEST_LEAK_GLOBAL, _BEST_NULL_DB_GLOBAL
    null_db = 20.0 * np.log10(max(leak, 1e-15))

    if leak < _BEST_LEAK_GLOBAL - 1e-15:
        _BEST_LEAK_GLOBAL = leak
        _BEST_NULL_DB_GLOBAL = null_db

    _GLOBAL_HIST.append({
        "oracle": int(_EVAL_COUNT),
        "leak": float(leak),
        "best_leak": float(_BEST_LEAK_GLOBAL),
        "null_db": float(null_db),
        "best_null_db": float(_BEST_NULL_DB_GLOBAL),
        "gain_lin": float(gain_lin),
        "accepted": bool(accepted),
        "phase": _CURRENT_PHASE,
        "gauge": _CURRENT_GAUGE,
    })


def get_counters():
    """Return (full_forward_evals, glcp_committed_updates) since the last reset."""
    return _EVAL_COUNT, _GLCP_UPDATES


def _bump_eval(k=1):
    global _EVAL_COUNT
    _EVAL_COUNT += k


def _bump_glcp(k=1):
    global _GLCP_UPDATES
    _GLCP_UPDATES += k


# --------------------------------------------------------------------------- #
# Manifold access helpers
# --------------------------------------------------------------------------- #
def _get_manifold(element):
    """Return (V_grid, c_grid).

    Supported layouts:
      * shared manifold:      (G,)
      * element-specific:     (N, G)

    The latter is useful when the measured hardware manifold differs across
    array elements.
    """
    V = np.asarray(element.V_grid)
    c = np.asarray(element.c_grid)
    return V, c


def _retract(target, c_grid, V_grid):
    """Project each desired coefficient onto the discrete hardware manifold.

    The returned coefficient always belongs to the supplied hardware grid.
    For a shared grid c_grid has shape (G,); for element-specific manifolds it
    has shape (N, G).
    """
    target = np.asarray(target)
    c_grid = np.asarray(c_grid)
    V_grid = np.asarray(V_grid)

    if c_grid.ndim == 1:
        score = np.real(np.conj(target)[:, None] * c_grid[None, :])
        idx = np.argmax(score, axis=1)
        return c_grid[idx], V_grid[idx], idx

    if c_grid.ndim != 2 or c_grid.shape[0] != len(target):
        raise ValueError(
            "c_grid must have shape (G,) or (N,G), with N=len(target)."
        )

    score = np.real(np.conj(target)[:, None] * c_grid)
    idx = np.argmax(score, axis=1)
    rows = np.arange(len(target))
    return c_grid[rows, idx], V_grid[rows, idx], idx


def _grid_candidates(c_grid, V_grid, i):
    """Return the candidate manifold points for element i."""
    if np.asarray(c_grid).ndim == 1:
        return np.asarray(c_grid), np.asarray(V_grid)
    return np.asarray(c_grid)[i], np.asarray(V_grid)[i]


def _af(c, u, N):
    """Array factor AF(u) = sum_n c_n exp(+j pi n u)."""
    return np.sum(c * np.exp(1j * np.pi * np.arange(N) * u))

# --------------------------------------------------------------------------- #
# Certified MR-LCMV solver
# --------------------------------------------------------------------------- #
def _dual_correction_certified(
    tgt0,
    A,
    Ginv,
    c_grid,
    V_grid,
    N,
    leak_target,
    max_dual_steps=32,
    patience=4,
    gain_floor=0.0,
    u_t=0.0,
    improvement_tol=1e-12,
    return_history=False,
    psi= None
):
    """Residual-driven dual correction on the finite hardware manifold.

    This is intentionally *not* presented as smooth Gauss-Newton convergence.
    Retraction makes the map discrete.  Instead, accepted states must strictly
    reduce the worst hard-null residual, while a requested null target provides
    an immediate success certificate.  ``max_dual_steps`` is only a safety cap.

    Returns
    -------
    c, V, leak, status : best manifold state, voltage state, residual, status
    """
    status= "safety_cap"
    if A is None:
        c, V, _ = _retract(tgt0, c_grid, V_grid)
        return c, V, 0.0, "no_nulls", []

    mu = Ginv @ (A.T @ tgt0)
    best_c = None
    best_V = None
    best_leak = np.inf
    best_gain = -np.inf
    no_improve = 0
    history = []
    #oracle_at_start = _GLCP_UPDATES          # absolute counter at entry
  
    for k in range(max_dual_steps):
        target = tgt0 - np.conj(A) @ mu
        c, V, _ = _retract(target, c_grid, V_grid)
        leak_vec = A.T @ c
        set_phase("dual", gauge=psi)          # psi is known in the caller
        _bump_eval()
        leak = float(np.max(np.abs(leak_vec))) if len(leak_vec) else 0.0
        gain = float(abs(_af(c, u_t, N)) / max(N, 1))
        null_depth_db = 20 * np.log10(leak + 1e-15)   # absolute null depth
        # A hard gain guard can be requested.  The first state is retained as a
        # fallback even if the requested guard is too aggressive.
        gain_ok = gain >= gain_floor
        accepted=False
        
        if best_c is None:
            best_c, best_V, best_leak, best_gain = c.copy(), V.copy(), leak, gain
            no_improve = 0
        elif gain_ok and leak < best_leak - improvement_tol:
            best_c, best_V, best_leak, best_gain = c.copy(), V.copy(), leak, gain
            no_improve = 0
            accepted= True
        else:
            no_improve += 1

        _record_oracle(leak, gain, accepted=accepted)
        history.append({
            "step": k,
            "leak": leak,
            "gain": gain,
            "accepted": accepted,
            "best_leak_so_far": best_leak,
            "oracle_calls": _EVAL_COUNT,   # cumulative oracles so far
            "null_depth_db": null_depth_db,
            "best_null_depth_db": 20 * np.log10(best_leak + 1e-15),
            "gain_lin": gain,
        })

        status= "safety_cap"
        # Requested null achieved: explicit certificate.
        if leak <= leak_target and gain_ok:
            status= "Target_reached"
            break

        # If the discrete manifold no longer gives a strictly better state,
        # stop at the best previously accepted state.
        if no_improve >= patience:
            status = "stagnated"
            break

        mu = mu + Ginv @ leak_vec
    
    if return_history:
        return best_c, best_V, best_leak, status, history

    return best_c, best_V, best_leak, "safety_cap"


def solve_coherent_lcmv(
    task,
    element,
    N,
    n_dual_steps=32,
    psi_samples=72,
    psi_grid=None,
    ridge=1e-6,
    null_target_db=-45.0,
    suppress_sll=False,
    sll_ceiling_db=None,
    n_sll_virtual=2,
    sll_outer_rounds=2,
    oversample_factor=8,
    return_history=False,
    dual_patience=4,
    dual_gain_floor=0.0,
    dual_improvement_tol=1e-12,
):
    """Certified MR-LCMV solve on the finite hardware manifold.

    The dual stage is residual-driven: it terminates when the requested null
    target is reached, when the discrete manifold becomes stagnant, or when
    the safety cap ``n_dual_steps`` is exhausted.  The cap is not the
    convergence criterion.

    The GLCP refinement uses hard null-preservation and a global gain floor.
    Thus a null certified by the dual stage cannot subsequently be degraded.
    """
    V_grid, c_grid = _get_manifold(element)
    u_t = task.target_angles[0]
    nulls = list(task.null_angles) if getattr(task, "null_angles", None) else []
    n = np.arange(N)
    s_t = np.exp(-1j * np.pi * n * u_t)
    null_target_lin = 10 ** (null_target_db / 20.0)

    if sll_ceiling_db is None and getattr(task, "sll_ceiling", None) is not None:
        sll_ceiling_db = 20 * np.log10(task.sll_ceiling * N)

    def build_A(null_list):
        if not null_list:
            return None, None
        A = np.column_stack([np.exp(1j * np.pi * n * um) for um in null_list])
        G = A.T @ np.conj(A)
        G += ridge * np.trace(G).real / max(len(null_list), 1) * np.eye(len(null_list))
        return A, np.linalg.inv(G)

    def solve_for_gauge(psi, null_list, A, Ginv):
        tgt0 = np.exp(1j * psi) * s_t
        c, V, leak, status,dual_hist = _dual_correction_certified(
            tgt0,
            A,
            Ginv,
            c_grid,
            V_grid,
            N,
            leak_target=null_target_lin,
            max_dual_steps=n_dual_steps,
            patience=dual_patience,
            gain_floor=dual_gain_floor,
            u_t=u_t,
            improvement_tol=dual_improvement_tol,
            return_history= True,
            psi=psi
        )
        return c, V, leak, status, dual_hist

    def score_solution(c):
        set_phase("score")                    # or "ptnr"
        _bump_eval()
        g_db = 20 * np.log10(abs(_af(c, u_t, N)) / N + 1e-12)
        g_lin = abs(_af(c, u_t, N)) / N
        leak=0
        if not nulls:
            return g_db, g_db
        else:
            leak = max(abs(_af(c, um, N)) for um in nulls)
        null_db = [20 * np.log10(abs(_af(c, um, N)) + 1e-12) for um in nulls]
        _record_oracle(leak, g_lin, accepted=False)
        worst = max(null_db)
        limit_db = 20 * np.log10(N) + null_target_db
        pen = max(0.0, worst - limit_db)
        return g_db - 0.5 * pen, g_db

    null_list = list(nulls)
    A, Ginv = build_A(null_list)
    best_pack = None

    if psi_grid is None:
        psis = np.linspace(0, 2 * np.pi, psi_samples, endpoint=False)
    else:
        psis = np.atleast_1d(np.asarray(psi_grid, dtype=float))

    dual_statuses = []
    dual_leaks = []
    for psi in psis:
        c, V, leak, status, hist1 = solve_for_gauge(psi, null_list, A, Ginv)
        sc, g_db = score_solution(c)
        dual_statuses.append(status)
        dual_leaks.append(leak)
        if best_pack is None or sc > best_pack[0]:
            best_pack = (sc, g_db, c, V, float(psi), leak, status, hist1)

    if suppress_sll and sll_ceiling_db is not None:
        from .utils import oversampled_fft
        psi = best_pack[4]
        for _ in range(sll_outer_rounds):
            c = best_pack[2]
            u_grid, F = oversampled_fft(c, oversample_factor)
            pdb = 20 * np.log10(np.abs(F) + 1e-12)
            mask = np.abs(u_grid - u_t) > task.beamwidth
            for um in null_list:
                mask &= np.abs(u_grid - um) > task.beamwidth
            cand = np.where(mask & (pdb > sll_ceiling_db))[0]
            if cand.size == 0:
                break
            worst = cand[np.argsort(pdb[cand])[-n_sll_virtual:]]
            for w in worst:
                uu = float(u_grid[w])
                if all(abs(uu - x) > 1e-3 for x in null_list):
                    null_list.append(uu)
            A, Ginv = build_A(null_list)
            local_best = None
            for dp in np.linspace(-0.3, 0.3, 7):
                c2, V2, leak2, status2, hist2 = solve_for_gauge(
                    (psi + dp) % (2 * np.pi), null_list, A, Ginv
                )
                sc2, g2 = score_solution(c2)
                if local_best is None or sc2 > local_best[0]:
                    local_best = (
                        sc2, g2, c2, V2, (psi + dp) % (2 * np.pi), leak2, status2, hist2
                    )
            best_pack = local_best

    sc, g_db, c_n, V_n, psi, dual_leak, dual_status,hist = best_pack

    if return_history:
        info = {
            "psi": psi,
            "norm_gain_db": g_db,
            "nulls_db": [20 * np.log10(abs(_af(c_n, um, N)) + 1e-12) for um in nulls],
            "null_list": null_list,
            "mean_abs_c": float(np.mean(np.abs(c_n))),
            "dual_residual_lin": float(dual_leak),
            "dual_status": dual_status,
            "dual_target_reached": bool(dual_status == "target_reached"),
            "dual_n_steps_cap": int(n_dual_steps),
            "history": hist,
        }
        return V_n, c_n, info
    return V_n, c_n


# --------------------------------------------------------------------------- #
# Certified Gain-Locked Coordinate Polish (GLCP)
# --------------------------------------------------------------------------- #
def gain_locked_polish(
    c_n,
    V_n,
    task,
    element,
    null_angles=None,
    rounds=32,
    gain_floor=0.995,
    null_tolerance=1e-12,
):
    """Finite-state gain-locked refinement with hard null preservation.

    Let c_in be the state entering GLCP.  The algorithm establishes two
    invariants relative to that state:

      |AF_t(c)| >= gain_floor * |AF_t(c_in)|

      |AF_m(c)| <= (1+null_tolerance) |AF_m(c_in)| for every hard null m.

    A coordinate move is committed only when it satisfies both invariants and
    strictly improves the worst-null residual.  Since the hardware manifold is
    finite, only finitely many strict improvements are possible. ``rounds`` is
    therefore a safety cap, not the convergence criterion.
    """
    V_grid, c_grid = _get_manifold(element)
    N = len(c_n)
    if null_angles is None:
        null_angles = list(getattr(task, "null_angles", []) or [])
    if not null_angles or not getattr(task, "target_angles", None):
        return V_n.copy(), c_n.copy()

    n = np.arange(N)
    u_t = task.target_angles[0]
    a_t = np.exp(1j * np.pi * n * u_t)
    A = np.column_stack([np.exp(1j * np.pi * n * um) for um in null_angles])

    c = c_n.copy().astype(np.complex128)
    V = V_n.copy().astype(float)
    St = np.sum(c * a_t)
    Sm = A.T @ c

    # These are fixed references, not recomputed after each round.  This makes
    # the gain and hard-null guarantees global over the complete refinement.
    gain_floor_abs = gain_floor * abs(St)
    null_cap = np.abs(Sm).copy()
    current_worst = float(np.max(np.abs(Sm)))

    for _ in range(rounds):
        moved = 0
        for i in range(N):
            candidates_c, candidates_V = _grid_candidates(c_grid, V_grid, i)
            old = c[i]

            dt = (candidates_c - old) * a_t[i]
            St_cand = St + dt
            gain_feasible = np.abs(St_cand) >= gain_floor_abs
            if not np.any(gain_feasible):
                continue

            dm = np.outer(candidates_c - old, A[i])
            Sm_cand = Sm[None, :] + dm
            abs_null = np.abs(Sm_cand)

            # Hard preservation: no established hard null may become worse.
            null_feasible = np.all(
                abs_null <= (1.0 + null_tolerance) * null_cap[None, :], axis=1
            )
            feasible = gain_feasible & null_feasible
            if not np.any(feasible):
                continue

            worst = np.max(abs_null, axis=1)
            # Strict descent in the primary worst-null objective.
            worst[~feasible] = np.inf
            best_j = int(np.argmin(worst))
            candidate_worst = float(worst[best_j])

            if candidate_worst < current_worst - null_tolerance:
                c[i] = candidates_c[best_j]
                V[i] = candidates_V[best_j]
                St = St_cand[best_j]
                Sm = Sm_cand[best_j]
                current_worst = candidate_worst
                _bump_glcp()
                moved += 1

        if moved == 0:
            break

    return V, c

# --------------------------------------------------------------------------- #
# Orchestrator: three selectable refinement modes
# --------------------------------------------------------------------------- #
def beamform_mrlcmv(
    task,
    element,
    N,
    rng=None,
    refine="glcp",
    gauge="ptnr",
    gauge_samples=24,
    suppress_sll=False,
    glcp_rounds=6,
    glcp_gain_floor=0.995,
    **kwargs,
):
    """
    Top-level MR-LCMV beamformer with a selectable refinement stage and a
    selectable global-gauge (offset psi) selection objective.

    refine:
        'none'  -> MR-LCMV anchor + dual correction only.
        'glcp'  -> MR-LCMV then Gain-Locked Coordinate Polish (recommended).
        'ap_lr' -> MR-LCMV used as a warm start for the legacy AP + local
                   refinement backbone (for comparison; AP typically degrades it).

    gauge: how the global offset psi is chosen. The offset does not change the
        inter-element phase taper, but it slides every element to a different
        absolute phase on the lossy manifold. Empirically the post-refinement
        *gain* is nearly psi-invariant (~0.6 dB), while achievable *null depth*
        varies by 50+ dB -- so scoring psi by gain (the 'gain' mode) lands on the
        wrong peak and leaves 10-16 dB of null depth on the table.
        'gain' -> pick psi by the pre-refinement gain/null score (legacy; one
                  internal sweep inside solve_coherent_lcmv).
        'ptnr' -> evaluate the *refined* solution (incl. GLCP) at each candidate
                  psi and pick the best peak-to-null ratio (recommended). Only
                  applies when refine in {'none','glcp'} and the task has nulls;
                  otherwise falls back to 'gain'.

    gauge_samples: number of uniform psi candidates for the 'ptnr' gauge.

    Returns a dict shaped like `beamformer.api.beamform`.
    """
    solve_kwargs = dict(
        n_dual_steps=kwargs.get("n_dual_steps", 32),
        dual_patience=kwargs.get("dual_patience", 4),
        dual_gain_floor=kwargs.get("dual_gain_floor", 0.0),
        dual_improvement_tol=kwargs.get("dual_improvement_tol", 1e-12),
        psi_samples=kwargs.get("psi_samples", 72),
        ridge=kwargs.get("ridge", 1e-6),
        null_target_db=kwargs.get("null_target_db", -45.0),
        suppress_sll=suppress_sll,
        sll_ceiling_db=kwargs.get("sll_ceiling_db", None),
        n_sll_virtual=kwargs.get("n_sll_virtual", 2),
        sll_outer_rounds=kwargs.get("sll_outer_rounds", 2),
        oversample_factor=kwargs.get("oversample_factor", 8),
    )
    nulls = list(task.null_angles) if getattr(task, "null_angles", None) else []
    u_t = task.target_angles[0]

    def solve_and_refine(psi_grid):
        V_n, c_n, info = solve_coherent_lcmv(
            task, element, N, return_history=True, psi_grid=psi_grid, **solve_kwargs
        )
        init = c_n.copy()
        if refine == "none":
            pass
        elif refine == "glcp":
            V_n, c_n = gain_locked_polish(
                c_n, V_n, task, element,
                null_angles=info["null_list"],
                rounds=glcp_rounds, gain_floor=glcp_gain_floor,
            )
        elif refine == "ap_lr":
            from .iterative import optimize_beam_ap
            V_n, _delta, c_n, _hist = optimize_beam_ap(
                task=task, element=element, N=N,
                initial_weights=c_n.copy(), initial_delta=0.0,
                K_max=kwargs.get("K_max", 60),
                projection_method=kwargs.get("projection_method", "weighted"),
                w_phase=kwargs.get("w_phase", 1.0), w_amp=kwargs.get("w_amp", 0.5),
                enable_ap=True, enable_local_refinement=True,
                use_pattern_cost=kwargs.get("use_pattern_cost", True),
            )
            info["ap_lr_history"] = _hist
        else:
            raise ValueError(f"Unknown refine mode: {refine!r} (use 'none', 'glcp', or 'ap_lr')")
        return V_n, c_n, info, init

    def ptnr_score(c):
        _bump_eval()                                 # one full forward-model eval
        g = 20 * np.log10(abs(_af(c, u_t, N)) + 1e-12)
        if not nulls:
            return 0.01 * g, g, -np.inf
        # Use the worst hard null for gauge selection.  This is consistent with
        # the certified refinement objective and prevents one weak null from
        # being hidden by averaging several very deep nulls.
        null_db = [20 * np.log10(abs(_af(c, um, N)) + 1e-12) for um in nulls]
        worst_nd = max(null_db)
        return (g - worst_nd) + 0.01 * g, g, worst_nd

    use_ptnr = (gauge == "ptnr") and nulls and (refine in ("none", "glcp"))
    if use_ptnr:
        best = None
        for psi in np.linspace(0, 2 * np.pi, gauge_samples, endpoint=False):
            V_n, c_n, info, init = solve_and_refine([psi])
            sc, _, _ = ptnr_score(c_n)
            if best is None or sc > best[0]:
                best = (sc, V_n, c_n, info, init, psi)
        _, V_n, c_n, info, initial_weights, psi = best
        info["gauge_psi"] = psi
    else:
        V_n, c_n, info, initial_weights = solve_and_refine(None)

    # ---- final certificate for the theorem-backed paths ----
    final_gain_db = 20 * np.log10(abs(_af(c_n, u_t, N)) / N + 1e-12)
    initial_gain_db = 20 * np.log10(abs(_af(initial_weights, u_t, N)) / N + 1e-12)
    final_nulls_db = [
        20 * np.log10(abs(_af(c_n, um, N)) + 1e-12) for um in nulls
    ]
    initial_nulls_lin = np.asarray(
        [abs(_af(initial_weights, um, N)) for um in nulls], dtype=float
    )
    final_nulls_lin = np.asarray(
        [abs(_af(c_n, um, N)) for um in nulls], dtype=float
    )
    if nulls:
        null_preserved = bool(
            np.all(final_nulls_lin <= (1.0 + 1e-10) * initial_nulls_lin)
        )
        final_worst_null_db = float(max(final_nulls_db))
    else:
        null_preserved = True
        final_worst_null_db = -np.inf

    gain_floor_abs = glcp_gain_floor * abs(_af(initial_weights, u_t, N))
    gain_preserved = bool(abs(_af(c_n, u_t, N)) >= gain_floor_abs - 1e-12)

    info["final_gain_db"] = float(final_gain_db)
    info["initial_gain_db"] = float(initial_gain_db)
    info["final_nulls_db"] = final_nulls_db
    info["final_worst_null_db"] = final_worst_null_db
    info["gain_preserved"] = gain_preserved
    info["nulls_preserved"] = null_preserved
    info["certified_refinement"] = bool(refine in ("none", "glcp") and
                                         gain_preserved and null_preserved)

    return {
        "voltages": V_n,
        "weights": c_n,
        "initial_weights": initial_weights,
        "info": info,
        "history": {"residuals": []},
    }



# --------------------------------------------------------------------------- #
# Theory-facing guarantees implemented by this file
# --------------------------------------------------------------------------- #
# 1) Hardware feasibility: every returned c is selected from the finite product
#    manifold M = M_1 x ... x M_N by _retract().
# 2) Certified dual termination: accepted dual states strictly improve the
#    worst-null residual; termination occurs at the requested target, at discrete
#    stagnation, or at a safety cap.
# 3) Certified GLCP invariants: gain is bounded relative to the incoming state and
#    every hard null is preserved relative to its incoming magnitude.  A move is
#    accepted only when the worst-null residual strictly decreases.
#
# These guarantees are intentionally stated for the discrete algorithm rather
# than as a global smooth Gauss-Newton convergence theorem.  The manifold
# retraction is discontinuous, so a classical contraction claim would require
# additional assumptions.


# --------------------------------------------------------------------------- #
# Convergence-based MR-LCMV  (append to coherent_lcmv.py)
# --------------------------------------------------------------------------- #
import math

def _pattern_u(c, u_grid):
    """AF magnitude in dB on a fine u-grid (for SLL detection)."""
    N = len(c)
    n = np.arange(N)[:, None]
    F = c[:, None] * np.exp(1j * np.pi * n * u_grid[None, :])
    return 20 * np.log10(np.abs(F.sum(axis=0)) + 1e-12)


def dual_correction_adaptive(tgt0, A, Ginv, c_grid, V_grid, N,
                             mu_start=None,
                             leak_target=None,
                             max_dual_steps=None,
                             patience=4,
                             track_gain=True,
                             gain_floor=0.9,
                             u_t=0.0):
    if A is None:
        c, V, _ = _retract(tgt0, c_grid, V_grid)
        return c, V, 0.0

    if max_dual_steps is None:
        max_dual_steps = 12 + int(round(2.0 * math.log(max(N, 2))))
    if leak_target is None:
        leak_target = 10 ** (-45.0 / 20.0)

    mu = mu_start if mu_start is not None else (Ginv @ (A.T @ tgt0))
    best_c, best_V, best_leak = None, None, math.inf
    no_improve = 0
    rel_tol = 0.05

    for _ in range(max_dual_steps):
        target = tgt0 - np.conj(A) @ mu
        c, V, _ = _retract(target, c_grid, V_grid)
        leak = A.T @ c
        _bump_eval()
        m = float(np.max(np.abs(leak))) if len(leak) else 0.0

        if track_gain:
            g = abs(np.sum(c * np.exp(1j * np.pi * np.arange(N) * u_t))) / N
            accept = (g >= gain_floor) or (best_c is None)
        else:
            accept = True

        if accept:
            if m < best_leak * (1 - rel_tol):
                best_c, best_V, best_leak = c.copy(), V.copy(), m
                no_improve = 0
            elif m < best_leak:
                best_c, best_V, best_leak = c.copy(), V.copy(), m
                no_improve += 1
            else:
                no_improve += 1
        else:
            no_improve += 1

        if m <= leak_target:
            break
        if no_improve >= patience:
            break
        mu = mu + Ginv @ leak

    if best_c is None:
        best_c, best_V = c, V
    return best_c, best_V, best_leak


def gain_locked_polish_adaptive(c_n, V_n, task, c_grid, V_grid, N,
                                null_angles=None,
                                gain_floor=0.995,
                                max_rounds=None):
    if null_angles is None:
        null_angles = list(getattr(task, "null_angles", []) or [])
    if not null_angles or not getattr(task, "target_angles", None):
        return V_n.copy(), c_n.copy()

    n = np.arange(N)
    u_t = task.target_angles[0]
    a_t = np.exp(1j * np.pi * n * u_t)
    A = np.column_stack([np.exp(1j * np.pi * n * um) for um in null_angles])

    c = c_n.copy().astype(np.complex128)
    V = V_n.copy().astype(float)
    St = np.sum(c * a_t)
    Sm = A.T @ c

    if max_rounds is None:
        max_rounds = N + 8

    for _ in range(max_rounds):
        floor = gain_floor * abs(St)
        moved = 0
        for i in range(N):
            old = c[i]
            candidates_c = c_grid if np.asarray(c_grid).ndim == 1 else c_grid[i]
            candidates_V = V_grid if np.asarray(V_grid).ndim == 1 else V_grid[i]
            dt = (candidates_c - old) * a_t[i]
            St_cand = St + dt
            feasible = np.abs(St_cand) >= floor
            if not np.any(feasible):
                continue
            dm = np.outer(candidates_c - old, A[i])
            Sm_cand = Sm[None, :] + dm
            nullpow = np.sum(np.abs(Sm_cand) ** 2, axis=1)
            nullpow[~feasible] = np.inf
            j = int(np.argmin(nullpow))
            if candidates_c[j] != old:
                c[i] = candidates_c[j]
                V[i] = candidates_V[j]
                St = St_cand[j]
                Sm = Sm_cand[j]
                _bump_glcp()
                moved += 1
        if moved == 0:
            break
    return V, c


def gauge_reachability(score_fn_full, gain_req_db=-0.5,
                       n_first=12, n_basins=2, n_regrid=9):
    psis = np.linspace(0.0, 2 * np.pi, n_first, endpoint=False)
    frontier = []
    for psi in psis:
        r = score_fn_full(psi, light=True)
        r["psi"] = psi
        frontier.append(r)

    def rank(r):
        if r["g_db"] < gain_req_db:
            return -math.inf
        return (r["g_db"] - r["nd_db"]) + 0.01 * r["g_db"]

    ranked = sorted(frontier, key=rank, reverse=True)
    winners = ranked[:n_basins]

    best = None
    for w in winners:
        half = np.pi / n_first
        local = np.linspace(w["psi"] - half, w["psi"] + half, n_regrid)
        for psi in local:
            cand = score_fn_full(psi % (2 * np.pi), light=False)
            cand["psi"] = psi % (2 * np.pi)
            if best is None or rank(cand) > rank(best):
                best = cand
    return best


def oversample_N(N, oversample=8):
    return oversample * N


def solve_coherent_lcmv_convergence(
    task, element, N,
    null_target_db=-45.0,
    gain_floor=0.995,
    suppress_sll=False,
    sll_ceiling_db=None,
    n_sll_virtual=2,
    sll_outer_rounds=2,
    gauge="ptnr",
    gauge_first=None,
    gauge_basins=3,
    ridge=1e-6,
    dual_patience=4,
    dual_gain_floor=0.9,
    verbose=False,
):
    V_grid = np.asarray(element.V_grid)
    c_grid = np.asarray(element.c_grid)
    u_t = task.target_angles[0]
    nulls = list(task.null_angles) if getattr(task, "null_angles", None) else []
    n = np.arange(N)
    s_t = np.exp(-1j * np.pi * n * u_t)
    leak_target = 10 ** (null_target_db / 20.0)

    if sll_ceiling_db is None and getattr(task, "sll_ceiling", None) is not None:
        sll_ceiling_db = 20 * np.log10(task.sll_ceiling * N)

    def build_A(null_list):
        if not null_list:
            return None, None
        A = np.column_stack([np.exp(1j * np.pi * n * um) for um in null_list])
        G = A.T @ np.conj(A)
        G += ridge * np.trace(G).real / max(len(null_list), 1) * np.eye(len(null_list))
        return A, np.linalg.inv(G)

    null_list = list(nulls)
    A, Ginv = build_A(null_list)

    def solve_at(psi, null_list, A, Ginv, light):
        tgt0 = np.exp(1j * psi) * s_t
        c, V, leak = dual_correction_adaptive(
            tgt0, A, Ginv, c_grid, V_grid, N,
            leak_target=leak_target, patience=dual_patience,
            gain_floor=dual_gain_floor, u_t=u_t)
        V, c = gain_locked_polish_adaptive(
            c, V, task, c_grid, V_grid, N,
            null_angles=null_list, gain_floor=gain_floor)
        g_db = 20 * np.log10(abs(_af(c, u_t, N)) / N + 1e-12)
        nd_lin = max(abs(_af(c, um, N)) for um in nulls) if nulls else 0.0
        nd_db = 20 * np.log10(nd_lin + 1e-12)
        return {"g_db": g_db, "nd_db": nd_db, "c": c, "V": V}

    if gauge == "reach":
        if gauge_first is None:
            gf = 28 + int(64 / max(N, 8))
        else:
            gf = gauge_first
        best = gauge_reachability(
            lambda psi, light: solve_at(psi, null_list, A, Ginv, light),
            n_first=gf, n_basins=gauge_basins, n_regrid=7)
    elif gauge == "ptnr":
        gf = gauge_first if gauge_first is not None else 24
        best = None
        for psi in np.linspace(0, 2 * np.pi, gf, endpoint=False):
            r = solve_at(psi, null_list, A, Ginv, light=False)
            sc = (r["g_db"] - r["nd_db"]) + 0.01 * r["g_db"]
            if best is None or sc > best[0]:
                best = (sc, r)
        best = best[1]
    else:
        best = solve_at(0.0, null_list, A, Ginv, light=False)

    c_n, V_n = best["c"], best["V"]
    psi = best.get("psi", 0.0)

    if suppress_sll and sll_ceiling_db is not None:
        u_grid = np.linspace(-1, 1, oversample_N(N))
        for _ in range(sll_outer_rounds):
            pdb = _pattern_u(c_n, u_grid)
            mask = np.abs(u_grid - u_t) > getattr(task, "beamwidth", 0.08)
            for um in null_list:
                mask &= np.abs(u_grid - um) > getattr(task, "beamwidth", 0.08)
            cand = np.where(mask & (pdb > sll_ceiling_db))[0]
            if cand.size == 0:
                break
            worst = cand[np.argsort(pdb[cand])[-n_sll_virtual:]]
            for w in worst:
                uu = float(u_grid[w])
                if all(abs(uu - x) > 1e-3 for x in null_list):
                    null_list.append(uu)
            A, Ginv = build_A(null_list)
            best = solve_at(psi % (2 * np.pi), null_list, A, Ginv, light=False)
            c_n, V_n = best["c"], best["V"]

    info = {
        "gauge_psi": psi,
        "norm_gain_db": 20 * np.log10(abs(_af(c_n, u_t, N)) / N + 1e-12),
        "nulls_db": [20 * np.log10(abs(_af(c_n, um, N)) + 1e-12) for um in nulls],
        "null_list": null_list,
        "mean_abs_c": float(np.mean(np.abs(c_n))),
    }
    return {"voltages": V_n, "weights": c_n, "info": info}


# --------------------------------------------------------------------------- #
# API-compatible wrapper  (same return shape as beamform_mrlcmv)
# --------------------------------------------------------------------------- #
def beamform_mrlcmv_convergence(task, element, N, **kwargs):
    """
    Convergence-based MR-LCMV with the same return signature as beamform_mrlcmv.
    Translates legacy parameter names so the top-level API does not break.
    """
    conv_kwargs = {}

    # Direct pass-through
    for key in ['null_target_db', 'gain_floor', 'suppress_sll', 'sll_ceiling_db',
                'n_sll_virtual', 'sll_outer_rounds', 'gauge', 'gauge_first',
                'gauge_basins', 'ridge', 'dual_patience', 'dual_gain_floor', 'verbose']:
        if key in kwargs:
            conv_kwargs[key] = kwargs[key]

    # Legacy name mappings (from beamform_mrlcmv -> convergence solver)
    if 'glcp_gain_floor' in kwargs and 'gain_floor' not in conv_kwargs:
        conv_kwargs['gain_floor'] = kwargs['glcp_gain_floor']
    if 'gauge_samples' in kwargs and 'gauge_first' not in conv_kwargs:
        conv_kwargs['gauge_first'] = kwargs['gauge_samples']

    result = solve_coherent_lcmv_convergence(task, element, N, **conv_kwargs)
    return {
        'voltages': result['voltages'],
        'weights': result['weights'],
        'initial_weights': result['weights'],
        'info': result.get('info', {}),
        'history': {'residuals': []},
    }

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def save_dual_history(info, path="dual_history.npz"):
    """Save the dual trajectory for later analysis."""
    hist = info.get("history", [])
    if not hist:
        print("No history to save")
        return

    # structured arrays
    step          = np.array([h["step"] for h in hist])
    leak          = np.array([h["leak"] for h in hist])
    best_leak     = np.array([h["best_leak_so_far"] for h in hist])
    null_db       = np.array([h["null_depth_db"] for h in hist])
    best_null_db  = np.array([h["best_null_depth_db"] for h in hist])
    gain_lin      = np.array([h["gain_lin"] for h in hist])
    oracle        = np.array([h["oracle_calls"] for h in hist])
    accepted      = np.array([h["accepted"] for h in hist], dtype=bool)

    np.savez_compressed(
        path,
        step=step,
        leak=leak,
        best_leak=best_leak,
        null_depth_db=null_db,
        best_null_depth_db=best_null_db,
        gain_lin=gain_lin,
        oracle_calls=oracle,
        accepted=accepted,
        dual_status=info.get("dual_status", ""),
        null_target_db=info.get("null_target_db", -45.0),
        final_gain_db=info.get("final_gain_db", np.nan),
    )
    # also a human-readable JSON sidecar
    json_path = Path(path).with_suffix(".json")
    with open(json_path, "w") as f:
        json.dump({
            "dual_status": info.get("dual_status"),
            "null_target_db": info.get("null_target_db"),
            "history": hist,
        }, f, indent=2)
    print(f"Saved → {path}  and  {json_path}")


import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ---------- IEEE double-column style ----------
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "legend.fontsize": 6.5,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.linewidth": 0.6,
    "lines.linewidth": 1.1,
})

def save_and_plot_global(save_prefix="fig_mrlcmv_oracle",
                         target_db=-45.0,
                         ymax_db=0.0,          # never show positive “depth”
                         ymin_db=None):        # auto if None
    if not _GLOBAL_HIST:
        print("No history recorded")
        return

    o     = np.array([h["oracle"] for h in _GLOBAL_HIST])
    leak  = np.array([h["leak"] for h in _GLOBAL_HIST])
    best  = np.array([h["best_leak"] for h in _GLOBAL_HIST])
    ndb   = np.array([h["null_db"] for h in _GLOBAL_HIST])
    bestn = np.array([h["best_null_db"] for h in _GLOBAL_HIST])
    phase = np.array([h["phase"] for h in _GLOBAL_HIST])
    gauge = [h["gauge"] for h in _GLOBAL_HIST]
    acc   = np.array([h["accepted"] for h in _GLOBAL_HIST])

    # clip display values (depth ≤ 0 dB)
    ndb_disp   = np.clip(ndb,   a_min=None, a_max=ymax_db)
    bestn_disp = np.clip(bestn, a_min=None, a_max=ymax_db)

    # ---------- gauge-change locations ----------
    gauge_changes = []
    prev = object()
    for i, g in enumerate(gauge):
        if g is not None and g != prev:
            gauge_changes.append(o[i])
            prev = g

    # ---------- only the steps that *improved* the global best ----------
    # (much cleaner than every accepted star)
    improve_mask = np.zeros_like(acc, dtype=bool)
    running = np.inf
    for i in range(len(best)):
        if acc[i] and best[i] < running - 1e-15:
            improve_mask[i] = True
            running = best[i]

    # =================================================================
    # Figure – null depth (the one you care about)
    # =================================================================
    fig, ax = plt.subplots(figsize=(3.45, 2.15))   # exact double-col width

    # subtle gauge-change lines (behind everything)
    for xc in gauge_changes:
        ax.axvline(xc, color="0.55", ls=":", lw=0.5, zorder=0)

    # current value (light)
    ax.plot(o, ndb_disp, color="0.75", lw=0.6, zorder=1, label="current")

    # best-so-far (the convergence curve)
    ax.plot(o, bestn_disp, color="C0", lw=1.3, zorder=2, label="best so far")

    # only improvement events (open circles – never cover the line)
    if np.any(improve_mask):
        ax.plot(o[improve_mask], bestn_disp[improve_mask],
                "o", ms=3.5, mfc="white", mec="k", mew=0.7,
                zorder=3, label="improvement")

    # target
    ax.axhline(target_db, color="C3", ls="--", lw=0.9, zorder=1,
               label=f"target ({target_db} dB)")

    # final-value annotation (clear, no clutter)
    final_db = bestn_disp[-1]
    ax.annotate(
        f"{final_db:.1f} dB",
        xy=(o[-1], final_db),
        xytext=(-28, 8), textcoords="offset points",
        fontsize=6.5, color="C0",
        arrowprops=dict(arrowstyle="-", color="C0", lw=0.6),
        zorder=4,
    )

    # y-limits
    if ymin_db is None:
        ymin_db = min(bestn_disp.min(), target_db) - 3
    ax.set_ylim(ymin_db, ymax_db + 0.5)
    ax.set_xlim(left=0)

    ax.set_xlabel("Oracle calls")
    ax.set_ylabel("Null depth (dB)")
    ax.yaxis.set_major_locator(ticker.MultipleLocator(10))
    ax.grid(True, which="major", ls=":", lw=0.4, zorder=0)

    # compact legend
    ax.legend(loc="lower right", frameon=True, fancybox=False,
              edgecolor="0.6", framealpha=0.92, borderpad=0.3,
              handlelength=1.4, handletextpad=0.4, labelspacing=0.25)

    fig.tight_layout(pad=0.25)
    fig.savefig(f"{save_prefix}_null_db.pdf")
    fig.savefig(f"{save_prefix}_null_db.png")
    plt.close(fig)

    # ---------- data export ----------
    np.savez_compressed(
        f"{save_prefix}_data.npz",
        oracle=o,
        leak=leak,
        best_leak=best,
        null_db=ndb,
        best_null_db=bestn,
        accepted=acc,
        improvement=improve_mask,
        phase=phase,
        gauge=np.array(gauge, dtype=object),
        gauge_changes=np.array(gauge_changes),
        target_db=target_db,
    )
    summary = {
        "total_oracle_calls": int(o[-1]),
        "final_best_null_db": float(final_db),
        "target_db": float(target_db),
        "n_improvements": int(improve_mask.sum()),
        "n_gauge_changes": len(gauge_changes),
    }
    with open(f"{save_prefix}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved {save_prefix}_null_db.pdf/.png + data + summary")
    print(f"Final null depth: {final_db:.2f} dB   (target {target_db} dB)")
    return summary


def plot_certified_convergence(info, save_prefix="certified_dual"):
    """Two figures: residual (linear) and null depth (dB)."""
    hist = info.get("history", [])
    if not hist:
        print("No dual history")
        return

    steps      = [h["step"] for h in hist]
    leak       = [h["leak"] for h in hist]
    best_leak  = [h["best_leak_so_far"] for h in hist]
    null_db    = [h["null_depth_db"] for h in hist]
    best_ndb   = [h["best_null_depth_db"] for h in hist]
    oracle     = [h["oracle_calls"] for h in hist]
    accepted   = [h["accepted"] for h in hist]
    target_db  = info.get("null_target_db", -45.0)
    target_lin = 10 ** (target_db / 20.0)
    status     = info.get("dual_status", "?")

    # ---------- Figure 1: residual (linear) ----------
    fig1, ax1 = plt.subplots(figsize=(8, 4.5))
    ax1.semilogy(steps, leak, "o-", alpha=0.5, label="current leak")
    ax1.semilogy(steps, best_leak, "s-", label="best accepted")
    acc_s = [s for s, a in zip(steps, accepted) if a]
    acc_b = [b for b, a in zip(best_leak, accepted) if a]
    ax1.semilogy(acc_s, acc_b, "k*", ms=11, label="accepted step")
    ax1.axhline(target_lin, color="r", ls="--", label=f"target ({target_db} dB)")
    ax1.set_xlabel("dual step")
    ax1.set_ylabel(r"$|A^T c|_\infty$ (linear)")
    ax1.set_title(f"Certified dual residual  –  status = {status}")
    ax1.legend(loc="lower left")
    ax1.grid(True, which="both", ls=":")
    fig1.tight_layout()
    fig1.savefig(f"{save_prefix}_residual.png", dpi=150)
    fig1.savefig(f"{save_prefix}_residual.pdf")

    # ---------- Figure 2: null depth (dB) + oracle axis ----------
    fig2, ax2 = plt.subplots(figsize=(8, 4.5))
    ax2.plot(steps, null_db, "o-", alpha=0.5, label="current null depth")
    ax2.plot(steps, best_ndb, "s-", label="best accepted")
    acc_ndb = [b for b, a in zip(best_ndb, accepted) if a]
    ax2.plot(acc_s, acc_ndb, "k*", ms=11, label="accepted step")
    ax2.axhline(target_db, color="r", ls="--", label=f"target ({target_db} dB)")
    ax2.set_xlabel("dual step")
    ax2.set_ylabel("null depth (dB)")
    ax2.set_title(f"Certified null depth  –  status = {status}")
    ax2.legend(loc="lower left")
    ax2.grid(True, ls=":")

    # secondary axis: cumulative oracle calls
    ax2b = ax2.twinx()
    ax2b.plot(steps, oracle, "g--", alpha=0.7, label="oracle calls")
    ax2b.set_ylabel("cumulative oracle calls", color="g")
    ax2b.tick_params(axis="y", labelcolor="g")
    

    fig2.tight_layout()
    fig2.savefig(f"{save_prefix}_null_depth.png", dpi=300)
    fig2.savefig(f"{save_prefix}_null_depth.pdf")

    plt.show()
    print(f"Figures saved as {save_prefix}_*.png/pdf")

if __name__ == "__main__":
    reset_counters()
    N=64
    target_u = np.random.uniform(-0.5, 0.5)
    null_u = np.random.uniform(-0.8, 0.8)
    while abs(null_u - target_u) < 0.15:
        null_u = np.random.uniform(-0.8, 0.8)
    from .iterative import Task
    from .element_model import MeasuredVaractor, SyntheticVaractor
    element= SyntheticVaractor(beta=0.8, folding=True)
    task = Task(type='nulled', target_angles=[target_u], null_angles=[null_u])
    out = beamform_mrlcmv(task, element, N, refine="glcp",  # or "none"
                        null_target_db=-80, return_history=True)  # if you expose it
    info = out["info"]
    print("status:", info["dual_status"],
        "final residual:", info["dual_residual_lin"],
        "evals:", get_counters())
    print(info)
        # save data
    # save_dual_history(info, path="certified_dual_history.npz")

    # # produce both plots
    # plot_certified_convergence(info, save_prefix="certified_dual")
    print("Final counters:", get_counters())
    # plot_global_convergence(save_prefix="certified_global")
    summary = save_and_plot_global( save_prefix="fig_mrlcmv_oracle",
    target_db=-60.0,
    ymax_db=0.0,
    ymin_db=-70
    )
    print(summary)
# --------------------------------------------------------------------------- #
# Theory-facing guarantees implemented by this file
# --------------------------------------------------------------------------- #
# The certified path (`beamform_mrlcmv`) uses a finite-manifold, monotone
# residual acceptance rule. The adaptive path (`beamform_mrlcmv_convergence`)
# remains available for benchmark comparison and uses residual/stagnation-based
# stopping with a safety cap.
#
# The theoretical guarantee is deliberately attached to the certified path:
# hardware-manifold feasibility, finite termination of accepted strict-descent
# moves, hard-null preservation during GLCP, and a fixed global gain floor.
# The adaptive path is an engineering variant and is not required to satisfy
# the theorem's strict-descent acceptance rule.
