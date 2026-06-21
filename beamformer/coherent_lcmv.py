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
# Manifold access helpers
# --------------------------------------------------------------------------- #
def _get_manifold(element):
    """Return (V_grid, c_grid) for fast vectorized retraction."""
    V = np.asarray(element.V_grid)
    c = np.asarray(element.c_grid)
    return V, c


def _retract(target, c_grid, V_grid):
    """
    Per-element manifold retraction by inner-product maximization.

        c_n = argmax_{c in grid} Re( c * conj(target_n) )

    target : (N,) complex desired field
    returns (c_sel (N,), V_sel (N,), idx (N,))
    """
    # score[n, g] = Re( c_grid[g] * conj(target[n]) )
    score = np.real(np.conj(target)[:, None] * c_grid[None, :])
    idx = np.argmax(score, axis=1)
    return c_grid[idx], V_grid[idx], idx


def _af(c, u, N):
    """Array factor AF(u) = sum_n c_n exp(+j pi n u)."""
    return np.sum(c * np.exp(1j * np.pi * np.arange(N) * u))


# --------------------------------------------------------------------------- #
# Core solver
# --------------------------------------------------------------------------- #
def solve_coherent_lcmv(
    task,
    element,
    N,
    n_dual_steps=8,
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
):
    """
    Solve the coherent gain-max / null-constrained problem on the hardware manifold.

    Parameters
    ----------
    task     : Task (uses target_angles[0] as main beam, null_angles as hard nulls)
    element  : ElementData with .V_grid / .c_grid
    N        : number of elements
    n_dual_steps : Gauss-Newton dual corrections per gauge (few-shot; 5-10 plenty)
    psi_samples  : resolution of the 1-D global-gauge (insertion-loss) sweep
    ridge        : Tikhonov term for the MxM Gram solve
    null_target_db : absolute null level (dB) used only for gauge scoring
    suppress_sll : if True, iteratively add worst sidelobe peaks as virtual nulls
    sll_ceiling_db : absolute SLL ceiling (dB). Defaults from task.sll_ceiling.
    n_sll_virtual  : sidelobe peaks added per SLL round
    sll_outer_rounds : active-set rounds for SLL suppression

    Returns
    -------
    V_n, c_n  (and info dict if return_history)
    """
    V_grid, c_grid = _get_manifold(element)
    u_t = task.target_angles[0]
    nulls = list(task.null_angles) if getattr(task, "null_angles", None) else []

    n = np.arange(N)
    s_t = np.exp(-1j * np.pi * n * u_t)            # matched target weight, AF(u_t)=N when c=s_t
    null_target_lin = 10 ** (null_target_db / 20.0)

    if sll_ceiling_db is None and getattr(task, "sll_ceiling", None) is not None:
        # task.sll_ceiling is relative to N; convert to absolute dB
        sll_ceiling_db = 20 * np.log10(task.sll_ceiling * N)

    def build_A(null_list):
        if not null_list:
            return None, None
        A = np.column_stack([np.exp(1j * np.pi * n * um) for um in null_list])  # (N, M)
        G = A.T @ np.conj(A)                                                    # (M, M)
        G += ridge * np.trace(G).real / max(len(null_list), 1) * np.eye(len(null_list))
        return A, np.linalg.inv(G)

    def solve_for_gauge(psi, null_list, A, Ginv):
        """Anchor + dual correction at a fixed gauge psi. Returns (c, V, leak)."""
        tgt0 = np.exp(1j * psi) * s_t
        if A is None:
            c, V, _ = _retract(tgt0, c_grid, V_grid)
            return c, V, 0.0
        # closed-form LCMV dual (optimal for the continuous problem) as warm start
        mu = Ginv @ (A.T @ tgt0)
        best = None
        for _ in range(n_dual_steps):
            target = tgt0 - np.conj(A) @ mu
            c, V, _ = _retract(target, c_grid, V_grid)
            leak = A.T @ c                          # (M,) null leakage
            score = max(abs(leak)) if len(leak) else 0.0
            if best is None or score < best[2]:
                best = (c.copy(), V.copy(), score)
            mu = mu + Ginv @ leak                    # Gauss-Newton dual step
        return best

    def score_solution(c):
        g_db = 20 * np.log10(abs(_af(c, u_t, N)) / N + 1e-12)
        pen = 0.0
        for um in nulls:
            nd = 20 * np.log10(abs(_af(c, um, N)) + 1e-12)
            pen += max(0.0, nd - (20 * np.log10(N) + null_target_db))
        return g_db - 0.5 * pen, g_db

    # ----- main: sweep the global gauge, keep the best gain-feasible solution -----
    null_list = list(nulls)
    A, Ginv = build_A(null_list)

    best_pack = None  # (score, g_db, c, V)
    # psi_grid lets an external offset-tuning algorithm drive the global-gauge
    # choice (pass a single value or a short candidate list). Defaults to a
    # uniform sweep over the full circle.
    if psi_grid is None:
        psis = np.linspace(0, 2 * np.pi, psi_samples, endpoint=False)
    else:
        psis = np.atleast_1d(np.asarray(psi_grid, dtype=float))
    for psi in psis:
        c, V, _ = solve_for_gauge(psi, null_list, A, Ginv)
        sc, g_db = score_solution(c)
        if best_pack is None or sc > best_pack[0]:
            best_pack = (sc, g_db, c, V, psi)

    # ----- optional SLL suppression via virtual-null active set -----
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
            # add the worst offending sidelobe peaks as virtual nulls
            worst = cand[np.argsort(pdb[cand])[-n_sll_virtual:]]
            for w in worst:
                uu = float(u_grid[w])
                if all(abs(uu - x) > 1e-3 for x in null_list):
                    null_list.append(uu)
            A, Ginv = build_A(null_list)
            # re-solve at the chosen gauge (and a small neighborhood)
            local_best = None
            for dp in np.linspace(-0.3, 0.3, 7):
                c2, V2, _ = solve_for_gauge((psi + dp) % (2 * np.pi), null_list, A, Ginv)
                sc2, g2 = score_solution(c2)
                if local_best is None or sc2 > local_best[0]:
                    local_best = (sc2, g2, c2, V2, (psi + dp) % (2 * np.pi))
            best_pack = local_best

    sc, g_db, c_n, V_n, psi = best_pack

    if return_history:
        info = {
            "psi": psi,
            "norm_gain_db": g_db,
            "nulls_db": [20 * np.log10(abs(_af(c_n, um, N)) + 1e-12) for um in nulls],
            "null_list": null_list,
            "mean_abs_c": float(np.mean(np.abs(c_n))),
        }
        return V_n, c_n, info
    return V_n, c_n


# --------------------------------------------------------------------------- #
# Gain-Locked Coordinate Polish (GLCP) -- the novel local refinement
# --------------------------------------------------------------------------- #
def gain_locked_polish(
    c_n,
    V_n,
    task,
    element,
    null_angles=None,
    rounds=6,
    gain_floor=0.995,
):
    """
    Greedy coordinate descent on the manifold *grid* that deepens nulls while the
    main-beam gain is held by a HARD constraint.

    For each element i we choose the grid point that minimizes total null power
    Sum_m |c^T a(u_m)|^2 subject to the coherent main-beam magnitude
    |c^T a(u_t)| staying >= gain_floor * (current). This is the key difference
    from `local_refinement.refine_local_active_set`, which uses a *soft* PTNR
    penalty and lets gain drift; here gain cannot drop below the floor, so the
    optimizer spends all its freedom on the nulls.

    Uses incremental running sums (St for the target, Sm for the nulls) so each
    element trial is O(G + G*M); total O(rounds * N * G * M). No FFT.

    null_angles : list of null directions to suppress (defaults to task.null_angles;
                  pass the SLL-augmented active set to also push down sidelobes).
    Returns (V_refined, c_refined).
    """
    V_grid, c_grid = _get_manifold(element)
    N = len(c_n)
    if null_angles is None:
        null_angles = list(getattr(task, "null_angles", []) or [])
    if not null_angles or not task.target_angles:
        return V_n.copy(), c_n.copy()

    n = np.arange(N)
    u_t = task.target_angles[0]
    a_t = np.exp(1j * np.pi * n * u_t)                                   # (N,)
    A = np.column_stack([np.exp(1j * np.pi * n * um) for um in null_angles])  # (N, M)

    c = c_n.copy().astype(np.complex128)
    V = V_n.copy().astype(float)
    St = np.sum(c * a_t)                       # running main-beam sum
    Sm = A.T @ c                               # running null sums (M,)

    for _ in range(rounds):
        floor = gain_floor * abs(St)
        for i in range(N):
            old = c[i]
            dt = (c_grid - old) * a_t[i]                     # (G,) delta to St
            St_cand = St + dt
            feasible = np.abs(St_cand) >= floor              # gain lock
            if not np.any(feasible):
                continue
            dm = np.outer(c_grid - old, A[i])                # (G, M) delta to Sm
            Sm_cand = Sm[None, :] + dm
            nullpow = np.sum(np.abs(Sm_cand) ** 2, axis=1)
            nullpow[~feasible] = np.inf
            j = int(np.argmin(nullpow))
            if c_grid[j] != old:
                c[i] = c_grid[j]
                V[i] = V_grid[j]
                St = St_cand[j]
                Sm = Sm_cand[j]
    return V, c


# --------------------------------------------------------------------------- #
# Orchestrator: three selectable refinement modes
# --------------------------------------------------------------------------- #
def beamform_mrlcmv(
    task,
    element,
    N,
    refine="glcp",
    gauge="ptnr",
    gauge_samples=24,
    suppress_sll=False,
    glcp_rounds=6,
    glcp_gain_floor=0.995,
    **kwargs
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
        n_dual_steps=kwargs.get("n_dual_steps", 8),
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
        g = 20 * np.log10(abs(_af(c, u_t, N)) + 1e-12)
        nd = np.mean([20 * np.log10(abs(_af(c, um, N)) + 1e-12) for um in nulls])
        return (g - nd) + 0.01 * g  # PTNR, tie-broken toward higher gain

    use_ptnr = (gauge == "ptnr") and nulls and (refine in ("none", "glcp"))
    if use_ptnr:
        best = None
        for psi in np.linspace(0, 2 * np.pi, gauge_samples, endpoint=False):
            V_n, c_n, info, init = solve_and_refine([psi])
            sc = ptnr_score(c_n)
            if best is None or sc > best[0]:
                best = (sc, V_n, c_n, info, init, psi)
        _, V_n, c_n, info, initial_weights, psi = best
        info["gauge_psi"] = psi
    else:
        V_n, c_n, info, initial_weights = solve_and_refine(None)

    return {
        "voltages": V_n,
        "weights": c_n,
        "initial_weights": initial_weights,
        "info": info,
        "history": {"residuals": []},
    }
