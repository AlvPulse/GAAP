"""
Unified, fair baselines for phase-only nullforming on a measured/lossy manifold.

Every solver minimizes the SAME scalar objective over the SAME decision variable
(element control voltages V on the real hardware manifold) and is scored by the
SAME u-space metrics. This makes the comparison about the *optimizer's ability to
exploit the hardware manifold*, not about convention or objective mismatches.

Families implemented (one modern representative each):
  Local            : PGD, Coordinate Descent, Trust-Region / Levenberg-Marquardt
  Manifold         : Riemannian Conjugate Gradient (unit-modulus, exact penalty)
  Splitting        : Scaled ADMM (unimodular) + manifold projection
  Global / DFO     : Simulated Annealing, Basin Hopping, Differential Evolution,
                     CMA-ES, Cross-Entropy Method
  Quantum-inspired : Simulated Bifurcation (ballistic bSB)
  Hardware-aware   : Perturbation-based nulling (measured manifold)
  Prior proposed   : OBH-ZKD
  Ours             : MR-LCMV (+GLCP)

Manifold-design methods (RCG, ADMM) optimize ideal unit-modulus phases and are
then projected onto the hardware manifold -- by construction they cannot exploit
the amplitude-phase coupling, which is exactly the contrast of interest.

Objective (minimize), with AF(u) = sum_n c_n exp(+j pi n u), c = element manifold:
    J(V) = mean_m |AF(u_m)|^2 / N^2  +  rho * max(0, g_floor - |AF(u_t)|/N)^2
i.e. minimize null power subject to a soft floor on coherent main-beam gain.
"""
import time
import numpy as np
from scipy.optimize import least_squares, differential_evolution, basinhopping, dual_annealing

from .controllers.metrics import get_metrics


# --------------------------------------------------------------------------- #
# Problem definition (shared objective + bookkeeping)
# --------------------------------------------------------------------------- #
class Problem:
    """Shared objective + bookkeeping.

    discrete=True (default) snaps control voltages onto the measured voltage grid
    before evaluating weights -- i.e. the element is a *measured lookup table*, the
    faithful model of real hardware (MeasuredVaractor does exactly this). Continuous
    gradient / least-squares methods then cannot finite-difference below the
    quantization step, which is the honest constraint on a measured device.

    discrete=False treats SyntheticVaractor's analytic c(V) as smooth/differentiable
    -- an optimistic upper bound for gradient methods that assumes a perfect
    differentiable hardware model (you never have this from measurements).
    """
    def __init__(self, task, element, N, gain_floor_frac=0.85, rho=50.0, discrete=True):
        self.discrete = discrete
        self.task, self.element, self.N = task, element, N
        self.n = np.arange(N)
        self.u_t = task.target_angles[0]
        self.nulls = list(task.null_angles) if getattr(task, "null_angles", None) else []
        self.a_t = np.exp(1j * np.pi * self.n * self.u_t)
        self.A = (np.column_stack([np.exp(1j * np.pi * self.n * um) for um in self.nulls])
                  if self.nulls else np.zeros((N, 0), complex))
        self.vmin = getattr(element, "v_min", 0.0)
        self.vmax = getattr(element, "v_max", 2 * np.pi)
        self.g_ceiling = self._ceiling()            # |AF|/N upper bound (no nulls)
        self.g_floor = gain_floor_frac * self.g_ceiling
        self.rho = rho
        self.eval_count = 0

    def _ceiling(self):
        cg = self.element.c_grid
        s_t = np.conj(self.a_t)
        best = 0.0
        for psi in np.linspace(0, 2 * np.pi, 181):
            tgt = np.exp(1j * psi) * s_t
            c = cg[np.argmax(np.real(np.conj(tgt)[:, None] * cg[None, :]), axis=1)]
            best = max(best, abs(c @ self.a_t) / self.N)
        return best

    def weights(self, V):
        V = np.clip(V, self.vmin, self.vmax)
        if self.discrete:
            Vg = self.element.V_grid
            idx = np.clip(np.searchsorted(Vg, V), 0, len(Vg) - 1)
            return self.element.c_grid[idx]
        return self.element.get_complex_weight(V)

    def cost(self, V):
        self.eval_count += 1
        c = self.weights(V)
        nulls = np.mean(np.abs(self.A.T @ c) ** 2) / self.N ** 2 if self.nulls else 0.0
        at = abs(c @ self.a_t) / self.N
        pen = max(0.0, self.g_floor - at) ** 2
        return nulls + self.rho * pen

    def metrics(self, c):
        """Return dict of u-space metrics for achieved complex weights c."""
        nd, ng, ptnr = get_metrics(c, self.task, self.element)
        if self.nulls:
            nulls_db = [20 * np.log10(abs(c @ np.exp(1j * np.pi * self.n * um)) + 1e-12)
                        for um in self.nulls]
            worst, avg = max(nulls_db), float(np.mean(nulls_db))
        else:
            worst = avg = -np.inf
        return dict(gain=ng, worst_null=worst, avg_null=avg,
                    ptnr=ng - worst if self.nulls else np.inf,
                    gain_loss=self.g_ceiling_db() - ng)

    def g_ceiling_db(self):
        return 20 * np.log10(self.g_ceiling + 1e-12)


def _warm_V(prob, rng=None):
    """Matched-filter manifold projection (shared warm start)."""
    s_t = np.conj(prob.a_t)
    cg, Vg = prob.element.c_grid, prob.element.V_grid
    idx = np.argmax(np.real(np.conj(s_t)[:, None] * cg[None, :]), axis=1)
    V = Vg[idx].astype(float)
    if rng is not None:
        V = np.clip(V + rng.normal(0, 0.1 * (prob.vmax - prob.vmin), prob.N), prob.vmin, prob.vmax)
    return V


# --------------------------------------------------------------------------- #
# Tier I -- Local continuous
# --------------------------------------------------------------------------- #
def solve_pgd(prob, rng, max_iter=400, lr=2.0, eps=1e-3):
    V = _warm_V(prob, rng)
    for _ in range(max_iter):
        J0 = prob.cost(V)
        g = np.empty(prob.N)
        for i in range(prob.N):
            Vp = V.copy(); Vp[i] = np.clip(Vp[i] + eps, prob.vmin, prob.vmax)
            g[i] = (prob.cost(Vp) - J0) / eps
        Vn = np.clip(V - lr * g, prob.vmin, prob.vmax)
        if np.max(np.abs(Vn - V)) < 1e-5:
            break
        V = Vn
    return V


def solve_cd(prob, rng, sweeps=8, grid=64):
    V = _warm_V(prob, rng)
    vs = np.linspace(prob.vmin, prob.vmax, grid)
    for _ in range(sweeps):
        improved = False
        for i in range(prob.N):
            best_v, best_J = V[i], prob.cost(V)
            for vt in vs:
                Vt = V.copy(); Vt[i] = vt
                J = prob.cost(Vt)
                if J < best_J:
                    best_v, best_J, improved = vt, J, True
            V[i] = best_v
        if not improved:
            break
    return V


def solve_trust_region(prob, rng, max_nfev=300):
    """Levenberg-Marquardt / trust-region on a residual form of the objective."""
    V0 = _warm_V(prob, rng)
    sqrt_rho = np.sqrt(prob.rho)

    def resid(V):
        prob.eval_count += 1
        c = prob.weights(V)
        r = list((prob.A.T @ c).real / prob.N) + list((prob.A.T @ c).imag / prob.N)
        at = abs(c @ prob.a_t) / prob.N
        r.append(sqrt_rho * max(0.0, prob.g_floor - at))
        return np.array(r)

    res = least_squares(resid, V0, bounds=(prob.vmin, prob.vmax),
                        method="trf", max_nfev=max_nfev)
    return res.x


# --------------------------------------------------------------------------- #
# Tier II -- Manifold optimization (ideal phases, then project)
# --------------------------------------------------------------------------- #
def _phase_cost_grad(prob, phi):
    """f(phi) and analytic grad for unit-modulus c=e^{j phi}, using the SAME
    penalized objective as Problem.cost so manifold methods are directly comparable:
        f = mean_m |c.a_m|^2/N^2 + rho*max(0, g_floor - |c.a_t|/N)^2
    """
    c = np.exp(1j * phi)
    f = 0.0
    g = np.zeros(prob.N)
    M = prob.A.shape[1]
    for m in range(M):
        a = prob.A[:, m]
        S = c @ a
        f += (abs(S) ** 2) / prob.N ** 2
        g += 2 * np.real(np.conj(S) * (1j * c * a)) / prob.N ** 2
    if M:
        f /= M; g /= M
    St = c @ prob.a_t
    at = abs(St) / prob.N
    if at < prob.g_floor:
        slack = prob.g_floor - at
        f += prob.rho * slack ** 2
        # d at/dphi = (1/N) Re(conj(St)/|St| * j c a_t)
        dat = np.real(np.conj(St) / (abs(St) + 1e-12) * (1j * c * prob.a_t)) / prob.N
        g += prob.rho * 2 * slack * (-dat)
    return f, g


def solve_rcg(prob, rng, max_iter=300):
    """Riemannian Conjugate Gradient on the unit-modulus manifold (torus angle coords).
    Minimizes the shared penalized objective over ideal phases, then projects to hardware."""
    s_t = np.conj(prob.a_t)
    phi = np.angle(np.exp(1j * (rng.uniform(0, 2 * np.pi)) ) * s_t)  # matched-filter phase + random gauge
    f, g = _phase_cost_grad(prob, phi); prob.eval_count += 1
    d = -g
    for _ in range(max_iter):
        # backtracking line search along d
        t = 0.5
        for _ls in range(20):
            phi_new = phi + t * d
            f_new, g_new = _phase_cost_grad(prob, phi_new); prob.eval_count += 1
            if f_new < f - 1e-4 * t * (g @ g):
                break
            t *= 0.5
        if np.max(np.abs(t * d)) < 1e-6:
            break
        beta = max(0.0, (g_new @ (g_new - g)) / (g @ g + 1e-12))  # Polak-Ribiere+
        phi, f, d, g = phi_new, f_new, -g_new + beta * d, g_new
    # project ideal phases to hardware manifold
    c_ideal = np.exp(1j * phi)
    cg, Vg = prob.element.c_grid, prob.element.V_grid
    idx = np.argmax(np.real(np.conj(c_ideal)[:, None] * cg[None, :]), axis=1)
    return Vg[idx].astype(float)


def solve_admm(prob, rng, max_iter=300, rho=1.0, gain_w=None):
    """Scaled ADMM for unimodular nulling, then manifold projection.

    Minimizes  w^H (A A^H) w  +  gain_w*||w - s_t||^2   s.t.  |w_n| = 1,
    where s_t is the matched filter. The anchor to s_t keeps the main beam
    pointed (without it the unimodular minimizer of pure null power collapses
    gain). gain_w defaults to #nulls to balance against the null gram diagonal.
    """
    N = prob.N
    Agram = prob.A @ prob.A.conj().T                  # null subspace gram (diag ~ #nulls)
    s_t = np.conj(prob.a_t)                            # matched filter (unit modulus)
    M = max(prob.A.shape[1], 1)
    gw = float(M) if gain_w is None else gain_w
    w = s_t.copy()                                     # warm start at matched filter
    u = w / (np.abs(w) + 1e-12)
    lam = np.zeros(N, complex)
    Mfac = np.linalg.inv(2 * Agram + (2 * gw + rho) * np.eye(N))
    for _ in range(max_iter):
        w = Mfac @ (2 * gw * s_t + rho * u - lam)      # quadratic w-update
        z = w + lam / rho
        u = z / (np.abs(z) + 1e-12)                    # unimodular projection
        lam = lam + rho * (w - u)
        prob.eval_count += 1
    cg, Vg = prob.element.c_grid, prob.element.V_grid
    idx = np.argmax(np.real(np.conj(u)[:, None] * cg[None, :]), axis=1)
    return Vg[idx].astype(float)


# --------------------------------------------------------------------------- #
# Tier IV -- Global / derivative-free
# --------------------------------------------------------------------------- #
def solve_sa(prob, rng, max_iter=400):
    res = dual_annealing(prob.cost, bounds=[(prob.vmin, prob.vmax)] * prob.N,
                         maxiter=max_iter, seed=int(rng.integers(1 << 30)),
                         x0=_warm_V(prob, rng))
    return res.x


def solve_basinhopping(prob, rng, niter=40):
    V0 = _warm_V(prob, rng)
    step = 0.15 * (prob.vmax - prob.vmin)

    def accept(**kw):
        return None  # use default Metropolis
    res = basinhopping(prob.cost, V0, niter=niter,
                       minimizer_kwargs=dict(method="L-BFGS-B",
                                             bounds=[(prob.vmin, prob.vmax)] * prob.N),
                       stepsize=step, seed=int(rng.integers(1 << 30)))
    return res.x


def solve_de(prob, rng, maxiter=120, popsize=15):
    res = differential_evolution(prob.cost, bounds=[(prob.vmin, prob.vmax)] * prob.N,
                                 maxiter=maxiter, popsize=popsize, tol=1e-8,
                                 seed=int(rng.integers(1 << 30)), polish=True, init="sobol")
    return res.x


def solve_cmaes(prob, rng, max_iter=300):
    import cma
    V0 = _warm_V(prob, rng)
    sigma0 = (prob.vmax - prob.vmin) / 4.0
    es = cma.CMAEvolutionStrategy(
        V0, sigma0,
        {"bounds": [prob.vmin, prob.vmax], "maxiter": max_iter, "verbose": -9,
         "seed": int(rng.integers(1 << 30))})
    es.optimize(prob.cost)
    return np.array(es.result.xbest)


def solve_cem(prob, rng, iters=80, pop=40, elite_frac=0.2):
    mu = _warm_V(prob, rng)
    sigma = np.full(prob.N, 0.3 * (prob.vmax - prob.vmin))
    n_elite = max(2, int(pop * elite_frac))
    best_V, best_J = mu.copy(), prob.cost(mu)
    for _ in range(iters):
        S = rng.normal(mu, sigma, size=(pop, prob.N))
        S = np.clip(S, prob.vmin, prob.vmax)
        J = np.array([prob.cost(s) for s in S])
        elite = S[np.argsort(J)[:n_elite]]
        mu, sigma = elite.mean(0), elite.std(0) + 1e-3
        if J.min() < best_J:
            best_J, best_V = J.min(), S[np.argmin(J)].copy()
    return best_V


# --------------------------------------------------------------------------- #
# Tier V -- Quantum-inspired (ballistic Simulated Bifurcation)
# --------------------------------------------------------------------------- #
def solve_sb(prob, rng, steps=400, dt=0.5, c0=0.5):
    """Ballistic SB dynamics over phase variables, then manifold projection."""
    x = rng.uniform(-0.1, 0.1, prob.N)                 # positions ~ phase offsets
    y = rng.uniform(-0.1, 0.1, prob.N)                 # momenta
    s_t = np.conj(prob.a_t)
    base = np.angle(s_t)
    for k in range(steps):
        a_t_pump = k / steps                           # 0 -> 1 bifurcation schedule
        _, g = _phase_cost_grad(prob, base + x); prob.eval_count += 1
        y += dt * (-(1.0 - a_t_pump) * x - c0 * g)
        x += dt * y
        x = np.clip(x, -np.pi, np.pi)
    c_ideal = np.exp(1j * (base + x))
    cg, Vg = prob.element.c_grid, prob.element.V_grid
    idx = np.argmax(np.real(np.conj(c_ideal)[:, None] * cg[None, :]), axis=1)
    return Vg[idx].astype(float)


# --------------------------------------------------------------------------- #
# Tier VI -- Hardware-aware perturbation nulling (measured manifold)
# --------------------------------------------------------------------------- #
def solve_perturbation(prob, rng, rounds=12, n_probe=3):
    """Sequential small-perturbation nulling on the measured manifold with a gain guard."""
    V = _warm_V(prob, rng)
    cg, Vg = prob.element.c_grid, prob.element.V_grid
    step = 0.3 * (prob.vmax - prob.vmin)
    for _ in range(rounds):
        for i in range(prob.N):
            J0 = prob.cost(V)
            best_v, best_J = V[i], J0
            for s in rng.uniform(-step, step, n_probe):
                Vt = V.copy(); Vt[i] = np.clip(V[i] + s, prob.vmin, prob.vmax)
                J = prob.cost(Vt)
                if J < best_J:
                    best_v, best_J = Vt[i], J
            V[i] = best_v
        step *= 0.7
    return V


# --------------------------------------------------------------------------- #
# Tier VII -- Semidefinite Relaxation (SDR) with Gaussian randomization
# --------------------------------------------------------------------------- #
def solve_sdr(prob, rng, n_rand=64):
    """Semidefinite-relaxation beamformer for unimodular nulling, then manifold
    projection -- a standard strong baseline for phase-only / constant-modulus
    array problems.

    The QCQP  max |w^H a_t|^2  s.t.  w^H R_n w <= 1, |w_n| = 1  relaxes to an SDP
    in W = w w^H; its solution is rank-1 in the noise-free single-target case and
    is then exactly the MVDR/Capon direction  w_cont ∝ R_n^{-1} a_t  (R_n = the
    null-direction covariance). We form that rank-1 SDR solution and recover a
    unimodular vector by Gaussian randomization (sampling phases consistent with
    W = w_cont w_cont^H), keeping the best draw under the shared objective, then
    project onto the hardware manifold. No external SDP solver required.
    """
    s_t = np.conj(prob.a_t)                            # matched filter (max-gain weight)
    if prob.A.shape[1] > 0:                            # project off the null subspace (= rank-1 SDR opt.)
        A = prob.A
        G = A.T @ np.conj(A)
        G += 1e-6 * np.trace(G).real / max(A.shape[1], 1) * np.eye(A.shape[1])
        P = np.eye(prob.N) - np.conj(A) @ np.linalg.solve(G, A.T)
        w_cont = P @ s_t                               # MVDR/LCMV continuous optimum
    else:
        w_cont = s_t.copy()
    cg, Vg = prob.element.c_grid, prob.element.V_grid
    base = np.angle(w_cont)

    def project(phases):                               # unimodular -> hardware manifold
        c_ideal = np.exp(1j * phases)
        idx = np.argmax(np.real(np.conj(c_ideal)[:, None] * cg[None, :]), axis=1)
        return Vg[idx].astype(float)

    best_V = project(base)
    best_J = prob.cost(best_V)
    for _ in range(n_rand):                            # Gaussian randomization of the rank-1 SDR sol.
        V = project(base + rng.normal(0.0, 0.3, prob.N))
        J = prob.cost(V)
        if J < best_J:
            best_V, best_J = V, J
    return best_V


# --------------------------------------------------------------------------- #
# Prior proposed -- OBH-ZKD (orbital basin hopping + zero-knowledge descent)
# --------------------------------------------------------------------------- #
def solve_obh_zkd(prob, rng, K=8, max_hops=10, T=0.1):
    dV_per_rad = (prob.vmax - prob.vmin) / (2 * np.pi)
    orbitals = []
    for k in range(K):
        dV = (k * np.deg2rad(45)) * dV_per_rad
        minJ = np.inf
        for _ in range(8):
            Vr = np.clip(rng.uniform(prob.vmin, prob.vmax, prob.N) + dV, prob.vmin, prob.vmax)
            minJ = min(minJ, prob.cost(Vr))
        orbitals.append((dV, minJ))
    orbitals.sort(key=lambda t: t[1])
    Vc, Jc, Vb, Jb = None, np.inf, None, np.inf
    for dV, _ in orbitals[:max_hops]:
        V = np.clip(rng.uniform(prob.vmin, prob.vmax, prob.N) + dV, prob.vmin, prob.vmax)
        # zero-knowledge coordinate descent
        delta = 0.5
        for _ in range(10):
            improved = False
            for i in range(prob.N):
                J0 = prob.cost(V)
                for sgn in (+1, -1):
                    Vt = V.copy(); Vt[i] = np.clip(V[i] + sgn * delta, prob.vmin, prob.vmax)
                    if prob.cost(Vt) < J0:
                        V[i] = Vt[i]; improved = True; break
            if not improved:
                delta *= 0.5
                if delta < 1e-3:
                    break
        J = prob.cost(V)
        if Vc is None or J < Jc or rng.random() < np.exp(-(J - Jc) / T):
            Vc, Jc = V.copy(), J
        if Jc < Jb:
            Vb, Jb = Vc.copy(), Jc
    return Vb


# --------------------------------------------------------------------------- #
# Ours
# --------------------------------------------------------------------------- #
def solve_ours(prob, rng, refine="glcp", gauge="ptnr", n_dual_steps=1):
    from .api import beamform
    from . import coherent_lcmv as CL
    CL.reset_counters()
    res = beamform(prob.task, prob.element, N=prob.N, solver="mrlcmv",n_dual_steps= n_dual_steps,
                   refine=refine, gauge=gauge)
    # Report the same cost currency as the iterative/DFO baselines: number of
    # full forward-model (array-factor) evaluations. GLCP's committed coordinate
    # moves use O(M) incremental updates (no full evaluation), so they are tracked
    # separately on the Problem and surfaced as a distinct column.
    full_evals, glcp_updates = CL.get_counters()
    prob.eval_count = full_evals
    prob.glcp_updates = glcp_updates
    return res["voltages"]


def solve_ours_none(prob, rng, gauge="ptnr"):
    """Bare MR-LCMV (closed-form anchor + dual + gauge), no GLCP polish."""
    return solve_ours(prob, rng, refine="none", gauge=gauge)

def solve_ours_certified(prob, rng, refine="glcp", gauge="ptnr", null_target=-60):
    """
    Adaptive/convergence MR-LCMV adapter for the SOTA benchmark.

    The benchmark interface is intentionally identical to all other solvers:
        solve_xxx(prob, rng, ...)

    rng is accepted for API compatibility and reproducibility. The MR-LCMV
    algorithm itself is deterministic for a fixed task/manifold.
    """
    from .api import beamform
    from . import MR_LCMV_certified as CL

    CL.reset_counters()

    res = beamform(
        prob.task,
        prob.element,
        N=prob.N,
        solver="mrlcmv-certified",
        refine=refine,
        gauge=gauge,
        null_target_db= null_target
    )

    full_evals, glcp_updates = CL.get_counters()

    prob.eval_count = full_evals
    prob.glcp_updates = glcp_updates

    return res["voltages"]

def solve_ours_adaptive(prob, rng, refine="glcp", gauge="ptnr"):
    """
    Adaptive/convergence MR-LCMV adapter for the SOTA benchmark.

    The benchmark interface is intentionally identical to all other solvers:
        solve_xxx(prob, rng, ...)

    rng is accepted for API compatibility and reproducibility. The MR-LCMV
    algorithm itself is deterministic for a fixed task/manifold.
    """
    from .api import beamform
    from . import MR_LCMV_certified as CL

    CL.reset_counters()

    res = beamform(
        prob.task,
        prob.element,
        N=prob.N,
        solver="mrlcmv-adaptive",
        refine=refine,
        gauge=gauge,
    )

    full_evals, glcp_updates = CL.get_counters()

    prob.eval_count = full_evals
    prob.glcp_updates = glcp_updates

    return res["voltages"]


# Registry: name -> (fn, family, stochastic?)
SOLVERS = {
    "PGD":               (solve_pgd,          "Local",          True),
    "Coordinate Descent":(solve_cd,           "Local",          True),
    "Trust-Region/LM":   (solve_trust_region, "Local",          True),
    "Riemannian CG":     (solve_rcg,          "Manifold",       True),
    "Scaled ADMM":       (solve_admm,         "Splitting",      True),
    "Simulated Annealing":(solve_sa,          "Global",         True),
    "Basin Hopping":     (solve_basinhopping, "Global",         True),
    "Differential Eq.":  (solve_de,           "Global",         True),
    "CMA-ES":            (solve_cmaes,        "Global",         True),
    "Cross-Entropy":     (solve_cem,          "Global",         True),
    "Simulated Bifurc.": (solve_sb,           "Quantum-insp.",  True),
    "SDR (randomized)":  (solve_sdr,          "Relaxation",     True),
    "Perturbation Null": (solve_perturbation, "Hardware-aware", True),
    "OBH-ZKD (prior)":   (solve_obh_zkd,      "Prior proposed", True),
    "MR-LCMV (ours)":    (solve_ours_none,    "Ours",           False),
    "MR-LCMV+GLCP (ours)":(solve_ours,        "Ours",           False),
    "MR-LCMV-certified":(solve_ours_certified,"Ours",           False),
    "MR-LCMV-adaptive": (solve_ours_adaptive, "Ours",           False,)
}


def run_solver(name, task, element, N, seed=0, discrete=True, **kw):
    """Run one solver; return metrics dict (incl. family, evals, wall_ms)."""
    fn, family, _ = SOLVERS[name]
    prob = Problem(task, element, N, discrete=discrete)
    rng = np.random.default_rng(seed)
    t = time.perf_counter()
    V = fn(prob, rng, **kw)
    ms = (time.perf_counter() - t) * 1e3
    c = prob.weights(V)
    m = prob.metrics(c)
    m.update(family=family, evals=prob.eval_count, ms=ms,
             glcp_updates=getattr(prob, "glcp_updates", 0))
    return m
