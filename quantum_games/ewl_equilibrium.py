"""
ewl_equilibrium.py — does entanglement actually change the equilibrium?

This replaces the repository's "GHZ Entanglement Consensus Theorem", which
asserted that entanglement doubles public-good approval but was implemented as a
75% chance of copying one voter (see CORRECTIONS.md §3). The question there was
the right one; the mechanism was not.

Here we test the real version of that question on the one game where it has a
published answer: the Eisert-Wilkens-Lewenstein (1999) quantised Prisoner's
Dilemma. Two claims are checked against the literature rather than asserted.

  EWL (Phys. Rev. Lett. 83, 3077):
      With a maximally entangling gate and the 2-parameter strategy set
      U(theta, alpha), the profile (Q, Q) with Q = diag(i, -i) is a Nash
      equilibrium paying (3, 3) — escaping the classical (D, D) trap at (1, 1).

  Benjamin & Hayden (Phys. Rev. Lett. 87, 069801) — the standard objection:
      That result depends on the restricted strategy set. Over the full SU(2)
      strategy space no pure-strategy Nash equilibrium survives, because every
      profile admits a profitable unilateral deviation.

Both are checked here, across a sweep of the entanglement parameter gamma.

Design notes, chosen deliberately:

  * Payoffs are computed analytically from the statevector. There is no
    sampling, so there is no estimator noise and results are exactly
    reproducible. Nothing here is fitted.
  * Equilibria are found by exhaustive search over a strategy grid, not by an
    optimiser. benchmark_qai_real.py documents why: this project's hill climb is
    noise-dominated and accepted 0 of 40 moves while reporting a converged-
    looking loss. A grid cannot fail that way.
  * Grid coarseness biases *towards* finding spurious equilibria (fewer
    deviations are tested). So "no pure NE found" is the conservative,
    trustworthy direction, and the (Q, Q) claim is separately re-checked against
    a dense random deviation search rather than trusted from the grid alone.

Usage:
    python3 q_ai_governance/ewl_equilibrium.py
    python3 q_ai_governance/ewl_equilibrium.py --out data/ewl_results.json
"""

import argparse
import json
import os

import numpy as np

# Prisoner's Dilemma payoffs, matching QuantumPrisonerDilemmaEnv in quantum_agent.py.
# Basis order is |CC>, |CD>, |DC>, |DD>.
PAYOFF_A = np.array([3.0, 0.0, 5.0, 1.0])
PAYOFF_B = np.array([3.0, 5.0, 0.0, 1.0])

SIGMA_X = np.array([[0, 1], [1, 0]], dtype=complex)


def entangling_gate(gamma):
    """J(gamma) = exp(i * gamma * sigma_x (x) sigma_x / 2).

    gamma = 0 leaves the game classical; gamma = pi/2 is maximal entanglement.
    """
    return (np.cos(gamma / 2) * np.eye(4, dtype=complex)
            + 1j * np.sin(gamma / 2) * np.kron(SIGMA_X, SIGMA_X))


def strategy(theta, alpha, beta=0.0):
    """A single-qubit strategy in SU(2).

    beta = 0 recovers the 2-parameter EWL strategy set. Allowing beta to vary is
    exactly the strategy-space widening of the Benjamin-Hayden objection.

        C = U(0, 0)        (cooperate, identity)
        D = U(pi, 0)       (defect)
        Q = U(0, pi/2)     (the EWL "quantum" strategy, diag(i, -i))
    """
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([
        [np.exp(1j * alpha) * c, 1j * np.exp(1j * beta) * s],
        [1j * np.exp(-1j * beta) * s, np.exp(-1j * alpha) * c],
    ], dtype=complex)


C_STRAT = strategy(0.0, 0.0)
D_STRAT = strategy(np.pi, 0.0)
Q_STRAT = strategy(0.0, np.pi / 2)


def payoff_matrices(strats_a, strats_b, gamma, chunk=256):
    """Exact expected payoffs for every profile in strats_a x strats_b.

    Returns (PA, PB), each shaped (len(strats_a), len(strats_b)).
    """
    jg = entangling_gate(gamma)
    j_dag = jg.conj().T
    # |psi_0> = J|CC>, carried as a 2x2 matrix so that (Ua (x) Ub)|psi> is Ua W Ub^T.
    w = (jg @ np.array([1, 0, 0, 0], dtype=complex)).reshape(2, 2)

    ua = np.asarray(strats_a)
    ub = np.asarray(strats_b)
    n_a, n_b = len(ua), len(ub)
    pa = np.empty((n_a, n_b))
    pb = np.empty((n_a, n_b))

    x = np.einsum("aij,jk->aik", ua, w)          # Ua W
    for lo in range(0, n_a, chunk):              # chunked to bound memory
        hi = min(lo + chunk, n_a)
        y = np.einsum("aik,bjk->abij", x[lo:hi], ub)   # (Ua W) Ub^T
        psi = np.einsum("xy,aby->abx", j_dag, y.reshape(hi - lo, n_b, 4))
        probs = np.abs(psi) ** 2
        pa[lo:hi] = probs @ PAYOFF_A
        pb[lo:hi] = probs @ PAYOFF_B
    return pa, pb


def pure_nash(pa, pb, tol=1e-9):
    """Boolean mask of pure-strategy Nash equilibria on the sampled grid.

    A profile is an equilibrium when neither player can strictly improve by
    switching to any other strategy in the grid, holding the other fixed.
    """
    best_a = pa.max(axis=0, keepdims=True)   # A's best reply to each b
    best_b = pb.max(axis=1, keepdims=True)   # B's best reply to each a
    return (pa >= best_a - tol) & (pb >= best_b - tol)


def build_grid(space, n_theta, n_phase):
    """Strategy grid and labels for either strategy space."""
    thetas = np.linspace(0.0, np.pi, n_theta)
    if space == "restricted":
        # EWL: alpha in [0, pi/2], beta fixed at 0.
        alphas = np.linspace(0.0, np.pi / 2, n_phase)
        params = [(t, a, 0.0) for t in thetas for a in alphas]
    elif space == "full":
        # Benjamin-Hayden: both phases free over the full circle.
        alphas = np.linspace(0.0, 2 * np.pi, n_phase, endpoint=False)
        betas = np.linspace(0.0, 2 * np.pi, n_phase, endpoint=False)
        params = [(t, a, b) for t in thetas for a in alphas for b in betas]
    else:
        raise ValueError(space)
    mats = np.array([strategy(*p) for p in params])
    return mats, params


def deviation_check(profile_a, profile_b, gamma, space, n_samples=200_000, seed=0):
    """Dense random search for a profitable unilateral deviation from a profile.

    Independent of the grid: draws strategies uniformly at random from `space`
    and asks whether either player can beat their current payoff. This is what
    actually tests the (Q, Q) claim, since a grid can miss deviations.
    """
    rng = np.random.default_rng(seed)
    thetas = rng.uniform(0.0, np.pi, n_samples)
    if space == "restricted":
        alphas = rng.uniform(0.0, np.pi / 2, n_samples)
        betas = np.zeros(n_samples)
    else:
        alphas = rng.uniform(0.0, 2 * np.pi, n_samples)
        betas = rng.uniform(0.0, 2 * np.pi, n_samples)
    devs = np.array([strategy(t, a, b) for t, a, b in zip(thetas, alphas, betas)])

    base_a, base_b = payoff_matrices([profile_a], [profile_b], gamma)
    base_a, base_b = float(base_a[0, 0]), float(base_b[0, 0])

    # A deviates against fixed B, then B deviates against fixed A.
    dev_a, _ = payoff_matrices(devs, [profile_b], gamma)
    _, dev_b = payoff_matrices([profile_a], devs, gamma)
    gain_a = float(dev_a[:, 0].max()) - base_a
    gain_b = float(dev_b[0, :].max()) - base_b
    return {
        "base_payoff": [round(base_a, 6), round(base_b, 6)],
        "best_deviation_gain_A": round(gain_a, 6),
        "best_deviation_gain_B": round(gain_b, 6),
        "is_nash": bool(gain_a <= 1e-6 and gain_b <= 1e-6),
        "samples": n_samples,
    }


def classical_correlated_equilibrium():
    """Best symmetric correlated-equilibrium payoff in the one-shot PD.

    This is the control that matters. The usual objection to any claimed quantum
    advantage in a game is that entanglement is just shared randomness — and
    shared randomness is exactly a correlated equilibrium. So if a correlated
    device could reach (3, 3), the quantum result would be unremarkable.

    It cannot: D strictly dominates C, so every correlated equilibrium places
    all its mass on (D, D). Solved here by linear programming rather than
    asserted.
    """
    from scipy.optimize import linprog

    # Variables: p(CC), p(CD), p(DC), p(DD).
    ua = {(0, 0): 3.0, (0, 1): 0.0, (1, 0): 5.0, (1, 1): 1.0}
    ub = {(0, 0): 3.0, (0, 1): 5.0, (1, 0): 0.0, (1, 1): 1.0}
    idx = {(0, 0): 0, (0, 1): 1, (1, 0): 2, (1, 1): 3}

    rows = []
    for a in (0, 1):                      # A's obedience constraints
        for a2 in (0, 1):
            if a == a2:
                continue
            row = np.zeros(4)
            for b in (0, 1):
                row[idx[(a, b)]] = -(ua[(a, b)] - ua[(a2, b)])   # <= 0 form
            rows.append(row)
    for b in (0, 1):                      # B's obedience constraints
        for b2 in (0, 1):
            if b == b2:
                continue
            row = np.zeros(4)
            for a in (0, 1):
                row[idx[(a, b)]] = -(ub[(a, b)] - ub[(a, b2)])
            rows.append(row)

    total = np.array([ua[k] + ub[k] for k in idx])   # maximise joint payoff
    res = linprog(c=-total, A_ub=np.array(rows), b_ub=np.zeros(len(rows)),
                  A_eq=np.ones((1, 4)), b_eq=[1.0], bounds=[(0, 1)] * 4)
    p = res.x
    label = {(0, 0): "CC", (0, 1): "CD", (1, 0): "DC", (1, 1): "DD"}
    return {
        "distribution": {label[k]: round(float(p[v]), 6) for k, v in idx.items()},
        "payoff_A": round(float(sum(p[idx[k]] * ua[k] for k in idx)), 6),
        "payoff_B": round(float(sum(p[idx[k]] * ub[k] for k in idx)), 6),
    }


def critical_gamma(verify=True, n_theta=601, n_alpha=301):
    """Entanglement threshold above which (Q, Q) becomes a Nash equilibrium.

    Derived rather than fitted. While B plays Q, A's best deviation in the
    restricted space is D, and its payoff falls with entanglement as

        u_A(D, Q) = (T - S) cos^2(gamma) + S

    which drops to the (Q, Q) payoff R when cos^2(gamma_c) = (R - S) / (T - S).
    For the payoffs used here (T=5, R=3, P=1, S=0) that is cos^2 = 3/5, i.e.
    gamma_c = arccos(sqrt(3/5)) ~ 0.684719 rad ~ 39.23 deg.

    With verify=True this is confirmed against a dense deterministic sweep
    instead of being taken on faith.
    """
    T, R, S = 5.0, 3.0, 0.0
    analytic = float(np.arccos(np.sqrt((R - S) / (T - S))))
    out = {"analytic_rad": round(analytic, 9),
           "analytic_deg": round(float(np.degrees(analytic)), 6),
           "sin2_gamma_c": round(float(np.sin(analytic) ** 2), 9)}
    if verify:
        thetas = np.linspace(0.0, np.pi, n_theta)
        alphas = np.linspace(0.0, np.pi / 2, n_alpha)
        devs = np.array([strategy(t, a, 0.0) for t in thetas for a in alphas])

        def best_deviation(gamma):
            pa, _ = payoff_matrices(devs, [Q_STRAT], gamma)
            return float(pa[:, 0].max())

        lo, hi = 0.0, np.pi / 2
        for _ in range(50):
            mid = (lo + hi) / 2
            if best_deviation(mid) > R + 1e-9:
                lo = mid
            else:
                hi = mid
        out["numeric_rad"] = round(hi, 9)
        out["abs_error"] = float(abs(hi - analytic))
    return out


def run(gammas, n_theta=13, n_phase=13, n_theta_r=41, n_phase_r=21):
    results = {"gammas": [], "correlated_equilibrium": classical_correlated_equilibrium(),
               "critical_gamma": critical_gamma()}

    for space in ("restricted", "full"):
        nt, npz = (n_theta_r, n_phase_r) if space == "restricted" else (n_theta, n_phase)
        mats, params = build_grid(space, nt, npz)
        print(f"\n=== {space} strategy space | {len(mats)} strategies "
              f"({len(mats)**2:,} profiles) ===")
        print(f"{'gamma':>8}{'pure NE':>10}{'best NE payoff':>17}{'(Q,Q) payoff':>15}"
              f"{'(Q,Q) NE?':>11}")
        for gamma in gammas:
            pa, pb = payoff_matrices(mats, mats, gamma)
            mask = pure_nash(pa, pb)
            n_ne = int(mask.sum())
            if n_ne:
                sym = (pa + pb) / 2.0
                best = float(sym[mask].max())
                best_s = f"{best:.3f}"
            else:
                best_s = "—"
            qq = deviation_check(Q_STRAT, Q_STRAT, gamma, space, n_samples=60_000)
            results["gammas"].append({
                "space": space, "gamma": round(float(gamma), 6),
                "grid_size": len(mats), "pure_nash_count": n_ne,
                "best_nash_symmetric_payoff": None if not n_ne else round(best, 6),
                "QQ": qq,
            })
            print(f"{gamma:>8.4f}{n_ne:>10}{best_s:>17}"
                  f"{str(qq['base_payoff']):>15}{str(qq['is_nash']):>11}")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/ewl_equilibrium_results.json")
    ap.add_argument("--n-gamma", type=int, default=6)
    args = ap.parse_args()

    gammas = np.linspace(0.0, np.pi / 2, args.n_gamma)
    print("EWL quantised Prisoner's Dilemma — exhaustive equilibrium search")
    print("Payoffs: (C,C)=(3,3) (C,D)=(0,5) (D,C)=(5,0) (D,D)=(1,1)")

    ce = classical_correlated_equilibrium()
    print(f"\nClassical baselines")
    print(f"  Nash equilibrium (D,D)          : (1.0, 1.0)")
    print(f"  Best correlated equilibrium     : "
          f"({ce['payoff_A']}, {ce['payoff_B']})  dist={ce['distribution']}")

    cg = results_cg = critical_gamma()
    print(f"\nEntanglement threshold for (Q,Q) as a Nash equilibrium")
    print(f"  analytic  arccos(sqrt((R-S)/(T-S))) = {cg['analytic_rad']:.6f} rad "
          f"({cg['analytic_deg']:.3f} deg)")
    print(f"  numeric   dense sweep               = {cg['numeric_rad']:.6f} rad "
          f"(|error| {cg['abs_error']:.2e})")

    results = run(gammas)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(results, open(args.out, "w"), indent=2)
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
