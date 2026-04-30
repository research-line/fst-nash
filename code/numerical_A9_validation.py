"""
Numerical validation of Hypothesis A9.3 (FST-RH3)
Tests: Phi_{n,Pmax}(sigma) -> lambda_n as sigma -> 1+

Vectorized computation with heatmap and convergence plots.
"""
import numpy as np
import mpmath as mp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

mp.mp.dps = 30  # sufficient for reference lambda_n


# ------------------------------------------------------------
# Primes via simple sieve
# ------------------------------------------------------------
def primes_up_to(N):
    N = int(N)
    if N < 2:
        return np.array([], dtype=int)
    sieve = np.ones(N + 1, dtype=bool)
    sieve[:2] = False
    for i in range(2, int(N**0.5) + 1):
        if sieve[i]:
            sieve[i * i : N + 1 : i] = False
    return np.nonzero(sieve)[0]


# ------------------------------------------------------------
# Vectorized Polylog Li_{1-k}(p^{-sigma}) over p-array
# Li_{1-k}(p^{-sigma}) = sum_{m>=1} m^{k-1} p^{-m sigma}
# ------------------------------------------------------------
def Li_polylog_1_minus_k_vec(p_array, sigma, k, mmax=50):
    p = p_array.astype(np.float64)
    # Adaptive: for p>=3, p^{-sigma} < 0.34, so mmax=50 gives ~1e-23 precision
    m = np.arange(1, mmax + 1, dtype=np.float64)[:, None]
    term = (m ** (k - 1)) * np.power(p[None, :], -m * sigma)
    return term.sum(axis=0)


# ------------------------------------------------------------
# H_{n,sigma}(p) vectorized, without archimedean term
# ------------------------------------------------------------
def H_n_sigma_vec(p_array, sigma, n, mmax=200):
    p = p_array.astype(np.float64)
    logp = np.log(p)
    H = np.zeros_like(p, dtype=np.float64)
    for k in range(1, n + 1):
        C = mp.binomial(n, k) * mp.factorial(n - 1) / mp.factorial(k - 1)
        C = float(C)
        Li = Li_polylog_1_minus_k_vec(p, sigma, k, mmax=mmax)
        H += C * ((-1) ** k) * (logp ** k) * Li
    return H


# ------------------------------------------------------------
# Phi_{n,Pmax}(sigma) vectorized
# ------------------------------------------------------------
def phi_n_cutoff(eps, alpha, n, mmax=200):
    sigma = 1.0 + float(eps)
    Pmax = int(np.floor(eps ** (-alpha)))
    ps = primes_up_to(Pmax)
    if ps.size == 0:
        return sigma, Pmax, np.nan

    # Gibbs weights
    w = ps.astype(np.float64) ** (-sigma)
    Z = w.sum()
    mu = w / Z

    # H_n_sigma(p)
    H = H_n_sigma_vec(ps, sigma, n, mmax=mmax)

    # log E[e^H] - E[H] (log-sum-exp trick)
    Hmax = H.max()
    sum_exp = np.sum(mu * np.exp(H - Hmax))
    logEexp = np.log(sum_exp) + Hmax
    EH = np.sum(mu * H)
    Phi = logEexp - EH

    return sigma, Pmax, Phi


# ------------------------------------------------------------
# Reference lambda_n via Riemann xi
# ------------------------------------------------------------
def xi(s):
    return (
        mp.mpf("0.5")
        * s
        * (s - 1)
        * mp.power(mp.pi, -s / 2)
        * mp.gamma(s / 2)
        * mp.zeta(s)
    )


def lambda_n_reference(n):
    """Compute lambda_n = L_n[log xi] at s=1 via numerical differentiation."""
    # Use the series definition with xi derivatives
    s0 = mp.mpf(1)
    # lambda_n = sum_{k=1}^{n} C(n,k) (-1)^{k+1} (d^k/ds^k log xi)(1)
    result = mp.mpf(0)
    for k in range(1, n + 1):
        C = mp.binomial(n, k) * mp.factorial(n - 1) / mp.factorial(k - 1)
        # k-th derivative of log(xi) at s=1
        dk = mp.diff(lambda s: mp.log(xi(s)), s0, k)
        result += C * ((-1) ** (k + 1)) * dk
    return result


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
if __name__ == "__main__":
    n = 1

    # Precomputed reference: lambda_1 from Li's criterion
    # lambda_1 = 1 - 1/2 * (gamma + ln(4pi) - 2) ≈ 0.023096...
    # More precisely from the formula lambda_n = sum_rho [1-(1-1/rho)^n]
    print(f"Computing reference lambda_{n}...")
    try:
        lam_ref = float(lambda_n_reference(n))
    except Exception:
        # Fallback: known value
        lam_ref = 0.0230957089680993
    print(f"Reference lambda_{n} = {lam_ref:.12e}")

    # Grid (reduced for speed)
    eps_values = np.array([1e-2, 5e-3, 2e-3, 1e-3, 5e-4])
    alpha_values = np.array([1.0, 1.25, 1.5, 1.75, 2.0])

    errors = np.zeros((len(eps_values), len(alpha_values)))
    Phi_vals = np.zeros_like(errors)

    mmax = 50

    print(f"\n{'eps':>10s}  {'alpha':>6s}  {'sigma':>10s}  {'Pmax':>8s}  {'Phi':>14s}  {'err':>10s}")
    print("-" * 70)

    for i, eps in enumerate(eps_values):
        for j, alpha in enumerate(alpha_values):
            sigma, Pmax, Phi = phi_n_cutoff(eps, alpha, n, mmax=mmax)
            Phi_vals[i, j] = Phi
            errors[i, j] = abs(Phi - lam_ref)
            print(
                f"{eps:10.1e}  {alpha:6.2f}  {sigma:10.6f}  {Pmax:8d}  {Phi:14.8e}  {errors[i, j]:10.3e}"
            )

    # --------------------------------------------------------
    # Heatmap: (eps, alpha) -> |Phi - lambda_n|
    # --------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(
        np.log10(errors + 1e-20),
        origin="lower",
        extent=[alpha_values[0], alpha_values[-1], 0, len(eps_values) - 1],
        aspect="auto",
        cmap="viridis",
    )
    ax.set_yticks(range(len(eps_values)))
    ax.set_yticklabels([f"{e:.0e}" for e in eps_values])
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"$\varepsilon$")
    ax.set_title(
        rf"$\log_{{10}}|\Phi_{{n,P_{{\max}}}}(\sigma) - \lambda_{n}|$"
    )
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(r"$\log_{10}$ absolute error")
    plt.tight_layout()
    plt.savefig("A9_heatmap.png", dpi=150)
    print("\nSaved: A9_heatmap.png")

    # --------------------------------------------------------
    # Convergence curves: Phi vs eps for fixed alpha
    # --------------------------------------------------------
    fig2, ax2 = plt.subplots(figsize=(7, 5))
    for j, alpha in enumerate(alpha_values):
        ax2.plot(
            eps_values,
            Phi_vals[:, j],
            marker="o",
            label=rf"$\alpha={alpha}$",
        )
    ax2.axhline(lam_ref, color="k", linestyle="--", label=rf"$\lambda_{n}$ ref")
    ax2.set_xscale("log")
    ax2.set_xlabel(r"$\varepsilon$")
    ax2.set_ylabel(r"$\Phi_{n,P_{\max}}(\sigma)$")
    ax2.set_title(
        rf"Convergence of $\Phi_{{n,P_{{\max}}}}(\sigma)$ to $\lambda_{n}$"
    )
    ax2.legend()
    plt.tight_layout()
    plt.savefig("A9_convergence.png", dpi=150)
    print("Saved: A9_convergence.png")

    plt.show()
