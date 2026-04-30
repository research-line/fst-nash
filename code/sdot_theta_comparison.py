"""
Numerical illustration: S_dot(pi) vs S_dot(0) for the Berry-Keating
saturation model with the first 50 primes as primitive orbits.

Computes the entropy production rate difference Delta_S_dot = S_dot(pi) - S_dot(0)
to verify that antiperiodic boundary conditions (theta=pi) maximize MEPP.

Author: Lukas Geiger
Date: March 2026
Part of: FST-RH Conjecture Paper
"""

import numpy as np
import json
import os

# === Physical constants (natural units: hbar = 1) ===
HBAR = 1.0

# Saturation boundaries (logarithmic)
# u_- = ln(l_p), u_+ = ln(L_max)
# l_p ~ 1.6e-35 m (Planck length)
# L_max ~ 4.4e26 m (Hubble radius)
L_PLANCK = 1.616e-35
L_MAX = 4.4e26
U_MINUS = np.log(L_PLANCK)
U_PLUS = np.log(L_MAX)
T = U_PLUS - U_MINUS  # ~ 141.3

# First 50 primes
PRIMES_50 = [
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29,
    31, 37, 41, 43, 47, 53, 59, 61, 67, 71,
    73, 79, 83, 89, 97, 101, 103, 107, 109, 113,
    127, 131, 137, 139, 149, 151, 157, 163, 167, 173,
    179, 181, 191, 193, 197, 199, 211, 223, 227, 229
]

# Model parameters
ALPHA = 1          # Dissipation scaling: E^alpha
EPSILON = 0.01     # Orbit modulation strength
N_MODES = 500      # Number of modes to sum over


def spectrum(theta, n_modes=N_MODES):
    """Discrete spectrum E_n(theta) = hbar * (2*pi*n - theta) / T"""
    n = np.arange(1, n_modes + 1)
    return HBAR * (2 * np.pi * n - theta) / T


def orbit_modulation(E, primes, a_p_func):
    """
    Compute the orbit interference term:
    1 + epsilon * sum_p a_p * cos(E/hbar * ln(p))

    Returns array of shape (len(E),)
    """
    mod = np.ones_like(E)
    for p in primes:
        a_p = a_p_func(p)
        phase = (E / HBAR) * np.log(p)
        mod += EPSILON * a_p * np.cos(phase)
    return mod


def sdot(theta, beta, primes=PRIMES_50, a_p_func=lambda p: 1.0/p):
    """
    Entropy production rate:
    S_dot(theta) = sum_n e^{-beta*E_n} * E_n^alpha * [1 + orbit interference]

    Only sums over modes with E_n > 0.
    """
    E = spectrum(theta)
    mask = E > 0
    E = E[mask]

    boltzmann = np.exp(-beta * E)
    dissipation = E ** ALPHA
    modulation = orbit_modulation(E, primes, a_p_func)

    return np.sum(boltzmann * dissipation * modulation)


def scan_beta_range():
    """Scan over a range of beta values to show robustness."""
    # beta range: from very hot (many modes active) to cold (few modes)
    betas = np.logspace(-3, 1, 50)

    results = {
        'T': T,
        'u_minus': U_MINUS,
        'u_plus': U_PLUS,
        'alpha': ALPHA,
        'epsilon': EPSILON,
        'n_modes': N_MODES,
        'n_primes': len(PRIMES_50),
        'primes': PRIMES_50,
        'a_p': '1/p',
        'scans': []
    }

    print(f"Saturation interval T = u+ - u- = {T:.2f}")
    print(f"  u- = ln(l_p) = {U_MINUS:.2f}")
    print(f"  u+ = ln(L_max) = {U_PLUS:.2f}")
    print(f"  E_1(0)   = {HBAR * 2 * np.pi / T:.6f}")
    print(f"  E_1(pi)  = {HBAR * np.pi / T:.6f}")
    print(f"  Ratio E_1(pi)/E_1(0) = 0.5 (by construction)")
    print()
    print(f"{'beta':>10s} | {'S_dot(0)':>14s} | {'S_dot(pi)':>14s} | {'Delta':>14s} | {'pi wins?':>8s}")
    print("-" * 72)

    pi_wins_count = 0
    for beta in betas:
        s0 = sdot(0.0, beta)
        spi = sdot(np.pi, beta)
        delta = spi - s0
        wins = delta > 0
        if wins:
            pi_wins_count += 1

        results['scans'].append({
            'beta': float(beta),
            'sdot_0': float(s0),
            'sdot_pi': float(spi),
            'delta': float(delta),
            'pi_wins': bool(wins)
        })

        print(f"{beta:10.4f} | {s0:14.6e} | {spi:14.6e} | {delta:+14.6e} | {'YES' if wins else 'no':>8s}")

    print()
    print(f"theta=pi wins in {pi_wins_count}/{len(betas)} cases ({100*pi_wins_count/len(betas):.0f}%)")

    # === Detailed analysis at a representative beta ===
    beta_rep = 0.1
    print(f"\n{'='*72}")
    print(f"Detailed analysis at beta = {beta_rep}")
    print(f"{'='*72}")

    # Baseline (no primes)
    s0_base = sdot(0.0, beta_rep, a_p_func=lambda p: 0.0)
    spi_base = sdot(np.pi, beta_rep, a_p_func=lambda p: 0.0)
    delta_base = spi_base - s0_base

    # With primes
    s0_full = sdot(0.0, beta_rep)
    spi_full = sdot(np.pi, beta_rep)
    delta_full = spi_full - s0_full

    print(f"Baseline (no orbit modulation):")
    print(f"  S_dot(0)  = {s0_base:.6e}")
    print(f"  S_dot(pi) = {spi_base:.6e}")
    print(f"  Delta     = {delta_base:+.6e} ({'pi wins' if delta_base > 0 else '0 wins'})")
    print()
    print(f"With 50-prime orbit modulation (epsilon={EPSILON}, a_p=1/p):")
    print(f"  S_dot(0)  = {s0_full:.6e}")
    print(f"  S_dot(pi) = {spi_full:.6e}")
    print(f"  Delta     = {delta_full:+.6e} ({'pi wins' if delta_full > 0 else '0 wins'})")
    print()
    print(f"Prime orbit contribution to Delta:")
    print(f"  Delta_prime = {delta_full - delta_base:+.6e}")
    print(f"  Primes {'reinforce' if (delta_full - delta_base) > 0 else 'counteract'} the MEPP advantage of theta=pi")

    results['detailed'] = {
        'beta': beta_rep,
        'baseline_delta': float(delta_base),
        'full_delta': float(delta_full),
        'prime_contribution': float(delta_full - delta_base),
        'primes_reinforce': bool((delta_full - delta_base) > 0)
    }

    # === Per-prime contribution analysis ===
    print(f"\n{'='*72}")
    print(f"Per-prime phase shift analysis")
    print(f"{'='*72}")
    print(f"{'p':>5s} | {'ln(p)':>8s} | {'Delta_p = pi*ln(p)/(2T)':>22s} | {'sin(Delta_p)':>12s}")
    print("-" * 55)

    for p in PRIMES_50[:20]:  # Show first 20
        lnp = np.log(p)
        delta_p = np.pi * lnp / (2 * T)
        sin_delta = np.sin(delta_p)
        print(f"{p:5d} | {lnp:8.4f} | {delta_p:22.6f} | {sin_delta:+12.6f}")
    print("  ... (30 more primes)")

    return results


def main():
    results = scan_beta_range()

    # Save results
    outpath = os.path.join(os.path.dirname(__file__), 'sdot_theta_results.json')
    with open(outpath, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {outpath}")


if __name__ == '__main__':
    main()
