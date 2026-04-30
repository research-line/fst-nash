#!/usr/bin/env python3
"""
protein_fold_nash.py
====================
Protein Folding as Nash Equilibrium Search

Toy continuous model implementing FST-III Prediction P1:
"The energy landscape of protein folding can be characterized by
Nash equilibrium stability analysis."

Model:
- Players: N residues (amino acids)
- Strategy space: Torsion angles theta_i in [-pi, pi]
- Global potential: F(theta) = sum of local preferences + pairwise contacts
- Nash equilibrium: gradient = 0 (no player benefits from unilateral deviation)
- Stability: Hessian eigenvalues > 0 (ESS condition)

Reference: FST-III Biological Game Theory (Geiger, 2026)

Usage:
    python protein_fold_nash.py [--residues N] [--steps S] [--seed R]
"""

import numpy as np
from typing import Tuple, Optional
import argparse


def torsion_potential(theta: np.ndarray, k: float = 1.0) -> float:
    """
    Local torsion angle preference (Ramachandran-like).
    Each residue prefers angles near 0 or +/- 2pi/3 (alpha-helix, beta-sheet).

    V_i(theta_i) = k * (1 - cos(3*theta_i)) / 2
    """
    return k * np.sum(1 - np.cos(3 * theta)) / 2


def contact_potential(theta: np.ndarray, epsilon: float = 0.5,
                      contact_range: int = 4) -> float:
    """
    Pairwise contact energy (simplified Lennard-Jones-like).
    Residues i and j interact if |i-j| >= contact_range.

    Distance approximated by cumulative torsion angle difference.
    """
    n = len(theta)
    energy = 0.0
    for i in range(n):
        for j in range(i + contact_range, n):
            # Effective distance based on path through chain
            d_eff = np.abs(np.sum(theta[i:j]))
            # Attractive well at d_eff ~ pi
            energy += epsilon * (d_eff - np.pi)**2 / (1 + d_eff**2)
    return energy


def total_potential(theta: np.ndarray, k: float = 1.0,
                    epsilon: float = 0.5) -> float:
    """
    Total free energy F(theta) = V_torsion + V_contact
    """
    return torsion_potential(theta, k) + contact_potential(theta, epsilon)


def gradient(theta: np.ndarray, delta: float = 1e-5,
             k: float = 1.0, epsilon: float = 0.5) -> np.ndarray:
    """
    Numerical gradient of F(theta).
    At Nash equilibrium: grad F = 0 (no player benefits from deviation).
    """
    n = len(theta)
    grad = np.zeros(n)
    f0 = total_potential(theta, k, epsilon)
    for i in range(n):
        theta_plus = theta.copy()
        theta_plus[i] += delta
        grad[i] = (total_potential(theta_plus, k, epsilon) - f0) / delta
    return grad


def hessian(theta: np.ndarray, delta: float = 1e-4,
            k: float = 1.0, epsilon: float = 0.5) -> np.ndarray:
    """
    Numerical Hessian matrix of F(theta).
    ESS condition: All eigenvalues > 0 (strict local minimum).
    """
    n = len(theta)
    H = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            theta_pp = theta.copy()
            theta_pm = theta.copy()
            theta_mp = theta.copy()
            theta_mm = theta.copy()

            theta_pp[i] += delta; theta_pp[j] += delta
            theta_pm[i] += delta; theta_pm[j] -= delta
            theta_mp[i] -= delta; theta_mp[j] += delta
            theta_mm[i] -= delta; theta_mm[j] -= delta

            H[i, j] = (total_potential(theta_pp, k, epsilon)
                      - total_potential(theta_pm, k, epsilon)
                      - total_potential(theta_mp, k, epsilon)
                      + total_potential(theta_mm, k, epsilon)) / (4 * delta**2)
    return H


def find_nash_equilibrium(n_residues: int = 10,
                          learning_rate: float = 0.1,
                          max_steps: int = 1000,
                          tolerance: float = 1e-6,
                          k: float = 1.0,
                          epsilon: float = 0.5,
                          seed: Optional[int] = None) -> Tuple[np.ndarray, dict]:
    """
    Find Nash equilibrium via gradient descent.

    Nash condition: grad F(theta*) = 0
    (No residue can reduce energy by unilateral angle change)

    Returns:
        theta_star: Equilibrium configuration
        info: Dict with convergence info and stability analysis
    """
    if seed is not None:
        np.random.seed(seed)

    # Initialize random configuration
    theta = np.random.uniform(-np.pi, np.pi, n_residues)

    history = []
    for step in range(max_steps):
        grad = gradient(theta, k=k, epsilon=epsilon)
        grad_norm = np.linalg.norm(grad)
        energy = total_potential(theta, k, epsilon)
        history.append({'step': step, 'energy': energy, 'grad_norm': grad_norm})

        if grad_norm < tolerance:
            break

        # Gradient descent step
        theta = theta - learning_rate * grad
        # Keep angles in [-pi, pi]
        theta = np.mod(theta + np.pi, 2 * np.pi) - np.pi

    # Stability analysis via Hessian
    H = hessian(theta, k=k, epsilon=epsilon)
    eigenvalues = np.linalg.eigvalsh(H)
    is_stable = np.all(eigenvalues > 0)

    info = {
        'converged': grad_norm < tolerance,
        'steps': step + 1,
        'final_energy': total_potential(theta, k, epsilon),
        'final_grad_norm': np.linalg.norm(gradient(theta, k=k, epsilon=epsilon)),
        'eigenvalues': eigenvalues,
        'is_ESS': is_stable,  # Evolutionarily Stable Strategy
        'history': history
    }

    return theta, info


def analyze_equilibrium(theta: np.ndarray, info: dict) -> None:
    """
    Print analysis of Nash equilibrium configuration.
    """
    print("\n" + "="*60)
    print("NASH EQUILIBRIUM ANALYSIS")
    print("="*60)

    print(f"\nConfiguration (N={len(theta)} residues):")
    print(f"  theta* = {np.round(theta, 3)}")

    print(f"\nConvergence:")
    print(f"  Steps: {info['steps']}")
    print(f"  Final energy F(theta*): {info['final_energy']:.6f}")
    print(f"  Gradient norm |grad F|: {info['final_grad_norm']:.2e}")
    print(f"  Converged: {info['converged']}")

    print(f"\nStability (Hessian eigenvalues):")
    for i, ev in enumerate(info['eigenvalues']):
        status = "+" if ev > 0 else "-" if ev < 0 else "0"
        print(f"  lambda_{i+1} = {ev:+.4f}  [{status}]")

    print(f"\nGame-Theoretic Interpretation:")
    print(f"  Nash Equilibrium: grad F = 0 => No player benefits from deviation")
    if info['is_ESS']:
        print(f"  ESS Status: STABLE (all eigenvalues > 0)")
        print(f"  => This is an Evolutionarily Stable Strategy")
        print(f"  => Perturbations will return to equilibrium")
    else:
        print(f"  ESS Status: UNSTABLE (some eigenvalues <= 0)")
        print(f"  => This is a saddle point or maximum")
        print(f"  => Configuration may transition to different state")

    print("\n" + "="*60)


def main():
    parser = argparse.ArgumentParser(
        description="Protein Folding as Nash Equilibrium Search (FST-III P1)"
    )
    parser.add_argument('--residues', '-n', type=int, default=8,
                        help='Number of residues (default: 8)')
    parser.add_argument('--steps', '-s', type=int, default=2000,
                        help='Max optimization steps (default: 2000)')
    parser.add_argument('--seed', '-r', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--lr', type=float, default=0.05,
                        help='Learning rate (default: 0.05)')
    args = parser.parse_args()

    print("Protein Folding as Nash Equilibrium Search")
    print("-" * 45)
    print(f"Model: {args.residues} residues, torsion angles as strategies")
    print(f"Finding Nash equilibrium via gradient descent...")

    theta_star, info = find_nash_equilibrium(
        n_residues=args.residues,
        max_steps=args.steps,
        learning_rate=args.lr,
        seed=args.seed
    )

    analyze_equilibrium(theta_star, info)

    return theta_star, info


if __name__ == "__main__":
    main()
