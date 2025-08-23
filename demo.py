#!/usr/bin/env python3
"""
GRAVIDY Demo Script

A simple demonstration of GRAVIDY's capabilities.
This script shows a quick example of each optimization type.
"""

import numpy as np
import matplotlib.pyplot as plt
from solver.gravidy_delta_klprox import GRAVIDY_Delta_KLprox
from solver.gravidy_box import GRAVIDY_box
from solver.gravidy_pos import GRAVIDY_pos
from utils.simplex_objective import SimplexLeastSquares
from utils.box_objective import BoxLeastSquares
from utils.pos_objective import PositiveLeastSquares

def demo_simplex():
    """Demo simplex optimization."""
    print("🎯 Simplex Optimization Demo")
    print("-" * 30)
    
    # Create a simple problem
    n = 20
    A = np.random.randn(n, n)
    b = np.random.randn(n)
    x_star = np.random.rand(n)
    x_star = x_star / np.sum(x_star)  # Project to simplex
    
    problem = SimplexLeastSquares(A, b)
    
    # Run GRAVIDY
    x, history = GRAVIDY_Delta_KLprox(problem, eta=30.0, max_outer=50, verbose=False)
    
    # Results
    error = np.linalg.norm(x - x_star)
    objective = problem.f(x)
    print(f"Final error: {error:.2e}")
    print(f"Final objective: {objective:.2e}")
    print(f"Simplex constraint satisfied: {np.abs(np.sum(x) - 1.0):.2e}")
    print()

def demo_box():
    """Demo box-constrained optimization."""
    print("📦 Box-Constrained Optimization Demo")
    print("-" * 40)
    
    # Create a simple problem
    n = 20
    A = np.random.randn(n, n)
    b = np.random.randn(n)
    x_star = np.random.rand(n) * 2 - 1  # Random in [-1, 1]
    
    lo = -1.0 * np.ones(n)
    hi = 1.0 * np.ones(n)
    
    problem = BoxLeastSquares(A, b, lo, hi)
    
    # Run GRAVIDY
    x, history = GRAVIDY_box(problem, eta=50.0, max_outer=50, verbose=False)
    
    # Results
    error = np.linalg.norm(x - x_star)
    objective = problem.f(x)
    in_bounds = np.all((x >= lo) & (x <= hi))
    print(f"Final error: {error:.2e}")
    print(f"Final objective: {objective:.2e}")
    print(f"Box constraints satisfied: {in_bounds}")
    print()

def demo_positive():
    """Demo positive orthant optimization."""
    print("➕ Positive Orthant Optimization Demo")
    print("-" * 40)
    
    # Create a simple problem
    n = 20
    A = np.abs(np.random.randn(n, n)) + 0.1  # Make A non-negative
    b = np.abs(np.random.randn(n)) + 0.1     # Make b non-negative
    x_star = np.abs(np.random.rand(n))       # Make x_star non-negative
    
    problem = PositiveLeastSquares(A, b)
    
    # Run GRAVIDY
    x, history = GRAVIDY_pos(problem, eta=30.0, max_outer=50, verbose=False)
    
    # Results
    error = np.linalg.norm(x - x_star)
    objective = problem.f(x)
    non_negative = np.all(x >= 0)
    print(f"Final error: {error:.2e}")
    print(f"Final objective: {objective:.2e}")
    print(f"Non-negativity satisfied: {non_negative}")
    print()

def main():
    """Run all demos."""
    print("🚀 GRAVIDY Demo")
    print("=" * 50)
    print("This demo shows GRAVIDY solving three types of optimization problems:")
    print("1. Simplex optimization (sum = 1, all ≥ 0)")
    print("2. Box-constrained optimization (each variable in [lo, hi])")
    print("3. Positive orthant optimization (all variables ≥ 0)")
    print()
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Run demos
    demo_simplex()
    demo_box()
    demo_positive()
    
    print("🎉 Demo completed!")
    print("\nTo run full benchmarks with comparisons:")
    print("  python simplex_gravidy_benchmark.py")
    print("  python box_gravidy_benchmark.py")
    print("  python pos_gravidy_benchmark.py")
    print("  python gravidy_st_benchmark.py")

if __name__ == "__main__":
    main()
