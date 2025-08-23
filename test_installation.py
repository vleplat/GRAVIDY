#!/usr/bin/env python3
"""
GRAVIDY Installation Test Script

This script verifies that all components of GRAVIDY are working correctly.
Run this after installation to ensure everything is set up properly.
"""

import sys
import importlib
import numpy as np
import matplotlib.pyplot as plt

def test_imports():
    """Test that all required packages can be imported."""
    print("🔍 Testing package imports...")
    
    # Core packages
    packages = [
        'numpy',
        'scipy', 
        'matplotlib'
    ]
    
    for package in packages:
        try:
            importlib.import_module(package)
            print(f"  ✅ {package}")
        except ImportError as e:
            print(f"  ❌ {package}: {e}")
            return False
    
    # GRAVIDY modules
    gravidy_modules = [
        'solver.gravidy_st_fast',
        'solver.gravidy_delta_klprox',
        'solver.gravidy_box',
        'solver.gravidy_pos',
        'utils.simplex_utils',
        'utils.box_objective',
        'utils.pos_objective'
    ]
    
    print("\n🔍 Testing GRAVIDY modules...")
    for module in gravidy_modules:
        try:
            importlib.import_module(module)
            print(f"  ✅ {module}")
        except ImportError as e:
            print(f"  ❌ {module}: {e}")
            return False
    
    return True

def test_basic_functionality():
    """Test basic numerical operations."""
    print("\n🧮 Testing basic functionality...")
    
    try:
        # Test NumPy
        x = np.random.randn(10)
        y = np.random.randn(10)
        z = x + y
        print("  ✅ NumPy operations")
        
        # Test SciPy
        from scipy.linalg import norm
        n = norm(x)
        print("  ✅ SciPy operations")
        
        # Test Matplotlib
        plt.figure(figsize=(4, 3))
        plt.plot(x, y, 'o-')
        plt.close()
        print("  ✅ Matplotlib plotting")
        
        return True
    except Exception as e:
        print(f"  ❌ Basic functionality test failed: {e}")
        return False

def test_benchmark_scripts():
    """Test that benchmark scripts can be imported."""
    print("\n📊 Testing benchmark scripts...")
    
    scripts = [
        'gravidy_st_benchmark',
        'simplex_gravidy_benchmark', 
        'box_gravidy_benchmark',
        'pos_gravidy_benchmark'
    ]
    
    for script in scripts:
        try:
            importlib.import_module(script)
            print(f"  ✅ {script}")
        except ImportError as e:
            print(f"  ❌ {script}: {e}")
            return False
    
    return True

def main():
    """Run all tests."""
    print("🚀 GRAVIDY Installation Test")
    print("=" * 40)
    
    # Check Python version
    print(f"🐍 Python version: {sys.version}")
    if sys.version_info < (3, 8):
        print("⚠️  Warning: Python 3.8+ recommended")
    
    # Run tests
    tests = [
        test_imports,
        test_basic_functionality,
        test_benchmark_scripts
    ]
    
    all_passed = True
    for test in tests:
        if not test():
            all_passed = False
    
    print("\n" + "=" * 40)
    if all_passed:
        print("🎉 All tests passed! GRAVIDY is ready to use.")
        print("\nNext steps:")
        print("1. Run a benchmark: python simplex_gravidy_benchmark.py")
        print("2. Check the README.md for detailed usage instructions")
        print("3. Explore the solver/ and utils/ directories")
    else:
        print("❌ Some tests failed. Please check the error messages above.")
        print("Try reinstalling dependencies: pip install -r requirements.txt")
    
    return all_passed

if __name__ == "__main__":
    main()
