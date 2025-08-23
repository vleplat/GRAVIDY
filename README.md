# GRAVIDY: Geometric Optimization Framework

A comprehensive Python framework for geometric optimization on various constraint sets including Stiefel manifolds, simplex, box constraints, and positive orthant (NNLS).

## 🚀 Quick Start Guide

This guide will walk you through installing and running GRAVIDY step-by-step, even if you're new to Python!

### 📋 Prerequisites

Before we start, you'll need:
- **Python 3.8 or higher** (we'll help you check this!)
- **Git** (to download our code)
- **Basic command line knowledge** (we'll guide you through everything)

### 🔧 Step 1: Check Your Python Installation

First, let's make sure you have Python installed:

**On Windows:**
```bash
python --version
```

**On Mac/Linux:**
```bash
python3 --version
```

You should see something like `Python 3.8.x` or higher. If you get an error, download Python from [python.org](https://www.python.org/downloads/).

### 📥 Step 2: Download GRAVIDY

**Option A: Using Git (recommended)**
Open your terminal/command prompt and run:

```bash
git clone https://github.com/vleplat/GRAVIDY.git
cd GRAVIDY
```

**Option B: Download ZIP file**
If the git command fails or you don't have Git installed:
1. Go to https://github.com/vleplat/GRAVIDY
2. Click the green "Code" button
3. Select "Download ZIP"
4. Extract the ZIP file to your desired location
5. Open terminal/command prompt and navigate to the extracted folder:
   ```bash
   cd GRAVIDY
   ```

### 🐍 Step 3: Create a Virtual Environment (Recommended)

Virtual environments keep your project dependencies separate from other Python projects. This is a best practice!

**On Windows:**
```bash
python -m venv gravidy_env
gravidy_env\Scripts\activate
```

**On Mac/Linux:**
```bash
python3 -m venv gravidy_env
source gravidy_env/bin/activate
```

You'll know it worked when you see `(gravidy_env)` at the beginning of your command line.

### 📦 Step 4: Install Dependencies

Now install the required packages:

```bash
pip install -r requirements.txt
```

This will install:
- **NumPy**: For numerical computations
- **Matplotlib**: For creating plots
- **SciPy**: For scientific computing

### ✅ Step 5: Verify Installation

Let's make sure everything is working:

```bash
python test_installation.py
```

This will test all components and tell you if everything is set up correctly.

### 🎬 Step 6: Try the Demo

Run a quick demo to see GRAVIDY in action:

```bash
python demo.py
```

This shows GRAVIDY solving three different types of optimization problems!

## 🎯 Running the Benchmarks

GRAVIDY includes 4 main benchmark scripts that compare different optimization algorithms:

### 1. **Stiefel Manifold Optimization** (`gravidy_st_benchmark.py`)
**What it does:** Optimizes functions on the Stiefel manifold (matrices with orthogonal columns)
**What it compares:**
- GRAVIDY–St (Fast) - Our main algorithm
- GRAVIDY–St (NR-Dense) - Dense variant
- Wen–Yin Cayley - Classical method
- RGD–QR - Riemannian gradient descent

**Run it:**
```bash
python gravidy_st_benchmark.py
```

### 2. **Simplex Optimization** (`simplex_gravidy_benchmark.py`)
**What it does:** Optimizes functions on the probability simplex (sum of components = 1, all ≥ 0)
**What it compares:**
- GRAVIDY–Δ (KL-prox) - Our KL-proximal method
- GRAVIDY–Δ (MGN variant) - Modified Gauss-Newton variant
- PGD (baseline) - Projected gradient descent
- APGD (Nesterov) - Accelerated projected gradient descent
- EMD (baseline) - Entropic mirror descent

**Run it:**
```bash
python simplex_gravidy_benchmark.py
```

### 3. **Box-Constrained Optimization** (`box_gravidy_benchmark.py`)
**What it does:** Optimizes functions with box constraints (each variable in [lo, hi])
**What it compares:**
- GRAVIDY–box - Our sigmoid reparameterization method
- APGD-box (Nesterov) - Accelerated projected gradient descent

**Run it:**
```bash
python box_gravidy_benchmark.py
```

### 4. **Positive Orthant Optimization (NNLS)** (`pos_gravidy_benchmark.py`)
**What it does:** Solves non-negative least squares problems (all variables ≥ 0)
**What it compares:**
- GRAVIDY–pos - Our exponential reparameterization method
- PGD+Nesterov - Accelerated projected gradient descent
- Proj-BB (Armijo) - Projected Barzilai-Borwein with line search
- MU (A≥0,b≥0) - Multiplicative updates (requires non-negative data)

**Run it:**
```bash
python pos_gravidy_benchmark.py
```

## 📊 Understanding the Results

Each benchmark will:
1. **Print a summary table** showing final errors, objective values, iterations, and timing
2. **Display interactive plots** showing convergence behavior
3. **Save publication-ready figures** in the `figs/` folder

### 📈 What the Plots Show

- **Error vs Iterations**: How quickly each method converges to the solution
- **Objective vs Time**: How the objective function decreases over time
- **Final Solutions**: Comparison of final results vs ground truth

### 📁 Generated Files

After running any benchmark, you'll find these files in the `figs/` folder:
- `*_benchmark.pdf` - Complete comparison plots
- `*_err_vs_it.pdf` - Error convergence plots
- `*_f_vs_time.pdf` - Objective vs time plots

## 🔬 Understanding GRAVIDY

### What Makes GRAVIDY Special?

GRAVIDY uses **implicit geometric integration** to solve constrained optimization problems:

1. **GRAVIDY–St**: Implicit Cayley step with inner solver for Stiefel manifolds
2. **GRAVIDY–Δ**: Implicit KL-proximal Newton-KKT for simplex constraints
3. **GRAVIDY–box**: Implicit reparameterization using sigmoid functions
4. **GRAVIDY–pos**: Implicit Euler in log-coordinates for positive orthant

### Key Advantages

- **Fast convergence**: Often reaches high accuracy in few iterations
- **Robust**: Handles ill-conditioned problems well
- **Theoretically sound**: Based on geometric integration principles
- **Practical**: Easy to use and tune

## 🛠️ Project Structure

```
GRAVIDY/
├── solver/                    # Optimization algorithms
│   ├── gravidy_st_*.py       # Stiefel manifold solvers
│   ├── gravidy_delta_*.py    # Simplex solvers
│   ├── gravidy_box.py        # Box constraint solver
│   ├── gravidy_pos.py        # Positive orthant solver
│   └── [competitor methods]  # Baseline algorithms
├── utils/                     # Utility functions
│   ├── objective.py          # Objective function definitions
│   ├── simplex_utils.py      # Simplex utilities
│   ├── box_objective.py      # Box constraint objectives
│   └── pos_objective.py      # Positive orthant objectives
├── *_gravidy_benchmark.py    # Main benchmark scripts
├── requirements.txt          # Python dependencies
└── README.md                # This file
```

## 🐛 Troubleshooting

### Common Issues

**"python: command not found"**
- Install Python from [python.org](https://www.python.org/downloads/)
- Make sure to check "Add Python to PATH" during installation

**"pip: command not found"**
- Try `python -m pip` instead of `pip`
- Or install pip: `python -m ensurepip --upgrade`

**"ModuleNotFoundError"**
- Make sure you activated your virtual environment
- Reinstall dependencies: `pip install -r requirements.txt`

**"Permission denied"**
- On Mac/Linux, you might need `sudo` for some commands
- Or use `--user` flag: `pip install --user -r requirements.txt`

### Getting Help

If you encounter issues:
1. Check that Python version is 3.8+
2. Verify virtual environment is activated
3. Reinstall dependencies
4. Check the error message for specific details

## 📚 Advanced Usage

### Customizing Parameters

Each benchmark script has parameters you can modify:
- `eta`: Step size parameter (higher = more aggressive)
- `max_iters`: Maximum iterations
- `n, m`: Problem dimensions
- `seed`: Random seed for reproducibility

### Adding New Problems

To add your own optimization problem:
1. Create objective function in `utils/`
2. Add solver in `solver/`
3. Create benchmark script following the existing pattern

## 📄 Citation

If you use GRAVIDY in your research, please cite our paper:

```bibtex
@article{gravidy2024,
  title={GRAVIDY: Geometric Optimization via Implicit Integration},
  author={Your Name and Co-authors},
  journal={Journal Name},
  year={2024}
}
```

## 🤝 Contributing

We welcome contributions! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

**Happy optimizing! 🚀**

If you have questions or need help, please open an issue on GitHub.
