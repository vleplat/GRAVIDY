# GRAVIDY - a new Geometric Optimization Framework

Gravidy (GRAdient flows with Vanishing Jacobian DYnamics) turns constraints into geometry and then solves the resulting gradient flows with A-stable implicit steps. Instead of projecting or penalizing, Gravidy reparameterizes the feasible set—positivity (orthant), simplex, box, or Stiefel manifold—so feasibility is built in. The Jacobian of the map is used as the local metric, which naturally “damps” motion at active faces (vanishing derivatives / rank loss) and makes complementary slackness a kinematic outcome; at stationarity the dynamics satisfy KKT automatically. Discretization is done with robust implicit integrators: backward Euler (orthant/box) with Modified Gauss–Newton or KL-prox inner solves, and a trapezoidal/Cayley step on Stiefel that stays exactly feasible. The result is monotone descent with no stepsize cap (A-stability), strong empirical robustness at large steps, and clean theory: global convergence in convex settings and linear rates under relative strong convexity—while handling rank-deficient simplex geometry without hacks. We illustrate on NNLS, simplex/box least squares, and orthogonality problems, where the implicit flows are both stable and fast.

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
- GRAVIDY–St (Fast) - Our main algorithm with Newton-Krylov inner solver
- Wen–Yin Cayley - Classical method
- RGD–QR - Riemannian gradient descent

**Run it:**
```bash
python gravidy_st_benchmark.py
```

### 2. **Simplex Optimization** (`simplex_gravidy_benchmark.py`)
**What it does:** Optimizes functions on the probability simplex (sum of components = 1, all ≥ 0)
**What it compares:**
- GRAVIDY–Δ (KL-prox) - Our KL-proximal method with Newton-KKT inner solver
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
- GRAVIDY–box (Newton) - Our sigmoid reparameterization with damped Newton
- GRAVIDY–box (MGN) - Our sigmoid reparameterization with Modified Gauss-Newton
- APGD-box (Nesterov) - Accelerated projected gradient descent

**Run it:**
```bash
python box_gravidy_benchmark.py
```

### 4. **Positive Orthant Optimization (NNLS)** (`pos_gravidy_benchmark.py`)
**What it does:** Solves non-negative least squares problems (all variables ≥ 0)
**What it compares:**
- GRAVIDY–pos (Newton) - Our exponential reparameterization with damped Newton
- GRAVIDY–pos (MGN) - Our exponential reparameterization with Modified Gauss-Newton
- PGD+Nesterov - Accelerated projected gradient descent
- Proj-BB (Armijo) - Projected Barzilai-Borwein with line search
- MU (A≥0,b≥0) - Multiplicative updates (requires non-negative data)

**Run it:**
```bash
python pos_gravidy_benchmark.py
```

## 📊 Paper-Grade Benchmarks

For research and publication purposes, GRAVIDY also includes paper-grade benchmarks with multi-seed averaging and detailed metrics:

### Paper-Grade Scripts
- `gravidy_st_benchmark_paper.py` - Stiefel with gradient norm and feasibility plots
- `simplex_gravidy_benchmark_paper.py` - Simplex with KKT residuals and objective gaps
- `box_gravidy_benchmark_paper.py` - Box with KKT residuals and objective gaps  
- `pos_gravidy_benchmark_paper.py` - Positive orthant with KKT residuals and objective gaps

**Run any paper benchmark:**
```bash
python [method]_gravidy_benchmark_paper.py
```

### Paper-Grade Features
- **Multi-seed averaging**: 10 trials with ±1 standard deviation bands
- **KKT residuals**: `||x - Π_C(x - ∇Φ(x))||_2` for vector domains
- **Gradient norms**: `||∇Φ(X)||_F` for Stiefel manifold
- **Feasibility violations**: `||X^T X - I||_F` for Stiefel
- **Objective gaps**: `|f(x_k) - f^*|` for convergence analysis
- **Runtime reporting**: Wall-clock time measurements

## 📈 Understanding the Results

Each benchmark will:
1. **Print a summary table** showing final errors, objective values, iterations, and timing
2. **Display interactive plots** showing convergence behavior
3. **Save publication-ready figures** in the `figs/` folder

### 📈 What the Plots Show

- **Error vs Iterations**: How quickly each method converges to the solution
- **Objective vs Time**: How the objective function decreases over time
- **KKT Residuals**: Optimality measure for constrained optimization
- **Final Solutions**: Comparison of final results vs ground truth

### 📁 Generated Files

After running any benchmark, you'll find these files in the `figs/` folder:

**Regular benchmarks:**
- `*_benchmark.pdf` - Complete comparison plots
- `*_err_vs_it.pdf` - Error convergence plots
- `*_f_vs_time.pdf` - Objective vs time plots

**Paper benchmarks:**
- `stiefel_grad_vs_it.pdf` - Gradient norm vs iterations (Stiefel)
- `stiefel_feas_vs_time.pdf` - Feasibility vs time (Stiefel)
- `simplex_err_vs_it.pdf` - Error vs iterations (Simplex)
- `simplex_f_vs_time.pdf` - Objective vs time (Simplex)
- `box_err_vs_it.pdf` - Error vs iterations (Box)
- `box_f_vs_time.pdf` - Objective vs time (Box)
- `pos_err_vs_it.pdf` - Error vs iterations (Positive orthant)
- `pos_f_vs_time.pdf` - Objective vs time (Positive orthant)

## 🔬 Understanding GRAVIDY

### What Makes GRAVIDY Special?

GRAVIDY uses **implicit geometric integration** to solve constrained optimization problems:

1. **GRAVIDY–St**: Implicit Cayley step with Newton-Krylov (GMRES) inner solver for Stiefel manifolds
2. **GRAVIDY–Δ**: Implicit KL-proximal Newton-KKT for simplex constraints
3. **GRAVIDY–box**: Implicit reparameterization using sigmoid functions with two inner solvers:
   - **Newton**: Damped Newton solver
   - **MGN**: Modified Gauss-Newton solver (recommended near active bounds)
4. **GRAVIDY–pos**: Implicit Euler in log-coordinates for positive orthant with two inner solvers:
   - **Newton**: Damped Newton solver
   - **MGN**: Modified Gauss-Newton solver

### Key Advantages

- **Fast convergence**: Often reaches high accuracy in few iterations
- **Robust**: Handles ill-conditioned problems well
- **Theoretically sound**: Based on geometric integration principles
- **Practical**: Easy to use and tune
- **Multiple inner solvers**: Choose between Newton and MGN based on problem characteristics

## 🛠️ Project Structure

```
GRAVIDY/
├── solver/                    # Optimization algorithms
│   ├── gravidy_st_fast.py    # Stiefel manifold solver (Newton-Krylov)
│   ├── gravidy_st_nk.py      # Stiefel manifold solver (Newton-Krylov)
│   ├── gravidy_st_nr_dense.py # Stiefel manifold solver (dense Newton)
│   ├── gravidy_delta.py      # Simplex solver (KL-proximal)
│   ├── gravidy_box.py        # Box constraint solver (Newton)
│   ├── gravidy_box_mgn.py    # Box constraint solver (MGN)
│   ├── gravidy_pos.py        # Positive orthant solver (Newton)
│   ├── gravidy_pos_mgn.py    # Positive orthant solver (MGN)
│   ├── wy_cayley.py          # Wen-Yin Cayley method
│   ├── rgd_qr.py             # Riemannian gradient descent
│   ├── pgd_simplex.py        # Projected gradient descent (simplex)
│   ├── apgd_simplex.py       # Accelerated PGD (simplex)
│   ├── emd_simplex.py        # Entropic mirror descent
│   ├── pgd_box.py            # Projected gradient descent (box)
│   ├── apgd_box.py           # Accelerated PGD (box)
│   ├── pgd_pos.py            # Projected gradient descent (positive)
│   ├── apgd_pos.py           # Accelerated PGD (positive)
│   ├── proj_bb_pos.py        # Projected Barzilai-Borwein
│   └── mu_pos.py             # Multiplicative updates
├── utils/                     # Utility functions
│   ├── objective.py          # Objective function definitions
│   ├── simplex_utils.py      # Simplex utilities
│   ├── box_objective.py      # Box constraint objectives
│   └── pos_objective.py      # Positive orthant objectives
├── *_gravidy_benchmark.py    # Regular benchmark scripts
├── *_gravidy_benchmark_paper.py # Paper-grade benchmark scripts
├── requirements.txt          # Python dependencies
├── test_installation.py      # Installation verification
├── demo.py                   # Quick demonstration
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

### Inner Solver Selection

For box and positive orthant problems, you can choose between inner solvers:
- **Newton**: Better for well-conditioned problems
- **MGN**: Better for ill-conditioned problems or near active bounds

### Adding New Problems

To add your own optimization problem:
1. Create objective function in `utils/`
2. Add solver in `solver/`
3. Create benchmark script following the existing pattern

## 📄 Citation

If you use GRAVIDY in your research, please cite our paper:

```bibtex
@article{gravidy2025,
  title={The Geometry of Constrained Optimization: Constrained Gradient Flows via Reparameterization: A-Stable Implicit Schemes, KKT from Stationarity, and Geometry-Respecting Algorithms},
  author={Valentin Leplat},
  journal={arXiv},
  year={2025}
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

For questions, bug reports, or need help, please contact:
**v dot leplat [at] innopolis dot ru**
