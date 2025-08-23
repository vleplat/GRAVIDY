# Contributing to GRAVIDY

We welcome contributions to GRAVIDY! This document provides guidelines for contributing to the project.

## 🎯 Types of Contributions

### 🐛 Bug Reports
- Use the GitHub issue tracker
- Include a minimal reproducible example
- Specify your Python version and operating system
- Include the full error traceback

### 💡 Feature Requests
- Describe the use case and motivation
- Explain why the feature would be valuable
- Consider implementation complexity

### 🔧 Code Contributions
- New optimization algorithms
- Performance improvements
- Better documentation
- Additional benchmark problems

## 🚀 Development Setup

1. **Fork and clone the repository**
```bash
git clone https://github.com/yourusername/GRAVIDY.git
cd GRAVIDY
```

2. **Create a virtual environment**
```bash
python -m venv gravidy_dev
source gravidy_dev/bin/activate  # On Windows: gravidy_dev\Scripts\activate
```

3. **Install in development mode**
```bash
pip install -r requirements.txt
python test_installation.py
```

4. **Run tests**
```bash
python demo.py
python simplex_gravidy_benchmark.py
```

## 📝 Code Standards

### Style Guidelines
- Follow PEP 8 for Python code style
- Use meaningful variable and function names
- Add docstrings for all public functions and classes
- Keep functions focused and reasonably sized

### Documentation
- Update README.md if adding new features
- Include examples for new functionality
- Document any new dependencies

### Testing
- Test new features with the existing benchmark scripts
- Ensure backward compatibility
- Include appropriate error handling

## 🔄 Pull Request Process

1. **Create a feature branch**
```bash
git checkout -b feature/your-feature-name
```

2. **Make your changes**
- Follow the code standards above
- Test your changes thoroughly
- Update documentation as needed

3. **Commit your changes**
```bash
git add .
git commit -m "Add: brief description of your changes"
```

4. **Push and create a pull request**
```bash
git push origin feature/your-feature-name
```

5. **Pull request guidelines**
- Use a clear, descriptive title
- Explain what your changes do and why
- Reference any related issues
- Include any performance impact information

## 🧪 Adding New Solvers

If you're adding a new optimization algorithm:

1. **Create the solver file**
```python
# solver/your_solver.py
def YourSolver(problem, max_iters=400, **kwargs):
    """
    Your solver description.
    
    Args:
        problem: Objective function instance
        max_iters: Maximum iterations
        **kwargs: Additional parameters
        
    Returns:
        x: Final solution
        history: List of (iter, obj_val, grad_norm, time) tuples
    """
    # Implementation here
    return x, history
```

2. **Add to benchmarks**
- Include your solver in the appropriate benchmark script
- Follow the existing pattern for timing and evaluation

3. **Update documentation**
- Add description to README.md
- Include any specific parameters or requirements

## 📊 Adding New Problem Types

For new constraint sets or objectives:

1. **Create objective class**
```python
# utils/your_objective.py
class YourObjective:
    def __init__(self, ...):
        # Problem setup
        
    def f(self, x):
        # Objective function
        
    def grad(self, x):
        # Gradient
        
    def project(self, x):
        # Projection onto constraint set
```

2. **Create benchmark script**
- Follow the pattern of existing benchmarks
- Include multiple algorithm comparisons
- Generate appropriate figures

## 🎨 Coding Examples

### Adding a New Algorithm
```python
def NewAlgorithm(problem, eta=1.0, max_iters=100, tol=1e-8, verbose=False):
    x = initialize_solution(problem)
    history = []
    
    for k in range(max_iters):
        obj_val = problem.f(x)
        grad = problem.grad(x)
        grad_norm = np.linalg.norm(grad)
        
        history.append((k, obj_val, grad_norm, time.time()))
        
        if grad_norm <= tol:
            break
            
        x = update_step(x, grad, eta)
    
    return x, history
```

### Adding Problem Generator
```python
def create_new_problem(n, seed=None):
    if seed is not None:
        np.random.seed(seed)
    
    # Generate problem data
    A = create_matrix(n)
    b = create_target(n)
    x_star = create_solution(n)
    
    problem = NewProblemClass(A, b)
    return A, b, x_star, problem
```

## ❓ Questions?

- Open an issue for questions about contributing
- Check existing issues and pull requests first
- Be respectful and constructive in all interactions

## 📄 License

By contributing to GRAVIDY, you agree that your contributions will be licensed under the MIT License.
