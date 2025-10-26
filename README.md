# RDD-Optimizer

An experimental project exploring randomized descent directions for poorly conditioned optimization problems.

## The Idea

Gradient descent on poorly conditioned problems is notoriously slow - it zigzags back and forth, taking tiny steps toward the optimum. I wondered: **what if we added some controlled randomness to the descent direction to help it escape this slow convergence pattern?**

### Randomized Descent Explained

Traditional gradient descent uses:
```
d = -∇f(x)    (always follow the negative gradient)
```

Randomized descent adds a random component:
```
d = -∇f(x) + η·u    (gradient + controlled random perturbation)
```

Where:
- `∇f(x)` is the gradient at point x
- `u` is a random vector (sampled from Uniform or Normal distribution)
- `η` is a hyperparameter controlling randomness magnitude
- We validate that `d·(-∇f) > 0` to guarantee descent (flip direction if needed)

**The hypothesis**: By introducing randomness, we might occasionally find better descent paths than the pure gradient, especially when dealing with elongated valleys characteristic of poorly conditioned problems.

## What I Found

**TL;DR**: It works beautifully in low dimensions (< 10), but performance falls off a cliff as dimensionality increases.

### Success Stories (2D-5D)

**2D Rosenbrock, κ=25,008:**
- Gradient Descent: Did not converge in 100k iterations
- Randomized Descent: 748 iterations ✓

**2D Rosenbrock, κ=133:**
- Gradient Descent: 804 iterations
- Randomized Descent: 91 iterations (8.8x faster!)

### The Dimensionality Problem

| Dim | GD Iters | RD Iters | Winner |
|-----|----------|----------|--------|
| 2   | 804      | 91       | RD 🎉  |
| 5   | 1144     | 764      | RD 🎉  |
| 10  | 2354     | 2248     | RD 🎉  |
| 20  | 5105     | 6190     | GD     |
| 50  | 14898    | 19043    | GD     |
| 100 | 37694    | 56807    | GD     |

**Why does it break down?** In high dimensions, the chance of randomly picking a good direction drops exponentially. Each dimension is essentially a coin flip - get the sign right or wrong. By 20 dimensions, you're trying to win a lottery.

## Repository Contents

- `KAUST_Numerical_Optimization_project_report.pdf` - Full research paper
- `Project_Experiments.ipynb` - Main experiments (2D Rosenbrock tests)
- `100_d_Experiment.ipynb` - High-dimensional scaling experiments
- `Randomization_and_normalization_ablations.ipynb` - Testing 8 different configurations
- `select_random_dim_10_coordinate_experiment.ipynb` - Randomizing only subset of dimensions
- `select_lowest_10_coordinates.ipynb` - Smart dimension selection attempt
- `100_d_Gradient_Variance_Based_Selection_Experiment.ipynb` - Variance-based selection

## Quick Start

```bash
git clone https://github.com/aallail/rdd-optimizer.git
cd rdd-optimizer
jupyter notebook Project_Experiments.ipynb
```

**Requirements**: `numpy`, `matplotlib`, `jupyter`, `scipy`

## Experiments Summary

**Experiments 1-2**: Proved the concept works in 2D with high condition numbers

**Experiment 3**: Discovered the dimensionality scaling problem (2D → 100D)

**Experiments 4-6**: Tried to fix high-D performance by smartly selecting which dimensions to randomize:
- Random subset
- Smallest gradient components
- Variance-based selection

None of these significantly helped. The fundamental problem remains.

**Experiment 7**: Ablation study on normalization and distributions
- Best: Normalized random vector from Uniform[-1,1] + normalized gradient
- Averaged over 100 trials: 2,952 iterations vs 4,063 for worst configuration

## Algorithm Pseudocode

```
Input: f(x), starting point x₀, tolerance ε, η
while |∇f(x)| > ε:
    g ← ∇f(x)
    u ← random_vector()
    d ← -g + η·u

    if d·(-g) ≤ 0:
        d ← -d

    t ← backtracking_line_search(f, x, d)
    x ← x + t·d

return x
```

## Key Takeaways

✅ **What worked:**
- Dramatic speedups in low-dimensional, poorly conditioned problems
- Simple to implement
- Sometimes converges where vanilla GD fails

❌ **What didn't:**
- Scales terribly with dimensionality
- Smart dimension selection strategies didn't help
- Adds computational overhead
- Stochastic behavior can be unpredictable

## Reflections

This was an experimental exploration of an interesting idea. While it didn't result in a universally applicable algorithm, the journey was valuable:

1. Sometimes simple ideas work in constrained settings
2. The curse of dimensionality is unforgiving
3. Intuition from low dimensions doesn't always scale
4. Negative results are results too

The method shows that randomization can genuinely help with poorly conditioned problems - just not in the high-dimensional regime where many modern ML problems live.

## Authors

**Ali Allail** & **Maryam Almaskin**
King Abdullah University of Science and Technology (KAUST), 2024

## References

[1] Zhang, L., Mahdavi, M., & Jin, R. (2013). Linear convergence with condition number independent access of full gradients. *NeurIPS*.

[2] Qu, Z., & Richtárik, P. (2016). Coordinate descent with arbitrary sampling I: Algorithms and complexity. *Optimization Methods and Software*.

[3] Kingma, D. P., & Ba, J. (2014). Adam: A method for stochastic optimization. *arXiv:1412.6980*.

[4] Qu, Z., & Richtárik, P. (2016). Coordinate descent with arbitrary sampling II: Expected separable overapproximation. *Optimization Methods and Software*.

---

*Sometimes the most interesting research is seeing where an idea breaks.*
