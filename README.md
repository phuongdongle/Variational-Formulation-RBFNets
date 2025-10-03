# Variational-Weak-Formulation-RBFNets


A pure-Python micro-library that tackles three canonical 1-D problems with
normalised radial basis functions (RBFs), weak boundary/obstacle enforcement,
and truncated-SVD stabilisation.

| PDE                | Strong form                                                                 | File                      |
|--------------------|------------------------------------------------------------------------------|---------------------------|
| Reaction–Diffusion | <img src="https://latex.codecogs.com/svg.image?-u''&plus;u=f" alt="-u'' + u = f"> | `rbf_reactiondiffusion.py` |
| Poisson            | <img src="https://latex.codecogs.com/svg.image?-u''=f" alt="-u'' = f">          | `rbf_poisson.py`            |
| Classical obstacle | <img src="https://latex.codecogs.com/svg.image?u(x)%5Cge%5Cpsi(x)%2C%5C;-u''(x)%5Cge%20f(x)%2C%5C;(u-%5Cpsi)(-u''-f)%3D0" alt="u ≥ ψ,  -u'' ≥ f,  (u-ψ)(-u''-f)=0"> | `rbf_obstacle.py`           |


## 📜 Problems in Detail

### 1  Reaction–Diffusion

<img src="https://latex.codecogs.com/svg.image?-u''(x)&plus;u(x)=f(x),\;\;x\in(a,b),\;\;u(a)=g_a,\;u(b)=g_b" alt="-u'' + u = f (a,b) with Dirichlet BCs">

### 2  Poisson

<img src="https://latex.codecogs.com/svg.image?-u''(x)=f(x),\;\;x\in(a,b),\;\;u(a)=g_a,\;u(b)=g_b" alt="-u'' = f with BCs">

The comparison of Approximation and Exact Solution for 1D - Poisson PDE as follows: 

![Convergence of Poisson solver](PoissonPDE/figures/fig1_poisson.png)

The variational method for Poisson PDE (1D) is shown to benchmark with other numerical methods: finite-difference, Galerkin, neural network (feed-forward). 

![Convergence of Poisson solver with other methods](PoissonPDE/figures/fig2_poisson_convergence.png)


### 3  Classical Obstacle

 <img src="https://latex.codecogs.com/svg.image?u(x)%5Cge%5Cpsi(x)%2C%5C;-u''(x)%5Cge%20f(x)%2C%5C;(u-%5Cpsi)(-u''-f)%3D0" alt="u ≥ ψ,  -u'' ≥ f,  (u-ψ)(-u''-f)=0">
Dirichlet data: `u(a)=g_a`, `u(b)=g_b`.

_Boundary conditions (and the obstacle constraint) are imposed weakly via a
quadratic penalty; the resulting ill-conditioned systems are solved with
truncated SVD (TSVD)._ 
