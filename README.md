# Variational-Weak-Formulation-RBFNets


A pure-Python micro-library that tackles three canonical 1-D problems with
normalised radial basis functions (RBFs), weak boundary/obstacle enforcement,
and truncated-SVD stabilisation.

| PDE                | Strong form                                                                 | File                |
|--------------------|------------------------------------------------------------------------------|---------------------|
| Reaction–Diffusion | <img src="https://latex.codecogs.com/svg.image?-u''&plus;u=f" alt="-u'' + u = f"> | `rbf_reactiondiffusion.py` |
| Poisson            | <img src="https://latex.codecogs.com/svg.image?-u''=f" alt="-u'' = f">          | `rbf_poisson.py`          |
| Classical obstacle | <img src="https://latex.codecogs.com/svg.image?%3F" alt="\begin{cases}u(x)\ge\psi(x)\\-u''(x)\ge f(x)\\(u-\psi)(-u''-f)=0\end{cases}\quad x\in(a,b)">                    | `rbf_obstacle.py`         |


## 📜 Problems in Detail

### 1  Reaction–Diffusion

<img src="https://latex.codecogs.com/svg.image?-u''(x)&plus;u(x)=f(x),\;\;x\in(a,b),\;\;u(a)=g_a,\;u(b)=g_b" alt="-u'' + u = f (a,b) with Dirichlet BCs">

### 2  Poisson

<img src="https://latex.codecogs.com/svg.image?-u''(x)=f(x),\;\;x\in(a,b),\;\;u(a)=g_a,\;u(b)=g_b" alt="-u'' = f with BCs">

### 3  Classical Obstacle

<img src="https://latex.codecogs.com/svg.image?\begin{cases}u(x)\ge\psi(x)\\-u''(x)\ge f(x)\\(u-\psi)(-u''-f)=0\end{cases}\quad x\in(a,b)" alt="obstacle complementarity">

Dirichlet data: `u(a)=g_a`, `u(b)=g_b`.

_Boundary conditions (and the obstacle constraint) are imposed weakly via a
quadratic penalty; the resulting ill-conditioned systems are solved with
truncated SVD (TSVD)._ 
