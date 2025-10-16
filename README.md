# Variational-Weak-Formulation-RBFNets


A pure-Python micro-library that tackles three canonical 1-D problems with
normalised radial basis functions (RBFs), weak boundary/obstacle enforcement,
and truncated-SVD stabilisation.

| PDE                | Strong form                                                                 | File                      |
|--------------------|------------------------------------------------------------------------------|---------------------------|
| Reaction–Diffusion | <img src="https://latex.codecogs.com/svg.image?-u''&plus;u=f" alt="-u'' + u = f"> | `rbf_reactiondiffusion.py` |
| Poisson            | <img src="https://latex.codecogs.com/svg.image?-u''=f" alt="-u'' = f">          | `rbf_poisson.py`            |
| Classical obstacle | <img src="https://latex.codecogs.com/svg.image?u(x)%5Cge%5Cpsi(x)%2C%5C;-u''(x)%5Cge%20f(x)%2C%5C;(u-%5Cpsi)(-u''-f)%3D0" alt="u ≥ ψ,  -u'' ≥ f,  (u-ψ)(-u''-f)=0"> | `rbf_obstacle.py`           |

#### Requirements
```text
numpy>=1.24
scipy>=1.10
matplotlib>=3.7
numba>=0.58
pillow>=10.0
```

## 🗂️ Repository Layout

```text
Variational-Weak-Formulation-RBFNets/
├── scripts/                   # reproducible experiments (make figures)
│   ├── 1D_Poisson_BenchmarkN.ipynb
│   ├── 1D_Poisson_BenchmarkRBFs.ipynb
│   ├── 1D_ReactionDiffusion_RBF.ipynb
│   ├── 1D_ReactionDiffusion_BenchmarkN.ipynb
│   ├── 1D_ReactionDiffusion_BenchmarkRBFs.ipynb
│   └── 1D_ObstacleProblem_BenchmarkN.ipynb
│   └── 1D_ObstacleProblem_BenchmarkN_Problem2.ipynb
│   └── 1D_ObstacleProblem_BenchmarkRBF.ipynb
├── PoissonPDE/
│   └── figures/
│       ├── fig1_poisson.png
│       ├── fig2_poisson_convergence.png
│       ├── fig3_poisson.png
│       ├── fig4_poisson.png
│       ├── fig14_poisson.png
│       └── fig15_poisson.png
├── ReactionDiffusion/
│   └── figures/
│       ├── fig1_reactiondiffusion.png
│       ├── fig2_reactiondiffusion_convergence.png
│       ├── fig3_reactiondiffusion.png
│       ├── fig13_reactiondiffusion.png
│       └── fig14_reactiondiffusion.png
├── requirements.txt
├── README.md
└── LICENSE
```



## 📜 Problems in Detail

### 1  Poisson

<img src="https://latex.codecogs.com/svg.image?-u''(x)=f(x),\;\;x\in(a,b),\;\;u(a)=g_a,\;u(b)=g_b" alt="-u'' = f with BCs">

The comparison of Approximation and Exact Solution for 1D - Poisson PDE as follows: 

![Convergence of Poisson solver](PoissonPDE/figures/fig1_poisson.png)

The variational method for Poisson PDE (1D) is shown to benchmark with other numerical methods: finite-difference, Galerkin, neural network (feed-forward). 

![Convergence of Poisson solver with other methods](PoissonPDE/figures/fig2_poisson_convergence.png) 

The expansion domain factor $T$ is to guarantee for error convergence versus parameter for choice of kernel shape $b = c(T, \tau) \times N$ in Gaussian RBF.

![T expansion factor for Poisson](PoissonPDE/figures/fig4_poisson.png) 

### 2  Reaction–Diffusion

<img src="https://latex.codecogs.com/svg.image?-u''(x)&plus;u(x)=f(x),\;\;x\in(a,b),\;\;u(a)=g_a,\;u(b)=g_b" alt="-u'' + u = f (a,b) with Dirichlet BCs">


The comparison of Approximation and Exact Solution for 1D - Poisson PDE as follows: 

![Convergence of RD solver](ReactionDiffusion/figures/fig1_reactiondiffusion.png)

The variational method for Poisson PDE (1D) is shown to benchmark with other numerical methods: finite-difference, Galerkin, neural network (feed-forward). 

![Convergence of Poisson solver with other methods](ReactionDiffusion/figures/fig2_reactiondiffusion_convergence.png) 

The expansion domain factor $T$ is to guarantee for error convergence versus parameter for choice of kernel shape $b = c(T, \tau) \times N$ in Gaussian RBF.

![T expansion factor for Poisson](ReactionDiffusion/figures/fig3_reactiondiffusion.png) 


### Kernel Comparison and Boundary-Penalty Parameter Sweeps

#### 1D Poisson PDE
<table>
  <tr>
    <td><img src="PoissonPDE/figures/fig3_poisson.png" width="400" alt="(a)"></td>
    <td><img src="PoissonPDE/figures/fig2_local_convergence_order.png" width="400" alt="(b)"></td>
  </tr>
  <tr>
    <td><img src="PoissonPDE/figures/fig13_poisson.png" width="400" alt="(c)"></td>
    <td><img src="PoissonPDE/figures/fig14_poisson.png" width="400" alt="(d)"></td>
  </tr>
</table>
<p><em>(a)–(d) Panel results for Poisson with normalized RBFs approximation: Comparison of basis functions, convergence order, boundary-penalty parameter sweeps.</em></p>

#### 1D Reaction-Diffusion PDE
<table>
  <tr>
    <td><img src="ReactionDiffusion/figures/fig4_reactiondiffusion.png" width="400" alt="(a)"></td>
    <td><img src="ReactionDiffusion/figures/fig2_local_convergence_order.png" width="400" alt="(b)"></td>
  </tr>
  <tr>
    <td><img src="ReactionDiffusion/figures/fig13_reactiondiffusion.png" width="400" alt="(c)"></td>
    <td><img src="ReactionDiffusion/figures/fig14_reactiondiffusion.png" width="400" alt="(d)"></td>
  </tr>
</table>
<p><em>(a)–(d) Panel results for Reaction-Diffusion with normalized RBFs approximation: Comparison of basis functions, convergence order, boundary-penalty parameter sweeps.</em></p>



### 3  Classical Obstacle

 <img src="https://latex.codecogs.com/svg.image?u(x)%5Cge%5Cpsi(x)%2C%5C;-u''(x)%5Cge%20f(x)%2C%5C;(u-%5Cpsi)(-u''-f)%3D0" alt="u ≥ ψ,  -u'' ≥ f,  (u-ψ)(-u''-f)=0">
Dirichlet data: `u(a)=g_a`, `u(b)=g_b`.

_Boundary conditions (and the obstacle constraint) are imposed weakly via a
quadratic penalty; the resulting ill-conditioned systems are solved with
truncated SVD (TSVD)._ 

#### 1D case: 

##### One-Bump Obstacle Problem: 



The comparison of Approximation and Exact Solution for 1D - Obstacle (Linear Problem) PDE as follows: 

![Convergence of Obstacle Single Bump solver](ClassicalObstacleProblem/figures/1D/One-Bump/fig_obstacle1D_single_bump.png)




##### Two-Bump Obstacle Problem: 



The comparison of Approximation and Exact Solution for 1D - Obstacle (Linear Problem) PDE as follows: 


![Convergence of Obstacle Double Bump solver](ClassicalObstacleProblem/figures/1D/Two-Bump/fig_obstacle1D_two_bump.png)





# (optional) create an environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Installation

```text
pip install -r requirements.txt
```







