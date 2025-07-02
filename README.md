# Variational-Weak-Formulation-RBFNets



# Mesh-Free RBF Solvers  
_A pure-Python micro-library that tackles three canonical 1-D problems with normalised radial basis functions, weak boundary/obstacle enforcement, and truncated-SVD stabilisation._

| PDE | Strong form | File | Demo |
|-----|-------------|------|------|
| ReactionDiffusion‐type | \(-u''+u=f\) | `rbf_reactiondiffusion.py` | `examples/demo_sin.py` |
| Poisson | \(-u''=f\) | `rbf_poisson.py` | `examples/demo_poly.py` |
| Classical obstacle | \(\max\{-u'',\,u-\psi\}=f\) | `rbf_obstacle.py` | `examples/demo_obstacle.py` |

---

## 📜 Problems in Detail  

### 1. Reaction-Diffusion 
\[
-u''(x) + u(x) = f(x), \quad x\!\in\!(a,b), \quad u(a)=g_a,\;u(b)=g_b.
\]

### 2. Poisson  
\[
-u''(x) = f(x), \quad x\!\in\!(a,b), \quad u(a)=g_a,\;u(b)=g_b.
\]

### 3. Classical Obstacle  
Find \(u\) such that  
\[
u(x)\ge\psi(x),\;
 -u''(x)\ge f(x),\;
 (u-\psi)\bigl(-u''-f\bigr)=0,\qquad x\!\in\!(a,b),
\]
with Dirichlet data \(u(a)=g_a,\;u(b)=g_b\).

All BCs (and the obstacle constraint) are imposed **weakly** via quadratic penalties; the ensuing ill-conditioned normal systems are inverted with **TSVD**.
