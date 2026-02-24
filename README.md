# radiative-bondi-products

## facc.tab
This is a 4-column ascii table of the dimensionless model grid of Paper I of our radiative Bondi works. The columns represent $(\tau_B, L_\infty, \beta, f_{\rm acc})$ respectively. Entries with $f_{\rm acc}= 0.0$ correspond to models with no steady-state solution. Each parameter $\tau_B$, $\tilde{L}_\infty$, $\beta$ spans a logarithmic range of $[10^{-3}, 10^3]$ with 21 points, e.g. --
```python
import numpy as np
grid = np.geomspace(1e-3, 1e3, 21)
tau, lum, beta = np.meshgrid(grid, grid, grid, indexing="ij")
```

 To read the data of `facc.tab` and place each column into a corresponding $(21\times 21\times 21)$ array in python for example:

```python
import numpy as np
data = np.loadtxt("facc.tab")
# Split columns
tau = data[:, 0].reshape((21, 21, 21))
lum = data[:, 1].reshape((21, 21, 21))
beta = data[:, 2].reshape((21, 21, 21))
facc = data[:, 3].reshape((21, 21, 21))
```
## Analytic Formula for $f_{\rm acc}$
Here is a simple python function for the analytic form of $f_{\rm acc}$ described in the appendix of our first paper. The inputs may be scalars or arrays with commensurate shapes.

```python
def facc(tau, lum, beta):
    tau, lum, beta = np.asarray(tau), np.asarray(lum), np.asarray(beta)
    tau, lum, beta = np.broadcast_arrays(tau, lum, beta)
    f = np.full(tau.shape, np.nan, dtype=float)

    a1 = 10
    a2 = 1
    a3 = 10
    
    # isothermal
    iso = (lum <= np.minimum(a3, a2/tau)) & (lum >= tau*beta*a1)
    f[iso] = 1.0

    # thin
    thin = (tau <= a2*lum/a3**2) & (lum >= a3)
    f[thin] = (lum[thin]/a3)**(-5/4)

    # thick 
    thick = (a2/tau < lum) & (lum <= a2*a3**2*tau)
    thick_boundary = (tau/a2)**(5/11)*(a1*a2*beta)**(8/11)
    thick1 = thick & (lum >= thick_boundary)
    thick2 = thick & (lum < thick_boundary)

    f[thick1] = (lum[thick1]*tau[thick1]/a2)**(-5/8)
    f[thick2] = (a1*tau[thick2]**2*beta[thick2]/a2)**(-5/11)

    return f.item() if f.ndim == 0 else f
```
