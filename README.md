# radiative-bondi-products

## facc.tab
This is a 4-column ascii table of the dimensionless model grid of Paper I of our radiative Bondi works. The columns represent (τ_B, L̃_∞, β, f_acc) respectively. Entries with $f_{\rm acc}= 0.0$ correspond to models with no steady-state solution. Each parameter $\tau_B$, $\tilde{L}_\infty$, $\beta$ spans a logarithmic range of $[10^{-3}, 10^3]$ with 21 points, e.g. --
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
