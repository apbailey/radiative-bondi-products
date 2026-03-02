# radiative-bondi-products
Scientific products associated with Bailey et al. works on radiative Bondi accretion 

## bonditools package
A lightweight package containing functions to use data and compute $f_{\rm acc}$

##  raw data
### /data/dimensionless_luminosity_facc.tab
This is a 4-column ascii table of the dimensionless model grid of Paper I of our radiative Bondi works. The columns represent $(\tau_B, L_\infty, \beta, f_{\rm acc})$ respectively. Entries with $f_{\rm acc}= 0.0$ correspond to models with no steady-state solution. Each parameter $\tau_B$, $\tilde{L}_\infty$, $\beta$ spans a logarithmic range of $[10^{-3}, 10^3]$ with 21 points. A simple wrapper (`interpolate_dimensionless_luminosity_table`) to read this data into a SciPy RegularGridInterpolator object exists in `bonditools.py`
