# radiative-bondi-products
Scientific products associated with Bailey et al. works on radiative Bondi accretion 

## How to use
Clone this repo and make sure your project knows where `bonditools.py` lives -- e.g. `sys.path.insert(1, PATH_TO_REPO)` -- then simply `import bonditools`. Make sure the `PATH_TO_TABLE` variable in `bonditools.py` points to the proper location of `facc.tab` if you intend to use the tabulated solutions for $f_{\rm acc}$. 

## bonditools.py
A lightweight collection of python scripts with two wrapper APIs (`compute_facc_from_cgs` and `compute_luminosity_facc_from_cgs`) for people to easily apply the results of our radiative Bondi calculations to planet population synthesis codes or other problems. See the docstrings in `bonditools.py` and the examples in `examples.ipynb` for further details.

## examples.ipynb
See this jupyter notebook for simple examples (planet population synthesis, reproducing figures from our paper, constructing a lookup table) on how to use these data/API for your problem.

## facc.tab
This is a 4-column ascii table of the dimensionless model grid of Paper I of our radiative Bondi works. The columns represent $(\tau_B, L_\infty, \beta, f_{\rm acc})$ respectively. Entries with $f_{\rm acc}= 0.0$ correspond to models with no steady-state solution. Each parameter $\tau_B$, $\tilde{L}_\infty$, $\beta$ spans a logarithmic range of $[10^{-3}, 10^3]$ with 21 points. There are some private table IO functions in `bonditools` (`_load_table()` and `_construct_table_interpolator`) to read this data, if you don't want to do it yourself.
