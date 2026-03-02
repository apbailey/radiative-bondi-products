"""
TODO: description of this file
"""

import numpy as np
import warnings 
from functools import lru_cache
from scipy.interpolate import RegularGridInterpolator

# ----------------------
# Physical constants (CGS)
# ----------------------
# Fundamental constants
G_CGS        = 6.67430e-8                # Gravitational constant [cm^3 g^-1 s^-2]
C_CGS        = 2.99792458e10             # Speed of light [cm s^-1]
KB_CGS       = 1.380649e-16              # Boltzmann constant [erg K^-1]
MP_CGS       = 1.67262192369e-24         # Proton mass [g]
SIGMA_SB_CGS = 5.670374419e-5            # Stefan-Boltzmann constant [erg cm^-2 s^-1 K^-4]
A_RAD_CGS    = 4 * SIGMA_SB_CGS / C_CGS  # Radiation constant [erg cm^-3 K^-4]

# Astronomical constants
MSUN_CGS     = 1.98847e33      # Solar mass [g]
LSUN_CGS     = 3.828e33        # Solar luminosity [erg s^-1]
AU_CGS       = 1.495978707e13  # Astronomical unit [cm]

@lru_cache(maxsize=None)
def interpolate_dimensionless_luminosity_table(method: str, extrapolate: bool):
    """
    Wrapper for using SciPy RegularGridInterpolator on dimensionless_luminosity_facc.tab
    
    Parameters
    ----------
    method : str, optional
        The method of interpolation to perform. Supported are "linear",
        "nearest", "slinear", "cubic", "quintic" and "pchip". This
        parameter will become the default for the object's ``__call__``
        method.

    extrapolate : bool, optional
        Whether to extrapolate points outside the interpolation domain.
        If True, then RegularGridInterpolators extrapolation is performed.
        If False, then np.nan is returned for points outside the domain.
        
    Returns
    -------
    A RegularGridInterpolator object

    """
    data = np.loadtxt("./tables/dimensionless_luminosity_facc.tab")
    data = data.reshape((4,21,21,21))
    span = data[0,:,0,0]
    bounds_error = False
    kwargs = dict(method=method, bounds_error=bounds_error)
    if extrapolate:
        kwargs["fill_value"] = None
    return RegularGridInterpolator((span, span, span), data[3], **kwargs)

def adiabatic_mdot(gamma, mass, rho, temp, mu):
    """
    Compute adibatic Bondi accretion rate in CGS units

    Parameters
    ----------
    gamma : float or array-like
        Ratio of specific heats
    mass : float or array-like
        Mass of the accretor (in grams)
    rho : float or array-like
        Mass density of the ambient medium (in g/cm^3)
    temp : float or array-like
        Temperature of the ambient medium (in K)
    mu : float or array-like
        Mean molecular weight

    Returns
    -------
    mdot : float or ndarray
        Bondi accretion rate (in g/s)
    """
    q = (2.0 / (5.0 - 3.0*gamma))**((5.0 - 3.0*gamma)/(2.0*gamma - 2.0))
    cs = np.sqrt(gamma*rho*KB_CGS*temp/mu/MP_CGS)
    mdot = q*np.pi*G_CGS**2*mass**2*rho/cs**3
    return mdot

def analytic_facc(tau, lum, beta):
    """
    Compute accretion factor from dimensionless parameters
    from the analytic formula (B4) of Paper I

    Parameters
    ----------
    tau : float or array-like
        Optical depth through the Bondi radius
    lum : float or array-like
        Dimensionless luminosity
    beta : float or array-like
        Dimensionless cooling time

    Returns
    -------
    f : float or ndarray
        Accretion suppression factor f_acc
    """
    tau, lum, beta = np.asarray(tau), np.asarray(lum), np.asarray(beta)
    if (beta < 1.0).any():
        warnings.warn(
            "facc_analytic: beta < 1 detected. "
            "Analytic facc is not recommended for beta < 1.",
            UserWarning,
            stacklevel=2
        )    
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


def facc_tabulated(tau, lum, beta, extrapolate=False):
    interpolator = interpolate_dimensionless_luminosity_table()
    return