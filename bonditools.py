"""
bonditools.py
=============
Utilities for computing the Bondi accretion suppression factor f_acc and the
self-consistent accretion luminosity following the analytic and numerical 
results of Paper I.

Public API
----------
compute_facc_from_cgs             -- f_acc from physical CGS parameters
compute_luminosity_facc_from_cgs  -- self-consistent (L, f_acc) via bisection

Private Functions
-----------------
_compute_facc_from_dimensionless             -- f_acc from (tau, L, beta)
_compute_luminosity_facc_from_dimensionless  -- L, f_acc from (tau, L_acc, beta, L_other)
_analytic_facc                    -- f_acc from (tau, L, beta), analytic formula
_tabulated_facc                   -- f_acc from (tau, L, beta), tabulated solutions

_sound_speed                      -- compute c_s from other CGS quantities
_bondi_radius                     -- compute Bondi radius from CGS quantities
_luminosity_scale                 -- compute luminosity scale l0 from CGS quantities
_beta                             -- compute beta from CGS quantities
_cgs_to_dimensionless             -- convert CGS to (tau, L,  beta)
_adiabatic_mdot                   -- compute adiabatic Bondi rate from CGS quantities

_luminosity_bracket               -- estimate brackets for root-finding L
_vectorized_bisect                -- bisection root-finder

_load_table                       -- load tabulated facc.tab
_construct_table_interpolator     -- create interpolator for table in logspace
_construct_lmin_interpolator      -- create interpolator for lmin(tau, beta)

_broadcast_float                  -- broadcast inputs and make floats
_validate_choice                  -- check optional inputs are valid

"""

from __future__ import annotations

import warnings
from functools import lru_cache
from typing import Literal

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.interpolate import RegularGridInterpolator

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------
Mode          = Literal["analytic", "tabulated", "mixed"]
Interpolation = Literal["linear", "nearest"]

# ---------------------------------------------------------------------------
# Physical constants (CGS)
# ---------------------------------------------------------------------------
G_CGS        = 6.67430e-8                  # gravitational constant  [cm^3 g^-1 s^-2]
C_CGS        = 2.99792458e10               # speed of light          [cm s^-1]
KB_CGS       = 1.380649e-16               # Boltzmann constant      [erg K^-1]
MP_CGS       = 1.67262192369e-24          # proton mass             [g]
SIGMA_SB_CGS = 5.670374419e-5             # Stefan-Boltzmann const  [erg cm^-2 s^-1 K^-4]
A_RAD_CGS    = 4.0 * SIGMA_SB_CGS / C_CGS # radiation constant      [erg cm^-3 K^-4]

# ---------------------------------------------------------------------------
# Analytic normalisations (fit to numerical solutions, Paper I)
# ---------------------------------------------------------------------------
A1 = 10.0
A2 = 1.0
A3 = 10.0

# ---------------------------------------------------------------------------
# Path to tabulated f_acc data
# ---------------------------------------------------------------------------
PATH_TO_TABLE: str = "./facc.tab"

# Tabulated domain boundaries (all three axes span [1e-3, 1e3])
_TAB_MIN: float = 0.001
_TAB_MAX: float = 1000.0

# Small offset used to nudge lmin just inside the valid domain
_SMALL: float = 1e-6

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_facc_from_cgs(
    mass:          float | np.ndarray,
    luminosity:    float | np.ndarray,
    density:       float | np.ndarray,
    temperature:   float | np.ndarray,
    gamma:         float | np.ndarray,
    mu:            float | np.ndarray,
    opacity:       float | np.ndarray,
    *,
    mode:          Mode          = "mixed",
    interpolation: Interpolation = "linear",
    extrapolate:   bool          = False,
) -> float | np.ndarray:
    """
    Compute the accretion suppression factor f_acc from physical CGS parameters.

    This is a thin wrapper around `_compute_facc_from_dimensionless` that
    converts CGS inputs to the dimensionless parameters (tau, lum, beta).

    Parameters
    ----------
    mass : float or array-like
        Planet mass [g].
    luminosity : float or array-like
        Planet luminosity [erg s^-1].
    density : float or array-like
        Ambient disk density [g cm^-3].
    temperature : float or array-like
        Ambient disk temperature [K].
    gamma : float or array-like
        Ratio of specific heats.
    mu : float or array-like
        Mean molecular weight [proton masses].
    opacity : float or array-like
        Disk opacity [cm^2 g^-1].  Use the Rosseland mean for optically thick
        conditions or the Planck mean for optically thin conditions.
    mode : {"mixed", "analytic", "tabulated"}, optional
        Computation method for f_acc.  Default "mixed".
    interpolation : {"linear", "nearest"}, optional
        Interpolation scheme for tabulated/mixed modes.  Default "linear".
    extrapolate : bool, optional
        Whether to extrapolate outside the tabulated domain.  Default False.

    Returns
    -------
    f : float or ndarray
        Accretion suppression factor f_acc.
    """
    mass, luminosity, density, temperature, gamma, mu, opacity = _broadcast_float(
        mass, luminosity, density, temperature, gamma, mu, opacity
    )
    tau, lum_dimless, beta = _cgs_to_dimensionless(
        mass, luminosity, density, temperature, gamma, mu, opacity
    )
    f = _compute_facc_from_dimensionless(
        tau, lum_dimless, beta,
        mode=mode, interpolation=interpolation, extrapolate=extrapolate,
    )
    return f.item() if f.ndim == 0 else f


def compute_luminosity_facc_from_cgs(
    mass:             float | np.ndarray,
    shock_efficiency: float | np.ndarray,
    shock_radius:     float | np.ndarray,
    density:          float | np.ndarray,
    temperature:      float | np.ndarray,
    gamma:            float | np.ndarray,
    mu:               float | np.ndarray,
    opacity:          float | np.ndarray,
    other_luminosity: float | np.ndarray = 0.0,
    *,
    mode:             Mode          = "mixed",
    interpolation:    Interpolation = "linear",
    extrapolate:      bool          = False,
    max_luminosity:   float         = 1e3,
    atol:             float         = 2e-12,
    rtol:             float         = np.float64(8.881784197001252e-16),
    maxiter:          int           = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute f_acc and the self-consistent total luminosity L = L_acc + L_other
    from physical CGS parameters.

    This is a thin wrapper around `_compute_luminosity_facc_from_dimensionless`
    that converts CGS inputs to the dimensionless parameters (tau, lum, beta).

    Physical Parameters
    -------------------
    mass : float or array-like
        Planet mass [g].
    shock_efficiency : float or array-like
        Accretion shock efficiency (dimensionless).
    shock_radius : float or array-like
        Radial location of accretion shock [cm].
    density : float or array-like
        Ambient disk density [g cm^-3].
    temperature : float or array-like
        Ambient disk temperature [K].
    gamma : float or array-like
        Ratio of specific heats.
    mu : float or array-like
        Mean molecular weight [proton masses].
    opacity : float or array-like
        Disk opacity [cm^2 g^-1].
    other_luminosity : float or array-like, optional
        Luminosity from non-accretion sources [erg s^-1].  Default 0.

    Keyword-Only Numerical Parameters
    ----------------------------------
    mode : {"mixed", "analytic", "tabulated"}, optional
        Computation method for f_acc.  Default "mixed".
    interpolation : {"linear", "nearest"}, optional
        Interpolation scheme for tabulated/mixed modes.  Default "linear".
    extrapolate : bool, optional
        Whether to extrapolate outside the tabulated domain.  Default False.
    max_luminosity : float, optional
        Upper bound on dimensionless L for bisection when extrapolate = True.  
        Must be >= 1e3.  Default 1e3.
    atol : float, optional
        Absolute tolerance for the bisection solver.  Default 2e-12.
    rtol : float, optional
        Relative tolerance.  Cannot be smaller than 4 * machine epsilon.
    maxiter : int, optional
        Maximum bisection iterations.  Default 100.

    Returns
    -------
    lum : ndarray
        Self-consistent total luminosity L [erg s^-1].
    facc : ndarray
        Accretion suppression factor f_acc.
    """
    # --- 0. Validate inputs -----------------------------------------------
    _validate_choice(mode, "mode", frozenset(("analytic", "tabulated", "mixed")))
    _validate_choice(interpolation, "interpolation", frozenset(("linear", "nearest")))

    # --- 1. Broadcast and convert CGS -> dimensionless --------------------
    (mass, shock_efficiency, shock_radius, density,
     temperature, gamma, mu, opacity, other_luminosity) = _broadcast_float(
        mass, shock_efficiency, shock_radius, density,
        temperature, gamma, mu, opacity, other_luminosity,
    )

    lum_acc_ad = (shock_efficiency * G_CGS * mass
                  * _adiabatic_mdot(gamma, mass, density, temperature, mu)
                  / shock_radius)
    tau, lum_acc_ad_dimless, beta = _cgs_to_dimensionless(
        mass, lum_acc_ad, density, temperature, gamma, mu, opacity
    )
    l0 = _luminosity_scale(_bondi_radius(mass, gamma, temperature, mu), temperature)
    lum_other_dimless = other_luminosity / l0

    # --- 2. Delegate to dimensionless solver ------------------------------
    lum_dimless, facc = _compute_luminosity_facc_from_dimensionless(
        tau, lum_acc_ad_dimless, beta, lum_other_dimless,
        mode=mode, interpolation=interpolation, extrapolate=extrapolate,
        max_luminosity=max_luminosity, atol=atol, rtol=rtol, maxiter=maxiter,
    )

    # --- 3. Convert back to CGS -------------------------------------------
    lum_out = lum_dimless * l0
    return (lum_out.item(), facc.item()) if lum_out.ndim == 0 else (lum_out, facc)


# ---------------------------------------------------------------------------
# Private - Functions to compute from dimensionless parameters
# ---------------------------------------------------------------------------
def _compute_facc_from_dimensionless(
    tau:           np.ndarray,
    lum:           np.ndarray,
    beta:          np.ndarray,
    *,
    mode:          Mode          = "mixed",
    interpolation: Interpolation = "linear",
    extrapolate:   bool          = False,
) -> np.ndarray:
    """
    Compute the accretion suppression factor f_acc from dimensionless parameters.
    Always returns an ndarray; scalar unwrapping is handled by the public callers.

    Parameters
    ----------
    tau : ndarray
        Optical depth through the Bondi radius.
    lum : ndarray
        Dimensionless luminosity.
    beta : ndarray
        Dimensionless cooling-time parameter.
    mode, interpolation, extrapolate
        See `compute_facc_from_cgs` for full descriptions.

    Returns
    -------
    f : ndarray
        Accretion suppression factor f_acc.
    """
    tau, lum, beta = _broadcast_float(tau, lum, beta)

    if mode == "analytic":
        f = _analytic_facc(tau, lum, beta)

    elif mode == "tabulated":
        f = _tabulated_facc(tau, lum, beta, interpolation=interpolation, extrapolate=extrapolate)

    else: # "mixed"
        f = np.full(tau.shape, np.nan, dtype=np.float64)
        mask_ana = beta <= 1.0
        mask_tab = ~mask_ana
        if mask_ana.any():
            f[mask_ana] = _analytic_facc(tau[mask_ana], lum[mask_ana], beta[mask_ana])
        if mask_tab.any():
            f[mask_tab] = _tabulated_facc(
                tau[mask_tab], lum[mask_tab], beta[mask_tab],
                interpolation=interpolation, extrapolate=extrapolate,
            )

    return f


def _compute_luminosity_facc_from_dimensionless(
    tau:               np.ndarray,
    lum_acc_ad_dimless: np.ndarray,
    beta:              np.ndarray,
    lum_other_dimless: np.ndarray,
    *,
    mode:              Mode          = "mixed",
    interpolation:     Interpolation = "linear",
    extrapolate:       bool          = False,
    max_luminosity:    float         = 1e3,
    atol:              float         = 2e-12,
    rtol:              float         = np.float64(8.881784197001252e-16),
    maxiter:           int           = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the self-consistent dimensionless luminosity and f_acc via bisection.

    Solves the fixed-point equation

        L = lum_acc_ad * f_acc(tau, L, beta) + L_other

    entirely in dimensionless units.

    Parameters
    ----------
    tau : ndarray
        Optical depth through the Bondi radius.
    lum_acc_ad_dimless : ndarray
        Dimensionless adiabatic accretion luminosity
    beta : ndarray
        Dimensionless cooling-time parameter.
    lum_other_dimless : ndarray
        Dimensionless luminosity from non-accretion sources.
    mode, interpolation, extrapolate, max_luminosity, atol, rtol, maxiter
        Forwarded to `_compute_facc_from_dimensionless` / `_vectorized_bisect`;
        see `compute_luminosity_facc_from_cgs` for full descriptions.

    Returns
    -------
    lum_dimless : ndarray
        Self-consistent dimensionless luminosity.
    facc : ndarray
        Accretion suppression factor f_acc.
    """
    tau, lum_acc_ad_dimless, beta, lum_other_dimless = _broadcast_float(
        tau, lum_acc_ad_dimless, beta, lum_other_dimless
    )

    # --- 1. Residual: L - (lum_acc_ad * facc + L_other) = 0 ----------------
    def residual(lum_dimless: np.ndarray) -> np.ndarray:
        facc = _compute_facc_from_dimensionless(
            tau, lum_dimless, beta,
            mode=mode, interpolation=interpolation, extrapolate=extrapolate,
        )
        return lum_acc_ad_dimless * facc + lum_other_dimless - lum_dimless

    # --- 2. Bracket the root ----------------------------------------------
    lmin, lmax = _luminosity_bracket(
        tau, beta, lum_acc_ad_dimless, lum_other_dimless,
        mode=mode, interpolation=interpolation, max_luminosity=max_luminosity,
    )

    # --- 3. Solve and recover facc ----------------------------------------
    lum_dimless = _vectorized_bisect(residual, lmin, lmax, atol=atol, rtol=rtol, maxiter=maxiter)
    facc = _compute_facc_from_dimensionless(
        tau, lum_dimless, beta,
        mode=mode, interpolation=interpolation, extrapolate=extrapolate,
    )
    return lum_dimless, facc


def _analytic_facc(tau: np.ndarray, lum: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """
    Analytic f_acc from eq. (B4) of Paper I.

    Same functionality as calling `_compute_facc_from_dimensionless` 
    with mode="analytic"

    Issues a `UserWarning` if any beta > 1 (analytic formula is unreliable
    in that regime).
    """
    if np.any(beta > 1.0):
        warnings.warn(
            "facc_analytic: beta > 1 detected - the analytic formula is not "
            "recommended for beta > 1.",
            UserWarning,
            stacklevel=3,
        )

    f = np.full(tau.shape, np.nan, dtype=np.float64)

    # Isothermal regime
    iso = (lum <= np.minimum(A3, A2 / tau)) & (lum >= tau * beta * A1)
    f[iso] = 1.0

    # Optically thin regime
    thin = (tau <= A2 * lum / A3**2) & (lum >= A3)
    f[thin] = (lum[thin] / A3) ** (-1.25)

    # Optically thick regime (two sub-cases)
    thick          = (A2 / tau < lum) & (lum <= A2 * A3**2 * tau)
    thick_boundary = (tau / A2) ** (5.0 / 11.0) * (A1 * A2 * beta) ** (8.0 / 11.0)
    thick1 = thick & (lum >= thick_boundary)
    thick2 = thick & (lum < thick_boundary)
    f[thick1] = (lum[thick1] * tau[thick1] / A2) ** (-0.625)
    f[thick2] = (A1 * tau[thick2]**2 * beta[thick2] / A2) ** (-5.0 / 11.0)

    return f


def _tabulated_facc(
    tau:           np.ndarray,
    lum:           np.ndarray,
    beta:          np.ndarray,
    interpolation: Interpolation = "linear",
    extrapolate:   bool          = False,
) -> np.ndarray:
    """
    Compute f_acc by interpolating the numerical grid from Paper I.

    Same functionality as calling `_compute_facc_from_dimensionless` 
    with mode="tabulated"
    """
    interpolator = _construct_table_interpolator(interpolation, extrapolate)

    if not extrapolate:
        oob = (tau < _TAB_MIN) | (tau > _TAB_MAX) | \
              (lum < _TAB_MIN) | (lum > _TAB_MAX) | \
              (beta < _TAB_MIN) | (beta > _TAB_MAX)
        n_oob = int(np.count_nonzero(oob))
        if n_oob:
            warnings.warn(
                f"{n_oob} point(s) outside the tabulated domain "
                f"[{_TAB_MIN}, {_TAB_MAX}]^3 - returning NaN there.  "
                "Set extrapolate=True to extrapolate instead.",
                RuntimeWarning,
                stacklevel=3,
            )

    return np.exp(interpolator((np.log(tau), np.log(lum), np.log(beta))))


# ---------------------------------------------------------------------------
# Private - Compute simple physics quantities
# ---------------------------------------------------------------------------

def _sound_speed(gamma: np.ndarray, temperature: np.ndarray, mu: np.ndarray) -> np.ndarray:
    """Isothermal sound speed [cm s^-1]."""
    return np.sqrt(gamma * KB_CGS * temperature / (mu * MP_CGS))


def _beta(
    density:     np.ndarray,
    cs:          np.ndarray,
    gamma:       np.ndarray,
    temperature: np.ndarray,
    tau:         np.ndarray,
) -> np.ndarray:
    """Dimensionless cooling-time parameter beta."""
    return (0.25 * density * cs**3
            / (gamma * (gamma - 1.0) * A_RAD_CGS * C_CGS * temperature**4 * tau))


def _bondi_radius(
    mass:        np.ndarray,
    gamma:       np.ndarray,
    temperature: np.ndarray,
    mu:          np.ndarray,
) -> np.ndarray:
    """Bondi radius [cm]."""
    return 0.5 * G_CGS * mass / _sound_speed(gamma, temperature, mu)**2


def _luminosity_scale(
    rbondi:      np.ndarray,
    temperature: np.ndarray,
) -> np.ndarray:
    """Luminosity scale l0 used to non-dimensionalise luminosity [erg s^-1]."""
    return 4.0 * np.pi * rbondi**2 * A_RAD_CGS * C_CGS * temperature**4


def _cgs_to_dimensionless(
    mass:        np.ndarray,
    luminosity:  np.ndarray,
    density:     np.ndarray,
    temperature: np.ndarray,
    gamma:       np.ndarray,
    mu:          np.ndarray,
    opacity:     np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (tau, lum_dimless, beta) from CGS physical parameters."""
    cs     = _sound_speed(gamma, temperature, mu)
    rbondi = _bondi_radius(mass, gamma, temperature, mu)
    tau    = density * opacity * rbondi
    lum    = luminosity / _luminosity_scale(rbondi, temperature)
    beta   = _beta(density, cs, gamma, temperature, tau)
    return tau, lum, beta


def _adiabatic_mdot(
    gamma:       np.ndarray,
    mass:        np.ndarray,
    density:     np.ndarray,
    temperature: np.ndarray,
    mu:          np.ndarray,
) -> np.ndarray:
    """
    Adiabatic Bondi accretion rate [g s^-1].

    Parameters
    ----------
    gamma       : array-like  Ratio of specific heats.
    mass        : array-like  Accretor mass [g].
    density     : array-like  Ambient density [g cm^-3].
    temperature : array-like  Ambient temperature [K].
    mu          : array-like  Mean molecular weight [proton masses].
    """
    q  = (2.0 / (5.0 - 3.0 * gamma)) ** ((5.0 - 3.0 * gamma) / (2.0 * gamma - 2.0))
    cs = _sound_speed(gamma, temperature, mu)
    return q * np.pi * G_CGS**2 * mass**2 * density / cs**3


# ---------------------------------------------------------------------------
# Private - root-finding
# ---------------------------------------------------------------------------

def _luminosity_bracket(
    tau:              np.ndarray,
    beta:             np.ndarray,
    lum_acc_ad_dimless: np.ndarray,
    lum_other_dimless: np.ndarray,
    *,
    mode:             Mode,
    interpolation:    Interpolation,
    max_luminosity:   float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (lmin, lmax) that bracket the self-consistent dimensionless luminosity.

    Lower bound
        Smallest L at which f_acc is not nan.
    Upper bound
        Estimated from the optically-thin regime for analytic. 
        Set by max_luminosity for tabulated.
    """
    if max_luminosity < 1e3:
        raise ValueError(f"max_luminosity must be >= 1e3, got {max_luminosity!r}")

    # --- analytic lower bound ---------------------------------------------
    lmin = np.asarray(np.minimum(np.minimum(A3, A2 / tau), A1 * tau * beta) * (1.0 + _SMALL))

    # --- analytic upper bound from thin-regime ----------------------------
    thin_lmin = np.maximum(A3, tau * A3**2 / A2)
    # if L > thin_lmin, then L = a*L^{-5/4} + b < a*(thin_lmin)^{-5/4} + b
    lmax = np.asarray(lum_acc_ad_dimless * (thin_lmin / A3) ** (-1.25) + lum_other_dimless)
    # fall back where no thin solution exists
    no_thin = lmax <= thin_lmin
    lmax[no_thin] = thin_lmin[no_thin]

    # --- tighten for tabulated / mixed modes ------------------------------
    if mode in ("mixed", "tabulated"):
        mask = beta > 1.0 if mode == "mixed" else np.ones(beta.shape, dtype=bool)
        if mask.any():
            lmin_interp = _construct_lmin_interpolator(interpolation)
            lmin[mask] = np.exp(
                lmin_interp((np.log(tau[mask]), np.log(beta[mask])))
            ) * (1.0 + _SMALL)
            lmax[mask] = max_luminosity

    return lmin, lmax


def _vectorized_bisect(
    f:       "Callable[[np.ndarray], np.ndarray]",
    xa:      np.ndarray,
    xb:      np.ndarray,
    atol:    float = 2e-12,
    rtol:    float = np.float64(8.881784197001252e-16),
    maxiter: int   = 100,
) -> np.ndarray:
    """
    Vectorized bisection root-finder.
    Default tolerances match `scipy.optimize.bisect`.

    Parameters
    ----------
    f : callable
        Accepts and returns arrays of the same shape.
    xa, xb : ndarray
        Lower / upper bracket bounds.
    atol : float
        Absolute tolerance (must be > 0).
    rtol : float
        Relative tolerance (must be >= 4 * machine epsilon).
    maxiter : int
        Maximum iterations.

    Returns
    -------
    roots : ndarray
        Approximate root for each bracket element.
    """
    if atol <= 0:
        raise ValueError(f"atol must be positive, got {atol!r}")
    min_rtol = 4.0 * np.finfo(float).eps
    if rtol < min_rtol:
        raise ValueError(f"rtol must be >= {min_rtol} (4*machine eps), got {rtol!r}")

    xa = np.asarray(xa, dtype=np.float64)
    xb = np.asarray(xb, dtype=np.float64)
    scalar_input = xa.ndim == 0
    xa = np.atleast_1d(xa)
    xb = np.atleast_1d(xb)
    fa = f(xa)
    fb = f(xb)
    roots = np.full_like(xa, np.nan)

    # Handle bracket endpoints that are already roots
    at_left  = np.isclose(fa, 0.0, atol=atol)
    at_right = np.isclose(fb, 0.0, atol=atol)
    roots[at_left]  = xa[at_left]
    roots[at_right] = xb[at_right]

    active = ~(at_left | at_right)

    # Warn about degenerate brackets (same sign on both ends)
    bad = active & (np.signbit(fa) == np.signbit(fb))
    if bad.any():
        warnings.warn(
            f"{int(bad.sum())} bracket(s) do not change sign - returning NaN.",
            RuntimeWarning,
            stacklevel=2,
        )
        active &= ~bad

    # Bisection loop (mirrors scipy's bisect.c)
    dm = xb - xa
    for _ in range(maxiter):
        if not active.any():
            break
        dm   *= 0.5
        xm    = xa + dm
        fm    = f(xm)
        # advance left endpoint where midpoint has same sign as left
        shift = active & (np.signbit(fm) == np.signbit(fa))
        xa[shift] = xm[shift]
        # mark converged
        conv = active & (np.abs(dm) <= atol + rtol * np.abs(xm))
        roots[conv]  = xm[conv]
        active[conv] = False

    # Any still-active brackets hit maxiter: return best estimate
    if active.any():
        roots[active] = xa[active]

    return roots.squeeze() if scalar_input else roots


# ---------------------------------------------------------------------------
# Private - interpolation / table IO
# ---------------------------------------------------------------------------

@lru_cache(maxsize=None)
def _load_table() -> np.ndarray:
    """Load and cache the raw f_acc table, shaped (4, 21, 21, 21)."""
    return np.loadtxt(PATH_TO_TABLE).T.reshape(4, 21, 21, 21)


@lru_cache(maxsize=None)
def _construct_table_interpolator(
    interpolation: Interpolation,
    extrapolate:   bool,
) -> RegularGridInterpolator:
    """
    Build (and cache) a `scipy.interpolate.RegularGridInterpolator`
    for log f_acc as a function of (log tau, log L, log beta).

    The table has shape (4, 21, 21, 21); axis 0 holds [tau, L, beta, f_acc].
    """
    data = _load_table()
    span = np.log(data[0, :, 0, 0])
    log_facc = np.log(np.where(data[3] == 0.0, np.nan, data[3]))

    kwargs: dict = dict(method=interpolation, bounds_error=False)
    if extrapolate:
        kwargs["fill_value"] = None
    return RegularGridInterpolator((span, span, span), log_facc, **kwargs)


@lru_cache(maxsize=None)
def _construct_lmin_interpolator(interpolation: Interpolation) -> RegularGridInterpolator:
    """
    Build (and cache) an interpolator for the minimum valid dimensionless
    luminosity log(L_min) as a function of (log tau, log beta).

    Used to set accurate lower bounds for the bisection in
    `compute_luminosity_facc_from_cgs` when using tabulated f_acc.
    """
    data = _load_table()
    log_facc = np.where(data[3] == 0.0, np.nan, data[3])
    span = np.log(data[1, 0, :, 0])
    logl_width = span[1] - span[0]

    if interpolation == "nearest":
        idx  = np.argmax(~np.isnan(log_facc), axis=1)  # shape (21, 21)
        lmin = span[idx] - 0.5 * logl_width
        return RegularGridInterpolator(
            (span, span), lmin, method="nearest", bounds_error=False, fill_value=None
        )

    # "linear": need first cell where all 8 corners of a 2x2x2 window are valid
    cells = sliding_window_view(log_facc, (2, 2, 2))
    valid = ~np.isnan(cells).any(axis=(-1, -2, -3))
    idx   = np.argmax(valid, axis=1) # shape (20, 20)
    lmin  = span[idx]
    mid_span = 0.5 * (span[:-1] + span[1:])
    return RegularGridInterpolator(
        (mid_span, mid_span), lmin, method="nearest", bounds_error=False, fill_value=None
    )

# ---------------------------------------------------------------------------
# Private helpers - general utilities
# ---------------------------------------------------------------------------

def _broadcast_float(*arrays) -> tuple[np.ndarray, ...]:
    """Broadcast and cast all inputs to float64 in a single pass."""
    return np.broadcast_arrays(*(np.asarray(a, dtype=np.float64) for a in arrays))


def _validate_choice(value: str, name: str, valid: frozenset[str]) -> None:
    if value not in valid:
        raise ValueError(f"Invalid {name} {value!r}. Expected one of: {', '.join(sorted(valid))}")
