"""
models.py
---------
Fit functions, model registry, and prior construction for lightcurve fitting.

Each model function has the signature:
    f(t, *params) -> np.ndarray
where the parameter order matches the corresponding entry in MODELS[key]['params'].

A y_offset term is included in every model so that a non-zero baseline within
the selected region is absorbed into the fit.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Model functions
# ---------------------------------------------------------------------------

def gaussian(t, amplitude, centre, sigma, y_offset):

    """
    Symmetric Gaussian peak on a constant baseline.

        f(t) = A * exp(-0.5 * ((t - mu) / sigma)^2) + y_offset

    Parameters
    ----------
    amplitude : float   Peak height above baseline (should be > 0).
    centre    : float   Peak position.
    sigma     : float   Standard deviation; must be > 0.
    y_offset  : float   Constant baseline level.
    """

    return amplitude * np.exp(-0.5 * ((t - centre) / sigma) ** 2) + y_offset


def rising_exponential(t, amplitude, tau_rise, y_offset):

    """
    Exponential rise on a constant baseline.

        f(t) = A * exp(t / tau_rise) + y_offset

    Parameters
    ----------
    amplitude : float
    tau_rise  : float
    y_offset  : float
    """

    return amplitude * np.exp(t / tau_rise) + y_offset


def decaying_exponential(t, amplitude, tau_decay, y_offset):

    """
    Exponential decay on a constant baseline.

        f(t) = A * exp(-t / tau_decay) + y_offset

    Parameters
    ----------
    amplitude : float
    tau_decay : float
    y_offset  : float
    """

    return amplitude * np.exp(-t / tau_decay) + y_offset


def crystal_ball(t, amplitude, centre, sigma, alpha, n, y_offset):

    """
    Crystal Ball function: Gaussian core with a power-law tail on one side.

    The standard Crystal Ball is defined piecewise:

        Let z = (t - centre) / sigma

        f(t) = A * exp(-0.5 * z^2)          if z > -|alpha|
             = A * (n/|alpha|)^n
                 * exp(-0.5 * alpha^2)
                 / (n/|alpha| - |alpha| - z)^n   otherwise

    The tail is on the *left* side (low-t side) when alpha > 0, modelling a
    sharp rise and a power-law decay — generally used in particle physics, but may be useful for astro.

    To flip the tail to the right (slow rise, sharp cutoff), pass alpha < 0.

    Parameters
    ----------
    amplitude : float   Peak height above baseline.
    centre    : float   Peak position (mode of the Gaussian core).
    sigma     : float   Width of the Gaussian core; must be > 0.
    alpha     : float   Transition point from Gaussian to power law (in units
                        of sigma). Conventionally > 0; sign sets tail side.
    n         : float   Power-law index; must be > 1 for normalisability.
    y_offset  : float   Constant baseline level.

    Note:
    For numerical stability, the power-law branch can overflow or go negative if
    n/|alpha| - |alpha| - z approaches zero.  A small floor is applied to
    the denominator to prevent division by zero without distorting the shape.
    """

    a = np.abs(alpha)
    z = (t - centre) / sigma

    # Gaussian branch
    gauss_branch = amplitude * np.exp(-0.5 * z ** 2)

    # Power-law branch pre-factor
    # C = (n/a)^n * exp(-0.5 * a^2)
    C = (n / a) ** n * np.exp(-0.5 * a ** 2)
    denom = np.maximum(n / a - a - z, 1e-10)   # floor avoids division by zero
    power_branch = amplitude * C / denom ** n

    # Piecewise selection: Gaussian where z > -a, power-law elsewhere
    result = np.where(z > -a, gauss_branch, power_branch)
    return result + y_offset


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
# Each entry maps a short string key to:
#   'func'        : the callable above
#   'params'      : ordered list of parameter names (matching func signature)
#   'label'       : human-readable name for UI display
#   'defaults'    : rough starting defaults (overridden by user sliders)
#   'bounds_hint' : (lower_fraction, upper_fraction) relative to guess,
#                   used by the slider initialiser as a starting range hint.
#                   These are just UI hints; the MCMC priors are set separately.

MODELS = {
    'gaussian': {
        'func':        gaussian,
        'params':      ['amplitude', 'centre', 'sigma', 'y_offset'],
        'label':       'Gaussian',
        'defaults':    [1.0, 0.0, 1.0, 0.0],
        'bounds_hint': (0.5, 1.5),
    },
    'rising_exp': {
        'func':        rising_exponential,
        'params': ['amplitude', 'tau_rise', 'y_offset'],
        'label':       'Rising Exponential',
        'defaults':    [1.0, 1.0, 0.0],
        'bounds_hint': (0.5, 1.5),
    },
    'decaying_exp': {
        'func':        decaying_exponential,
        'params':      ['amplitude', 'tau_decay', 'y_offset'],
        'label':       'Decaying Exponential',
        'defaults':    [1.0, 1.0, 0.0],
        'bounds_hint': (0.5, 1.5),
    },
    'crystal_ball': {
        'func':        crystal_ball,
        'params':      ['amplitude', 'centre', 'sigma', 'alpha', 'n', 'y_offset'],
        'label':       'Crystal Ball',
        'defaults':    [1.0, 0.0, 1.0, 1.0, 2.0, 0.0],
        'bounds_hint': (0.5, 1.5),
    },
}

# Convenience tuple for UI menus / radio buttons
MODEL_KEYS   = list(MODELS.keys())
MODEL_LABELS = [MODELS[k]['label'] for k in MODEL_KEYS]


# ---------------------------------------------------------------------------
# Prior construction
# ---------------------------------------------------------------------------

# Small absolute floor so that a guess of zero does not produce a zero-width prior.
# Units are whatever the user's axes are, so this is intentionally tiny — it just prevents a degenerate prior, it is not physically meaningful.

_PRIOR_ABS_FLOOR = 1e-6

# Fraction above and below the initial guess that the prior should span.
PRIOR_FRACTION = 0.30


def build_priors(param_names, guesses, prior_fraction=PRIOR_FRACTION):

    """
    Construct uniform prior bounds for each parameter given initial guesses.

    For each parameter p_i with guess g_i:

        lower_i = g_i - max(prior_fraction * |g_i|, _PRIOR_ABS_FLOOR)
        upper_i = g_i + max(prior_fraction * |g_i|, _PRIOR_ABS_FLOOR)

    Parameters
    ----------
    param_names    : list of str   Parameter names (for labelling only).
    guesses        : list of float Initial guess values, one per parameter.
    prior_fraction : float         Fractional half-width (default 0.30 → ±30%).

    Returns
    -------
    priors : dict
        Maps each parameter name to {'lower': float, 'upper': float}.

    Example
    -------
    >>> priors = build_priors(['amplitude', 'centre', 'sigma', 'y_offset'], [5.0, 100.0, 2.0, 0.1])
    >>> priors['amplitude']
    {'lower': 3.5, 'upper': 6.5}
    """

    if len(param_names) != len(guesses):
        raise ValueError(
            f"param_names has {len(param_names)} entries but guesses has "
            f"{len(guesses)}."
        )

    priors = {}
    for name, g in zip(param_names, guesses):
        half_width = max(prior_fraction * abs(g), _PRIOR_ABS_FLOOR)
        priors[name] = {
            'lower': g - half_width,
            'upper': g + half_width,
        }
    return priors


def build_priors_from_curvefit(param_names, popt, pcov,
                                n_sigma=2.0):
    """
    Construct uniform prior bounds from a curve_fit result.

    Uses the fitted parameter values and their standard deviations (from the
    diagonal of the covariance matrix) to set prior bounds as:

        lower_i = popt_i - n_sigma * std_i
        upper_i = popt_i + n_sigma * std_i

    Falls back to the flat ±30% rule for any parameter whose covariance is
    non-finite (fit did not converge for that parameter).

    Parameters
    ----------
    param_names : list of str
    popt        : array-like   Best-fit parameters from curve_fit.
    pcov        : 2-D array    Covariance matrix from curve_fit.
    n_sigma     : float        Number of sigma to use as prior half-width.

    Returns
    -------
    priors : dict
        Same format as build_priors.
    """

    stds = np.sqrt(np.diag(pcov))
    priors = {}
    for name, g, s in zip(param_names, popt, stds):
        if np.isfinite(s) and s > 0:
            priors[name] = {
                'lower': g - n_sigma * s,
                'upper': g + n_sigma * s,
            }
        else:
            # Fallback: flat ±30% around the curve_fit central value
            half_width = max(PRIOR_FRACTION * abs(g), _PRIOR_ABS_FLOOR)
            priors[name] = {
                'lower': g - half_width,
                'upper': g + half_width,
            }
    return priors


# ---------------------------------------------------------------------------
# Convenience: evaluate a named model
# ---------------------------------------------------------------------------

def evaluate(model_key, t, params):

    """
    Evaluate a model at times t given a parameter list or dict.

    Parameters
    ----------
    model_key : str         Key into MODELS (e.g. 'gaussian').
    t         : array-like  Time values.
    params    : list or dict

        If list => must be in the same order as MODELS[model_key]['params'].
        If dict => keys must match parameter names.

    Returns
    -------
    flux : np.ndarray
    """

    if model_key not in MODELS:
        raise KeyError(
            f"Unknown model '{model_key}'. "
            f"Available: {list(MODELS.keys())}"
        )
    func        = MODELS[model_key]['func']
    param_names = MODELS[model_key]['params']

    if isinstance(params, dict):
        param_list = [params[n] for n in param_names]
    else:
        param_list = list(params)

    return func(np.asarray(t), *param_list)