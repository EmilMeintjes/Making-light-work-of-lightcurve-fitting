"""
fitting_models.py
-----------------
Model definitions, evaluation, and prior construction for the nova-like
lightcurve fitting pipeline.

Public API
----------
MODEL_KEYS     : list[str]   — canonical model key strings
MODEL_LABELS   : list[str]   — human-readable display names (same order)
MODEL_EQUATIONS: dict        — LaTeX-style equation string per model key
PRIOR_FRACTION : float       — default ±fraction for uniform priors (0.30)

evaluate(model_key, t, params)        -> np.ndarray
build_priors(param_names, guesses, prior_fraction=PRIOR_FRACTION,
             fix_y_offset=False)      -> dict
build_priors_from_curvefit(param_names, popt, pcov, n_sigma=2.0,
                           fix_y_offset=False)  -> dict
param_names_for(model_key)            -> list[str]
"""

import numpy as np

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

MODEL_KEYS = [
    'gaussian',
    'rising_exp',
    'decaying_exp',
    'crystal_ball',
]

MODEL_LABELS = [
    'Gaussian',
    'Rising Exp',
    'Decaying Exp',
    'Crystal Ball',
]

# Human-readable equation strings shown in the initialiser window.
# Uses simple ASCII math that renders well in a matplotlib Text object.
# t is always region-relative (zero at region start).
MODEL_EQUATIONS = {
    'gaussian': (
        'f(t) = A · exp(−0.5·((t − centre) / σ)²) + y_offset'
    ),
    'rising_exp': (
        'f(t) = A · exp(t / τ_rise) + y_offset'
        '\n'
        '  t = 0 at region start;  τ_rise > 0'
    ),
    'decaying_exp': (
        'f(t) = A · exp(−t / τ_decay) + y_offset'
        '\n'
        '  t = 0 at region start;  τ_decay > 0'
    ),
    'crystal_ball': (
        'f(t) = A · CB(t; centre, σ, α, n) + y_offset'
        '\n'
        '  Gaussian core + power-law tail.  α > 0 → tail on left side.'
    ),
}

# ---------------------------------------------------------------------------
# Prior constants
# ---------------------------------------------------------------------------

PRIOR_FRACTION       = 0.30   # default ±30% uniform prior
_Y_OFFSET_PRIOR_FRACTION = 0.10   # tighter ±10% for y_offset
_PRIOR_ABS_FLOOR     = 1e-6   # minimum half-width to avoid zero-width priors

# ---------------------------------------------------------------------------
# Parameter name lists
# ---------------------------------------------------------------------------

_PARAM_NAMES = {
    'gaussian':     ['amplitude', 'centre', 'sigma',        'y_offset'],
    'rising_exp':   ['amplitude', 'tau_rise',               'y_offset'],
    'decaying_exp': ['amplitude', 'tau_decay',              'y_offset'],
    'crystal_ball': ['amplitude', 'centre', 'sigma', 'alpha', 'n', 'y_offset'],
}


def param_names_for(model_key):
    """Return the ordered list of parameter names for the given model."""
    try:
        return list(_PARAM_NAMES[model_key])
    except KeyError:
        raise ValueError(f"Unknown model key: '{model_key}'.  "
                         f"Valid keys: {MODEL_KEYS}")


# ---------------------------------------------------------------------------
# Model functions
# ---------------------------------------------------------------------------

def _gaussian(t, params):
    A   = params['amplitude']
    mu  = params['centre']
    sig = params['sigma']
    off = params['y_offset']
    return A * np.exp(-0.5 * ((t - mu) / sig) ** 2) + off


def _rising_exp(t, params):
    A   = params['amplitude']
    tau = params['tau_rise']
    off = params['y_offset']
    return A * np.exp(t / tau) + off


def _decaying_exp(t, params):
    A   = params['amplitude']
    tau = params['tau_decay']
    off = params['y_offset']
    return A * np.exp(-t / tau) + off


def _crystal_ball(t, params):
    A     = params['amplitude']
    mu    = params['centre']
    sig   = params['sigma']
    alpha = params['alpha']
    n     = params['n']
    off   = params['y_offset']

    z     = (t - mu) / sig
    abs_a = abs(alpha)

    # Gaussian side
    gauss = np.exp(-0.5 * z ** 2)

    # Power-law tail (activated where z < -abs_a for alpha>0)
    C      = (n / abs_a) ** n * np.exp(-0.5 * abs_a ** 2)
    D      = n / abs_a - abs_a
    denom  = np.where(np.abs(D - z) < 1e-10, 1e-10, D - z)
    power  = C / (denom ** n)

    result = np.where(z > -abs_a, gauss, power)
    return A * result + off


_MODEL_FUNCS = {
    'gaussian':     _gaussian,
    'rising_exp':   _rising_exp,
    'decaying_exp': _decaying_exp,
    'crystal_ball': _crystal_ball,
}

# Callable wrappers for scipy.optimize.curve_fit  (positional args).
# These are used only in initialiser.py.
def _make_cf_wrapper(model_key):
    names = _PARAM_NAMES[model_key]
    func  = _MODEL_FUNCS[model_key]
    def wrapper(t, *args):
        params = dict(zip(names, args))
        return func(t, params)
    wrapper.__name__ = model_key
    return wrapper

CURVE_FIT_FUNCS = {k: _make_cf_wrapper(k) for k in MODEL_KEYS}


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(model_key, t, params):
    """
    Evaluate the named model at array t with the given params.

    Parameters
    ----------
    model_key : str
    t         : array-like   Region-relative time (t_shifted = t - region start).
    params    : dict or list/array
                If dict, keys must match param_names_for(model_key).
                If list/array, values are matched positionally.

    Returns
    -------
    np.ndarray
    """
    t = np.asarray(t, dtype=float)

    if isinstance(params, dict):
        p = params
    else:
        p = dict(zip(_PARAM_NAMES[model_key], params))

    try:
        return _MODEL_FUNCS[model_key](t, p)
    except KeyError:
        raise ValueError(f"Unknown model key: '{model_key}'")


# ---------------------------------------------------------------------------
# Prior construction
# ---------------------------------------------------------------------------

def build_priors(param_names, guesses,
                 prior_fraction=PRIOR_FRACTION,
                 fix_y_offset=False):
    """
    Build uniform priors from initial guesses.

    Parameters
    ----------
    param_names   : list[str]
    guesses       : list[float]   Same length and order as param_names.
    prior_fraction: float         Half-width as a fraction of |guess|.
    fix_y_offset  : bool          If True, y_offset is given a razor-thin
                                  prior (±0.1% of the guess) so the sampler
                                  effectively treats it as fixed.  Use this
                                  to resolve Gaussian amplitude/offset
                                  degeneracy when the region never returns
                                  to baseline on both sides.

    Returns
    -------
    dict  {param_name: {'lower': float, 'upper': float}}
    """
    if len(param_names) != len(guesses):
        raise ValueError(
            f"param_names has {len(param_names)} entries but "
            f"guesses has {len(guesses)}."
        )

    priors = {}
    for name, g in zip(param_names, guesses):
        if name == 'y_offset':
            if fix_y_offset:
                frac = 0.001          # effectively fixed — ±0.1%
            else:
                frac = _Y_OFFSET_PRIOR_FRACTION
        else:
            frac = prior_fraction

        half_width = max(frac * abs(g), _PRIOR_ABS_FLOOR)
        priors[name] = {'lower': g - half_width, 'upper': g + half_width}

    return priors


def build_priors_from_curvefit(param_names, popt, pcov,
                               n_sigma=2.0,
                               fix_y_offset=False):
    """
    Build priors from curve_fit results, with covariance-based widths.

    Covariance-derived widths are used only when they are finite and not more
    than 10× the flat-prior fallback; otherwise the flat rule is used.
    y_offset always uses the flat rule (covariance is unreliable for offsets
    due to amplitude/y_offset anti-correlation).

    Parameters
    ----------
    param_names  : list[str]
    popt         : array-like    curve_fit optimal parameters
    pcov         : 2-D array     curve_fit covariance matrix
    n_sigma      : float         prior half-width = n_sigma * std
    fix_y_offset : bool          Same meaning as in build_priors().

    Returns
    -------
    dict  {param_name: {'lower': float, 'upper': float}}
    """
    stds   = np.sqrt(np.diag(pcov))
    priors = {}

    for name, g, s in zip(param_names, popt, stds):
        if name == 'y_offset':
            if fix_y_offset:
                frac = 0.001
            else:
                frac = _Y_OFFSET_PRIOR_FRACTION
            half_width = max(frac * abs(g), _PRIOR_ABS_FLOOR)
        else:
            half_width_fb  = max(PRIOR_FRACTION * abs(g), _PRIOR_ABS_FLOOR)
            half_width_cf  = n_sigma * s
            if np.isfinite(s) and s > 0 and half_width_cf <= 10.0 * half_width_fb:
                half_width = half_width_cf
            else:
                half_width = half_width_fb

        priors[name] = {'lower': g - half_width, 'upper': g + half_width}

    return priors