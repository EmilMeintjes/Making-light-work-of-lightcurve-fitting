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


def bazin(t, amplitude, t0, tau_rise, tau_fall, y_offset):

    """
    Bazin et al. (2009) transient lightcurve function: a sigmoid-modulated
    exponential decay that smoothly rises to a peak near t0 and then falls.

        f(t) = A * exp(-(t - t0) / tau_fall) / (1 + exp(-(t - t0) / tau_rise)) + y_offset

    Unlike the separate rising/decaying exponentials, this is a single
    smooth function covering the full rise-peak-fall shape with no
    discontinuities, making it suitable for fitting an entire lightcurve
    in one go.

    Parameters
    ----------
    amplitude : float   Overall scale of the outburst.
    t0        : float   Reference time near the peak.
    tau_rise  : float   Rise timescale; must be > 0.
    tau_fall  : float   Decay timescale; must be > 0.
    y_offset  : float   Constant baseline level.
    """

    x = t - t0
    return amplitude * np.exp(-x / tau_fall) / (1.0 + np.exp(-x / tau_rise)) + y_offset


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


def gaussian_exp_wings(t, amplitude, centre, sigma, tau_rise, tau_fall, y_offset):

    """
    Gaussian core with exponential wings — a single, continuous rise-peak-fall
    transient profile that combines an exponential rise, a Gaussian peak and an
    exponential decay into one function with shared parameters.

        Let z = (t - centre) / sigma,
            a_L = sigma / tau_rise,   a_R = sigma / tau_fall

        f(t) = A * exp(+a_L * z + 0.5 * a_L^2) + y_offset   if z < -a_L  (rise)
             = A * exp(-0.5 * z^2)             + y_offset   if -a_L<=z<=a_R
             = A * exp(-a_R * z + 0.5 * a_R^2) + y_offset   if z >  a_R  (fall)

    Why this form
    -------------
    The two exponential wings are joined to the Gaussian core at z = -a_L
    (left) and z = +a_R (right).  The join points and wing amplitudes are
    *derived* from sigma, tau_rise and tau_fall so that both the value and the
    first derivative are continuous there — the curve has no kinks.  tau_rise
    and tau_fall are the e-folding timescales of the rising and falling wings
    respectively (a larger tau => a shallower, more slowly varying wing).

    Unlike fitting three separate regions, this is one model over one region:
    the peak position, width, baseline and both wing timescales are fit
    together, so the rise, peak and decay are guaranteed to meet seamlessly.

    Parameters
    ----------
    amplitude : float   Peak height above baseline.
    centre    : float   Peak position (mu).
    sigma     : float   Gaussian core width; must be > 0.
    tau_rise  : float   Rising-wing e-folding timescale; must be > 0.
    tau_fall  : float   Falling-wing e-folding timescale; must be > 0.
    y_offset  : float   Constant baseline level.
    """

    t   = np.asarray(t, dtype=float)
    z   = (t - centre) / sigma
    a_L = sigma / tau_rise
    a_R = sigma / tau_fall

    # All three branches are evaluated then selected; the unused branches can
    # overflow to +inf for extreme z, which np.where simply discards — so we
    # silence the harmless overflow warning rather than let it print.
    with np.errstate(over='ignore', invalid='ignore'):
        core  = np.exp(-0.5 * z ** 2)
        left  = np.exp(a_L * z + 0.5 * a_L ** 2)
        right = np.exp(-a_R * z + 0.5 * a_R ** 2)
        g = np.where(z < -a_L, left, np.where(z > a_R, right, core))

    return amplitude * g + y_offset


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
#   'latex'       : the model equation as a matplotlib-mathtext string (the
#                   surrounding $...$ are included), shown beside the sliders.
#   'symbols'     : per-parameter mathtext symbol (no $...$), in the same order
#                   as 'params'; used to label each slider with the symbol it
#                   corresponds to in 'latex'.

MODELS = {
    'gaussian': {
        'func':        gaussian,
        'params':      ['amplitude', 'centre', 'sigma', 'y_offset'],
        'label':       'Gaussian',
        'defaults':    [1.0, 0.0, 1.0, 0.0],
        'bounds_hint': (0.5, 1.5),
        'latex':       r'$f(t) = A\,\exp\!\left[-\frac{(t-\mu)^2}{2\sigma^2}\right] + c$',
        'symbols':     ['A', r'\mu', r'\sigma', 'c'],
    },
    'rising_exp': {
        'func':        rising_exponential,
        'params': ['amplitude', 'tau_rise', 'y_offset'],
        'label':       'Rising Exponential',
        'defaults':    [1.0, 1.0, 0.0],
        'bounds_hint': (0.5, 1.5),
        'latex':       r'$f(t) = A\,e^{\,t/\tau_r} + c$',
        'symbols':     ['A', r'\tau_r', 'c'],
    },
    'decaying_exp': {
        'func':        decaying_exponential,
        'params':      ['amplitude', 'tau_decay', 'y_offset'],
        'label':       'Decaying Exponential',
        'defaults':    [1.0, 1.0, 0.0],
        'bounds_hint': (0.5, 1.5),
        'latex':       r'$f(t) = A\,e^{-t/\tau_d} + c$',
        'symbols':     ['A', r'\tau_d', 'c'],
    },
    'crystal_ball': {
        'func':        crystal_ball,
        'params':      ['amplitude', 'centre', 'sigma', 'alpha', 'n', 'y_offset'],
        'label':       'Crystal Ball',
        'defaults':    [1.0, 0.0, 1.0, 1.0, 2.0, 0.0],
        'bounds_hint': (0.5, 1.5),
        'latex':       (r'$f(t) = A\,e^{-z^2/2} + c\ \ (z > -\alpha),'
                        r'\quad z = \frac{t-\mu}{\sigma}$' '\n'
                        r'$(\mathrm{power\ law,\ index\ } n,\ \mathrm{for}\ z < -\alpha)$'),
        'symbols':     ['A', r'\mu', r'\sigma', r'\alpha', 'n', 'c'],
    },
    'bazin': {
        'func':        bazin,
        'params':      ['amplitude', 't0', 'tau_rise', 'tau_fall', 'y_offset'],
        'label':       'Bazin (rise-peak-fall)',
        'defaults':    [1.0, 0.0, 1.0, 1.0, 0.0],
        'bounds_hint': (0.5, 1.5),
        'latex':       r'$f(t) = A\,\frac{e^{-(t-t_0)/\tau_f}}{1 + e^{-(t-t_0)/\tau_r}} + c$',
        'symbols':     ['A', 't_0', r'\tau_r', r'\tau_f', 'c'],
    },
    'gauss_exp_wings': {
        'func':        gaussian_exp_wings,
        'params':      ['amplitude', 'centre', 'sigma', 'tau_rise', 'tau_fall', 'y_offset'],
        'label':       'Gaussian + Exp. wings',
        'defaults':    [1.0, 0.0, 1.0, 1.0, 1.0, 0.0],
        'bounds_hint': (0.5, 1.5),
        'latex':       (r'$f(t) = A\,e^{-z^2/2} + c,\quad z = \frac{t-\mu}{\sigma}$' '\n'
                        r'$(\mathrm{exp.\ rise}\ \tau_r\ \mathrm{for}\ z<-\sigma/\tau_r,\ '
                        r'\mathrm{exp.\ decay}\ \tau_f\ \mathrm{for}\ z>\sigma/\tau_f)$'),
        'symbols':     ['A', r'\mu', r'\sigma', r'\tau_r', r'\tau_f', 'c'],
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

_Y_OFFSET_PRIOR_FRACTION = 0.10  # ±10% around the user's visual estimate — module level

def build_priors(param_names, guesses, prior_fraction=PRIOR_FRACTION):
    """
    Construct uniform prior bounds for each parameter given initial guesses.

    For each parameter p_i with guess g_i:

        lower_i = g_i - max(prior_fraction * |g_i|, _PRIOR_ABS_FLOOR)
        upper_i = g_i + max(prior_fraction * |g_i|, _PRIOR_ABS_FLOOR)

    y_offset uses _Y_OFFSET_PRIOR_FRACTION (±10%) regardless of prior_fraction,
    to prevent the amplitude/y_offset degeneracy in Gaussian fits.
    """

    if len(param_names) != len(guesses):
        raise ValueError(
            f"param_names has {len(param_names)} entries but guesses has "
            f"{len(guesses)}."
        )

    priors = {}
    for name, g in zip(param_names, guesses):
        frac       = _Y_OFFSET_PRIOR_FRACTION if name == 'y_offset' else prior_fraction
        half_width = max(frac * abs(g), _PRIOR_ABS_FLOOR)
        priors[name] = {
            'lower': g - half_width,
            'upper': g + half_width,
        }
    return priors


def build_priors_from_curvefit(param_names, popt, pcov, n_sigma=2.0):
    """
    Construct uniform prior bounds from a curve_fit result.

    For y_offset: always uses the flat ±10% rule around the fitted value,
    ignoring the covariance — the covariance on y_offset is unreliable when
    amplitude and offset are anti-correlated (Gaussian fits).

    For all other parameters: uses n_sigma * std from the covariance, but
    falls back to flat ±30% if the covariance-derived width exceeds 10x the
    flat fallback (ill-conditioned fit).
    """

    stds   = np.sqrt(np.diag(pcov))
    priors = {}
    for name, g, s in zip(param_names, popt, stds):

        if name == 'y_offset':
            # Always use tight flat prior for offset — covariance unreliable
            half_width = max(_Y_OFFSET_PRIOR_FRACTION * abs(g), _PRIOR_ABS_FLOOR)
        else:
            half_width_cf = n_sigma * s
            half_width_fb = max(PRIOR_FRACTION * abs(g), _PRIOR_ABS_FLOOR)
            # Use CF width only if it's well-conditioned
            if np.isfinite(s) and s > 0 and half_width_cf <= 10.0 * half_width_fb:
                half_width = half_width_cf
            else:
                half_width = half_width_fb

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