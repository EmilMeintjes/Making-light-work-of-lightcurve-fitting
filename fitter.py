"""
fitter.py
---------
Stage 3: MCMC fitting via PyAutoFit / emcee.

For each initialised region in the RegionStore, this module:
  1. Reads the initial guesses and priors from the store
  2. Builds a PyAutoFit model with Uniform priors on every parameter
  3. Runs emcee via PyAutoFit's Emcee search
  4. Saves posterior samples + summary to disk via persistence.save_mcmc_results

PyAutoFit wraps emcee so that the model definition, prior specification, and
sampler configuration all live in one place.  The Analysis class below is the
only thing that needs to know about the likelihood function.

Dependencies
------------
    pip install pyautofit emcee

Typical usage
-------------
    from fitter import run_fitter
    run_fitter(t, flux, uncertainty,
               regions_file='regions.json',
               results_dir='results/')
"""

# System imports
import os
import shutil

# External imports
import numpy as np

try:
    import autofit as af
    _HAS_AUTOFIT = True
except ImportError:
    _HAS_AUTOFIT = False

# Local imports
from fitting_models import evaluate, param_names_for
from persistence import RegionStore, save_mcmc_results


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

N_WALKERS = 60
N_STEPS   = 1500
N_BURN    = 300    # steps to discard as burn-in before saving


# ---------------------------------------------------------------------------
# PyAutoFit model wrapper
# ---------------------------------------------------------------------------

class _ParameterSet:
    """
    A plain Python class whose attributes are the free parameters.

    PyAutoFit maps each attribute to a prior during model construction.
    The attribute names must match the parameter names in the model registry.
    """

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class _Analysis(af.Analysis if _HAS_AUTOFIT else object):
    """
    PyAutoFit Analysis: defines the log-likelihood for a given model + dataset.

    Parameters
    ----------
    t          : np.ndarray   Time values for this region.
    flux       : np.ndarray   Flux values.
    uncertainty: np.ndarray or None
    model_key  : str          Key into fitting_models (e.g. 'gaussian').
    param_names: list of str  Parameter names, in order.
    """

    def __init__(self, t, flux, uncertainty, model_key, param_names):
        self.t           = t
        self.flux        = flux
        self.uncertainty = uncertainty
        self.model_key   = model_key
        self.param_names = param_names

    @property
    def _xp(self):
        """Force numpy as the array backend — disables JAX dispatch.

        Some PyAutoFit versions attempt to route array operations through JAX
        when running on Apple Silicon (M-series chips).  Explicitly returning
        numpy here forces the pure-numpy path on all platforms.
        """
        return np

    @property
    def _use_jax(self):
        """Explicitly disable JAX on all platforms."""
        return False

    def log_likelihood_function(self, instance):
        """
        Gaussian log-likelihood.

        If uncertainties are provided:
            ln L = -0.5 * sum( ((flux - model) / sigma)^2 )

        If not:
            ln L = -0.5 * sum( (flux - model)^2 )  [unweighted]
        """
        params = {p: getattr(instance, p) for p in self.param_names}
        try:
            model_flux = evaluate(self.model_key, self.t, params)
        except Exception:
            return -np.inf

        if not np.all(np.isfinite(model_flux)):
            return -np.inf

        residuals = self.flux - model_flux

        if self.uncertainty is not None:
            sigma = np.where(self.uncertainty > 0,
                             self.uncertainty, 1e-10)
            log_l = -0.5 * np.sum((residuals / sigma) ** 2)
        else:
            log_l = -0.5 * np.sum(residuals ** 2)

        return float(log_l) if np.isfinite(log_l) else -np.inf


# ---------------------------------------------------------------------------
# Prior construction helper
# ---------------------------------------------------------------------------

def _build_af_priors(param_names, priors_dict):
    """
    Convert the persistence-format priors dict into PyAutoFit UniformPrior
    objects.

    Parameters
    ----------
    param_names  : list of str
    priors_dict  : dict  {param_name: {'lower': float, 'upper': float}, ...}

    Returns
    -------
    af_priors : dict  {param_name: af.UniformPrior}
    """
    af_priors = {}
    for name in param_names:
        if name in priors_dict:
            lo = priors_dict[name]['lower']
            hi = priors_dict[name]['upper']
        else:
            # Fallback: wide uninformative prior centred on zero
            lo, hi = -1e3, 1e3
        af_priors[name] = af.UniformPrior(lower_limit=lo, upper_limit=hi)
    return af_priors


# ---------------------------------------------------------------------------
# Single-region fitter
# ---------------------------------------------------------------------------

def _fit_one_region(t_r, f_r, u_r, region, results_dir,
                    n_walkers, n_steps, n_burn, force_refit=False):
    """
    Run MCMC for a single region.  Returns the summary dict or None on failure.

    Parameters
    ----------
    force_refit : bool
        If True, delete any existing PyAutoFit output directory for this
        segment before fitting.  Necessary when a previous run crashed and
        left a partial or corrupt state.

        If False (default), raise a RuntimeError if a PyAutoFit directory
        already exists so the caller can decide what to do.  This prevents
        silent deletion of a completed, good fit.
    """
    if not _HAS_AUTOFIT:
        raise ImportError(
            "PyAutoFit is not installed.  "
            "Install it with:  pip install pyautofit emcee"
        )

    sid         = region['segment_id']
    model_key   = region['model']
    param_names = param_names_for(model_key)          # ← new API
    guesses     = region.get('initial_guesses', {})
    priors_dict = region.get('priors', {})

    if not guesses:
        print(f"  Region #{sid}: no initial guesses — skipping.  "
              "Run the initialiser first.")
        return None

    # --- Handle stale PyAutoFit output directory ---
    paf_path = os.path.join(os.path.abspath(results_dir),
                            'pyautofit', f'seg{sid:04d}')

    if os.path.exists(paf_path):
        if force_refit:
            print(f"  force_refit=True: removing stale PyAutoFit directory "
                  f"for seg #{sid}.")
            shutil.rmtree(paf_path)
        else:
            raise RuntimeError(
                f"PyAutoFit output directory already exists for seg #{sid}.\n"
                f"  Path: {paf_path}\n"
                "This may be from a completed run or a crashed run.\n"
                "  - To refit from scratch, pass force_refit=True\n"
                "    (or use --force on the command line).\n"
                "  - To keep the existing result, skip this segment."
            )

    os.makedirs(paf_path, exist_ok=True)

    # --- Build PyAutoFit model ---
    af_priors = _build_af_priors(param_names, priors_dict)
    model     = af.Model(_ParameterSet, **af_priors)

    # --- Analysis ---
    analysis = _Analysis(t_r, f_r, u_r, model_key, param_names)

    # --- Emcee search ---
    search = af.Emcee(
        path_prefix=paf_path,
        name=f'seg{sid:04d}',
        nwalkers=n_walkers,
        nsteps=n_steps,
    )

    print(f"  Running emcee: {n_walkers} walkers x {n_steps} steps ...")
    result = search.fit(model=model, analysis=analysis)

    # --- Extract samples ---
    all_params = np.array(result.samples.parameter_lists)   # (n_total, n_params)

    if n_burn > 0 and all_params.shape[0] > n_burn:
        samples = all_params[n_burn:]
    else:
        samples = all_params

    try:
        lnprob = np.array(result.samples.log_likelihood_list)[n_burn:]
    except Exception:
        lnprob = None

    # --- Persist ---
    metadata = {
        'model':     model_key,
        'start':     region['start'],
        'end':       region['end'],
        't_ref':     region['start'],
        'n_walkers': n_walkers,
        'n_steps':   n_steps,
        'n_burn':    n_burn,
        'note':      region.get('note', ''),
    }

    summary = save_mcmc_results(
        results_dir, sid,
        samples=samples,
        param_names=param_names,
        lnprob=lnprob,
        metadata=metadata,
    )
    return summary


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_fitter(t, flux, uncertainty=None,
               regions_file='regions.json',
               results_dir='results',
               n_walkers=N_WALKERS,
               n_steps=N_STEPS,
               n_burn=N_BURN,
               segment_ids=None,
               force_refit=False):
    """
    Run MCMC fitting for all initialised regions (or a specified subset).

    Parameters
    ----------
    t, flux       : array-like   Full lightcurve.
    uncertainty   : array-like or None
    regions_file  : str          JSON file from the selector.
    results_dir   : str          Directory for .npy and .json output files.
    n_walkers     : int          Number of emcee walkers (default 60).
    n_steps       : int          Number of emcee steps (default 1500).
    n_burn        : int          Burn-in steps to discard (default 300).
    segment_ids   : list of int or None
                    If given, only fit these segments.
    force_refit   : bool
                    If True, delete stale PyAutoFit output directories before
                    fitting.  Use to recover from crashed runs.
                    If False (default), a RuntimeError is raised if a prior
                    PyAutoFit directory is detected.

    Returns
    -------
    summaries : dict  {segment_id: summary_dict}
    """
    if not _HAS_AUTOFIT:
        raise ImportError(
            "PyAutoFit is not installed.  "
            "Install it with:  pip install pyautofit emcee"
        )

    t    = np.asarray(t,    dtype=float)
    flux = np.asarray(flux, dtype=float)
    if uncertainty is not None:
        uncertainty = np.asarray(uncertainty, dtype=float)

    store = RegionStore(regions_file)
    store.load()

    if len(store) == 0:
        print("No regions found.  Run the selector first.")
        return {}

    if segment_ids is not None:
        todo = [store.get(sid) for sid in segment_ids]
    else:
        todo    = [r for r in store if r.get('initial_guesses')]
        skipped = [r['segment_id'] for r in store
                   if not r.get('initial_guesses')]
        if skipped:
            print(f"Skipping un-initialised regions: {skipped}")

    if not todo:
        print("No initialised regions to fit.")
        return {}

    print(f"\nFitting {len(todo)} region(s): "
          f"{[r['segment_id'] for r in todo]}")

    summaries = {}
    for i, region in enumerate(todo):
        sid = region['segment_id']
        print(f"\n--- Region #{sid}  ({i+1}/{len(todo)})  "
              f"[{region['model']}]  "
              f"{region['start']:.6g} -> {region['end']:.6g} ---")

        # Clip data to region; shift to region-relative time coords.
        # All time-related fitted parameters are in these coordinates,
        # matching what the initialiser showed on the sliders.
        mask = (t >= region['start']) & (t <= region['end'])
        t_r  = t[mask] - region['start']   # region-relative
        f_r  = flux[mask]
        u_r  = uncertainty[mask] if uncertainty is not None else None

        if len(t_r) == 0:
            print(f"  WARNING: no data points inside region #{sid} — skipping.")
            continue

        try:
            summary = _fit_one_region(
                t_r, f_r, u_r, region, results_dir,
                n_walkers=n_walkers,
                n_steps=n_steps,
                n_burn=n_burn,
                force_refit=force_refit,
            )
            if summary is not None:
                summaries[sid] = summary
                print(f"  Done.  Median parameters:")
                for pname, s in summary['statistics'].items():
                    print(f"    {pname:<20}  "
                          f"{s['median']:.6g} "
                          f"+{s['err_hi']:.4g} / -{s['err_lo']:.4g}")
        except Exception as exc:
            print(f"  ERROR fitting region #{sid}: {exc}")

    return summaries