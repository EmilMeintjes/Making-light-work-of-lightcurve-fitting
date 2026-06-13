"""
fitter.py
---------
Stage 3: MCMC fitting via PyAutoFit / emcee.

For each initialised region in the RegionStore, this module:
  1. Reads the initial guesses and priors from the store
  2. Builds a PyAutoFit model with Uniform priors on every parameter
  3. Runs emcee via PyAutoFit's DynestyStatic or Emcee search
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
import contextlib
import os
import re
import shutil

# External imports
import numpy as np

# Check PyAutoFit is installed; will flag later
try:
    import autofit as af
    _HAS_AUTOFIT = True
except ImportError:
    _HAS_AUTOFIT = False

# Local imports
from fitting_models import MODELS, evaluate
from persistence import RegionStore, save_mcmc_results


def _force_noninteractive_mpl():
    """
    Force matplotlib onto the non-interactive 'Agg' backend for fitting.

    `selector.py` / `initialiser.py` set the interactive 'TkAgg' backend at
    import time.  During a fit, PyAutoFit creates matplotlib figures for its
    own visualisation; with the GUI progress window, that happens on a worker
    thread while Tkinter's main loop runs on the main thread.  Letting those
    figures use 'TkAgg' means touching Tk from the worker thread, which on
    macOS corrupts the event loop and segfaults at shutdown.  'Agg' draws to
    memory/files only, so it is thread-safe and needs no GUI.  The pure-tkinter
    progress window is unaffected (it does not use matplotlib at all).
    """
    import matplotlib
    if matplotlib.get_backend().lower() != 'agg':
        try:
            matplotlib.use('Agg', force=True)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

N_WALKERS   = 60
N_STEPS     = 1500
N_BURN      = 300    # steps to discard as burn-in before saving

# PyAutoFit's emcee wrapper computes an autocorrelation estimate over
# get_chain()[:-check_size] (check_size defaults to 100).  If n_steps <=
# check_size that slice is empty and emcee raises a cryptic
# "index 0 is out of bounds for axis 0 with size 0".  We require a margin
# above this so the failure surfaces early with a clear message instead.
_EMCEE_CHECK_SIZE = 100
_MIN_STEPS        = _EMCEE_CHECK_SIZE + 1   # absolute minimum for emcee to run

# emcee only needs walkers >= 2*n_params; far more wastes compute with no
# benefit.  Guard against accidental huge values (e.g. a typo like 10000).
_MAX_WALKERS_PER_PARAM = 50


# ---------------------------------------------------------------------------
# Progress reporting
# ---------------------------------------------------------------------------
# `progress`, when not None, is a `progress_window.ProgressReporter` (or
# anything with the same `.log()` / `.set_region()` / `.set_mcmc_progress()`
# interface). It lets run_fitter_gui() show a GUI progress bar + log window
# instead of printing to the terminal. run_fitter() itself stays usable
# headlessly: pass progress=None (the default) and it behaves exactly as
# before, printing to stdout.

def _log(progress, message):
    """Print *message* and, if a progress reporter is given, forward it too."""
    print(message)
    if progress is not None:
        progress.log(message)


class _TqdmCapture:

    """
    File-like object that intercepts stdout/stderr while `search.fit()` runs.

    emcee's `EnsembleSampler.sample(..., progress=True)` prints a tqdm
    progress bar using carriage-return ('\\r') updates rather than newlines.
    This class buffers writes, splits them on '\\r'/'\\n', and for each
    resulting line either:
      - extracts the percentage from a tqdm-style "NN%|...|" line and
        forwards it to `reporter.set_mcmc_progress(NN / 100)`, or
      - forwards any other line to `reporter.log(line)`.

    This is what turns the terminal tqdm bar into GUI progress-bar updates.
    """

    _PERCENT_RE = re.compile(r'(\d+)%\|')

    def __init__(self, reporter):
        self._reporter = reporter
        self._buf = ''

    def write(self, text):
        self._buf += text
        while True:
            idx_r = self._buf.find('\r')
            idx_n = self._buf.find('\n')
            candidates = [i for i in (idx_r, idx_n) if i != -1]
            if not candidates:
                break
            idx = min(candidates)
            line, self._buf = self._buf[:idx], self._buf[idx + 1:]
            self._handle_line(line)
        return len(text)

    def _handle_line(self, line):
        line = line.strip()
        if not line:
            return
        match = self._PERCENT_RE.search(line)
        if match:
            self._reporter.set_mcmc_progress(int(match.group(1)) / 100.0)
        else:
            self._reporter.log(line)

    def flush(self):
        pass

    def isatty(self):
        return False


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


class _Analysis(af.Analysis):

    """
    PyAutoFit Analysis: defines the log-likelihood for a given model + dataset.

    Parameters
    ----------
    t          : np.ndarray   Time values for this region.
    flux       : np.ndarray   Flux values.
    uncertainty: np.ndarray or None
    model_key  : str          Key into models.MODELS.
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
        numpy here forces the pure-numpy path on all platforms, regardless of
        what PyAutoFit detects about the host hardware.
        """
        return np

    @property
    def _use_jax(self):
        """Explicitly disable JAX on all platforms. Ensure compatibility with M5 silicon chips +laziness hehe"""
        return False

    def log_likelihood_function(self, instance):

        """
        Gaussian log-likelihood.

        If uncertainties are provided:
            ln L = -0.5 * sum( ((flux - model) / sigma)^2 )

        If not:
            ln L = -0.5 * sum( (flux - model)^2 )  [unweighted]
        """

        params = [getattr(instance, p) for p in self.param_names]
        try:
            model_flux = evaluate(self.model_key, self.t, params)
        except Exception:
            return -np.inf

        if not np.all(np.isfinite(model_flux)):
            return -np.inf

        residuals = self.flux - model_flux

        if self.uncertainty is not None:
            # Guard against zero or negative uncertainties
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
                    n_walkers, n_steps, n_burn, force_refit=False,
                    progress=None):
    """
    Run MCMC for a single region.  Returns the summary dict or None on failure.

    Parameters
    ----------
    force_refit : bool
        If True, delete any existing PyAutoFit output directory for this
        segment before fitting.  This is necessary when a previous run crashed
        and left a partial or corrupt state that causes PyAutoFit to fail on
        resumption.

        If False (default), raise a RuntimeError if a PyAutoFit directory
        already exists so the caller can decide what to do.  This prevents
        silent deletion of a completed, good fit.

    progress : progress_window.ProgressReporter or None
        If given, emcee's tqdm progress bar is captured and converted into
        `progress.set_mcmc_progress(fraction)` calls instead of being printed.
    """

    if not _HAS_AUTOFIT:
        raise ImportError(
            "PyAutoFit is not installed.  "
            "Install it with:  pip install pyautofit emcee"
        )

    sid         = region['segment_id']
    model_key   = region['model']
    param_names = MODELS[model_key]['params']
    guesses     = region.get('initial_guesses', {})
    priors_dict = region.get('priors', {})

    if not guesses:
        _log(progress, f"  Region #{sid}: no initial guesses — skipping.  "
                        "Run the initialiser first.")
        return None

    # --- Handle stale PyAutoFit output directory ---
    # PyAutoFit writes search state (search.pkl, chain files) into paf_path.
    # If a previous run crashed, these files can be incomplete or inconsistent,
    # causing PyAutoFit to fail on any subsequent attempt to fit the same
    # segment.  We detect this situation and either abort (force_refit=False)
    # or wipe the directory (force_refit=True).
    paf_path = os.path.join(os.path.abspath(results_dir),
                            'pyautofit', f'seg{sid:04d}')

    if os.path.exists(paf_path):
        if force_refit:
            _log(progress, f"  force_refit=True: removing stale PyAutoFit "
                            f"directory for seg #{sid}.")
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

    # Create a model from _ParameterSet with one prior per attribute
    model = af.Model(_ParameterSet, **af_priors)

    # --- Analysis ---
    analysis = _Analysis(t_r, f_r, u_r, model_key, param_names)

    # --- Emcee search ---
    search = af.Emcee(
        path_prefix  = paf_path,
        name         = f'seg{sid:04d}',
        nwalkers     = n_walkers,
        nsteps       = n_steps,
    )

    _log(progress, f"  Running emcee: {n_walkers} walkers x {n_steps} steps ...")
    if progress is not None:
        progress.set_mcmc_progress(0.0)
        capture = _TqdmCapture(progress)
        with contextlib.redirect_stdout(capture), contextlib.redirect_stderr(capture):
            result = search.fit(model=model, analysis=analysis)
        progress.set_mcmc_progress(1.0)
    else:
        result = search.fit(model=model, analysis=analysis)

    # --- Extract samples ---
    # PyAutoFit stores the full chain via result.samples
    # result.samples.parameter_lists: list of lists, shape (n_total, n_params)
    all_params = np.array(result.samples.parameter_lists)   # (n_total, n_params)

    # Burn-in discard
    if n_burn > 0 and all_params.shape[0] > n_burn:
        samples = all_params[n_burn:]
    else:
        samples = all_params

    # log-probabilities
    try:
        lnprob = np.array(result.samples.log_likelihood_list)[n_burn:]
    except Exception:
        lnprob = None

    # --- Persist ---
    metadata = {
        'model':     model_key,
        'start':     region['start'],
        'end':       region['end'],
        't_ref':     region['start'],   # time offset; add to params to get abs. coords
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
               force_refit=False,
               progress=None):

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
                    If given, only fit these segments.  Useful for re-fitting
                    a specific region without re-running all.
    force_refit   : bool
                    If True, delete any stale PyAutoFit output directories
                    before fitting.  Use this to recover from crashed runs.
                    If False (default), a RuntimeError is raised if a prior
                    PyAutoFit directory is detected, so you can decide
                    consciously whether to overwrite it.
    progress      : progress_window.ProgressReporter or None
                    If given, status messages and MCMC progress are sent here
                    instead of (in addition to) being printed — see
                    run_fitter_gui() for the GUI version that supplies this.

    Returns
    -------
    summaries : dict  {segment_id: summary_dict}   One entry per fitted region.
    """

    if not _HAS_AUTOFIT:
        raise ImportError(
            "PyAutoFit is not installed.  "
            "Install it with:  pip install pyautofit emcee"
        )

    # Make sure PyAutoFit's own plotting never touches Tk (see helper docstring).
    _force_noninteractive_mpl()

    if n_steps < _MIN_STEPS:
        raise ValueError(
            f"n_steps={n_steps} is too small.  PyAutoFit's emcee "
            f"autocorrelation check needs more than {_EMCEE_CHECK_SIZE} steps "
            f"(it analyses the chain minus the last {_EMCEE_CHECK_SIZE}), so "
            f"fewer would crash inside emcee.  Use at least {_MIN_STEPS} "
            f"(a few hundred or more is recommended; the default is {N_STEPS}). "
            f"On the command line: --steps {N_STEPS}"
        )

    t    = np.asarray(t,    dtype=float)
    flux = np.asarray(flux, dtype=float)
    if uncertainty is not None:
        uncertainty = np.asarray(uncertainty, dtype=float)

    store = RegionStore(regions_file)
    store.load()

    if len(store) == 0:
        _log(progress, "No regions found.  Run the selector first.")
        return {}

    # Determine which segments to fit
    if segment_ids is not None:
        todo = [store.get(sid) for sid in segment_ids]
    else:
        todo = [r for r in store if r.get('initial_guesses')]
        skipped = [r['segment_id'] for r in store
                   if not r.get('initial_guesses')]
        if skipped:
            _log(progress, f"Skipping un-initialised regions: {skipped}")

    if not todo:
        _log(progress, "No initialised regions to fit.")
        return {}

    # Guard against an unreasonably large walker count (e.g. a typo like
    # 10000): emcee only needs walkers >= 2*n_params, so cap relative to the
    # smallest model being fit.
    min_params  = min(len(MODELS[r['model']]['params']) for r in todo)
    max_walkers = _MAX_WALKERS_PER_PARAM * min_params
    if n_walkers > max_walkers:
        raise ValueError(
            f"n_walkers={n_walkers} is excessive for a model with as few as "
            f"{min_params} parameters.  emcee only needs walkers >= "
            f"2*n_params; more than ~{_MAX_WALKERS_PER_PARAM}*n_params "
            f"(={max_walkers} here) just wastes compute.  Use something like "
            f"{N_WALKERS} (the default).  On the command line: "
            f"--walkers {N_WALKERS}"
        )

    _log(progress, f"\nFitting {len(todo)} region(s): "
                    f"{[r['segment_id'] for r in todo]}")

    summaries = {}
    for i, region in enumerate(todo):
        sid = region['segment_id']
        _log(progress, f"\n--- Region #{sid}  ({i+1}/{len(todo)})  "
                        f"[{region['model']}]  "
                        f"{region['start']:.6g} -> {region['end']:.6g} ---")
        if progress is not None:
            progress.set_region(i + 1, len(todo), sid, label=f"[{region['model']}]")

        # Clip data to the region and shift time to region-relative coords.
        # All fitted parameters involving time (centre, t_peak, etc.) are
        # therefore in units of (time since region start), matching what the
        # initialiser showed the user on the sliders.
        mask   = (t >= region['start']) & (t <= region['end'])
        t_r    = t[mask] - region['start']   # region-relative
        f_r    = flux[mask]
        u_r    = uncertainty[mask] if uncertainty is not None else None

        if len(t_r) == 0:
            _log(progress, f"  WARNING: no data points inside region #{sid} — skipping.")
            continue

        try:
            summary = _fit_one_region(
                t_r, f_r, u_r, region, results_dir,
                n_walkers=n_walkers,
                n_steps=n_steps,
                n_burn=n_burn,
                force_refit=force_refit,
                progress=progress,
            )
            if summary is not None:
                summaries[sid] = summary
                _log(progress, "  Done.  Median parameters:")
                for pname, s in summary['statistics'].items():
                    _log(progress,
                         f"    {pname:<20}  "
                         f"{s['median']:.6g} "
                         f"+{s['err_hi']:.4g} / -{s['err_lo']:.4g}")
        except Exception as exc:
            _log(progress, f"  ERROR fitting region #{sid}: {exc}")

    return summaries


# ---------------------------------------------------------------------------
# GUI entry point
# ---------------------------------------------------------------------------

def run_fitter_gui(t, flux, uncertainty=None, **kwargs):

    """
    Like `run_fitter`, but shows a Tk window with an overall "region i of n"
    progress bar, a "current region" MCMC progress bar (driven by emcee's
    tqdm output), and a separate scrolling log window — instead of printing
    to the terminal.

    Runs `run_fitter` in a background thread so the Tk windows stay
    responsive; blocks until fitting completes (the windows can be closed
    early without interrupting the fit — see `ProgressWindow`).

    Parameters
    ----------
    t, flux, uncertainty : as for `run_fitter`.
    **kwargs              : forwarded to `run_fitter` (regions_file,
                             results_dir, n_walkers, n_steps, n_burn,
                             segment_ids, force_refit).  Do not pass
                             `progress` — it is supplied by this function.

    Returns
    -------
    summaries : dict  {segment_id: summary_dict}, as for `run_fitter`.

    Raises
    ------
    Whatever exception `run_fitter` raised, re-raised here after the windows
    are closed.
    """

    # Switch matplotlib off the interactive Tk backend *here on the main
    # thread*, before the worker thread starts — so PyAutoFit's visualisation
    # never touches Tk from the worker (which would segfault on macOS).
    _force_noninteractive_mpl()

    from progress_window import ProgressWindow

    window = ProgressWindow()
    result, error = window.run(run_fitter, args=(t, flux),
                                kwargs=dict(uncertainty=uncertainty, **kwargs))
    if error is not None:
        raise error
    return result if result is not None else {}
