"""
plots.py
--------
Publication-quality output plots for the nova-like lightcurve fitter.

Three plot types
----------------
1. Overview plot  (plot_overview)
   The full lightcurve with every fitted model overplotted in its region
   colour.  Median model + 1-sigma (16/84 percentile) shaded band.

2. Per-region fit plot  (plot_region_fit)
   Data within the region, median model, and 16/84 shaded band.  Parameter
   summary printed as a legend.

3. Corner plot  (plot_corner)
   Posterior distributions for all parameters of one region, using the
   `corner` package.

All functions accept an optional `save_path` argument.  If given, the figure
is saved there instead of (or as well as) being displayed.

Dependencies
------------
    pip install corner
"""

# System imports
import os

# External imports
import numpy as np
import matplotlib.pyplot as plt

try:
    import corner as corner_pkg
    _HAS_CORNER = True
except ImportError:
    _HAS_CORNER = False

# Local imports
from fitting_models import evaluate, param_names_for
from persistence import RegionStore, load_mcmc_results


# ---------------------------------------------------------------------------
# Human-readable model labels (mirrors _KEY_TO_LABEL in initialiser.py)
# Defined here so plots.py has no dependency on initialiser.
# ---------------------------------------------------------------------------

_KEY_TO_LABEL = {
    'gaussian':     'Gaussian',
    'rising_exp':   'Rising exp.',
    'decaying_exp': 'Decaying exp.',
    'crystal_ball': 'Crystal ball',
}


# ---------------------------------------------------------------------------
# Colour map: one colour per model type (matches selector.py)
# ---------------------------------------------------------------------------

_MODEL_COLOURS = {
    'gaussian':     'cornflowerblue',
    'rising_exp':   'mediumseagreen',
    'decaying_exp': 'tomato',
    'crystal_ball': 'mediumpurple',
}
_ALPHA_BAND  = 0.25
_ALPHA_SHADE = 0.12
_LW_MEDIAN   = 2.0
_N_DRAW      = 300   # number of posterior draws used to build the shaded band


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _posterior_band(model_key, t_fine, samples, param_names, n_draw=_N_DRAW):
    """
    Draw *n_draw* random posterior samples and evaluate the model at each.

    Returns
    -------
    median : np.ndarray   Median model at each point in t_fine.
    lo     : np.ndarray   16th percentile.
    hi     : np.ndarray   84th percentile.
    """
    rng    = np.random.default_rng()
    idx    = rng.choice(len(samples),
                        size=min(n_draw, len(samples)),
                        replace=False)
    draws  = samples[idx]                      # (n_draw, n_params)
    curves = np.empty((len(idx), len(t_fine)))

    for j, row in enumerate(draws):
        params = {p: row[i] for i, p in enumerate(param_names)}
        try:
            curves[j] = evaluate(model_key, t_fine, params)
        except Exception:
            curves[j] = np.nan

    median = np.nanpercentile(curves, 50, axis=0)
    lo     = np.nanpercentile(curves, 16, axis=0)
    hi     = np.nanpercentile(curves, 84, axis=0)
    return median, lo, hi


def _param_label(name, stats):
    """Format a parameter summary string for legend / annotation."""
    s = stats[name]
    return (f"{name} = "
            f"{s['median']:.4g}"
            f" +{s['err_hi']:.3g}"
            f" / -{s['err_lo']:.3g}")


def _save_or_show(fig, save_path, show):
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[plots] Saved -> {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


# ---------------------------------------------------------------------------
# 1. Overview plot
# ---------------------------------------------------------------------------

def plot_overview(t, flux, uncertainty=None,
                  regions_file='regions.json',
                  results_dir='results',
                  xlabel='Time', ylabel='Flux',
                  title='Lightcurve overview',
                  save_path=None, show=True):
    """
    Plot the full lightcurve with all fitted models overplotted.

    For regions that have MCMC results, the median model and 16/84 band are
    shown.  For regions that only have initial guesses (not yet fitted), the
    guess curve is shown as a dashed line.

    Parameters
    ----------
    t, flux       : array-like   Full lightcurve.
    uncertainty   : array-like or None
    regions_file  : str
    results_dir   : str
    xlabel, ylabel, title : str
    save_path     : str or None   If given, save to this path.
    show          : bool          Whether to call plt.show().
    """
    t    = np.asarray(t,    dtype=float)
    flux = np.asarray(flux, dtype=float)
    if uncertainty is not None:
        uncertainty = np.asarray(uncertainty, dtype=float)

    store = RegionStore(regions_file)
    store.load()

    fig, ax = plt.subplots(figsize=(16, 5))

    # --- Raw data ---
    if uncertainty is not None:
        ax.errorbar(t, flux, yerr=uncertainty,
                    fmt='o', ms=3, alpha=0.5,
                    color='steelblue', ecolor='lightgrey', elinewidth=0.8,
                    label='Data', zorder=2)
    else:
        ax.plot(t, flux, 'o', ms=3, alpha=0.5, color='steelblue',
                label='Data', zorder=2)

    # --- Per-region models ---
    print(f"[plots] store has {len(store.regions)} region(s): {store.ids()}")
    for region in store.regions:
        print(f"[plots] entering loop for seg #{region['segment_id']}")
        sid       = region['segment_id']
        model_key = region['model']
        colour    = _MODEL_COLOURS.get(model_key, 'grey')
        label     = f"#{sid} {_KEY_TO_LABEL.get(model_key, model_key)}"

        ax.axvspan(region['start'], region['end'],
                   alpha=_ALPHA_SHADE, color=colour, zorder=1)

        # t_fine in absolute coords for plotting; shifted for model evaluation
        t_fine         = np.linspace(region['start'], region['end'], 500)
        t_fine_shifted = t_fine - region['start']

        try:
            res         = load_mcmc_results(results_dir, sid)
            samples     = res['samples']
            param_names = res['param_names']
            print(f"[plots] seg #{sid}: {len(samples)} samples, "
                  f"params={param_names}, "
                  f"t_fine_shifted range [{t_fine_shifted.min():.4g}, "
                  f"{t_fine_shifted.max():.4g}]")
            median, lo, hi = _posterior_band(model_key, t_fine_shifted,
                                             samples, param_names)
            print(f"[plots] seg #{sid}: median range "
                  f"[{np.nanmin(median):.4g}, {np.nanmax(median):.4g}]")
            ax.plot(t_fine, median, color=colour, lw=_LW_MEDIAN,
                    label=label, zorder=4)
            ax.fill_between(t_fine, lo, hi,
                            color=colour, alpha=_ALPHA_BAND, zorder=3)

        except FileNotFoundError:
            guesses = region.get('initial_guesses', {})
            if guesses:
                param_names = param_names_for(model_key)
                params      = {p: guesses.get(p, 0.0) for p in param_names}
                try:
                    y_guess = evaluate(model_key, t_fine_shifted, params)
                    ax.plot(t_fine, y_guess, color=colour, lw=1.2,
                            ls='--', alpha=0.7,
                            label=f"{label} (guess)", zorder=3)
                except Exception as exc:
                    print(f"[plots] seg #{sid} guess plot failed: {exc}")

        except Exception as exc:
            print(f"[plots] seg #{sid} MCMC plot failed: {exc}")
            import traceback
            traceback.print_exc()

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xscale(store.xscale)
    ax.set_yscale(store.yscale)
    ax.legend(fontsize=8, loc='upper right')
    fig.tight_layout()

    _save_or_show(fig, save_path, show)
    return fig


# ---------------------------------------------------------------------------
# 2. Per-region fit plot
# ---------------------------------------------------------------------------

def plot_region_fit(t, flux, uncertainty=None,
                    segment_id=None,
                    regions_file='regions.json',
                    results_dir='results',
                    xlabel='Time', ylabel='Flux',
                    save_path=None, show=True):
    """
    Plot the data and posterior fit for a single region.

    Shows:
      - Data within the region (with error bars if available)
      - Median model (solid line)
      - 16/84 percentile band (shaded)
      - Parameter summary in the legend

    Parameters
    ----------
    segment_id : int   The region to plot.  Required.
    """
    if segment_id is None:
        raise ValueError("segment_id must be specified.")

    t    = np.asarray(t,    dtype=float)
    flux = np.asarray(flux, dtype=float)
    if uncertainty is not None:
        uncertainty = np.asarray(uncertainty, dtype=float)

    store = RegionStore(regions_file)
    store.load()
    print(f"[plots] store has {len(store.regions)} region(s): {store.ids()}")
    region = store.get(segment_id)
    xscale = store.xscale
    yscale = store.yscale

    # Load MCMC results
    res         = load_mcmc_results(results_dir, segment_id)
    samples     = res['samples']
    param_names = res['param_names']
    summary     = res['summary']
    model_key   = region['model']
    colour      = _MODEL_COLOURS.get(model_key, 'grey')

    # Clip data to region; shift time to region-relative coords for display.
    # Fitted parameters involving time are in these same shifted coordinates.
    t_ref = region['start']
    mask  = (t >= region['start']) & (t <= region['end'])
    t_r   = t[mask] - t_ref   # region-relative
    f_r   = flux[mask]
    u_r   = uncertainty[mask] if uncertainty is not None else None

    t_fine         = np.linspace(t_r.min(), t_r.max(), 500)
    median, lo, hi = _posterior_band(model_key, t_fine, samples, param_names)

    fig, ax = plt.subplots(figsize=(10, 5))

    if u_r is not None:
        ax.errorbar(t_r, f_r, yerr=u_r,
                    fmt='o', ms=4, alpha=0.7,
                    color='steelblue', ecolor='lightgrey', elinewidth=0.8,
                    label='Data', zorder=2)
    else:
        ax.plot(t_r, f_r, 'o', ms=4, alpha=0.7,
                color='steelblue', label='Data', zorder=2)

    ax.plot(t_fine, median, color=colour, lw=_LW_MEDIAN,
            label='Median model', zorder=4)
    ax.fill_between(t_fine, lo, hi,
                    color=colour, alpha=_ALPHA_BAND,
                    label='16/84 percentile', zorder=3)

    # Parameter summary in the legend via invisible proxy lines
    for pname in param_names:
        lbl = _param_label(pname, summary['statistics'])
        ax.plot([], [], ' ', label=lbl)

    ax.set_xlabel(xlabel + f' (relative to {t_ref:.6g})')
    ax.set_ylabel(ylabel)
    ax.set_title(
        f"Seg #{segment_id}  [{_KEY_TO_LABEL.get(model_key, model_key)}]  "
        f"note: \"{region.get('note', '')}\""
    )
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    ax.legend(fontsize=8, loc='upper right')
    fig.tight_layout()

    _save_or_show(fig, save_path, show)
    return fig


# ---------------------------------------------------------------------------
# 3. Corner plot
# ---------------------------------------------------------------------------

def plot_corner(segment_id,
                results_dir='results',
                regions_file='regions.json',
                save_path=None, show=True):
    """
    Corner plot of the posterior distribution for *segment_id*.

    Requires the `corner` package (pip install corner).
    Vertical lines mark the 16th, 50th, and 84th percentiles on each 1-D
    histogram.
    """
    if not _HAS_CORNER:
        raise ImportError(
            "The `corner` package is required for corner plots.  "
            "Install it with:  pip install corner"
        )

    res         = load_mcmc_results(results_dir, segment_id)
    samples     = res['samples']
    param_names = res['param_names']
    summary     = res['summary']

    # Build labels with units/values
    labels = []
    for pname in param_names:
        s   = summary['statistics'][pname]
        lbl = (f"{pname}\n"
               f"{s['median']:.4g} +{s['err_hi']:.3g}/-{s['err_lo']:.3g}")
        labels.append(lbl)

    fig = corner_pkg.corner(
        samples,
        labels=labels,
        quantiles=[0.16, 0.50, 0.84],
        show_titles=True,
        title_fmt='.4g',
        title_kwargs={'fontsize': 9},
        label_kwargs={'fontsize': 9},
        color='steelblue',
        hist_kwargs={'color': 'steelblue', 'alpha': 0.7},
    )

    model_key = summary.get('metadata', {}).get('model', '')
    fig.suptitle(
        f"Posterior — Seg #{segment_id}  "
        f"[{_KEY_TO_LABEL.get(model_key, model_key)}]  "
        f"({summary['n_draws']} draws)",
        fontsize=10, y=1.01,
    )

    _save_or_show(fig, save_path, show)
    return fig


# ---------------------------------------------------------------------------
# Convenience: generate all output plots for every fitted region
# ---------------------------------------------------------------------------

def plot_all(t, flux, uncertainty=None,
             regions_file='regions.json',
             results_dir='results',
             output_dir='plots',
             xlabel='Time', ylabel='Flux',
             show=False):
    """
    Generate and save all plots for every fitted region:
      - One overview PNG
      - One fit PNG per region
      - One corner PNG per region

    Parameters
    ----------
    output_dir : str    Directory in which to write the PNG files.
    show       : bool   Whether to display each figure interactively as well
                        as saving it.  Default False (save only).
    """
    os.makedirs(output_dir, exist_ok=True)

    store = RegionStore(regions_file)
    store.load()

    # --- Overview ---
    overview_path = os.path.join(output_dir, 'overview.png')
    plot_overview(
        t, flux, uncertainty,
        regions_file=regions_file,
        results_dir=results_dir,
        xlabel=xlabel, ylabel=ylabel,
        save_path=overview_path, show=show,
    )

    # --- Per-region ---
    for region in store:
        sid = region['segment_id']

        # Check results exist before trying to plot
        stem         = os.path.join(os.path.abspath(results_dir),
                                    f'seg{sid:04d}')
        summary_path = stem + '_summary.json'
        if not os.path.exists(summary_path):
            print(f"[plots] No results for seg #{sid} — skipping fit/corner.")
            continue

        fit_path    = os.path.join(output_dir, f'seg{sid:04d}_fit.png')
        corner_path = os.path.join(output_dir, f'seg{sid:04d}_corner.png')

        try:
            plot_region_fit(
                t, flux, uncertainty,
                segment_id=sid,
                regions_file=regions_file,
                results_dir=results_dir,
                xlabel=xlabel, ylabel=ylabel,
                save_path=fit_path, show=show,
            )
        except Exception as exc:
            print(f"[plots] Fit plot for seg #{sid} failed: {exc}")

        try:
            plot_corner(
                sid,
                results_dir=results_dir,
                regions_file=regions_file,
                save_path=corner_path, show=show,
            )
        except Exception as exc:
            print(f"[plots] Corner plot for seg #{sid} failed: {exc}")