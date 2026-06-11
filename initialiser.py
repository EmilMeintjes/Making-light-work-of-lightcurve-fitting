"""
initialiser.py
--------------
Stage 2: interactive parameter initialisation window.

For each region produced by the selector the user is presented with:
  - A plot of the data within the selected region
  - One slider per model parameter (with editable min / max / step)
  - A live model curve that updates as sliders move
  - An optional curve_fit pass (button) that refines the slider values
  - Accept / Reject buttons to decide whether to keep the curve_fit result
  - A "Run MCMC" button that writes the finalised guesses + priors to the
    RegionStore and signals that fitting should proceed

Controls
--------
  Sliders         : drag to change parameter values; model updates live
  Min / Max / Step: text boxes beside each slider to change its range/step
  Curve Fit       : run scipy.optimize.curve_fit from the current slider pos
  Accept CF / Reject CF : keep or discard the curve_fit result
  Confirm & Next  : save guesses + priors to JSON and move to the next region
  Skip            : leave this region un-initialised and move on
  q (key)         : close the window (same as finishing the last region)
"""

# Sytem imports
#import sys
import warnings

# External imports
import matplotlib.pyplot as plt
import matplotlib.widgets as mwidgets
import numpy as np
from scipy.optimize import curve_fit, OptimizeWarning

#Local imports
from fitting_models import MODELS, evaluate, build_priors, build_priors_from_curvefit
from persistence import RegionStore


# ---------------------------------------------------------------------------
# Layout constants  (all in figure-fraction units unless noted)
# ---------------------------------------------------------------------------

_FIG_W        = 14.0

_AX_LEFT      = 0.07
_AX_RIGHT     = 0.93

# Slider row geometry — these are the fixed physical allocations per row
_SL_LEFT      = 0.07
_SL_WIDTH     = 0.52
_SL_HEIGHT    = 0.030   # figure-fraction height of each slider row
_SL_GAP       = 0.012   # gap between consecutive slider rows
_TB_WIDTH     = 0.07    # width of each min/max/step textbox
_TB_GAP       = 0.008
_TB_H         = 0.028

# Button strip at the bottom
_BTN_Y        = 0.02
_BTN_H        = 0.05
_BTN_W        = 0.13

# Fixed figure-fraction allocations (independent of n_params)
_PLOT_TOP_PAD = 0.05    # space above the plot axes
_PLOT_BOT_PAD = 0.04    # gap between plot bottom and first slider
_SLIDER_BOT_PAD = 0.02  # gap between last slider and button strip
_PLOT_MIN_H   = 0.28    # minimum figure-fraction height for the plot axes


# ---------------------------------------------------------------------------
# Colour scheme
# ---------------------------------------------------------------------------

_COL_DATA     = 'steelblue'
_COL_MODEL    = 'tomato'
_COL_CF       = 'mediumseagreen'
_COL_REGION   = 'lightyellow'


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_region(t, flux, unc, start, end):

    """
    Return the data arrays clipped to [start, end].
    """

    mask = (t >= start) & (t <= end)
    t_r  = t[mask]
    f_r  = flux[mask]
    u_r  = unc[mask] if unc is not None else None
    return t_r, f_r, u_r


def _default_slider_bounds(param_name, guess):

    """
    Return sensible (min, max, step) defaults for a slider given a guess.

    Rules:
      - Range is 5x the abs(guess) on each side, floored at 1.0 so that a
        zero guess still gives a usable range.
      - Step is 1 % of the range, floored at 1e-4.
    """

    half  = max(5.0 * abs(guess), 1.0)
    lo    = guess - half
    hi    = guess + half
    step  = max((hi - lo) * 0.01, 1e-4)
    return lo, hi, step


# ---------------------------------------------------------------------------
# Single-region initialiser window
# ---------------------------------------------------------------------------

def _run_one(t_r, f_r, u_r, region, store, xlabel, ylabel, xscale='linear', yscale='linear'):

    """
    Open the initialisation window for *region* and block until the user
    clicks Confirm or Skip.

    Returns
    -------
    'confirmed', 'skipped', or 'quit'
    """

    model_key   = region['model']
    model_entry = MODELS[model_key]
    func        = model_entry['func']
    param_names = model_entry['params']
    n_params    = len(param_names)

    # --- Recover any previously saved guesses (e.g. re-opening a region) ---
    saved_guesses = region.get('initial_guesses', {})
    defaults      = model_entry['defaults']
    init_vals     = [
        saved_guesses.get(p, defaults[i])
        for i, p in enumerate(param_names)
    ]

    # --- Compute figure height from the bottom up so sliders never overlap ---
    # Fixed space consumed regardless of n_params (all in inches): button strip + padding, then one row per param, then plot + padding.
    # We work in inches so the figure height is physically sensible, then
    # convert everything to figure-fraction for add_axes().

    _ROW_IN    = 0.42    # inches per slider row (height + gap)
    _BTN_IN    = 0.55    # inches for the button strip
    _PLOT_IN   = 3.20    # inches for the data/model plot
    _PAD_IN    = 0.55    # top + inter-section padding inches

    fig_h = _PAD_IN + _PLOT_IN + n_params * _ROW_IN + _BTN_IN
    fig   = plt.figure(figsize=(_FIG_W, fig_h))
    fig.patch.set_facecolor(_COL_REGION)

    # Convert fixed inch values to figure-fractions
    btn_frac    = _BTN_IN   / fig_h
    plot_frac   = _PLOT_IN  / fig_h
    row_frac    = _ROW_IN   / fig_h
    pad_frac    = 0.03                  # small gap between sections

    # Bottom edges in figure-fraction (building upward from 0)
    btn_bottom  = 0.01
    # Slider block sits above the button strip
    sliders_bottom = btn_bottom + btn_frac + pad_frac
    # Plot sits above the slider block
    plot_bottom = sliders_bottom + n_params * row_frac + pad_frac
    plot_height = plot_frac

    # ------------------------------------------------------------------
    # Data / model axes
    # ------------------------------------------------------------------

    ax_plot = fig.add_axes([_AX_LEFT, plot_bottom,
                            _AX_RIGHT - _AX_LEFT, plot_height])

    if u_r is not None:
        ax_plot.errorbar(t_r, f_r, yerr=u_r,
                         fmt='o', ms=4, alpha=0.7,
                         color=_COL_DATA, ecolor='lightgrey', elinewidth=0.8,
                         label='Data', zorder=2)
    else:
        ax_plot.plot(t_r, f_r, 'o', ms=4, alpha=0.7,
                     color=_COL_DATA, label='Data', zorder=2)

    t_fine     = np.linspace(t_r.min(), t_r.max(), 500)
    model_line, = ax_plot.plot(t_fine,
                               evaluate(model_key, t_fine, init_vals),
                               color=_COL_MODEL, lw=2.0,
                               label='Current guess', zorder=3)
    cf_line,    = ax_plot.plot([], [], color=_COL_CF, lw=1.5,
                               ls='--', label='Curve fit', zorder=4)

    ax_plot.set_xlabel(xlabel)
    ax_plot.set_ylabel(ylabel)
    ax_plot.set_xscale(xscale)
    ax_plot.set_yscale(yscale)
    ax_plot.set_ylim(np.min(f_r) - 0.1*np.min(f_r), np.max(f_r) + 0.1*np.max(f_r))

    ax_plot.set_title(
        f"Seg #{region['segment_id']}  |  model: {model_entry['label']}  |  "
        f"note: \"{region.get('note', '')}\"",
        fontsize=9,
    )
    ax_plot.legend(fontsize=8, loc='upper right')

    # ------------------------------------------------------------------
    # Slider rows  (one per parameter, stacked below the plot)
    # ------------------------------------------------------------------
    # Slider rows are placed bottom-up starting from sliders_bottom.
    # row_top tracks the top edge of the next row to place.
    row_top = sliders_bottom + n_params * row_frac

    sliders    = []   # matplotlib Slider objects
    tb_mins    = []   # TextBox for min
    tb_maxs    = []   # TextBox for max
    tb_steps   = []   # TextBox for step
    sl_axes    = []   # the slider Axes (needed for reconnecting callbacks)

    current_vals = list(init_vals)            # mutable, shared via closure

    for i, (pname, val) in enumerate(zip(param_names, init_vals)):
        lo, hi, step = _default_slider_bounds(pname, val)
        row_bottom   = row_top - row_frac

        # Label
        ax_lbl = fig.add_axes([_SL_LEFT - 0.005, row_bottom,
                                0.12, _SL_HEIGHT])
        ax_lbl.axis('off')
        ax_lbl.text(1.0, 0.5, pname, ha='right', va='center', fontsize=8)

        # Slider
        ax_sl = fig.add_axes([_SL_LEFT + 0.12, row_bottom,
                               _SL_WIDTH, _SL_HEIGHT])
        sl = mwidgets.Slider(ax_sl, '', lo, hi,
                             valinit=val, valstep=step,
                             color='steelblue')
        sl.label.set_visible(False)
        sl.valtext.set_fontsize(8)
        sl_axes.append(ax_sl)
        sliders.append(sl)

        # Min textbox
        x_tb = _SL_LEFT + 0.12 + _SL_WIDTH + _TB_GAP
        ax_mn = fig.add_axes([x_tb, row_bottom, _TB_WIDTH, _TB_H])
        tb_mn = mwidgets.TextBox(ax_mn, 'min ', initial=f'{lo:.4g}',
                                 color='white', hovercolor='lightyellow')
        tb_mn.label.set_fontsize(7)
        tb_mins.append(tb_mn)

        # Max textbox
        x_tb += _TB_WIDTH + _TB_GAP
        ax_mx = fig.add_axes([x_tb, row_bottom, _TB_WIDTH, _TB_H])
        tb_mx = mwidgets.TextBox(ax_mx, 'max ', initial=f'{hi:.4g}',
                                 color='white', hovercolor='lightyellow')
        tb_mx.label.set_fontsize(7)
        tb_maxs.append(tb_mx)

        # Step textbox
        x_tb += _TB_WIDTH + _TB_GAP
        ax_st = fig.add_axes([x_tb, row_bottom, _TB_WIDTH, _TB_H])
        tb_st = mwidgets.TextBox(ax_st, 'step', initial=f'{step:.4g}',
                                 color='white', hovercolor='lightyellow')
        tb_st.label.set_fontsize(7)
        tb_steps.append(tb_st)

        row_top = row_bottom

    # ------------------------------------------------------------------
    # Buttons
    # ------------------------------------------------------------------

    btn_y = btn_bottom
    x_btn = _AX_LEFT

    ax_cf   = fig.add_axes([x_btn,              btn_y, _BTN_W, _BTN_H])
    btn_cf  = mwidgets.Button(ax_cf, 'Curve Fit',
                              color='lightyellow', hovercolor='khaki')
    x_btn  += _BTN_W + 0.01

    ax_acf  = fig.add_axes([x_btn,              btn_y, _BTN_W, _BTN_H])
    btn_acf = mwidgets.Button(ax_acf, 'Accept CF',
                              color='lightgreen', hovercolor='mediumseagreen')
    btn_acf.ax.set_visible(False)   # hidden until a CF result exists
    x_btn  += _BTN_W + 0.01

    ax_rcf  = fig.add_axes([x_btn,              btn_y, _BTN_W, _BTN_H])
    btn_rcf = mwidgets.Button(ax_rcf, 'Reject CF',
                              color='lightsalmon', hovercolor='tomato')
    btn_rcf.ax.set_visible(False)
    x_btn  += _BTN_W + 0.02

    ax_ok   = fig.add_axes([x_btn,              btn_y, _BTN_W, _BTN_H])
    btn_ok  = mwidgets.Button(ax_ok, 'Confirm & Next',
                              color='lightgreen', hovercolor='mediumseagreen')
    x_btn  += _BTN_W + 0.01

    ax_skip = fig.add_axes([x_btn,              btn_y, _BTN_W, _BTN_H])
    btn_skip = mwidgets.Button(ax_skip, 'Skip',
                               color='lightgrey', hovercolor='silver')

    # Status text in plot
    status = ax_plot.text(
        0.01, 0.03, 'Adjust sliders, then Confirm or run Curve Fit.',
        transform=ax_plot.transAxes, va='bottom', fontsize=8,
        bbox=dict(boxstyle='round', fc='wheat', alpha=0.85), zorder=10,
    )

    # ------------------------------------------------------------------
    # Shared mutable state
    # ------------------------------------------------------------------

    result     = {'action': None}   # filled on button press
    cf_popt    = [None]             # best-fit params from curve_fit
    cf_pcov    = [None]

    # ------------------------------------------------------------------
    # Live model update
    # ------------------------------------------------------------------

    def _update_model(_val=None):
        for i, sl in enumerate(sliders):
            current_vals[i] = sl.val
        try:
            y_model = evaluate(model_key, t_fine, current_vals)
            model_line.set_ydata(y_model)
        except Exception:
            pass
        fig.canvas.draw_idle()

    for sl in sliders:
        sl.on_changed(_update_model)

    # ------------------------------------------------------------------
    # Min / Max / Step textbox callbacks (rebuild slider range)
    # ------------------------------------------------------------------

    def _make_range_cb(idx):
        def cb(_text):
            try:
                lo_new   = float(tb_mins[idx].text)
                hi_new   = float(tb_maxs[idx].text)
                step_new = float(tb_steps[idx].text)
            except ValueError:
                return
            if lo_new >= hi_new or step_new <= 0:
                return
            # Clamp current value into the new range
            val_now = np.clip(current_vals[idx], lo_new, hi_new)
            ax_old  = sl_axes[idx]
            # Remove the old slider axes and recreate in the same position
            pos  = ax_old.get_position()
            ax_old.remove()
            ax_new = fig.add_axes(pos)
            sl_new = mwidgets.Slider(ax_new, '', lo_new, hi_new,
                                     valinit=val_now, valstep=step_new,
                                     color='steelblue')
            sl_new.label.set_visible(False)
            sl_new.valtext.set_fontsize(8)
            sl_new.on_changed(_update_model)
            sliders[idx]  = sl_new
            sl_axes[idx]  = ax_new
            current_vals[idx] = val_now
            _update_model()
        return cb

    for idx in range(n_params):
        cb = _make_range_cb(idx)
        tb_mins[idx].on_submit(cb)
        tb_maxs[idx].on_submit(cb)
        tb_steps[idx].on_submit(cb)

    # ------------------------------------------------------------------
    # Curve Fit button
    # ------------------------------------------------------------------

    def on_curve_fit(_event):
        p0 = list(current_vals)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', OptimizeWarning)
                sigma = u_r if u_r is not None else None
                popt, pcov = curve_fit(
                    func, t_r, f_r,
                    p0=p0,
                    sigma=sigma,
                    absolute_sigma=(sigma is not None),
                    maxfev=10_000,
                )
            cf_popt[0] = popt
            cf_pcov[0] = pcov
            y_cf = evaluate(model_key, t_fine, popt)
            cf_line.set_xdata(t_fine)
            cf_line.set_ydata(y_cf)
            btn_acf.ax.set_visible(True)
            btn_rcf.ax.set_visible(True)
            status.set_text(
                'Curve fit succeeded.  Accept to use these values, '
                'or Reject to keep sliders.'
            )
        except RuntimeError as exc:
            status.set_text(f'Curve fit failed: {exc}')
            cf_popt[0] = None
        fig.canvas.draw_idle()

    btn_cf.on_clicked(on_curve_fit)

    # ------------------------------------------------------------------
    # Accept / Reject curve_fit
    # ------------------------------------------------------------------

    def on_accept_cf(_event):
        if cf_popt[0] is None:
            return
        for i, sl in enumerate(sliders):
            # Move each slider to the fitted value, clamping to its range
            lo_sl = sl.valmin
            hi_sl = sl.valmax
            sl.set_val(np.clip(cf_popt[0][i], lo_sl, hi_sl))
        model_line.set_ydata(evaluate(model_key, t_fine, cf_popt[0]))
        btn_acf.ax.set_visible(False)
        btn_rcf.ax.set_visible(False)
        status.set_text('Curve fit accepted.  Click Confirm & Next when ready.')
        fig.canvas.draw_idle()

    def on_reject_cf(_event):
        cf_line.set_xdata([])
        cf_line.set_ydata([])
        cf_popt[0] = None
        cf_pcov[0] = None
        btn_acf.ax.set_visible(False)
        btn_rcf.ax.set_visible(False)
        status.set_text('Curve fit rejected.  Use sliders to set initial guess.')
        fig.canvas.draw_idle()

    btn_acf.on_clicked(on_accept_cf)
    btn_rcf.on_clicked(on_reject_cf)

    # ------------------------------------------------------------------
    # Confirm & Next
    # ------------------------------------------------------------------

    def on_confirm(_event):
        guesses = {p: float(current_vals[i])
                   for i, p in enumerate(param_names)}

        # Build priors: from CF covariance if available, else from guesses
        if cf_popt[0] is not None and cf_pcov[0] is not None:
            priors = build_priors_from_curvefit(
                param_names, cf_popt[0], cf_pcov[0]
            )
            cf_result = {
                'popt':    cf_popt[0].tolist(),
                'pcov':    cf_pcov[0].tolist(),
                'success': True,
            }
        else:
            priors    = build_priors(param_names, list(guesses.values()))
            cf_result = None

        store.update_guesses(
            region['segment_id'],
            guesses=guesses,
            priors=priors,
            curvefit_result=cf_result,
        )
        result['action'] = 'confirmed'
        plt.close(fig)

    btn_ok.on_clicked(on_confirm)

    # ------------------------------------------------------------------
    # Skip
    # ------------------------------------------------------------------

    def on_skip(_event):
        result['action'] = 'skipped'
        plt.close(fig)

    btn_skip.on_clicked(on_skip)

    # ------------------------------------------------------------------
    # Key handler
    # ------------------------------------------------------------------

    def on_key(event):
        if event.key == 'q':
            result['action'] = 'quit'
            plt.close(fig)

    fig.canvas.mpl_connect('key_press_event', on_key)

    # ------------------------------------------------------------------
    # Initial draw
    # ------------------------------------------------------------------

    plt.tight_layout(rect=[0, _BTN_Y + _BTN_H + 0.01, 1, 1])
    plt.show()

    return result['action'] or 'skipped'


# ---------------------------------------------------------------------------
# Public entry point: iterate over all un-initialised regions
# ---------------------------------------------------------------------------

def run_initialiser(t, flux, uncertainty=None,
                    regions_file='regions.json',
                    xlabel='Time', ylabel='Flux',
                    segment_ids=None):

    """
    Open an initialisation window for each region in *regions_file* that has
    not yet been given initial guesses, or for the IDs listed in *segment_ids*.

    Time is displayed relative to region['start'] so the user works with
    small, intuitive numbers rather than raw MJD values.  The shift is
    purely for display and slider initialisation; stored guesses are in
    shifted coordinates, and the fitter receives the same shifted arrays.

    Parameters
    ----------
    t, flux       : array-like   Full lightcurve (not pre-clipped).
    uncertainty   : array-like or None
    regions_file  : str          JSON file written by the selector.
    xlabel, ylabel: str          Axis labels (should match the selector call).
                                 The x label will have ' (relative)' appended.
    segment_ids   : list of int or None
    """

    t    = np.asarray(t,    dtype=float)
    flux = np.asarray(flux, dtype=float)
    if uncertainty is not None:
        uncertainty = np.asarray(uncertainty, dtype=float)

    store = RegionStore(regions_file)
    store.load()
    xscale = store.xscale
    yscale = store.yscale

    if len(store) == 0:
        print("No regions found in the region file.  Run the selector first.")
        return store

    # Determine which segments to process
    if segment_ids is not None:
        todo = [store.get(sid) for sid in segment_ids]
    else:
        # All regions that have not been initialised yet
        todo = [r for r in store
                if not r.get('initial_guesses')]
        if not todo:
            print("All regions already have initial guesses.  "
                  "Pass segment_ids=[...] to re-initialise specific ones.")
            return store

    print(f"\nInitialising {len(todo)} region(s): "
          f"{[r['segment_id'] for r in todo]}")

    for i, region in enumerate(todo):
        sid    = region['segment_id']
        t_ref  = region['start']   # shift origin to region start
        print(f"\n--- Region #{sid}  ({i+1}/{len(todo)}) "
              f"[{region['model']}]  "
              f"{region['start']:.6g} -> {region['end']:.6g}  "
              f"(t_ref = {t_ref:.6g}) ---")

        t_r, f_r, u_r = _extract_region(
            t, flux, uncertainty, region['start'], region['end']
        )

        if len(t_r) == 0:
            print(f"  WARNING: no data points inside region #{sid} — skipping.")
            continue

        # Shift time to be relative to the region start
        t_r_shifted = t_r - t_ref
        xlabel_rel  = xlabel + ' (relative)'

        action = _run_one(t_r_shifted, f_r, u_r, region, store,
                          xlabel_rel, ylabel, xscale=xscale, yscale=yscale)
        print(f"  -> {action}")

        if action == 'quit':
            print("Initialiser closed early.")
            break

    return store


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import json, tempfile, os
    from persistence import save_regions_json

    rng       = np.random.default_rng(0)
    t_test    = np.linspace(0, 200, 600)
    flux_test = (
        0.3
        + 4.0 * np.exp(-0.5 * ((t_test - 60)  / 7) ** 2)
        + 2.5 * np.exp(-0.5 * ((t_test - 150) / 5) ** 2)
        + 0.12 * rng.standard_normal(600)
    )
    unc_test  = np.abs(0.10 + 0.02 * rng.standard_normal(600))

    tmpdir = tempfile.mkdtemp()
    jpath  = os.path.join(tmpdir, 'regions.json')

    save_regions_json(jpath, [
        {'segment_id': 1, 'start': 40.0,  'end': 85.0,
         'model': 'gaussian',   'note': 'main peak',
         'initial_guesses': {}, 'curvefit_result': None, 'priors': {}},
        {'segment_id': 2, 'start': 130.0, 'end': 170.0,
         'model': 'crystal_ball', 'note': 'secondary peak',
         'initial_guesses': {}, 'curvefit_result': None, 'priors': {}},
    ])

    store = run_initialiser(
        t_test, flux_test, uncertainty=unc_test,
        regions_file=jpath,
        xlabel='Days since discovery',
        ylabel='Flux density (arbitrary)',
    )

    print("\nFinal store state:")
    for r in store:
        print(f"  #{r['segment_id']}  guesses: {r['initial_guesses']}")