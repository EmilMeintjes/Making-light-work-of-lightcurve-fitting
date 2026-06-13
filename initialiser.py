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
  - A "Confirm & Next" button that writes the finalised guesses + priors to
    the RegionStore and signals that fitting should proceed

Controls
--------
  Sliders         : drag to change parameter values; model updates live
  Min / Max / Step: text boxes beside each slider to change its range/step
  Curve Fit       : run scipy.optimize.curve_fit from the current slider pos
  Accept CF / Reject CF : keep or discard the curve_fit result
  Undo / Redo     : step back/forward through Curve Fit, Accept/Reject CF
                    and range-edit actions (continuous slider drags are not
                    recorded — drag back to undo those)
  Confirm & Next  : save guesses + priors to JSON and move to the next region
                    (prompts if a curve_fit result hasn't been reviewed yet)
  Skip            : leave this region un-initialised and move on
  q (key)         : close the window, with a confirmation prompt (same as
                    finishing the last region if confirmed)

Any errors (bad regions file, failed save, invalid ranges) are shown as
small popup dialogs via ui_helpers rather than printed to the terminal.
"""

# Sytem imports
#import sys
import warnings

# External imports
import matplotlib
try:
    matplotlib.use('TkAgg')
except Exception as exc:
    print(f"Warning: could not select the 'TkAgg' backend ({exc}); "
          f"falling back to matplotlib's default backend. Some interactive "
          f"features (dropdowns, popup dialogs) may not work correctly.")

import matplotlib.pyplot as plt
import matplotlib.widgets as mwidgets
from matplotlib.lines import Line2D
import numpy as np
from scipy.optimize import curve_fit, OptimizeWarning

#Local imports
from fitting_models import MODELS, evaluate, build_priors, build_priors_from_curvefit
from persistence import RegionStore
from ui_helpers import (
    ToggleButton,
    any_textbox_capturing,
    ask_yes_no,
    ask_yes_no_cancel,
    show_error,
    show_warning,
)


# ---------------------------------------------------------------------------
# Layout constants (all in figure-fraction units unless noted)
# ---------------------------------------------------------------------------

_FIG_W        = 14.0

_AX_LEFT      = 0.07
_AX_RIGHT     = 0.93

# Slider row geometry — these are the fixed physical allocations per row
_SL_LEFT      = 0.07
_SL_WIDTH     = 0.46    # shrunk to leave room for the value readout + columns
_SL_HEIGHT    = 0.030   # figure-fraction height of each slider row
_SL_GAP       = 0.012   # gap between consecutive slider rows
_VAL_GAP      = 0.075   # gap after the slider for its value readout text
_TB_WIDTH     = 0.07    # width of each min/max/step textbox
_TB_GAP       = 0.008
_TB_H         = 0.028

# Button strip at the bottom
_BTN_Y        = 0.02
_BTN_H        = 0.05
_BTN_W        = 0.105   # shrunk from 0.13 to fit Undo/Redo buttons in the row

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
    _TOP_IN    = 0.45    # top margin (room for the plot title)
    _HDR_IN    = 0.85    # gap below plot: x-axis labels + headers + equation
    _GAP_IN    = 0.12    # gap between the slider block and the button strip

    fig_h = (_TOP_IN + _PLOT_IN + _HDR_IN
             + n_params * _ROW_IN + _GAP_IN + _BTN_IN)
    fig   = plt.figure(figsize=(_FIG_W, fig_h))
    fig.patch.set_facecolor(_COL_REGION)

    # Convert fixed inch values to figure-fractions
    btn_frac    = _BTN_IN  / fig_h
    plot_frac   = _PLOT_IN / fig_h
    row_frac    = _ROW_IN  / fig_h
    hdr_frac    = _HDR_IN  / fig_h     # gap holding the column headers + x-axis
    gap_frac    = _GAP_IN  / fig_h

    # Bottom edges in figure-fraction (building upward from 0)
    btn_bottom  = 0.01
    # Slider block sits above the button strip
    sliders_bottom = btn_bottom + btn_frac + gap_frac
    # The plot sits a clear gap (hdr_frac) above the top of the slider block,
    # so its x-axis labels and the min/max/step column headers don't collide.
    plot_bottom = sliders_bottom + n_params * row_frac + hdr_frac
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

    # Column x-positions for the min/max/step "table" to the right of the
    # sliders.  Computed once so the per-row boxes, the column headers and the
    # separator lines all line up.  _VAL_GAP reserves space between the slider
    # and the first column for the slider's own value readout, which would
    # otherwise overlap the Minimum column.
    col_min_x  = _SL_LEFT + 0.12 + _SL_WIDTH + _VAL_GAP
    col_max_x  = col_min_x + _TB_WIDTH + _TB_GAP
    col_step_x = col_max_x + _TB_WIDTH + _TB_GAP

    # Header titles + framing lines for the range/step table, drawn once above
    # the rows.  The per-box inline labels ('min'/'max'/'step') are dropped in
    # favour of these column headers, which gives each number its full box
    # width and stops the three boxes looking squashed together.
    block_top = sliders_bottom + n_params * row_frac
    header_y  = block_top + 0.010
    for cx, label in ((col_min_x,  'Minimum'),
                      (col_max_x,  'Maximum'),
                      (col_step_x, 'Step')):
        fig.text(cx + _TB_WIDTH / 2, header_y, label,
                 ha='center', va='bottom', fontsize=8, fontweight='bold')

    # Vertical separators: left edge of the table, between each column, and
    # the right edge — so the three columns read as a clean table.
    sep_xs = (col_min_x  - _TB_GAP / 2,
              col_max_x  - _TB_GAP / 2,
              col_step_x - _TB_GAP / 2,
              col_step_x + _TB_WIDTH + _TB_GAP / 2)
    for sx in sep_xs:
        fig.add_artist(Line2D([sx, sx], [sliders_bottom, block_top],
                              transform=fig.transFigure,
                              color='grey', lw=0.8, alpha=0.5))
    # Horizontal rule under the headers (top of the table).
    fig.add_artist(Line2D([sep_xs[0], sep_xs[-1]], [block_top, block_top],
                          transform=fig.transFigure,
                          color='grey', lw=0.8, alpha=0.5))

    # Model equation in proper maths notation, far left in the gap between the
    # plot's x-axis and the slider block — a legend for the slider symbols.
    # Anchored just above the slider table and grown upward; the extra _HDR_IN
    # gap keeps even the two-line Crystal Ball clear of the plot's x-axis.
    fig.text(_AX_LEFT, block_top + 0.008, model_entry.get('latex', ''),
             ha='left', va='bottom', fontsize=12)

    # Per-parameter mathtext symbols matching the equation above (fall back to
    # the plain parameter name if a model has no 'symbols' entry).
    symbols = model_entry.get('symbols', param_names)

    for i, (pname, val) in enumerate(zip(param_names, init_vals)):
        lo, hi, step = _default_slider_bounds(pname, val)
        row_bottom   = row_top - row_frac

        # Label: the equation symbol (large mathtext) with the plain parameter
        # name beneath it in small grey, so the slider is unambiguous.
        ax_lbl = fig.add_axes([_SL_LEFT - 0.005, row_bottom,
                                0.12, _SL_HEIGHT])
        ax_lbl.axis('off')
        ax_lbl.text(1.0, 0.66, f'${symbols[i]}$',
                    ha='right', va='center', fontsize=13)
        ax_lbl.text(1.0, 0.16, pname,
                    ha='right', va='center', fontsize=6.5, color='dimgrey')

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

        # Min / Max / Step boxes — no inline label (the column headers above
        # name them); this gives each number the full box width.
        ax_mn = fig.add_axes([col_min_x, row_bottom, _TB_WIDTH, _TB_H])
        tb_mn = mwidgets.TextBox(ax_mn, '', initial=f'{lo:.4g}',
                                 color='white', hovercolor='lightyellow')
        tb_mins.append(tb_mn)

        ax_mx = fig.add_axes([col_max_x, row_bottom, _TB_WIDTH, _TB_H])
        tb_mx = mwidgets.TextBox(ax_mx, '', initial=f'{hi:.4g}',
                                 color='white', hovercolor='lightyellow')
        tb_maxs.append(tb_mx)

        ax_st = fig.add_axes([col_step_x, row_bottom, _TB_WIDTH, _TB_H])
        tb_st = mwidgets.TextBox(ax_st, '', initial=f'{step:.4g}',
                                 color='white', hovercolor='lightyellow')
        tb_steps.append(tb_st)

        row_top = row_bottom

    # ------------------------------------------------------------------
    # Buttons
    # ------------------------------------------------------------------

    btn_y = btn_bottom
    x_btn = _AX_LEFT

    ax_undo  = fig.add_axes([x_btn,              btn_y, _BTN_W, _BTN_H])
    undo_btn = ToggleButton(ax_undo, 'Undo', enabled=False)
    x_btn   += _BTN_W + 0.01

    ax_redo  = fig.add_axes([x_btn,              btn_y, _BTN_W, _BTN_H])
    redo_btn = ToggleButton(ax_redo, 'Redo', enabled=False)
    x_btn   += _BTN_W + 0.02

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
    x_btn   += _BTN_W + 0.025/3.75

    # "Fix y_offset" checkbox — pins y_offset at its slider value during MCMC,
    # removing it from the search to break the amplitude/offset degeneracy.
    # Only shown when the model actually has a y_offset parameter.
    _has_y_offset = 'y_offset' in param_names
    ax_fix = fig.add_axes([x_btn, btn_y, _BTN_W, _BTN_H])
    chk_fix = mwidgets.CheckButtons(ax_fix, ['Fix y_offset?'], [False])
    ax_fix.set_visible(_has_y_offset)
    ax_fix.set_visible(_has_y_offset)
    print("patches:", ax_fix.patches)
    print("lines:", ax_fix.lines)
    print(dir(chk_fix))

    # Scale up the checkbox box and cross lines for legibility.
    # matplotlib >= 3.7 removed .rectangles and .lines from CheckButtons;
    # the patches and lines now live directly on ax_fix.
    for patch in ax_fix.patches:
        patch.set_width(patch.get_width() * 1.4)
        patch.set_height(patch.get_height() * 1.4)
    for line in ax_fix.lines:
        line.set_linewidth(2.5)

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

    # Undo/redo history of discrete actions (Curve Fit / Accept / Reject /
    # range edits).  Continuous slider drags are intentionally NOT recorded
    # here — dragging a slider back is itself the "undo" for that change.
    history             = []
    redo_stack          = []
    _suppress_range_cb  = [False]   # guards against re-entrant range callbacks

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
    # Slider range/value helper (shared by range-edit and undo/redo)
    # ------------------------------------------------------------------

    def _set_slider(idx, lo, hi, step, val):
        """
        Replace slider *idx* with a new one spanning [lo, hi] with the given
        *step*, set to *val* (clamped into range), and sync its min/max/step
        textboxes to match.

        This recreates the underlying `Slider` because matplotlib sliders
        cannot have their `valmin`/`valmax`/`valstep` changed in place. The
        textbox updates are wrapped with `_suppress_range_cb` so that syncing
        them here does not re-trigger the range-edit callback below.
        """

        val = float(np.clip(val, lo, hi))

        ax_old = sl_axes[idx]
        pos    = ax_old.get_position()
        ax_old.remove()
        ax_new = fig.add_axes(pos)

        sl_new = mwidgets.Slider(ax_new, '', lo, hi,
                                 valinit=val, valstep=step,
                                 color='steelblue')
        sl_new.label.set_visible(False)
        sl_new.valtext.set_fontsize(8)
        sl_new.on_changed(_update_model)

        sliders[idx]      = sl_new
        sl_axes[idx]      = ax_new
        current_vals[idx] = val

        _suppress_range_cb[0] = True
        try:
            tb_mins[idx].set_val(f'{lo:.4g}')
            tb_maxs[idx].set_val(f'{hi:.4g}')
            tb_steps[idx].set_val(f'{step:.4g}')
        finally:
            _suppress_range_cb[0] = False

    # ------------------------------------------------------------------
    # Undo / redo
    # ------------------------------------------------------------------

    def _snapshot():
        """Capture enough state to fully restore the window's current look."""
        return {
            'vals':       list(current_vals),
            'ranges':     [(sl.valmin, sl.valmax, sl.valstep) for sl in sliders],
            'cf_popt':    None if cf_popt[0] is None else np.array(cf_popt[0]),
            'cf_pcov':    None if cf_pcov[0] is None else np.array(cf_pcov[0]),
            'cf_visible': bool(btn_acf.ax.get_visible()),
        }

    def _restore(snapshot):
        """Apply a snapshot produced by `_snapshot` back onto the window."""
        for i, (lo, hi, step) in enumerate(snapshot['ranges']):
            _set_slider(i, lo, hi, step, snapshot['vals'][i])

        cf_popt[0] = snapshot['cf_popt']
        cf_pcov[0] = snapshot['cf_pcov']

        if snapshot['cf_visible'] and cf_popt[0] is not None:
            y_cf = evaluate(model_key, t_fine, cf_popt[0])
            cf_line.set_xdata(t_fine)
            cf_line.set_ydata(y_cf)
            btn_acf.ax.set_visible(True)
            btn_rcf.ax.set_visible(True)
        else:
            cf_line.set_xdata([])
            cf_line.set_ydata([])
            btn_acf.ax.set_visible(False)
            btn_rcf.ax.set_visible(False)

        _update_model()

    def _update_undo_redo_buttons():
        """Grey out Undo/Redo whenever their respective stacks are empty."""
        undo_btn.set_enabled(len(history) > 0)
        redo_btn.set_enabled(len(redo_stack) > 0)

    def _push_history():
        """
        Record the current state as a step that can be undone, and clear any
        existing redo history (a new action invalidates old "future" states).
        Call this *before* making the change it should allow undoing.
        """
        history.append(_snapshot())
        redo_stack.clear()
        _update_undo_redo_buttons()

    def _do_undo(_event=None):
        if not history:
            return
        redo_stack.append(_snapshot())
        _restore(history.pop())
        _update_undo_redo_buttons()
        status.set_text('Undid last action.')
        fig.canvas.draw_idle()

    def _do_redo(_event=None):
        if not redo_stack:
            return
        history.append(_snapshot())
        _restore(redo_stack.pop())
        _update_undo_redo_buttons()
        status.set_text('Redid last action.')
        fig.canvas.draw_idle()

    undo_btn.on_clicked(_do_undo)
    redo_btn.on_clicked(_do_redo)

    # ------------------------------------------------------------------
    # Min / Max / Step textbox callbacks (rebuild slider range)
    # ------------------------------------------------------------------

    def _make_range_cb(idx):
        def cb(_text):
            if _suppress_range_cb[0]:
                return
            try:
                lo_new   = float(tb_mins[idx].text)
                hi_new   = float(tb_maxs[idx].text)
                step_new = float(tb_steps[idx].text)
            except ValueError:
                show_warning(
                    fig, 'Invalid range',
                    f'Min, max and step for "{param_names[idx]}" must all '
                    f'be numbers.'
                )
                return
            if lo_new >= hi_new or step_new <= 0:
                show_warning(
                    fig, 'Invalid range',
                    f'For "{param_names[idx]}", min must be less than max, '
                    f'and step must be greater than zero.'
                )
                return
            _push_history()
            _set_slider(idx, lo_new, hi_new, step_new, current_vals[idx])
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
        pre_snapshot = _snapshot()
        p0 = list(current_vals)

        # Constrain curve_fit to the slider ranges shown to the user.
        # Without bounds, models like the Gaussian can drift to wildly
        # degenerate solutions (e.g. amplitude/sigma/y_offset orders of
        # magnitude larger than the data) that fit numerically but are
        # meaningless, and which then poison the MCMC priors.
        lo = [sl.valmin for sl in sliders]
        hi = [sl.valmax for sl in sliders]
        eps = [1e-9 * (h - l) for l, h in zip(lo, hi)]
        p0  = [min(max(v, l + e), h - e) for v, l, h, e in zip(p0, lo, hi, eps)]

        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', OptimizeWarning)
                sigma = u_r if u_r is not None else None
                popt, pcov = curve_fit(
                    func, t_r, f_r,
                    p0=p0,
                    sigma=sigma,
                    absolute_sigma=(sigma is not None),
                    bounds=(lo, hi),
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
            history.append(pre_snapshot)
            redo_stack.clear()
            _update_undo_redo_buttons()
        except (RuntimeError, ValueError) as exc:
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
        _push_history()
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
        _push_history()
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
        # Don't let an unreviewed curve_fit result silently get lost (or
        # silently get used) — make the user decide.
        if cf_popt[0] is not None and btn_acf.ax.get_visible():
            answer = ask_yes_no_cancel(
                fig, 'Unreviewed curve fit',
                'A curve fit result has not been accepted or rejected yet.\n\n'
                'Accept it and use those values?\n\n'
                'Yes = accept the curve fit\n'
                'No = discard it and use the current slider values\n'
                'Cancel = go back without confirming'
            )
            if answer is None:
                return
            elif answer:
                on_accept_cf(None)
            else:
                on_reject_cf(None)

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

        fix_y = _has_y_offset and chk_fix.get_status()[0]

        try:
            store.update_guesses(
                region['segment_id'],
                guesses=guesses,
                priors=priors,
                curvefit_result=cf_result,
                fix_y_offset=fix_y,
            )
        except Exception as exc:
            show_error(fig, 'Could not save region', str(exc))
            return

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
        # Don't let single-key shortcuts fire while the user is typing in a
        # min/max/step textbox (e.g. a stray 'q' while editing a range).
        if any_textbox_capturing(tb_mins + tb_maxs + tb_steps):
            return
        if event.key == 'q':
            if ask_yes_no(fig, 'Quit initialiser',
                          'Quit without saving this region?'):
                result['action'] = 'quit'
                plt.close(fig)

    fig.canvas.mpl_connect('key_press_event', on_key)

    # ------------------------------------------------------------------
    # Initial draw
    # ------------------------------------------------------------------

    _update_undo_redo_buttons()

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
    try:
        store.load()
    except ValueError as exc:
        show_error(
            None, 'Could not load regions file',
            f'Failed to read "{regions_file}":\n\n{exc}\n\n'
            f'Run the selector to create or fix this file before '
            f'initialising regions.'
        )
        return store

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
            # Everything is already initialised — rather than just bail out to
            # the terminal, ask the user whether they want to re-do the
            # initialisation or keep the existing values and move on.
            n = len(store)
            redo = ask_yes_no(
                None, 'Already initialised',
                f"All {n} region(s) already have initial guesses.\n\n"
                "Re-initialise them now?\n\n"
                "  • Yes  — re-open every region to adjust its guesses.\n"
                "  • No   — keep the existing values and continue.",
            )
            if not redo:
                print("Keeping existing initial guesses for all regions.")
                return store
            print("Re-initialising all regions at user request.")
            todo = list(store)

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
            msg = f"Region #{sid} contains no data points — skipping."
            print(f"  WARNING: {msg}")
            show_warning(None, 'Empty region', msg)
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