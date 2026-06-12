"""
selector.py
-----------
Stage 1: Interactive region selector for lightcurve fitting.

Controls
--------
Left-click (x2)  : set region START then END (order sorted automatically).
Enter / button   : confirm the pending region and write to JSON.
Undo Last button : undo the most recently confirmed region.
Delete # button  : delete the region whose ID is typed in the adjacent box.
Escape           : cancel the current in-progress selection.
q                : save all regions and close (keyboard, only outside text boxes).
Model dropdown   : click the model button to open a selection panel; click a
                   model name to select it.
Zoom / Pan tools : toolbar modes work normally; selector ignores those clicks.

Notes
-----
Keyboard shortcuts u / d are intentionally removed.  All region management is
done through GUI buttons to avoid conflicts with the notes text-box input.
"""

import matplotlib.pyplot as plt
import matplotlib.widgets as mwidgets
import numpy as np

from fitting_models import MODEL_KEYS, MODEL_LABELS
from persistence import RegionStore

# ---------------------------------------------------------------------------
# Colours
# ---------------------------------------------------------------------------

_MODEL_COLOURS = {
    'gaussian':     'cornflowerblue',
    'rising_exp':   'mediumseagreen',
    'decaying_exp': 'tomato',
    'crystal_ball': 'mediumpurple',
}
_PENDING_COLOUR = 'orange'

# ---------------------------------------------------------------------------
# Layout constants (inches, bottom-up)
# ---------------------------------------------------------------------------
_FIG_W        = 16.0
_PLOT_H       = 4.8   # main axes height (inches)
_CTRL_H       = 1.80  # control strip height
_FIG_H        = _PLOT_H + _CTRL_H
_CTRL_BOTTOM  = 0.02  # figure fraction
_CTRL_TOP     = _CTRL_H / _FIG_H
_PLOT_BOTTOM  = _CTRL_TOP + 0.01
_PLOT_TOP     = 0.97


# ---------------------------------------------------------------------------
# Main selector
# ---------------------------------------------------------------------------

def run_selector(t, flux, uncertainty=None,
                 regions_file='regions.json',
                 xlabel='Time', ylabel='Flux',
                 xscale='linear', yscale='linear',
                 title=None):
    """
    Display the lightcurve and let the user interactively define fit regions.

    Parameters
    ----------
    t, flux      : array-like
    uncertainty  : array-like or None
    regions_file : str    Path to the JSON file (created if absent).
    xlabel       : str    Time-axis label.
    ylabel       : str    Flux-axis label.
    xscale       : str    'linear' or 'log'.  Stored in the JSON so downstream
                          modules pick it up automatically.
    yscale       : str    'linear' or 'log'.  Same.
    title        : str or None

    Notes on scale
    --------------
    If the JSON file already exists and contains scale metadata, the stored
    values take precedence over the xscale/yscale arguments here.  To change
    the scale of an existing session, edit the JSON metadata block directly.
    """
    t    = np.asarray(t,    dtype=float)
    flux = np.asarray(flux, dtype=float)
    if uncertainty is not None:
        uncertainty = np.asarray(uncertainty, dtype=float)

    store   = RegionStore(regions_file, xscale=xscale, yscale=yscale)
    store.load()
    regions = store.regions   # live reference

    # ------------------------------------------------------------------
    # Shared mutable state
    # ------------------------------------------------------------------
    state = {
        'clicks':          [],
        'vlines':          [],
        'patch':           None,
        'pending':         None,
        'dropdown_open':   False,
        'dropdown_axes':   [],   # axes created for the open dropdown panel
        'dropdown_buttons': [],
        'active_textbox':  None, # tracks which TextBox (if any) has focus
    }

    # ------------------------------------------------------------------
    # Figure layout
    # ------------------------------------------------------------------
    fig = plt.figure(figsize=(_FIG_W, _FIG_H))

    # Main plot axes
    ax = fig.add_axes([0.06, _PLOT_BOTTOM, 0.90, _PLOT_TOP - _PLOT_BOTTOM])

    # Helper to convert inches -> figure fractions
    def _fy(y_inch):
        return y_inch / _FIG_H

    def _fx(x_inch):
        return x_inch / _FIG_W

    # --- Control strip layout (all y in figure fractions) ---
    #
    # Row 1 (bottom):  [Undo Last]  [Delete ID box]  [Delete Btn]
    # Row 2 (top):     [Model dropdown trigger]  [Notes label+box]  [Confirm btn]
    #
    row1_y   = _fy(0.0)
    row1_h   = _fy(0.40)
    row2_y   = _fy(0.54)
    row2_h   = _fy(0.40)
    btn_w    = _fx(1.30)   # standard button width
    wide_w   = _fx(1.60)   # wider buttons
    gap      = _fx(0.10)
    left     = _fx(0.60)

    # Row 1 — Undo, Delete ID textbox, Delete button
    ax_undo  = fig.add_axes([left,           row1_y, wide_w,       row1_h])
    ax_delid = fig.add_axes([left + wide_w + gap,
                              row1_y,
                              _fx(0.55), row1_h])
    ax_delbtn = fig.add_axes([left + wide_w + gap + _fx(0.55) + gap,
                               row1_y,
                               btn_w, row1_h])

    btn_undo   = mwidgets.Button(ax_undo,   'Undo Last',
                                 color='lightyellow',    hovercolor='khaki')
    delid_box  = mwidgets.TextBox(ax_delid,  '', initial='')
    btn_delete = mwidgets.Button(ax_delbtn, 'Delete #',
                                 color='lightsalmon',    hovercolor='tomato')

    # Small label above delete ID box
    ax_delid.set_title('ID', fontsize=7, pad=1)

    # Row 2 — model dropdown trigger, notes, confirm
    model_btn_x = left
    model_btn_w = _fx(1.50)
    ax_modelbtn = fig.add_axes([model_btn_x, row2_y, model_btn_w, row2_h])
    btn_model   = mwidgets.Button(ax_modelbtn,
                                  f'▾ {MODEL_LABELS[0]}',
                                  color='lightsteelblue', hovercolor='steelblue')
    btn_model.label.set_fontsize(8)

    notes_x = model_btn_x + model_btn_w + _fx(0.20)
    notes_w = _fx(6.50)
    ax_notes_lbl = fig.add_axes([notes_x, row2_y + _fy(0.25),
                                  notes_w, _fy(0.15)])
    ax_notes_lbl.axis('off')
    ax_notes_lbl.text(0.0, 0.5, 'Notes for this region:',
                      va='center', fontsize=8)
    ax_notes  = fig.add_axes([notes_x, row2_y, notes_w, row2_h])
    notes_box = mwidgets.TextBox(ax_notes, '', initial='')

    confirm_x = notes_x + notes_w + _fx(0.15)
    confirm_w = _fx(1.20)
    ax_confirm = fig.add_axes([confirm_x, row2_y, confirm_w, row2_h])
    btn_confirm = mwidgets.Button(ax_confirm, 'Confirm\n(Enter)',
                                  color='lightgreen', hovercolor='mediumseagreen')

    # Status text (lives in the main axes)
    status_txt = ax.text(
        0.01, 0.97, 'Left-click: set START',
        transform=ax.transAxes, va='top', fontsize=9,
        bbox=dict(boxstyle='round', fc='wheat', alpha=0.85), zorder=10,
    )

    # Track selected model key (mutable list so closures can write to it)
    selected_model = [MODEL_KEYS[0]]

    # ------------------------------------------------------------------
    # TextBox focus tracking
    # ------------------------------------------------------------------
    # When a TextBox has keyboard focus, suppress all navigation shortcuts
    # so that typing normal characters doesn't trigger undo / delete / quit.

    def _on_notes_focus(_event):
        state['active_textbox'] = 'notes'

    def _on_notes_unfocus(_event):
        if state['active_textbox'] == 'notes':
            state['active_textbox'] = None

    def _on_delid_focus(_event):
        state['active_textbox'] = 'delid'

    def _on_delid_unfocus(_event):
        if state['active_textbox'] == 'delid':
            state['active_textbox'] = None

    notes_box.on_submit(lambda _: None)   # keep box alive
    # matplotlib TextBox fires 'begin_typing' / 'stop_typing' on focus
    notes_box.on_text_change(lambda _: None)

    fig.canvas.mpl_connect('axes_enter_event', lambda e:
        _on_notes_focus(e)   if e.inaxes is ax_notes  else
        _on_delid_focus(e)   if e.inaxes is ax_delid  else
        None)
    fig.canvas.mpl_connect('axes_leave_event', lambda e:
        _on_notes_unfocus(e) if e.inaxes is ax_notes  else
        _on_delid_unfocus(e) if e.inaxes is ax_delid  else
        None)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _apply_scales():
        ax.set_xscale(store.xscale)
        ax.set_yscale(store.yscale)

    def _draw_data():
        if uncertainty is not None:
            ax.errorbar(t, flux, yerr=uncertainty,
                        fmt='o', ms=3, alpha=0.6,
                        color='steelblue', ecolor='lightgrey', elinewidth=0.8,
                        zorder=2)
        else:
            ax.plot(t, flux, 'o', ms=3, alpha=0.6, color='steelblue', zorder=2)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(
            title or (
                'Radio lightcurve  |  '
                'Click: START → END  |  Enter/button: confirm  |  '
                'Esc: cancel  |  q: save and quit'
            ),
            fontsize=9,
        )
        _apply_scales()

    def _shade_confirmed():
        for r in regions:
            colour = _MODEL_COLOURS.get(r['model'], 'grey')
            ax.axvspan(r['start'], r['end'], alpha=0.20, color=colour, zorder=1)
            mid = 0.5 * (r['start'] + r['end'])
            ax.text(mid, ax.get_ylim()[1], f"#{r['segment_id']}",
                    ha='center', va='top', fontsize=7, color='dimgrey', zorder=5)

    def _full_redraw():
        nonlocal status_txt
        ax.cla()
        _draw_data()
        _shade_confirmed()
        status_txt = ax.text(
            0.01, 0.97,
            f'{len(regions)} region(s) saved  |  Click to set START',
            transform=ax.transAxes, va='top', fontsize=9,
            bbox=dict(boxstyle='round', fc='wheat', alpha=0.85), zorder=10,
        )
        fig.canvas.draw_idle()

    def _clear_in_progress():
        for vl in state['vlines']:
            try:
                vl.remove()
            except Exception:
                pass
        state['vlines'] = []
        if state['patch'] is not None:
            try:
                state['patch'].remove()
            except Exception:
                pass
            state['patch'] = None
        state['clicks']  = []
        state['pending'] = None
        fig.canvas.draw_idle()

    def _update_status(msg):
        status_txt.set_text(msg)
        fig.canvas.draw_idle()

    # ------------------------------------------------------------------
    # Model dropdown
    # ------------------------------------------------------------------

    def _close_dropdown():
        for btn_ax in state['dropdown_axes']:
            btn_ax.set_visible(False)
            try:
                fig.delaxes(btn_ax)
            except Exception:
                pass
        state['dropdown_axes']    = []
        state['dropdown_buttons'] = []
        state['dropdown_open']    = False
        fig.canvas.draw_idle()

    def _open_dropdown(_event=None):
        if state['dropdown_open']:
            _close_dropdown()
            return

        state['dropdown_open'] = True
        # Build a column of buttons above the trigger button, in figure fractions
        # Each button: same x and width as btn_model trigger, stacked upward
        bx = model_btn_x
        bw = model_btn_w
        bh = row2_h + _fy(0.04)  # slightly taller items

        for i, (key, label) in enumerate(zip(MODEL_KEYS, MODEL_LABELS)):
            by = row2_y + row2_h + _fy(0.04) + i * (bh + _fy(0.02))
            bax = fig.add_axes([bx, by, bw, bh])
            colour = (
                'lightsteelblue'
                if key == selected_model[0]
                else 'white'
            )
            btn = mwidgets.Button(bax, label, color=colour,
                                  hovercolor='lightsteelblue')
            btn.label.set_fontsize(8)

            # Closure over key / label
            def _make_callback(k, lbl):
                def _cb(_ev):
                    selected_model[0] = k
                    btn_model.label.set_text(f'▾ {lbl}')
                    _close_dropdown()
                    _update_status(f'Model set to [{k}]  |  Click to set START')
                return _cb

            btn.on_clicked(_make_callback(key, label))
            state['dropdown_axes'].append(bax)
            state['dropdown_buttons'].append(btn)

        fig.canvas.draw_idle()

    btn_model.on_clicked(_open_dropdown)

    # ------------------------------------------------------------------
    # Confirm region
    # ------------------------------------------------------------------

    def _confirm_region(_event=None):
        _close_dropdown()
        if state['pending'] is None:
            _update_status('Nothing to confirm — click START then END first.')
            return

        model_key = selected_model[0]
        note      = notes_box.text.strip()
        seg_id    = store.next_id()

        region = {
            'segment_id':      seg_id,
            'start':           state['pending']['start'],
            'end':             state['pending']['end'],
            'model':           model_key,
            'note':            note,
            'initial_guesses': {},
            'curvefit_result': None,
            'priors':          {},
        }
        store.add(region)

        colour = _MODEL_COLOURS.get(model_key, 'grey')
        if state['patch'] is not None:
            try:
                state['patch'].remove()
            except Exception:
                pass
            state['patch'] = None
        ax.axvspan(region['start'], region['end'], alpha=0.20, color=colour, zorder=1)
        mid = 0.5 * (region['start'] + region['end'])
        ax.text(mid, ax.get_ylim()[1], f"#{seg_id}",
                ha='center', va='top', fontsize=7, color='dimgrey', zorder=5)

        for vl in state['vlines']:
            try:
                vl.remove()
            except Exception:
                pass
        state['vlines'] = []
        state['clicks']  = []
        state['pending'] = None
        notes_box.set_val('')

        _update_status(
            f'Saved #{seg_id} [{model_key}]  |  '
            f'{len(regions)} total  |  Click to set next START'
        )

    btn_confirm.on_clicked(_confirm_region)

    # ------------------------------------------------------------------
    # Undo button
    # ------------------------------------------------------------------

    def _on_undo(_event=None):
        _close_dropdown()
        _clear_in_progress()
        removed = store.undo_last()
        if removed is not None:
            _full_redraw()
            _update_status(
                f'Undone #{removed["segment_id"]}  |  '
                f'{len(regions)} remaining  |  Click to set START'
            )
        else:
            _update_status('Nothing to undo  |  Click to set START')

    btn_undo.on_clicked(_on_undo)

    # ------------------------------------------------------------------
    # Delete button
    # ------------------------------------------------------------------

    def _on_delete(_event=None):
        _close_dropdown()
        _clear_in_progress()
        if not regions:
            _update_status('No regions to delete')
            return
        raw = delid_box.text.strip()
        if not raw:
            _update_status(
                f'Type a segment ID in the box first.  '
                f'Existing IDs: {store.ids()}'
            )
            return
        try:
            target = int(raw)
        except ValueError:
            _update_status(f'"{raw}" is not a valid integer ID')
            return
        try:
            store.remove(target)
            delid_box.set_val('')
            _full_redraw()
            _update_status(
                f'Deleted #{target}  |  '
                f'{len(regions)} remaining  |  Click to set START'
            )
        except KeyError:
            _update_status(f'ID #{target} not found  |  Existing: {store.ids()}')

    btn_delete.on_clicked(_on_delete)

    # ------------------------------------------------------------------
    # Mouse click
    # ------------------------------------------------------------------

    def on_click(event):
        # Close the dropdown if the user clicks anywhere outside it
        if state['dropdown_open']:
            if event.inaxes not in state['dropdown_axes'] \
                    and event.inaxes is not ax_modelbtn:
                _close_dropdown()

        if event.inaxes is not ax or event.button != 1:
            return
        if fig.canvas.toolbar is not None and fig.canvas.toolbar.mode != '':
            return
        x = event.xdata
        if x is None:
            return

        state['clicks'].append(x)

        if len(state['clicks']) == 1:
            vl = ax.axvline(x, color='green', ls='--', lw=1.2, zorder=4)
            state['vlines'].append(vl)
            _update_status(f'START = {x:.6g}  |  Now click END')

        elif len(state['clicks']) == 2:
            start, end = sorted(state['clicks'])
            for vl in state['vlines']:
                try:
                    vl.remove()
                except Exception:
                    pass
            state['vlines'] = [
                ax.axvline(start, color='green', ls='--', lw=1.2, zorder=4),
                ax.axvline(end,   color='red',   ls='--', lw=1.2, zorder=4),
            ]
            if state['patch'] is not None:
                try:
                    state['patch'].remove()
                except Exception:
                    pass
            state['patch']   = ax.axvspan(start, end, alpha=0.30,
                                          color=_PENDING_COLOUR, zorder=1)
            state['pending'] = {'start': start, 'end': end}
            _update_status(
                f'[{start:.6g} → {end:.6g}]  |  '
                'Add a note if desired, then press Enter or Confirm'
            )

        elif len(state['clicks']) > 2:
            _clear_in_progress()
            _update_status('Selection reset — click to set START again')

        fig.canvas.draw_idle()

    # ------------------------------------------------------------------
    # Key press  — ONLY 'enter', 'escape', and 'q' are active.
    # All other shortcuts are removed; use GUI buttons instead.
    # Suppressed entirely when a TextBox has keyboard focus.
    # ------------------------------------------------------------------

    def on_key(event):
        # If the user is typing in any TextBox, suppress ALL shortcuts.
        # We detect this by checking if the event originated inside a
        # TextBox axes — axes_enter_event tracking covers most cases, but
        # matplotlib's internal focus transfer for TextBox is unreliable
        # across backends.  As a belt-and-suspenders check we also test
        # event.inaxes directly.
        textbox_axes = {ax_notes, ax_delid}
        if state['active_textbox'] is not None:
            return
        if event.inaxes in textbox_axes:
            return

        key = event.key

        if key == 'enter':
            _confirm_region()

        elif key == 'escape':
            _close_dropdown()
            _clear_in_progress()
            _update_status('Cancelled  |  Click to set START')

        elif key == 'q':
            _close_dropdown()
            _clear_in_progress()
            store.save()
            print(f"\nSaved {len(regions)} region(s) to '{regions_file}'")
            print(f"  xscale={store.xscale}  yscale={store.yscale}")
            for r in regions:
                print(
                    f"  #{r['segment_id']:3d}  "
                    f"{r['start']:.6g} -> {r['end']:.6g}  "
                    f"[{r['model']}]  \"{r['note']}\""
                )
            plt.close(fig)

    # ------------------------------------------------------------------
    # Initial draw
    # ------------------------------------------------------------------
    _draw_data()
    _shade_confirmed()
    if regions:
        _update_status(
            f'{len(regions)} region(s) loaded  |  '
            f'xscale={store.xscale}  yscale={store.yscale}  |  '
            'Click to set START'
        )

    fig.canvas.mpl_connect('button_press_event', on_click)
    fig.canvas.mpl_connect('key_press_event',    on_key)
    plt.tight_layout(rect=[0, _CTRL_TOP - 0.01, 1, 1])
    plt.show()

    return store


# ---------------------------------------------------------------------------
# Testing
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    rng       = np.random.default_rng(42)
    t_test    = np.linspace(0, 200, 800)
    flux_test = (
        0.5
        + 3.0 * np.exp(-0.5 * ((t_test - 60)  / 8) ** 2)
        + 2.0 * np.exp(-0.5 * ((t_test - 140) / 5) ** 2)
        + 0.1 * rng.standard_normal(800)
    )
    unc_test = np.abs(0.08 + 0.02 * rng.standard_normal(800))

    store = run_selector(
        t_test, flux_test, uncertainty=unc_test,
        regions_file='/tmp/test_regions.json',
        xlabel='Days since discovery',
        ylabel='Flux density (arbitrary)',
        xscale='linear', yscale='linear',
    )
    print("\nReturned store:", store)