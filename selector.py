"""
selector.py
-----------
Stage 1: Interactive region selector for nova-like lightcurve fitting.

Controls
--------
Left-click (x2)  : set region START then END (order sorted automatically).
Enter / button   : confirm the pending region and write to JSON.
u                : undo the most recently confirmed region.
d                : delete by ID (prompts in terminal).
Escape           : cancel the current in-progress selection.
q                : save all regions and close.
Zoom / Pan tools : toolbar modes work normally; selector ignores those clicks.
"""

# Sytem imports
#import sys

#External imports
import matplotlib.pyplot as plt
import matplotlib.widgets as mwidgets
import numpy as np

#Local imports
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
    xscale       : str    'linear' or 'log'.  Stored in the JSON file so all
                          downstream modules pick it up automatically.
    yscale       : str    'linear' or 'log'.  Same.
    title        : str or None

    Notes on scale
    --------------
    If the JSON file already exists and contains scale metadata, the stored
    values take precedence over the xscale/yscale arguments here.  This means
    re-opening an existing session always restores the original choice.
    To change the scale of an existing session, edit the JSON metadata block
    or call store.set_scales() before running.
    """

    t    = np.asarray(t,    dtype=float)
    flux = np.asarray(flux, dtype=float)
    if uncertainty is not None:
        uncertainty = np.asarray(uncertainty, dtype=float)

    # --- Load (or create) the region store ---
    store = RegionStore(regions_file, xscale=xscale, yscale=yscale)
    store.load()   # stored scales override constructor args if file exists
    regions = store.regions   # live reference — mutations are reflected in store

    # ------------------------------------------------------------------
    # Shared mutable state
    # ------------------------------------------------------------------
    state = {
        'clicks':  [],
        'vlines':  [],
        'patch':   None,
        'pending': None,
    }

    # ------------------------------------------------------------------
    # Figure layout
    # ------------------------------------------------------------------
    fig = plt.figure(figsize=(16, 7))
    ax  = fig.add_axes([0.07, 0.30, 0.88, 0.62])

    # Model radio buttons
    ax_radio = fig.add_axes([0.07, 0.05, 0.18, 0.20])
    ax_radio.set_title('Model', fontsize=9, pad=3)
    radio = mwidgets.RadioButtons(ax_radio, MODEL_LABELS, activecolor='steelblue')

    for label in radio.labels:
        label.set_fontsize(9)

    if hasattr(radio, "circles"):
        for circle in radio.circles:
            circle.set_radius(0.06)

    # Notes text box
    ax_notes_label = fig.add_axes([0.28, 0.18, 0.50, 0.04])
    ax_notes_label.axis('off')
    ax_notes_label.text(0.0, 0.5, 'Notes for this region:',
                        va='center', fontsize=9)
    ax_notes  = fig.add_axes([0.28, 0.10, 0.50, 0.07])
    notes_box = mwidgets.TextBox(ax_notes, '', initial='')

    # Confirm button
    ax_confirm  = fig.add_axes([0.80, 0.10, 0.10, 0.07])
    btn_confirm = mwidgets.Button(ax_confirm, 'Confirm\n(Enter)',
                                  color='lightgreen', hovercolor='mediumseagreen')

    # Status text
    status_txt = ax.text(
        0.01, 0.97, 'Left-click: set START',
        transform=ax.transAxes, va='top', fontsize=9,
        bbox=dict(boxstyle='round', fc='wheat', alpha=0.85), zorder=10,
    )

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
                'Nova-like lightcurve  |  '
                'Click: START->END  |  Enter: confirm  |  '
                'u: undo  |  d: delete by ID  |  Esc: cancel  |  q: quit'
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

    def _selected_model_key():
        return MODEL_KEYS[MODEL_LABELS.index(radio.value_selected)]

    # ------------------------------------------------------------------
    # Confirm region
    # ------------------------------------------------------------------

    def _confirm_region(_event=None):
        if state['pending'] is None:
            _update_status('Nothing to confirm — click START then END first.')
            return

        model_key = _selected_model_key()
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
        store.add(region)   # validates, fills defaults, saves JSON

        # Commit shading
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
    # Mouse click
    # ------------------------------------------------------------------

    def on_click(event):
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
                f'[{start:.6g} -> {end:.6g}]  |  '
                'Add a note if desired, then press Enter or Confirm'
            )

        elif len(state['clicks']) > 2:
            _clear_in_progress()
            _update_status('Selection reset — click to set START again')

        fig.canvas.draw_idle()

    # ------------------------------------------------------------------
    # Key press
    # ------------------------------------------------------------------

    def on_key(event):
        key = event.key

        if key == 'enter':
            _confirm_region()

        elif key == 'escape':
            _clear_in_progress()
            _update_status('Cancelled  |  Click to set START')

        elif key == 'u':
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

        elif key == 'd':
            _clear_in_progress()
            if not regions:
                _update_status('No regions to delete')
                return
            print(f"\nDelete by ID.  Existing IDs: {store.ids()}")
            try:
                target = int(input("Enter segment ID to delete (or 0 to cancel): "))
            except (ValueError, EOFError):
                _update_status('Delete cancelled')
                return
            if target == 0:
                _update_status('Delete cancelled')
                return
            try:
                store.remove(target)
                _full_redraw()
                _update_status(
                    f'Deleted #{target}  |  '
                    f'{len(regions)} remaining  |  Click to set START'
                )
            except KeyError:
                _update_status(f'ID #{target} not found  |  Click to set START')

        elif key == 'q':
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
    plt.tight_layout()
    #plt.tight_layout(rect=[0, 0.28, 1, 1])
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
        xscale=xscale, yscale=yscale,
    )
    print("\nReturned store:", store)