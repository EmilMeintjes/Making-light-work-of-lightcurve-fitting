"""
selector.py
-----------
Stage 1: Interactive region selector for nova-like lightcurve fitting.

Controls
--------
Left-click (x2)  : set region START then END (order sorted automatically).
Model dropdown   : choose which model this region will be fitted with.
Enter / Confirm  : confirm the pending region and write to JSON.
                   Greyed out until both START and END are set.
Undo / u         : undo the most recently confirmed region.
Redo             : re-apply the most recently undone region.
                   Both buttons grey out automatically when there is nothing
                   left to undo/redo.
d                : delete by ID (opens a small dialog, no terminal needed).
Escape           : cancel the current in-progress selection.
q                : save all regions and close.
Zoom / Pan tools : toolbar modes work normally; selector ignores those clicks.

Notes box
---------
Typing into the "Notes" box never triggers the single-key shortcuts above
(u/d/q/Escape) — the figure-level key handler checks whether the notes box
currently has keyboard focus and, if so, lets the text box handle the key.
"""

# System imports
#import sys

# External imports — force the 'TkAgg' backend *before* importing pyplot so
# that ui_helpers can rely on fig.canvas.get_tk_widget() existing (used for
# the model dropdown and all popup dialogs).
import matplotlib
try:
    matplotlib.use('TkAgg')
except Exception as exc:
    print(f"WARNING: could not set matplotlib backend to 'TkAgg' ({exc}). "
          "The model dropdown and popup dialogs may not work correctly.")

import matplotlib.pyplot as plt
import matplotlib.widgets as mwidgets
import numpy as np
import tkinter as tk
from tkinter import ttk

#Local imports
from fitting_models import MODEL_KEYS, MODEL_LABELS
from persistence import RegionStore
from ui_helpers import (
    ToggleButton, any_textbox_capturing, ask_integer,
    place_over_axes, show_error, show_warning,
)

# ---------------------------------------------------------------------------
# Colours
# ---------------------------------------------------------------------------

_MODEL_COLOURS = {
    'gaussian':     'cornflowerblue',
    'rising_exp':   'mediumseagreen',
    'decaying_exp': 'tomato',
    'crystal_ball': 'mediumpurple',
    'bazin':        'goldenrod',
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
    try:
        store.load()   # stored scales override constructor args if file exists
    except ValueError as exc:
        show_error(
            None, "Could not load regions file",
            f"Failed to load '{regions_file}':\n\n{exc}\n\n"
            "Fix or remove the file and try again."
        )
        return store
    regions = store.regions   # live reference — mutations are reflected in store

    # ------------------------------------------------------------------
    # Shared mutable state
    # ------------------------------------------------------------------
    # Undo/redo are driven by an action history rather than a single "remove
    # last region" step, so that *deletes* (not just adds) can be reversed —
    # an accidental delete should be recoverable.  Each entry is a
    # ('add' | 'delete', region) tuple recording the action the user took;
    # undo applies its inverse, redo re-applies it.
    state = {
        'clicks':     [],
        'vlines':     [],
        'patch':      None,
        'pending':    None,
        'undo_stack': [],   # actions the user performed, newest last
        'redo_stack': [],   # actions undone, available to Redo
    }

    # ------------------------------------------------------------------
    # Figure layout
    # ------------------------------------------------------------------
    fig = plt.figure(figsize=(16, 7))
    ax  = fig.add_axes([0.07, 0.30, 0.88, 0.62])

    # Model dropdown (a real Tk combobox, overlaid on the figure canvas)
    ax_model_label = fig.add_axes([0.07, 0.205, 0.18, 0.035])
    ax_model_label.axis('off')
    ax_model_label.text(0.0, 0.5, 'Model:', va='center', fontsize=9)

    model_var = tk.StringVar(value=MODEL_LABELS[0])
    model_combo = ttk.Combobox(
        master=fig.canvas.get_tk_widget(), textvariable=model_var,
        values=MODEL_LABELS, state='readonly',
    )
    place_over_axes(fig, model_combo, (0.07, 0.155, 0.18, 0.045))

    # Undo / Redo buttons
    ax_undo = fig.add_axes([0.07, 0.05, 0.085, 0.07])
    undo_btn = ToggleButton(ax_undo, 'Undo (u)', enabled=False)

    ax_redo = fig.add_axes([0.165, 0.05, 0.085, 0.07])
    redo_btn = ToggleButton(ax_redo, 'Redo', enabled=False)

    # Notes text box
    ax_notes_label = fig.add_axes([0.28, 0.18, 0.50, 0.04])
    ax_notes_label.axis('off')
    ax_notes_label.text(0.0, 0.5, 'Notes for this region:',
                        va='center', fontsize=9)
    ax_notes  = fig.add_axes([0.28, 0.10, 0.50, 0.07])
    notes_box = mwidgets.TextBox(ax_notes, '', initial='')

    # Confirm button — greyed out until a START and END have been clicked
    ax_confirm  = fig.add_axes([0.80, 0.10, 0.10, 0.07])
    confirm_btn = ToggleButton(ax_confirm, 'Confirm\n(Enter)',
                               color='lightgreen', hovercolor='mediumseagreen',
                               enabled=False)

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
                'Click: START->END  |  Enter/Confirm: confirm  |  '
                'Undo/Redo buttons  |  d: delete by ID  |  Esc: cancel  |  q: quit'
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
        confirm_btn.set_enabled(False)
        fig.canvas.draw_idle()

    def _update_status(msg):
        status_txt.set_text(msg)
        fig.canvas.draw_idle()

    def _update_undo_redo_buttons():
        undo_btn.set_enabled(len(state['undo_stack']) > 0)
        redo_btn.set_enabled(len(state['redo_stack']) > 0)

    def _selected_model_key():
        return MODEL_KEYS[MODEL_LABELS.index(model_var.get())]

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
        try:
            store.add(region)   # validates, fills defaults, saves JSON
        except (ValueError, TypeError) as exc:
            show_error(fig, 'Could not save region', str(exc))
            return

        # A brand-new user action invalidates any pending redo history, and is
        # recorded so it can be undone.
        state['undo_stack'].append(('add', region))
        state['redo_stack'] = []

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
        confirm_btn.set_enabled(False)
        _update_undo_redo_buttons()

        _update_status(
            f'Saved #{seg_id} [{model_key}]  |  '
            f'{len(regions)} total  |  Click to set next START'
        )

    confirm_btn.on_clicked(_confirm_region)

    # ------------------------------------------------------------------
    # Undo / Redo
    # ------------------------------------------------------------------

    def _apply_inverse(action, region):
        """
        Apply the inverse of *action* on *region* (used by undo).  Returns a
        short human-readable description of what happened, or None on failure
        (in which case an error dialog has already been shown).

        Because the undo/redo stacks are strictly LIFO inverses, a region is
        only ever re-added once every later action referencing the same
        segment_id has been undone first — so re-adding never collides with a
        live id.
        """
        if action == 'add':
            store.remove(region['segment_id'])
            return f'Removed #{region["segment_id"]}'
        else:  # 'delete'
            try:
                store.add(region)
            except (ValueError, TypeError) as exc:
                show_error(fig, 'Could not restore region', str(exc))
                return None
            return f'Restored #{region["segment_id"]}'

    def _apply_action(action, region):
        """Apply *action* on *region* (used by redo).  Returns a description
        or None on failure."""
        if action == 'add':
            try:
                store.add(region)
            except (ValueError, TypeError) as exc:
                show_error(fig, 'Could not redo', str(exc))
                return None
            return f'Re-added #{region["segment_id"]}'
        else:  # 'delete'
            store.remove(region['segment_id'])
            return f'Re-deleted #{region["segment_id"]}'

    def _do_undo(_event=None):
        """
        Undo the most recent action (an add or a delete) and push it onto the
        redo stack.  The 'u' key and the Undo button both call this.
        """
        if not state['undo_stack']:
            _update_status('Nothing to undo  |  Click to set START')
            return
        _clear_in_progress()
        action, region = state['undo_stack'][-1]
        desc = _apply_inverse(action, region)
        if desc is None:
            return   # failed; leave the stack untouched
        state['undo_stack'].pop()
        state['redo_stack'].append((action, region))
        _full_redraw()
        _update_status(
            f'Undone: {desc}  |  {len(regions)} region(s)  |  Click to set START'
        )
        _update_undo_redo_buttons()

    def _do_redo(_event=None):
        """Re-apply the most recently undone action (if any)."""
        if not state['redo_stack']:
            return
        _clear_in_progress()
        action, region = state['redo_stack'][-1]
        desc = _apply_action(action, region)
        if desc is None:
            return   # failed; leave the stack untouched
        state['redo_stack'].pop()
        state['undo_stack'].append((action, region))
        _full_redraw()
        _update_status(
            f'Redone: {desc}  |  {len(regions)} region(s)  |  Click to set START'
        )
        _update_undo_redo_buttons()

    undo_btn.on_clicked(_do_undo)
    redo_btn.on_clicked(_do_redo)

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

            if start == end:
                show_warning(
                    fig, 'Invalid region',
                    'Start and end are the same point — click two '
                    'different times to define a region.'
                )
                _clear_in_progress()
                _update_status('Selection reset — click to set START again')
                fig.canvas.draw_idle()
                return

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
            confirm_btn.set_enabled(True)
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
        # While the user is typing a note, let the text box handle every
        # key — otherwise typing e.g. "undo" would trigger the 'u' and 'd'
        # shortcuts below on every keystroke.
        if any_textbox_capturing([notes_box]):
            return

        key = event.key

        if key == 'enter':
            _confirm_region()

        elif key == 'escape':
            _clear_in_progress()
            _update_status('Cancelled  |  Click to set START')

        elif key == 'u':
            _do_undo()

        elif key == 'd':
            _clear_in_progress()
            if not regions:
                _update_status('No regions to delete')
                return
            # Show each ID alongside its model and the user's note, so they
            # can tell which region an integer refers to (e.g. "the rising").
            lines = []
            for r in sorted(regions, key=lambda x: x['segment_id']):
                note = (r.get('note') or '').strip()
                desc = f"[{r['model']}]"
                if note:
                    desc += f"  — {note}"
                lines.append(
                    f"  #{r['segment_id']}: {desc}  "
                    f"({r['start']:.6g} → {r['end']:.6g})"
                )
            target = ask_integer(
                fig, 'Delete region',
                "Existing regions:\n\n"
                + "\n".join(lines)
                + "\n\nEnter the segment ID to delete (Cancel to abort):",
            )
            if target is None:
                _update_status('Delete cancelled')
                return
            # Grab a copy of the region before removing it, so Undo can put it
            # back exactly as it was.
            doomed = next((dict(r) for r in regions
                           if r['segment_id'] == target), None)
            try:
                store.remove(target)
            except KeyError:
                show_warning(fig, 'Not found', f'No region with ID #{target}.')
                _update_status(f'ID #{target} not found  |  Click to set START')
                _update_undo_redo_buttons()
                return
            # Record the delete so it can be undone, and invalidate redo.
            state['undo_stack'].append(('delete', doomed))
            state['redo_stack'] = []
            _full_redraw()
            _update_status(
                f'Deleted #{target} (Undo to restore)  |  '
                f'{len(regions)} remaining  |  Click to set START'
            )
            _update_undo_redo_buttons()

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
    _update_undo_redo_buttons()
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
        xscale='linear', yscale='linear',
    )
    print("\nReturned store:", store)