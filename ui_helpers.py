"""
ui_helpers.py
-------------
Small, reusable Tk/matplotlib helpers shared by the interactive stages
(selector.py, initialiser.py).

Why Tk?
-------
The interactive stages are built from `matplotlib.widgets` inside a single
`plt.show()` window using the 'TkAgg' backend.  matplotlib's own widget set
has no dropdown / combobox and no real modal dialogs, but the 'TkAgg' canvas
is itself a child of an ordinary Tk `Toplevel` window — so we can create
normal `tkinter`/`ttk` widgets (Combobox, messagebox, simpledialog) and either
place them as siblings over the canvas, or use them as standalone popups
parented to that same window.

Both selector.py and initialiser.py call `matplotlib.use('TkAgg')` before
importing pyplot, so `fig.canvas.get_tk_widget()` is guaranteed to exist by
the time these helpers are used.
"""

import tkinter as tk
from tkinter import ttk

import matplotlib.widgets as mwidgets


# ---------------------------------------------------------------------------
# Tk root / canvas helpers
# ---------------------------------------------------------------------------

def get_tk_root(fig):
    """
    Return the Tk root/toplevel window hosting *fig*'s canvas.

    Raises
    ------
    RuntimeError
        If *fig* was not created with the 'TkAgg' backend (i.e. has no
        `get_tk_widget()` method), with a message explaining how to fix it.
    """

    canvas = fig.canvas
    if not hasattr(canvas, 'get_tk_widget'):
        raise RuntimeError(
            "ui_helpers requires the 'TkAgg' matplotlib backend, but this "
            f"figure's canvas is {type(canvas).__name__}.  Call "
            "matplotlib.use('TkAgg') before importing pyplot."
        )
    return canvas.get_tk_widget().winfo_toplevel()


def place_over_axes(fig, widget, rect):
    """
    Overlay a Tk *widget* on top of *fig*'s canvas at a figure-fraction
    *rect* = (x, y, width, height), using the same bottom-left-origin
    convention as `Figure.add_axes`.

    The widget's master must be `fig.canvas.get_tk_widget()` (or another
    widget sharing the same toplevel).  Position is recomputed on every
    canvas resize via `<Configure>`, so the overlay stays correctly placed
    if the window is resized.

    Returns
    -------
    callable
        The reposition function, in case the caller wants to force an
        immediate re-layout (e.g. after changing *rect*).
    """

    canvas_widget = fig.canvas.get_tk_widget()
    x, y, w, h = rect

    def _reposition(_event=None):
        width_px  = canvas_widget.winfo_width()
        height_px = canvas_widget.winfo_height()
        if width_px <= 1 or height_px <= 1:
            return
        widget.place(
            in_=canvas_widget,
            x=x * width_px,
            y=(1.0 - y - h) * height_px,
            width=w * width_px,
            height=h * height_px,
        )

    canvas_widget.bind('<Configure>', _reposition, add='+')
    canvas_widget.after_idle(_reposition)
    return _reposition


# ---------------------------------------------------------------------------
# Dialogs
# ---------------------------------------------------------------------------
# These are custom `Toplevel`-based dialogs rather than `tkinter.messagebox` /
# `simpledialog` ones.  The native message boxes (especially on macOS, where
# they are OS-drawn alerts) ignore any attempt to position them, so on a
# multi-monitor setup they get stranded in a corner of the wrong display.  A
# plain `Toplevel` honours `geometry()` on every platform, so we build the
# dialogs ourselves and place each one centred on the mouse pointer — which
# puts it where the user is looking, on whichever monitor they are using.
#
# Each dialog accepts fig=None (e.g. before any figure window has been
# created, such as a failure while loading the regions file) and falls back to
# a throwaway hidden Tk root so the popup still appears.

_DIALOG_CANCELLED = object()   # sentinel: dialog closed without a choice


def _make_hidden_root():
    """Create a hidden, throwaway Tk root to parent a standalone dialog."""
    root = tk.Tk()
    root.withdraw()
    return root


def _dialog_parent(fig):
    if fig is None:
        return _make_hidden_root(), True
    try:
        return get_tk_root(fig), False
    except RuntimeError:
        return _make_hidden_root(), True


def _centre_on_pointer(top):
    """
    Position the Toplevel *top* centred on the current mouse pointer.

    Using the pointer (rather than `winfo_screenwidth()/winfo_screenheight()`)
    avoids the multi-monitor trap where those report the *combined* desktop
    size, so the computed "centre" lands on the seam between displays and the
    window is shoved to a screen edge.  The pointer is always within the
    monitor the user is actually working on.
    """
    top.update_idletasks()
    w  = top.winfo_width()
    h  = top.winfo_height()
    px = top.winfo_pointerx()
    py = top.winfo_pointery()
    top.geometry(f'+{px - w // 2}+{py - h // 2}')


def _run_dialog(fig, title, message, buttons, *, default=None):
    """
    Show a modal dialog with *message* and one button per (label, value) pair
    in *buttons*; return the value of the button clicked.

    If the window is closed via its title-bar control, returns *default*.
    """
    parent, owned = _dialog_parent(fig)
    chosen = {'value': default}

    top = tk.Toplevel(parent)
    top.title(title)
    top.configure(padx=18, pady=16)
    top.resizable(False, False)

    ttk.Label(top, text=message, justify='left', wraplength=460
              ).pack(anchor='w')

    def _choose(value):
        chosen['value'] = value
        top.destroy()

    bar = ttk.Frame(top)
    bar.pack(anchor='e', pady=(16, 0))
    for i, (label, value) in enumerate(buttons):
        btn = ttk.Button(bar, text=label, command=lambda v=value: _choose(v))
        btn.pack(side='left', padx=4)
        if i == len(buttons) - 1:
            btn.focus_set()   # last button (usually the affirmative) is default

    top.protocol('WM_DELETE_WINDOW', lambda: _choose(default))
    top.transient(parent if not owned else None)
    _centre_on_pointer(top)
    top.grab_set()
    top.wait_window()

    if owned:
        parent.destroy()
    return chosen['value']


def show_error(fig, title, message):
    """Show a modal error dialog parented to *fig*'s window (or standalone)."""
    _run_dialog(fig, title, message, [('OK', None)])


def show_warning(fig, title, message):
    """Show a modal warning dialog parented to *fig*'s window (or standalone)."""
    _run_dialog(fig, title, message, [('OK', None)])


def show_info(fig, title, message):
    """Show a modal info dialog parented to *fig*'s window (or standalone)."""
    _run_dialog(fig, title, message, [('OK', None)])


def ask_yes_no(fig, title, message):
    """Show a Yes/No dialog and return True (Yes) or False (No)."""
    return bool(_run_dialog(fig, title, message,
                            [('No', False), ('Yes', True)], default=False))


def ask_yes_no_cancel(fig, title, message):
    """Show a Yes/No/Cancel dialog; returns True, False, or None (Cancel)."""
    value = _run_dialog(
        fig, title, message,
        [('Cancel', _DIALOG_CANCELLED), ('No', False), ('Yes', True)],
        default=_DIALOG_CANCELLED,
    )
    return None if value is _DIALOG_CANCELLED else value


def ask_integer(fig, title, prompt):
    """
    Show a numeric-entry dialog; returns an int, or None if cancelled or if
    the entry was left blank / non-numeric.
    """
    parent, owned = _dialog_parent(fig)
    chosen = {'value': None}

    top = tk.Toplevel(parent)
    top.title(title)
    top.configure(padx=18, pady=16)
    top.resizable(False, False)

    ttk.Label(top, text=prompt, justify='left', wraplength=460).pack(anchor='w')

    var = tk.StringVar()
    entry = ttk.Entry(top, textvariable=var)
    entry.pack(fill='x', pady=(10, 0))
    entry.focus_set()

    def _ok(_event=None):
        try:
            chosen['value'] = int(var.get().strip())
        except (ValueError, TypeError):
            chosen['value'] = None
        top.destroy()

    def _cancel(_event=None):
        chosen['value'] = None
        top.destroy()

    bar = ttk.Frame(top)
    bar.pack(anchor='e', pady=(16, 0))
    ttk.Button(bar, text='Cancel', command=_cancel).pack(side='left', padx=4)
    ttk.Button(bar, text='OK', command=_ok).pack(side='left', padx=4)

    entry.bind('<Return>', _ok)
    entry.bind('<Escape>', _cancel)
    top.protocol('WM_DELETE_WINDOW', _cancel)
    top.transient(parent if not owned else None)
    _centre_on_pointer(top)
    top.grab_set()
    top.wait_window()

    if owned:
        parent.destroy()
    return chosen['value']


# ---------------------------------------------------------------------------
# Key-capture guard
# ---------------------------------------------------------------------------

def any_textbox_capturing(textboxes):
    """
    Return True if any `matplotlib.widgets.TextBox` in *textboxes* currently
    has keyboard focus (`capturekeystrokes`).

    Figure-level `key_press_event` handlers should check this first and
    return early if True, so that typing into a text box (e.g. a value like
    'quit' or a note containing the letter 'u') does not also trigger global
    single-key shortcuts such as undo ('u'), delete ('d') or quit ('q').
    """

    return any(getattr(tb, 'capturekeystrokes', False) for tb in textboxes)


# ---------------------------------------------------------------------------
# Enable/disable-able button
# ---------------------------------------------------------------------------

class ToggleButton:
    """
    Wrapper around `matplotlib.widgets.Button` that can be greyed out and
    made unclickable.

    Used for actions that are only valid in certain states (e.g. "Confirm"
    before a region is fully selected, "Undo"/"Redo" when there is nothing
    to undo/redo).  When disabled, the button face and label are dimmed and
    its callback becomes a no-op; `set_enabled(True)` restores the original
    appearance and behaviour.

    Parameters
    ----------
    ax              : matplotlib Axes for the button.
    label           : str    Button text.
    color           : str    Face colour when enabled.
    hovercolor      : str    Hover colour when enabled.
    disabled_color  : str    Face (and hover) colour when disabled.
    enabled         : bool   Initial state.
    """

    def __init__(self, ax, label, color='lightgrey', hovercolor='gray35',
                 disabled_color='whitesmoke', enabled=True):
        self.button = mwidgets.Button(ax, label, color=color, hovercolor=hovercolor)
        self._enabled_color  = color
        self._enabled_hover  = hovercolor
        self._disabled_color = disabled_color
        self._enabled_label_color  = self.button.label.get_color()
        self._disabled_label_color = 'darkgrey'
        self._callback = None
        self._enabled  = True   # set properly by set_enabled below
        self.button.on_clicked(self._on_click)
        self.set_enabled(enabled)

    def on_clicked(self, callback):
        """Register *callback* to run on click, but only while enabled."""
        self._callback = callback

    def _on_click(self, event):
        if self._enabled and self._callback is not None:
            self._callback(event)

    def set_enabled(self, enabled):
        """Enable or grey-out the button, updating its appearance immediately."""
        self._enabled = bool(enabled)
        if self._enabled:
            self.button.color      = self._enabled_color
            self.button.hovercolor = self._enabled_hover
            self.button.label.set_color(self._enabled_label_color)
        else:
            self.button.color      = self._disabled_color
            self.button.hovercolor = self._disabled_color
            self.button.label.set_color(self._disabled_label_color)
        self.button.ax.set_facecolor(self.button.color)
        self.button.ax.figure.canvas.draw_idle()

    @property
    def ax(self):
        return self.button.ax
