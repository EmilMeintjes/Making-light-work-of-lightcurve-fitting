"""
progress_window.py
-------------------
A small Tk window pair used to show MCMC fitting progress without needing
the terminal:

  - "MCMC Fitting Progress": an overall "region i of n" progress bar plus a
    "current region" progress bar driven by emcee's tqdm output (the
    terminal-based tqdm bar that would otherwise print to stdout/stderr).
  - "Fitting Log": a scrolling, read-only text window showing the same
    messages that would normally be printed to the terminal.

Usage
-----
    window = ProgressWindow()
    summaries, error = window.run(run_fitter, args=(t, flux),
                                   kwargs=dict(uncertainty=unc, ...))
    if error is not None:
        raise error

`run_fitter` (or any target function) must accept a `progress=` keyword
argument; `window.run` injects `window.reporter` as that argument.  The
target runs in a background thread so the Tk main loop stays responsive.

Thread safety
-------------
`ProgressReporter`'s methods are the only thing the worker thread touches.
They just push events onto a `queue.Queue`.  `ProgressWindow._poll`, which
runs on the main thread via `root.after`, drains the queue and updates the
Tk widgets — Tk widgets must never be touched directly from the worker
thread.
"""

import queue
import threading
import tkinter as tk
from tkinter import ttk

_MAX_LOG_LINES = 2000   # cap to avoid unbounded memory growth on long runs


class ProgressReporter:

    """
    Thread-safe handle passed to the fitting function as `progress=`.

    Each method just enqueues an event; `ProgressWindow._poll` (running on
    the main/Tk thread) is responsible for turning these into widget updates.
    """

    def __init__(self):
        self._queue = queue.Queue()

    def log(self, message):
        """Append a line to the log window."""
        self._queue.put(('log', str(message)))

    def set_region(self, index, total, segment_id, label=''):
        """
        Called once per region, before fitting it starts.

        Parameters
        ----------
        index      : int   1-based position of this region (e.g. 2 of 5).
        total      : int   Total number of regions being fitted.
        segment_id : int   The region's segment_id, for display.
        label      : str   Extra context (e.g. the model name).
        """
        self._queue.put(('region', index, total, segment_id, label))

    def set_mcmc_progress(self, fraction):
        """Update the "current region" progress bar, fraction in [0, 1]."""
        self._queue.put(('mcmc', fraction))

    def done(self, result=None, error=None):
        """Signal that the target function has returned (or raised)."""
        self._queue.put(('done', result, error))


class ProgressWindow:

    """
    Tk window pair showing overall + per-region MCMC progress, and a
    separate scrolling log window.

    Create one, then call `run(target, args, kwargs)` to execute `target`
    in a background thread while the windows are shown.
    """

    def __init__(self, title='MCMC Fitting Progress'):
        self.reporter = ProgressReporter()
        self.result   = None
        self.error    = None
        self._thread  = None

        self.root = tk.Tk()
        self.root.title(title)
        self.root.minsize(520, 300)
        self.root.protocol('WM_DELETE_WINDOW', self._on_close)

        # A single padded frame holds everything so nothing is clipped against
        # the window edge; the window then sizes itself to fit the frame.
        frame = ttk.Frame(self.root, padding=14)
        frame.pack(fill='both', expand=True)
        frame.columnconfigure(0, weight=1)

        bar_len = 480
        row = 0

        ttk.Label(frame, text='Overall progress').grid(
            row=row, column=0, sticky='w', pady=(0, 2)); row += 1
        self.region_bar = ttk.Progressbar(frame, orient='horizontal',
                                           length=bar_len, mode='determinate')
        self.region_bar.grid(row=row, column=0, sticky='ew'); row += 1
        self.region_label = ttk.Label(frame, text='Waiting to start...')
        self.region_label.grid(row=row, column=0, sticky='w', pady=(2, 10))
        row += 1

        ttk.Label(frame, text='Current region: MCMC progress').grid(
            row=row, column=0, sticky='w', pady=(0, 2)); row += 1
        self.mcmc_bar = ttk.Progressbar(frame, orient='horizontal',
                                         length=bar_len, mode='determinate',
                                         maximum=100)
        self.mcmc_bar.grid(row=row, column=0, sticky='ew'); row += 1
        self.mcmc_label = ttk.Label(frame, text='0%')
        self.mcmc_label.grid(row=row, column=0, sticky='w', pady=(2, 10))
        row += 1

        ttk.Separator(frame, orient='horizontal').grid(
            row=row, column=0, sticky='ew', pady=6); row += 1

        self.status_label = ttk.Label(frame, text='Running...',
                                       font=('TkDefaultFont', 12, 'bold'))
        self.status_label.grid(row=row, column=0, sticky='w', pady=(0, 8))
        row += 1

        self.close_btn = ttk.Button(frame, text='Close', command=self._on_close,
                                     state='disabled')
        self.close_btn.grid(row=row, column=0, sticky='e'); row += 1

        # Separate log window
        self.log_win = tk.Toplevel(self.root)
        self.log_win.title('Fitting log')
        self.log_win.geometry('640x400')
        self.log_win.protocol('WM_DELETE_WINDOW', lambda: None)  # closes with main window only

        self.log_text = tk.Text(self.log_win, state='disabled', wrap='word',
                                 font=('Courier', 10))
        scrollbar = ttk.Scrollbar(self.log_win, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scrollbar.set)
        self.log_text.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')

    # ------------------------------------------------------------------
    # Running the target function
    # ------------------------------------------------------------------

    def run(self, target, args=(), kwargs=None):
        """
        Run `target(*args, **kwargs, progress=self.reporter)` in a background
        thread, show the windows, and block until the user closes them *and*
        the target has finished.

        Returns
        -------
        (result, error) : the target's return value and `None`, or
                          (None, exception) if the target raised.
        """

        kwargs = dict(kwargs or {})
        kwargs['progress'] = self.reporter

        def _worker():
            try:
                result = target(*args, **kwargs)
                self.reporter.done(result=result)
            except Exception as exc:
                self.reporter.done(error=exc)

        self._thread = threading.Thread(target=_worker, daemon=True)
        self._thread.start()

        self._poll()
        self.root.mainloop()

        # The window may have been closed before the worker finished; make
        # sure we wait for it and pick up its final 'done' event either way.
        self._thread.join()
        self._drain(apply_to_widgets=False)

        return self.result, self.error

    # ------------------------------------------------------------------
    # Queue draining / widget updates
    # ------------------------------------------------------------------

    def _poll(self):
        self._drain(apply_to_widgets=True)
        try:
            self.root.after(100, self._poll)
        except tk.TclError:
            pass  # window already destroyed

    def _drain(self, apply_to_widgets):
        try:
            while True:
                item = self.reporter._queue.get_nowait()
                if apply_to_widgets:
                    self._handle(item)
                elif item[0] == 'done':
                    self.result, self.error = item[1], item[2]
        except queue.Empty:
            pass

    def _handle(self, item):
        kind = item[0]

        if kind == 'log':
            self._append_log(item[1])

        elif kind == 'region':
            _, index, total, segment_id, label = item
            self.region_bar['maximum'] = total
            self.region_bar['value']   = index - 1
            text = f'Region {index} of {total} — segment #{segment_id}'
            if label:
                text += f'  {label}'
            self.region_label.config(text=text)
            self.mcmc_bar['value'] = 0
            self.mcmc_label.config(text='0%')

        elif kind == 'mcmc':
            pct = max(0, min(100, int(round(item[1] * 100))))
            self.mcmc_bar['value'] = pct
            self.mcmc_label.config(text=f'{pct}%')

        elif kind == 'done':
            self.result, self.error = item[1], item[2]
            if self.error is not None:
                self.status_label.config(text=f'Error: {self.error}',
                                         foreground='firebrick')
                self._append_log(f'ERROR: {self.error}')
            else:
                self.status_label.config(
                    text='✔  Fitting complete — click Close to continue',
                    foreground='forest green')
                self.region_bar['value'] = self.region_bar['maximum']
                self.mcmc_bar['value']   = 100
                self.mcmc_label.config(text='100%')
                self._append_log('=== Fitting complete. '
                                 'Close the windows to continue. ===')
            self.close_btn.config(text='Close', state='normal')
            self.close_btn.focus_set()
            # Bring the window forward and force a repaint so the finished
            # state is unmistakable (otherwise the window can look "blank"
            # after the worker thread releases the CPU).
            try:
                self.root.lift()
                self.root.update_idletasks()
            except tk.TclError:
                pass

    def _append_log(self, text):
        self.log_text.config(state='normal')
        self.log_text.insert('end', text + '\n')
        n_lines = int(self.log_text.index('end-1c').split('.')[0])
        if n_lines > _MAX_LOG_LINES:
            self.log_text.delete('1.0', f'{n_lines - _MAX_LOG_LINES}.0')
        self.log_text.see('end')
        self.log_text.config(state='disabled')

    # ------------------------------------------------------------------
    # Closing
    # ------------------------------------------------------------------

    def _on_close(self):
        if self._thread is not None and self._thread.is_alive():
            from tkinter import messagebox
            if messagebox.askyesno(
                'Fitting in progress',
                'MCMC fitting is still running in the background.\n\n'
                'Closing this window will hide the progress display, but '
                'fitting will continue and results will still be saved.\n\n'
                'Close anyway?',
                parent=self.root,
            ):
                self.root.destroy()
        else:
            self.root.destroy()