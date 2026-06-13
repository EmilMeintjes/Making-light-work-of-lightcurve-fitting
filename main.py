"""
main.py
-------
Top-level entry point for the nova-like lightcurve fitting pipeline.

Pipeline stages
---------------
  Stage 1  selector    -- interactively mark fit regions on the lightcurve
  Stage 2  initialiser -- set parameter initial guesses per region via sliders
  Stage 3  fitter      -- run MCMC (PyAutoFit/emcee) for each region
  Stage 4  plots       -- generate overview + per-region fit and corner plots

Typical usage
-------------
  1. Edit the DATA LOADING section below (or subclass / replace it) to load
     your lightcurve into t, flux, [uncertainty].

  2. Run stages interactively:

       python main.py --stage select
       python main.py --stage init
       python main.py --stage fit
       python main.py --stage plot

     Or run the full pipeline end-to-end (stages run sequentially; interactive
     stages block until the window is closed):

       python main.py --stage all

  3. To re-fit or re-initialise specific regions:

       python main.py --stage init --ids 2 5
       python main.py --stage fit  --ids 2 5

Command-line arguments
----------------------
  --stage   {select, init, fit, plot, all}   Which stage(s) to run.
  --ids     [int ...]                        Restrict init/fit to these IDs.
  --data    str                              Path to data file (optional
                                             override of the DATA section).
  --regions str                             Path to regions JSON file.
  --results str                              Path to results directory.
  --plots   str                              Path to plots output directory.
  --walkers int                              emcee walkers  (default 60).
  --steps   int                              emcee steps    (default 1500).
  --burn    int                              Burn-in steps  (default 300).
"""

# System imports
import argparse
#import os
import sys

#External imports
import numpy as np

# local imports
from selector    import run_selector
from initialiser import run_initialiser
from fitter      import run_fitter, run_fitter_gui
from plots       import plot_all


# ===========================================================================
# DATA LOADING
# ===========================================================================
# Replace or extend this function to load your specific data format.
# It must return:
#   t           : 1-D np.ndarray   Time axis (any units — label via XLABEL)
#   flux        : 1-D np.ndarray   Flux (any units — label via YLABEL)
#   uncertainty : 1-D np.ndarray or None   Per-point 1-sigma errors
#
# The pipeline does not care about absolute units; consistency is all that matters.
# The user is responsible for knowing what the axis values mean.

def load_data(data_path=None):

    """
    Load lightcurve data.

    Default implementation: expects a whitespace-delimited text file with
    columns  [time, flux]  or  [time, flux, uncertainty].
    Lines beginning with '#' are treated as comments.

    Replace this function body with your own loading logic as needed.
    """

    if data_path is None:
        raise ValueError(
            "No data path provided.  Pass --data <path> on the command line "
            "or edit the load_data() function in main.py."
        )

    data = np.loadtxt(data_path, comments='#')

    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(
            f"Expected at least 2 columns in '{data_path}', "
            f"got shape {data.shape}."
        )

    t    = data[:, 0]
    flux = data[:, 1]
    unc  = data[:, 2] if data.shape[1] >= 3 else None
    return t, flux, unc


# ===========================================================================
# CONFIGURATION  (edit these defaults to match your project)
# ===========================================================================

REGIONS_FILE = 'regions.json'
RESULTS_DIR  = 'results'
PLOTS_DIR    = 'plots'

XLABEL = 'Time'    # e.g. 'MJD', 'Days since discovery', 'Phase'
YLABEL = 'Flux'    # e.g. 'Flux density [mJy]', 'Magnitude'

N_WALKERS = 60
N_STEPS   = 1500
N_BURN    = 300


# ===========================================================================
# Stage runners
# ===========================================================================

def stage_select(t, flux, unc, args):
    print("\n=== Stage 1: Region selection ===")
    regions = run_selector(
        t, flux, uncertainty=unc,
        regions_file=args.regions,
        xlabel=XLABEL, ylabel=YLABEL,
        xscale=args.xscale, yscale=args.yscale,
    )
    print(f"Selection complete.  {len(regions)} region(s) saved to "
          f"'{args.regions}'.")


def stage_init(t, flux, unc, args):
    print("\n=== Stage 2: Parameter initialisation ===")
    ids = args.ids if args.ids else None
    run_initialiser(
        t, flux, uncertainty=unc,
        regions_file=args.regions,
        xlabel=XLABEL, ylabel=YLABEL,
        segment_ids=ids,
    )
    print("Initialisation complete.")


def stage_fit(t, flux, unc, args):
    print("\n=== Stage 3: MCMC fitting ===")
    ids = args.ids if args.ids else None
    fit_fn = run_fitter if args.no_gui else run_fitter_gui
    summaries = fit_fn(
        t, flux, uncertainty=unc,
        regions_file=args.regions,
        results_dir=args.results,
        n_walkers=args.walkers,
        n_steps=args.steps,
        n_burn=args.burn,
        segment_ids=ids,
        force_refit=args.force,
    )
    print(f"\nFitting complete.  {len(summaries)} region(s) fitted.")


def stage_plot(t, flux, unc, args):
    print("\n=== Stage 4: Plots ===")
    plot_all(
        t, flux, uncertainty=unc,
        regions_file=args.regions,
        results_dir=args.results,
        output_dir=args.plots,
        xlabel=XLABEL, ylabel=YLABEL,
        show=False,
    )
    print(f"Plots written to '{args.plots}'.")


# ===========================================================================
# CLI
# ===========================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description='Nova-like lightcurve fitting pipeline.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument('--stage',
                   choices=['select', 'init', 'fit', 'plot', 'all'],
                   default='all',
                   help='Pipeline stage to run (default: all).')
    p.add_argument('--data',
                   default=None,
                   help='Path to the lightcurve data file.')
    p.add_argument('--ids',
                   nargs='+', type=int, default=None,
                   help='Segment IDs to restrict init/fit to.')
    p.add_argument('--regions',
                   default=REGIONS_FILE,
                   help=f'Regions JSON file (default: {REGIONS_FILE}).')
    p.add_argument('--results',
                   default=RESULTS_DIR,
                   help=f'Results directory (default: {RESULTS_DIR}).')
    p.add_argument('--plots',
                   default=PLOTS_DIR,
                   help=f'Plots directory (default: {PLOTS_DIR}).')
    p.add_argument('--walkers', type=int, default=N_WALKERS,
                   help=f'emcee walkers (default: {N_WALKERS}).')
    p.add_argument('--steps',   type=int, default=N_STEPS,
                   help=f'emcee steps (default: {N_STEPS}).')
    p.add_argument('--burn',    type=int, default=N_BURN,
                   help=f'Burn-in steps to discard (default: {N_BURN}).')
    p.add_argument('--force',   action='store_true', default=False,
                   help='Delete stale PyAutoFit directories before fitting '
                        '(use to recover from crashed runs).')
    p.add_argument('--no-gui',  action='store_true', default=False,
                   help='During the fit stage, print progress to the '
                        'terminal instead of opening progress/log windows.')
    p.add_argument('--xscale', default='linear', choices=['linear', 'log'],
                   help='X-axis scale (default: linear). Stored in regions JSON.')
    p.add_argument('--yscale', default='linear', choices=['linear', 'log'],
                   help='Y-axis scale (default: linear). Stored in regions JSON.')
    return p.parse_args()


def main():
    args = parse_args()

    # Load data
    try:
        t, flux, unc = load_data(args.data)
    except ValueError as exc:
        print(f"ERROR loading data: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(t)} data points.")

    stage = args.stage

    if stage in ('select', 'all'):
        stage_select(t, flux, unc, args)

    if stage in ('init', 'all'):
        stage_init(t, flux, unc, args)

    if stage in ('fit', 'all'):
        stage_fit(t, flux, unc, args)

    if stage in ('plot', 'all'):
        stage_plot(t, flux, unc, args)

if __name__ == '__main__':
    main()