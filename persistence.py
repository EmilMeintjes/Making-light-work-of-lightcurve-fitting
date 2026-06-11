"""
persistence.py
--------------
Canonical persistence layer for nova-like lightcurve fitting.

JSON file format
----------------
The regions file now uses a top-level object rather than a bare list:

    {
      "metadata": {
        "xscale": "linear",
        "yscale": "log"
      },
      "regions": [ {region}, ... ]
    }

This makes the file self-describing: every downstream module reads the
axis scales from the same source of truth rather than requiring them to
be passed explicitly at every call site.

MCMC results
------------
Stored as plain numpy .npy arrays + a summary JSON — no extra dependencies.

    <results_dir>/seg<ID>_samples.npy    (n_draws, n_params) float64
    <results_dir>/seg<ID>_lnprob.npy     (n_draws,) float64  [optional]
    <results_dir>/seg<ID>_summary.json   median, p16, p84 per parameter

Typical usage
-------------
    from persistence import RegionStore, save_mcmc_results, load_mcmc_results

    store = RegionStore('regions.json', xscale='linear', yscale='log')
    store.load()

    region = store.get(segment_id=3)
    store.update_guesses(segment_id=3, guesses={...}, priors={...})

    save_mcmc_results('results/', segment_id=3, samples=..., param_names=...)
    summary = load_mcmc_results('results/', segment_id=3)
"""

#System imports
import json
import os

#External imports
import numpy as np

# ---------------------------------------------------------------------------
# Valid axis scale values
# ---------------------------------------------------------------------------

_VALID_SCALES = {'linear', 'log'}


def _check_scale(name, value):
    if value not in _VALID_SCALES:
        raise ValueError(
            f"{name} must be 'linear' or 'log', got '{value}'"
        )

# ---------------------------------------------------------------------------
# Low-level JSON helpers
# ---------------------------------------------------------------------------

def load_regions_json(path):

    """
    Load the regions file at *path*.

    Returns a tuple (metadata, regions) where:
        metadata : dict   Top-level metadata (xscale, yscale, ...).
        regions  : list   List of region dicts.

    Returns ({}, []) if the file does not exist.
    Raises ValueError if the file exists but cannot be parsed.
    """

    if not os.path.exists(path):
        return {}, []
    try:
        with open(path, 'r') as fh:
            data = json.load(fh)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Could not parse region file '{path}': {exc}"
        ) from exc

    # Support old flat-list format (pre-metadata) — treat as empty metadata
    if isinstance(data, list):
        return {}, data

    metadata = data.get('metadata', {})
    regions  = data.get('regions',  [])
    return metadata, regions


def save_regions_json(path, regions, metadata=None):

    """
    Write regions and metadata to *path* as pretty-printed JSON.

    Parameters
    ----------
    path     : str
    regions  : list of dict
    metadata : dict or None   If None, writes an empty metadata block.
    """

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    payload = {
        'metadata': metadata or {},
        'regions':  regions,
    }
    with open(path, 'w') as fh:
        json.dump(payload, fh, indent=2)


# ---------------------------------------------------------------------------
# RegionStore
# ---------------------------------------------------------------------------

_REQUIRED_KEYS = {'segment_id', 'start', 'end', 'model'}
_OPTIONAL_KEYS = {'note', 'initial_guesses', 'curvefit_result', 'priors'}


class RegionStore:

    """
    Managed, file-backed store for fit region definitions.

    Parameters
    ----------
    path   : str   Path to the JSON file.  Created on first save if absent.
    xscale : str   'linear' or 'log' — stored in the file metadata.
    yscale : str   'linear' or 'log' — stored in the file metadata.

    If the file already exists and contains scale metadata, the stored values
    take precedence over the constructor arguments so that reloading a session
    does not silently override a previous choice.  Pass xscale/yscale
    explicitly only when creating a new session.

    Attributes
    ----------
    path    : str
    xscale  : str
    yscale  : str
    regions : list of dict
    """

    def __init__(self, path, xscale='linear', yscale='linear'):
        _check_scale('xscale', xscale)
        _check_scale('yscale', yscale)
        self.path    = os.path.abspath(path)
        self.xscale  = xscale
        self.yscale  = yscale
        self.regions = []

    # ------------------------------------------------------------------
    # Load / save
    # ------------------------------------------------------------------

    def load(self):

        """
        Load from the JSON file into self.regions (and update scale attrs).

        Safe to call on a non-existent file — leaves regions=[] and keeps
        the constructor-supplied scale values.
        """

        metadata, regions = load_regions_json(self.path)
        self.regions = regions
        # Stored scales take precedence over constructor defaults
        if 'xscale' in metadata:
            self.xscale = metadata['xscale']
        if 'yscale' in metadata:
            self.yscale = metadata['yscale']
        self._validate_all()
        return self.regions

    def save(self):

        """
        Persist regions + metadata to the JSON file immediately.
        """

        metadata = {'xscale': self.xscale, 'yscale': self.yscale}
        save_regions_json(self.path, self.regions, metadata=metadata)

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def get(self, segment_id):

        """
        Return the region dict for *segment_id*.  Raises KeyError if absent.
        """

        for r in self.regions:
            if r['segment_id'] == segment_id:
                return r
        raise KeyError(f"No region with segment_id={segment_id}")

    def ids(self):

        """
        Return a sorted list of all segment IDs.
        """

        return sorted(r['segment_id'] for r in self.regions)

    def __len__(self):
        return len(self.regions)

    def __iter__(self):
        return iter(self.regions)

    def __repr__(self):
        return (
            f"RegionStore(path='{self.path}', "
            f"xscale='{self.xscale}', yscale='{self.yscale}', "
            f"n_regions={len(self.regions)}, ids={self.ids()})"
        )

    # ------------------------------------------------------------------
    # Mutations — all persist immediately
    # ------------------------------------------------------------------

    def add(self, region):

        """
        Add *region* to the store and save.

        The dict must contain all _REQUIRED_KEYS; optional keys are filled
        with defaults if absent.  Returns the stored dict.
        """

        self._validate_one(region)
        region = self._fill_defaults(region)
        self.regions.append(region)
        self.save()
        return region

    def remove(self, segment_id):

        """
        Remove the region with *segment_id* and save.  Raises KeyError if absent.
        """

        before = len(self.regions)
        self.regions = [r for r in self.regions if r['segment_id'] != segment_id]
        if len(self.regions) == before:
            raise KeyError(f"No region with segment_id={segment_id}")
        self.save()

    def undo_last(self):

        """
        Remove and return the most recently added region, then save.
        Returns None if the store was already empty.
        """

        if not self.regions:
            return None
        removed = self.regions.pop()
        self.save()
        return removed

    def update_guesses(self, segment_id, guesses, priors=None, curvefit_result=None):

        """
        Store initial parameter guesses (and optionally priors / curve_fit
        result) for *segment_id*.

        Parameters
        ----------
        guesses         : dict  {param_name: float}
        priors          : dict or None
            {param_name: {'lower': float, 'upper': float}}
        curvefit_result : dict or None
            {'popt': list, 'pcov': list-of-lists, 'success': bool}
        """

        r = self.get(segment_id)
        r['initial_guesses'] = guesses
        if priors is not None:
            r['priors'] = priors
        if curvefit_result is not None:
            r['curvefit_result'] = curvefit_result
        self.save()

    def set_scales(self, xscale=None, yscale=None):

        """
        Update the stored axis scales and save.

        Useful for changing the scale mid-session without reloading.
        """

        if xscale is not None:
            _check_scale('xscale', xscale)
            self.xscale = xscale
        if yscale is not None:
            _check_scale('yscale', yscale)
            self.yscale = yscale
        self.save()

    def next_id(self):

        """
        Return the next available integer segment ID.
        """
        if not self.regions:
            return 1
        return max(r['segment_id'] for r in self.regions) + 1

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate_one(self, region):
        missing = _REQUIRED_KEYS - set(region.keys())
        if missing:
            raise ValueError(
                f"Region dict is missing required keys: {missing}  "
                f"Got: {set(region.keys())}"
            )
        if not isinstance(region['segment_id'], int):
            raise TypeError(
                f"segment_id must be int, got {type(region['segment_id'])}"
            )

    def _validate_all(self):
        for i, r in enumerate(self.regions):
            try:
                self._validate_one(r)
            except (ValueError, TypeError) as exc:
                raise ValueError(
                    f"Region at index {i} in '{self.path}' is invalid: {exc}"
                ) from exc

    @staticmethod
    def _fill_defaults(region):
        region.setdefault('note',            '')
        region.setdefault('initial_guesses', {})
        region.setdefault('curvefit_result', None)
        region.setdefault('priors',          {})
        return region


# ---------------------------------------------------------------------------
# MCMC results persistence
# ---------------------------------------------------------------------------

def _results_stem(results_dir, segment_id):
    return os.path.join(os.path.abspath(results_dir), f'seg{segment_id:04d}')


def save_mcmc_results(results_dir, segment_id, samples, param_names, lnprob=None, metadata=None):

    """
    Persist MCMC posterior samples and a summary statistics file.

    Files written:
        seg<ID>_samples.npy    (n_draws, n_params)
        seg<ID>_lnprob.npy     (n_draws,)            [if lnprob provided]
        seg<ID>_summary.json   median + 16th/84th percentiles

    Parameters
    ----------
    results_dir  : str
    segment_id   : int
    samples      : array-like   Flattened chain, shape (n_draws, n_params).
    param_names  : list of str
    lnprob       : array-like or None
    metadata     : dict or None   Extra info (model, bounds, n_walkers, etc.).

    Returns
    -------
    summary : dict
    """

    os.makedirs(os.path.abspath(results_dir), exist_ok=True)
    stem    = _results_stem(results_dir, segment_id)
    samples = np.asarray(samples, dtype=float)

    if samples.ndim != 2:
        raise ValueError(
            f"samples must be 2-D (n_draws, n_params), got shape {samples.shape}"
        )
    if samples.shape[1] != len(param_names):
        raise ValueError(
            f"samples has {samples.shape[1]} columns but param_names has "
            f"{len(param_names)} entries."
        )

    summary = {
        'segment_id':  segment_id,
        'param_names': param_names,
        'n_draws':     int(samples.shape[0]),
        'statistics':  {},
    }
    for i, name in enumerate(param_names):
        col = samples[:, i]
        lo, med, hi = np.percentile(col, [16, 50, 84])
        summary['statistics'][name] = {
            'median':  float(med),
            'p16':     float(lo),
            'p84':     float(hi),
            'err_lo':  float(med - lo),
            'err_hi':  float(hi - med),
        }
    if metadata:
        summary['metadata'] = metadata

    npy_path = stem + '_samples.npy'
    np.save(npy_path, samples)
    print(f"[persistence] Samples saved  -> {npy_path}")

    if lnprob is not None:
        lnp_path = stem + '_lnprob.npy'
        np.save(lnp_path, np.asarray(lnprob, dtype=float))
        print(f"[persistence] lnprob  saved  -> {lnp_path}")

    summary_path = stem + '_summary.json'
    with open(summary_path, 'w') as fh:
        json.dump(summary, fh, indent=2)
    print(f"[persistence] Summary saved  -> {summary_path}")

    return summary


def load_mcmc_results(results_dir, segment_id):

    """
    Load posterior samples and summary for *segment_id*.

    Returns
    -------
    dict with keys: 'samples', 'param_names', 'lnprob', 'summary'
    """

    stem = _results_stem(results_dir, segment_id)

    summary_path = stem + '_summary.json'
    if not os.path.exists(summary_path):
        raise FileNotFoundError(
            f"No summary file found for segment_id={segment_id} "
            f"in '{results_dir}'  (expected: {summary_path})"
        )
    with open(summary_path, 'r') as fh:
        summary = json.load(fh)

    npy_path = stem + '_samples.npy'
    if not os.path.exists(npy_path):
        raise FileNotFoundError(
            f"No sample file found for segment_id={segment_id} "
            f"in '{results_dir}'  (expected: {npy_path})"
        )
    samples = np.load(npy_path)

    lnp_path = stem + '_lnprob.npy'
    lnprob   = np.load(lnp_path) if os.path.exists(lnp_path) else None

    return {
        'samples':     samples,
        'param_names': summary['param_names'],
        'lnprob':      lnprob,
        'summary':     summary,
    }


def print_summary(results_dir, segment_id):

    """
    Pretty-print median +/- 1-sigma for a segment (reads summary JSON only).
    """

    stem         = _results_stem(results_dir, segment_id)
    summary_path = stem + '_summary.json'
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"Summary file not found: {summary_path}")

    with open(summary_path, 'r') as fh:
        summary = json.load(fh)

    print(f"\nMCMC summary - segment #{segment_id} "
          f"({summary.get('n_draws', '?')} draws)")
    print(f"{'Parameter':<20}  {'Median':>12}  {'-1sigma':>10}  {'+1sigma':>10}")
    print("-" * 58)
    for name, s in summary['statistics'].items():
        print(
            f"{name:<20}  {s['median']:>12.6g}"
            f"  {s['err_lo']:>10.4g}  {s['err_hi']:>10.4g}"
        )
    if 'metadata' in summary:
        print(f"\nMetadata: {summary['metadata']}")