"""Lazy data access layer over the two large training datasets.

The CSVs are large (~140 MB claims / ~480 MB provider), so frames are loaded on
first use and cached. Single-record lookups are served from an index keyed by the
identifier column (``Claim_ID`` / ``provider_npi``).
"""
from __future__ import annotations

import threading

import pandas as pd

from . import config


class DatasetStore:
    """Thread-safe, lazily-loaded access to the claim and provider datasets."""

    def __init__(self):
        self._lock = threading.Lock()
        self._claims: pd.DataFrame | None = None
        self._providers: pd.DataFrame | None = None
        self._claim_index: pd.DataFrame | None = None
        self._provider_index: pd.DataFrame | None = None

    # ------------------------------------------------------------- loaders
    def claims_frame(self) -> pd.DataFrame:
        with self._lock:
            if self._claims is None:
                self._claims = pd.read_csv(config.CLAIM_DATA_PATH)
            return self._claims

    def providers_frame(self) -> pd.DataFrame:
        with self._lock:
            if self._providers is None:
                self._providers = pd.read_csv(config.PROVIDER_DATA_PATH)
            return self._providers

    # ------------------------------------------------------------- lookups
    def _claim_indexed(self) -> pd.DataFrame:
        with self._lock:
            if self._claim_index is None:
                frame = self.claims_frame()
                self._claim_index = frame.set_index(config.CLAIM_ID_COLUMN, drop=False)
            return self._claim_index

    def _provider_indexed(self) -> pd.DataFrame:
        with self._lock:
            if self._provider_index is None:
                frame = self.providers_frame()
                self._provider_index = frame.set_index(config.PROVIDER_ID_COLUMN, drop=False)
            return self._provider_index

    def get_claim(self, claim_id) -> pd.Series | None:
        idx = self._claim_indexed()
        try:
            row = idx.loc[claim_id]
        except KeyError:
            return None
        # A duplicated id returns a DataFrame; take the first match.
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        return row

    def get_provider(self, provider_npi) -> pd.Series | None:
        idx = self._provider_indexed()
        try:
            row = idx.loc[provider_npi]
        except KeyError:
            return None
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        return row

    # ------------------------------------------------------------- samples
    def claim_sample(self, n: int) -> pd.DataFrame:
        # Read only the first n rows to avoid loading the full ~140 MB file.
        return pd.read_csv(config.CLAIM_DATA_PATH, nrows=n)

    def provider_sample(self, n: int) -> pd.DataFrame:
        # Read only the first n rows to avoid loading the full ~480 MB file.
        return pd.read_csv(config.PROVIDER_DATA_PATH, nrows=n)


# Module-level singleton.
store = DatasetStore()
