"""Download and cache NFL regular-season player statistics from nflverse.

Source
------
nflverse-data, release tag ``stats_player`` (files ``stats_player_reg_<season>.csv``),
produced with ``nflfastR::calculate_stats()`` from NFL play-by-play data.

    https://github.com/nflverse/nflverse-data/releases/tag/stats_player

Licence: the nflverse-data repository is released under Creative Commons
Attribution 4.0 International (CC BY 4.0),
https://github.com/nflverse/nflverse-data/blob/master/LICENSE.md .
Attribution: "Data: nflverse (nflverse-data, stats_player release), CC BY 4.0".

Design
------
* Files are downloaded once into a local cache directory (``data/nflverse`` by
  default; ``*.csv`` is git-ignored in this repository) and re-used afterwards.
* Every file is verified against a pinned SHA-256 digest recorded when the
  chapter was written. nflverse occasionally re-publishes release assets with
  corrections; if that happens the loader raises instead of silently returning
  different data, because the numbers quoted in the chapter would no longer be
  reproducible. Pass ``verify=False`` to accept the newer upstream files
  deliberately.
* There is no synthetic fallback: a failed download raises ``RuntimeError``.
"""

from __future__ import annotations

import hashlib
import logging
import shutil
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, Iterable, Optional

import pandas as pd

logger = logging.getLogger(__name__)

RELEASE_URL = "https://github.com/nflverse/nflverse-data/releases/tag/stats_player"
ASSET_URL_TEMPLATE = (
    "https://github.com/nflverse/nflverse-data/releases/download/"
    "stats_player/stats_player_reg_{season}.csv"
)
LICENSE_NAME = "CC BY 4.0"
LICENSE_URL = "https://github.com/nflverse/nflverse-data/blob/master/LICENSE.md"

#: SHA-256 digests of the release assets as downloaded on 2026-10-19 for
#: Chapter 7. Seasons not listed here are downloaded but not verified.
PINNED_SHA256: Dict[int, str] = {
    2012: "6cdfbf3b291cf1349b33fe8e90542d8ae54cca3726769866f94feaf866b329b5",
    2013: "342fd27eef619a2aec4a74abf8f689e6b14fab526ae1300095b2654a3981882b",
    2014: "9f7aee274dddd6b27ec66ae493507aebf5779ed81d4735a53b87bbf28030f7a1",
    2015: "91cc6edbfe16bdc08c680c58ea6d8525c6b7668ef9b94ebdcc6ef9156fcd92d0",
    2016: "17063fa01736c41b30ffbb15a284f64611fd8a517bfae5d4cdb0ebb19da78801",
    2017: "ee3ed8e1b9f7998b7b7b43ec1139e1d69186818ad3e2ce100e4467f163f1ca3c",
    2018: "8d66bc2dde26951038f6cb68edfcc9082c36423ec7df8185aa7b1b61906c866b",
    2019: "324b77e6027839c15bad5d32a6d83d07f998e71e0a1e682a37543efeb5219064",
    2020: "ac90bb97fd0cb8f8ae7783226c50f5db64bb375e99eaedbd32e97b8b16cf1a94",
    2021: "2edb2190fa41c92eed8e997c0ba1ff6c737a0fe6d996129487fe34cb63264c6f",
    2022: "af2d534b208af43d4b17bcba242dc949f2d40b367253f78c882c19eee9dfc937",
    2023: "0cc79526a678907817713a7f603b75c4424a1cd4dfc98daa2b3018af80312b9c",
    2024: "fbad4e9f1b892e3870a51a4eea808d692cde14ed6dbacd67cce3a0def8ddcbaa",
    2025: "aaa8559478dd8d58569c93e278239e70e9411e79ee649c99d8d506030cd42b12",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, dest: Path, timeout: float = 60.0) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "FrontierML-textbook/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response, tmp.open("wb") as fh:
            shutil.copyfileobj(response, fh)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(
            f"Could not download {url}: {exc}. This chapter needs the real nflverse "
            "data and deliberately has no synthetic fallback. Check your network "
            "connection or download the file manually into the cache directory."
        ) from exc
    tmp.replace(dest)


def fetch_player_stats(
    seasons: Iterable[int],
    cache_dir: str | Path = "../data/nflverse",
    verify: bool = True,
) -> Dict[int, Path]:
    """Ensure ``stats_player_reg_<season>.csv`` exists locally for each season.

    Returns a mapping season -> local path. Raises ``RuntimeError`` if a
    download fails and ``ValueError`` if a pinned checksum does not match.
    """
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    paths: Dict[int, Path] = {}
    for season in seasons:
        dest = cache / f"stats_player_reg_{season}.csv"
        if not dest.exists():
            url = ASSET_URL_TEMPLATE.format(season=season)
            print(f"Downloading {url}")
            _download(url, dest)
        digest = _sha256(dest)
        expected: Optional[str] = PINNED_SHA256.get(season)
        if verify and expected is not None and digest != expected:
            raise ValueError(
                f"{dest} has SHA-256 {digest}, but the chapter was written with {expected}.\n"
                "nflverse periodically rebuilds its release files, so the downloaded data "
                "differ from the data behind the numbers quoted in the text.\n"
                "To continue with the new files, call load_receiving_seasons(..., verify=False) "
                "in the data-loading cell (or set PINNED_SHA256 = {} in utils/nfl_data.py). "
                "Results, and therefore some numbers quoted in the markdown, may then differ.\n"
                "If you suspect a corrupted download instead, delete the file and rerun to "
                "download it again."
            )
        paths[season] = dest
    return paths


def load_receiving_seasons(
    seasons: Iterable[int],
    cache_dir: str | Path = "../data/nflverse",
    verify: bool = True,
) -> pd.DataFrame:
    """Load regular-season player rows for the given seasons as one DataFrame."""
    seasons = list(seasons)
    paths = fetch_player_stats(seasons, cache_dir=cache_dir, verify=verify)
    frames = [pd.read_csv(paths[s], low_memory=False) for s in seasons]
    return pd.concat(frames, ignore_index=True)


def file_digests(seasons: Iterable[int], cache_dir: str | Path = "../data/nflverse") -> Dict[int, str]:
    """Return SHA-256 digests of cached files (for recording provenance)."""
    cache = Path(cache_dir)
    return {s: _sha256(cache / f"stats_player_reg_{s}.csv") for s in seasons}
