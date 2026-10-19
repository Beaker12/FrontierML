"""
Web scraping utilities for FrontierML.

This module provides helper functions for collecting real-world data from various sources
including web scraping, API interactions, and data preprocessing.

References:
    - Lawson, R. (2015). Web scraping with Python. Packt Publishing.
    - Richardson, L. (2007). Beautiful Soup Documentation. 
      https://www.crummy.com/software/BeautifulSoup/

Author: FrontierML Team
Date: 2025-08-12
"""

import requests
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from typing import Dict, List, Optional, Union, Any
import time
import logging
from urllib.parse import urljoin, urlparse
import json
from pathlib import Path
import io

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WebScraper:
    """
    A comprehensive web scraping utility class.
    
    This class provides methods for scraping data from websites with proper
    error handling, rate limiting, and data cleaning capabilities following
    ethical scraping practices as outlined by Lawson (2015).
    
    Parameters
    ----------
    delay : float, default=1.0
        Delay between requests in seconds to be respectful to servers
    timeout : int, default=30
        Request timeout in seconds
        
    Attributes
    ----------
    session : requests.Session
        HTTP session for maintaining connection state
        
    References
    ----------
    Lawson, R. (2015). Web scraping with Python. Packt Publishing.
    """
    
    def __init__(self, delay: float = 1.0, timeout: int = 30):
        """
        Initialize the WebScraper.
        
        Parameters
        ----------
        delay : float, default=1.0
            Delay between requests in seconds to be respectful to servers.
        timeout : int, default=30
            Request timeout in seconds.
        """
        self.delay = delay
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def get_page(self, url: str) -> Optional[BeautifulSoup]:
        """
        Fetch and parse a web page.
        
        Parameters
        ----------
        url : str
            URL to fetch.
            
        Returns
        -------
        BeautifulSoup or None
            Parsed HTML content or None if failed.
        """
        try:
            logger.info(f"Fetching: {url}")
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            
            # Add delay to be respectful
            time.sleep(self.delay)
            
            return BeautifulSoup(response.content, 'html.parser')
            
        except requests.RequestException as e:
            logger.error(f"Error fetching {url}: {e}")
            return None
    
    def extract_table(self, soup: BeautifulSoup, table_selector: str = 'table') -> pd.DataFrame:
        """
        Extract table data from HTML.
        
        Parameters
        ----------
        soup : BeautifulSoup
            Parsed HTML content.
        table_selector : str, default='table'
            CSS selector for the table.
            
        Returns
        -------
        pd.DataFrame
            Extracted table data.
        """
        tables = soup.select(table_selector)
        if not tables:
            logger.warning("No tables found with the given selector")
            return pd.DataFrame()
        
        # Use pandas to parse HTML tables
        table_html = str(tables[0])
        df = pd.read_html(table_html)[0]
        
        return df
    
    def extract_links(self, soup: BeautifulSoup, base_url: str, 
                     link_selector: str = 'a') -> List[str]:
        """
        Extract all links from a page.
        
        Parameters
        ----------
        soup : BeautifulSoup
            Parsed HTML content.
        base_url : str
            Base URL for resolving relative links.
        link_selector : str, default='a'
            CSS selector for links.
            
        Returns
        -------
        List[str]
            List of absolute URLs.
        """
        links = []
        for link in soup.select(link_selector):
            href = link.get('href')
            if href:
                absolute_url = urljoin(base_url, href)
                links.append(absolute_url)
        
        return list(set(links))  # Remove duplicates


class APIClient:
    """
    A utility class for interacting with web APIs.
    
    This class provides methods for making API requests with proper error handling,
    authentication, and data processing.
    """
    
    def __init__(self, base_url: str, api_key: Optional[str] = None, 
                 rate_limit: float = 1.0):
        """
        Initialize the API client.
        
        Parameters
        ----------
        base_url : str
            Base URL for the API.
        api_key : str, optional
            API key for authentication.
        rate_limit : float, default=1.0
            Minimum time between requests in seconds.
        """
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.rate_limit = rate_limit
        self.last_request_time = 0
        
        self.session = requests.Session()
        if api_key:
            self.session.headers.update({'Authorization': f'Bearer {api_key}'})
    
    def get(self, endpoint: str, params: Optional[Dict] = None) -> Optional[Dict]:
        """
        Make a GET request to the API.
        
        Parameters
        ----------
        endpoint : str
            API endpoint (without base URL).
        params : dict, optional
            Query parameters.
            
        Returns
        -------
        dict or None
            JSON response or None if failed.
        """
        # Rate limiting
        time_since_last = time.time() - self.last_request_time
        if time_since_last < self.rate_limit:
            time.sleep(self.rate_limit - time_since_last)
        
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        
        try:
            logger.info(f"API request: {url}")
            response = self.session.get(url, params=params)
            response.raise_for_status()
            
            self.last_request_time = time.time()
            
            return response.json()
            
        except requests.RequestException as e:
            logger.error(f"API request failed: {e}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            return None


def clean_numeric_data(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    Clean numeric columns in a DataFrame.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    columns : List[str]
        List of column names to clean.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with cleaned numeric columns.
    """
    df_clean = df.copy()
    
    for col in columns:
        if col in df_clean.columns:
            # Remove common non-numeric characters
            df_clean[col] = df_clean[col].astype(str).str.replace(r'[,$%\s]', '', regex=True)
            
            # Convert to numeric, coercing errors to NaN
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    
    return df_clean


def save_data(data: Union[pd.DataFrame, Dict], filepath: Union[str, Path], 
              format: str = 'csv') -> None:
    """
    Save data to file in various formats.
    
    Parameters
    ----------
    data : pd.DataFrame or dict
        Data to save.
    filepath : str or Path
        Output file path.
    format : str, default='csv'
        Output format ('csv', 'json', 'excel').
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    if format.lower() == 'csv' and isinstance(data, pd.DataFrame):
        data.to_csv(filepath, index=False)
    elif format.lower() == 'json':
        if isinstance(data, pd.DataFrame):
            data.to_json(filepath, orient='records', indent=2)
        else:
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
    elif format.lower() == 'excel' and isinstance(data, pd.DataFrame):
        data.to_excel(filepath, index=False)
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    logger.info(f"Data saved to: {filepath}")


def load_data(filepath: Union[str, Path]) -> Union[pd.DataFrame, Dict]:
    """
    Load data from file.
    
    Parameters
    ----------
    filepath : str or Path
        Input file path.
        
    Returns
    -------
    pd.DataFrame or dict
        Loaded data.
    """
    filepath = Path(filepath)
    
    if filepath.suffix.lower() == '.csv':
        return pd.read_csv(filepath)
    elif filepath.suffix.lower() == '.json':
        with open(filepath, 'r') as f:
            return json.load(f)
    elif filepath.suffix.lower() in ['.xlsx', '.xls']:
        return pd.read_excel(filepath)
    else:
        raise ValueError(f"Unsupported file format: {filepath.suffix}")


def scrape_nfl_wr_stats(year: int, max_players: int = 40) -> Optional[pd.DataFrame]:
    """
    Scrape NFL WR statistics for a specific year using WebScraper.
    
    Parameters
    ----------
    year : int
        Season year to scrape
    max_players : int, default=40
        Maximum number of top WRs to collect
        
    Returns
    -------
    pd.DataFrame or None
        DataFrame with WR statistics or None if scraping fails
        
    References
    ----------
    Data source: Pro Football Reference (https://www.pro-football-reference.com/)
    """
    try:
        # Initialize scraper with respectful delay
        scraper = WebScraper(delay=2.0, timeout=30)
        
        # Pro Football Reference URL for receiving stats
        url = f"https://www.pro-football-reference.com/years/{year}/receiving.htm"
        
        logger.info(f"Scraping {year} WR stats from: {url}")
        soup = scraper.get_page(url)
        
        if soup is None:
            logger.error(f"Failed to fetch page for {year}")
            return None
        
        # Extract the main receiving stats table
        stats_table = soup.find('table', {'id': 'receiving'})
        if stats_table is None:
            logger.error(f"No receiving stats table found for {year}")
            return None
        
        # Convert table to DataFrame
        df = pd.read_html(str(stats_table))[0]
        
        # Clean up the DataFrame
        df = df.dropna(subset=['Player'])  # Remove empty rows
        df = df[df['Player'] != 'Player']  # Remove header rows that repeat
        
        # Filter for Wide Receivers
        if 'Pos' in df.columns:
            df = df[df['Pos'].str.contains('WR', na=False)]
        
        # Rename columns to match our schema
        column_mapping = {
            'Player': 'player_name',
            'Pos': 'position', 
            'Rec': 'receptions',
            'Tgt': 'targets',
            'Yds': 'receiving_yards',
            'Y/R': 'yards_per_reception',
            'Lng': 'longest_reception',
            'TD': 'receiving_tds',
            '1D': 'first_downs',
            'Ctch%': 'catch_rate',
            'Y/Tgt': 'yards_per_target'
        }
        
        # Apply column mapping where columns exist
        df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})
        
        # Add season column
        df['season'] = year
        
        # Clean numeric columns
        numeric_cols = ['receptions', 'targets', 'receiving_yards', 'yards_per_reception',
                       'longest_reception', 'receiving_tds', 'first_downs', 'catch_rate', 
                       'yards_per_target']
        
        df = clean_numeric_data(df, numeric_cols)
        
        # Calculate additional performance metrics
        df['performance_score'] = (
            df['receiving_yards'] * 0.02 + 
            df['receiving_tds'] * 6 + 
            df['receptions'] * 0.5
        )
        
        df['receptions_per_game'] = df['receptions'] / 17  # NFL season length
        df['yards_per_game'] = df['receiving_yards'] / 17
        
        # Binary features for neural network analysis
        df['high_volume'] = (df['receptions'] > 80).astype(int)
        df['big_play_threat'] = (df['longest_reception'] > 40).astype(int) 
        df['red_zone_threat'] = (df['receiving_tds'] > 6).astype(int)
        df['elite_efficiency'] = (df['yards_per_reception'] > 15).astype(int)
        
        # Filter out rows with insufficient data
        df = df.dropna(subset=['receptions', 'receiving_yards'])
        df = df[df['receptions'] > 0]  # Must have at least 1 reception
        
        # Sort by performance and take top players
        df = df.sort_values('performance_score', ascending=False).head(max_players)
        
        logger.info(f"Successfully scraped {len(df)} WR players for {year}")
        return df
        
    except Exception as e:
        logger.error(f"Failed to scrape data for {year}: {e}")
        return None


def scrape_multi_year_nfl_data(start_year: int = 2019, num_years: int = 5, 
                              players_per_year: int = 40) -> Optional[pd.DataFrame]:
    """
    Scrape NFL WR data across multiple years using WebScraper.
    
    Parameters
    ----------
    start_year : int, default=2019
        Starting year for scraping
    num_years : int, default=5
        Number of years to scrape
    players_per_year : int, default=40
        Target number of players per year
        
    Returns
    -------
    pd.DataFrame or None
        Combined DataFrame with all years or None if scraping fails
        
    References
    ----------
    Data source: Pro Football Reference (https://www.pro-football-reference.com/)
    """
    all_data = []
    years = list(range(start_year, start_year + num_years))
    
    for year in years:
        logger.info(f"Scraping data for {year}...")
        year_data = scrape_nfl_wr_stats(year, players_per_year)
        
        if year_data is not None and not year_data.empty:
            all_data.append(year_data)
            logger.info(f"Collected {len(year_data)} players for {year}")
        else:
            logger.warning(f"Failed to get data for {year}")
        
        # Be respectful to the server between years
        time.sleep(3)
    
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        logger.info(f"Total players scraped: {len(combined_df)} across {len(years)} seasons")
        return combined_df
    else:
        logger.error("No data could be scraped from any year")
        return None


def download_nyc_taxi_data(year: int = 2024, month: int = 1, 
                          taxi_type: str = 'yellow', 
                          sample_size: Optional[int] = 10000,
                          save_path: Optional[str] = None,
                          data_type: str = 'trips') -> pd.DataFrame:
    """
    Download NYC TLC taxi trip record data from the official dataset.

    ``data_type='zones'`` returns the taxi-zone lookup table (see
    :func:`get_taxi_zone_data`) instead of trip records; any other value than
    ``'trips'`` or ``'zones'`` raises ``ValueError``. For cached, provenance-
    recording access prefer :func:`load_tlc_trip_records`,
    :func:`load_taxi_zone_lookup` and :func:`load_taxi_zone_geometry`.

    Note that TLC trip files from mid-2016 onwards contain zone IDs
    (``PULocationID``/``DOLocationID``) rather than latitude/longitude.
    
    This function downloads parquet files from the NYC Taxi and Limousine Commission
    public dataset hosted on AWS S3. Data includes pickup/dropoff locations,
    timestamps, fares, and trip details.
    
    Parameters
    ----------
    year : int, default=2024
        Year of taxi data to download (2009-2025)
    month : int, default=1
        Month of taxi data to download (1-12)
    taxi_type : str, default='yellow'
        Type of taxi data ('yellow', 'green', 'fhv', 'fhvhv')
    sample_size : int or None, default=10000
        Number of random samples to return. If None, returns full dataset
    save_path : str or None, default=None
        Path to save the downloaded data. If None, saves to data/raw/
        
    Returns
    -------
    pd.DataFrame
        Taxi trip data with columns including pickup/dropoff locations,
        timestamps, passenger count, trip distance, and fare information
        
    References
    ----------
    NYC Taxi and Limousine Commission (TLC) Trip Record Data was accessed
    from https://registry.opendata.aws/nyc-tlc-trip-records-pds/
    
    Notes
    -----
    Data schema varies slightly by year and taxi type. Function handles
    common column standardization for clustering analysis.
    """
    if data_type == 'zones':
        return get_taxi_zone_data()
    if data_type != 'trips':
        raise ValueError(f"data_type must be 'trips' or 'zones', got {data_type!r}")

    # Construct URL for NYC TLC data
    month_str = f"{month:02d}"
    filename = f"{taxi_type}_tripdata_{year}-{month_str}.parquet"
    base_url = "https://d37ci6vzurychx.cloudfront.net/trip-data/"
    url = base_url + filename
    
    logger.info(f"Downloading {taxi_type} taxi data for {year}-{month_str}")
    logger.info(f"URL: {url}")
    
    try:
        # Download parquet file directly into pandas
        response = requests.get(url, timeout=300)
        response.raise_for_status()
        
        # Try to read parquet data from bytes
        try:
            df = pd.read_parquet(io.BytesIO(response.content))
        except Exception as parquet_error:
            logger.warning(f"Parquet reading failed: {parquet_error}")
            # Fallback: save to temp file and read
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as tmp_file:
                tmp_file.write(response.content)
                tmp_file.flush()
                df = pd.read_parquet(tmp_file.name)
                
        logger.info(f"Successfully downloaded {len(df):,} trip records")
        
        # Standardize column names for clustering analysis
        df = _standardize_taxi_columns(df, taxi_type)
        
        # Sample data if requested
        if sample_size is not None and len(df) > sample_size:
            df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
            logger.info(f"Sampled {sample_size:,} records for analysis")
        
        # Save to file if path provided
        if save_path is None:
            save_path = f"../data/raw/nyc_taxi_{taxi_type}_{year}_{month:02d}.csv"
        
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        if save_path.endswith('.parquet'):
            df.to_parquet(save_path)
        else:
            df.to_csv(save_path, index=False)
        logger.info(f"Data saved to {save_path}")
        
        return df
        
    except requests.RequestException as e:
        logger.error(f"Failed to download data: {e}")
        raise
    except Exception as e:
        logger.error(f"Error processing data: {e}")
        raise


def _standardize_taxi_columns(df: pd.DataFrame, taxi_type: str) -> pd.DataFrame:
    """
    Standardize column names across different taxi data schemas.
    
    Parameters
    ----------
    df : pd.DataFrame
        Raw taxi data
    taxi_type : str
        Type of taxi data for schema detection
        
    Returns
    -------
    pd.DataFrame
        Data with standardized column names for clustering analysis
    """
    # Create column mapping for different schemas
    column_mappings = {
        'yellow': {
            'tpep_pickup_datetime': 'pickup_datetime',
            'tpep_dropoff_datetime': 'dropoff_datetime',
            'pickup_longitude': 'pickup_longitude',
            'pickup_latitude': 'pickup_latitude', 
            'dropoff_longitude': 'dropoff_longitude',
            'dropoff_latitude': 'dropoff_latitude',
            'PULocationID': 'pickup_location_id',
            'DOLocationID': 'dropoff_location_id'
        },
        'green': {
            'lpep_pickup_datetime': 'pickup_datetime',
            'lpep_dropoff_datetime': 'dropoff_datetime',
            'pickup_longitude': 'pickup_longitude',
            'pickup_latitude': 'pickup_latitude',
            'dropoff_longitude': 'dropoff_longitude', 
            'dropoff_latitude': 'dropoff_latitude',
            'PULocationID': 'pickup_location_id',
            'DOLocationID': 'dropoff_location_id'
        }
    }
    
    # Apply column mapping if available
    if taxi_type in column_mappings:
        mapping = column_mappings[taxi_type]
        existing_cols = {old: new for old, new in mapping.items() if old in df.columns}
        df = df.rename(columns=existing_cols)
    
    # Filter to essential columns for clustering
    essential_cols = []
    for col in ['pickup_datetime', 'dropoff_datetime', 
                'pickup_latitude', 'pickup_longitude',
                'dropoff_latitude', 'dropoff_longitude',
                'pickup_location_id', 'dropoff_location_id',
                'passenger_count', 'trip_distance', 'total_amount']:
        if col in df.columns:
            essential_cols.append(col)
    
    if essential_cols:
        df = df[essential_cols].copy()
    
    # Clean coordinate data
    if 'pickup_latitude' in df.columns and 'pickup_longitude' in df.columns:
        # Remove invalid coordinates (NYC bounds approximately)
        valid_coords = (
            (df['pickup_latitude'].between(40.4, 41.0)) & 
            (df['pickup_longitude'].between(-74.5, -73.5)) &
            (df['dropoff_latitude'].between(40.4, 41.0)) & 
            (df['dropoff_longitude'].between(-74.5, -73.5))
        )
        df = df[valid_coords].copy()
        logger.info(f"Filtered to {len(df):,} records with valid NYC coordinates")
    
    return df


def get_taxi_zone_data() -> pd.DataFrame:
    """
    Download NYC taxi zone lookup table for location mapping.
    
    Returns
    -------
    pd.DataFrame
        Taxi zone lookup data with zone IDs, names, and borough information
        
    References
    ----------
    NYC Taxi Zone Lookup Table from:
    https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv
    """
    url = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv"
    
    try:
        logger.info("Downloading NYC taxi zone lookup table")
        df = pd.read_csv(url)
        logger.info(f"Downloaded {len(df)} taxi zones")
        return df
    except Exception as e:
        logger.error(f"Failed to download taxi zone data: {e}")
        raise


# ---------------------------------------------------------------------------
# Cached, provenance-recording loaders for NYC TLC data (used by Chapter 8).
#
# These functions never fall back to synthetic data: any download, parsing or
# validation problem raises an exception.
# ---------------------------------------------------------------------------

TLC_TRIP_BASE_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data/"
TLC_ZONE_LOOKUP_URL = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv"
TLC_ZONE_SHAPEFILE_URL = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zones.zip"
TLC_TERMS_URL = "https://www.nyc.gov/home/terms-of-use.page"
TLC_DATA_PAGE_URL = "https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page"


def fetch_cached(url: str, cache_dir: Union[str, Path], filename: Optional[str] = None,
                 timeout: int = 600) -> Path:
    """
    Download ``url`` into ``cache_dir`` once and record its provenance.

    The first download writes ``<filename>.provenance.json`` next to the file
    with the URL, UTC access timestamp, size in bytes and SHA-256. Later calls
    reuse the cached file and leave the provenance record unchanged, so the
    recorded access time is the time the bytes were actually obtained.

    Raises
    ------
    requests.HTTPError
        If the server returns an error status.
    """
    import hashlib
    from datetime import datetime, timezone

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    filename = filename or Path(urlparse(url).path).name
    if not filename or "/" in filename or filename.startswith("."):
        raise ValueError(f"Unsafe cache filename derived from {url!r}: {filename!r}")
    path = cache_dir / filename
    prov_path = cache_dir / f"{filename}.provenance.json"

    if not (path.exists() and prov_path.exists()):
        logger.info(f"Downloading {url}")
        tmp_path = cache_dir / f"{filename}.part"
        sha = hashlib.sha256()
        n_bytes = 0
        with requests.get(url, stream=True, timeout=timeout) as response:
            response.raise_for_status()
            with open(tmp_path, "wb") as fh:
                for chunk in response.iter_content(chunk_size=1 << 20):
                    fh.write(chunk)
                    sha.update(chunk)
                    n_bytes += len(chunk)
        if n_bytes == 0:
            tmp_path.unlink()
            raise IOError(f"Empty response from {url}")
        tmp_path.replace(path)
        provenance = {
            "url": url,
            "accessed_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "bytes": n_bytes,
            "sha256": sha.hexdigest(),
        }
        prov_path.write_text(json.dumps(provenance, indent=2))
    return path


def read_provenance(path: Union[str, Path]) -> Dict[str, Any]:
    """Return the provenance record written by :func:`fetch_cached` for ``path``."""
    path = Path(path)
    prov_path = path.with_name(f"{path.name}.provenance.json")
    return json.loads(prov_path.read_text())


def load_tlc_trip_records(year: int, month: int, taxi_type: str = "yellow",
                          cache_dir: Union[str, Path] = "../data/raw/tlc",
                          columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Load one monthly TLC trip-record parquet file (cached on first use).

    Parameters
    ----------
    year, month : int
        Month of data, e.g. ``2024, 1`` for ``yellow_tripdata_2024-01.parquet``.
    taxi_type : str
        ``'yellow'``, ``'green'``, ``'fhv'`` or ``'fhvhv'``.
    cache_dir : path
        Directory for the cached parquet file and its provenance record.
    columns : list of str, optional
        Subset of columns to read; a missing column raises an error.

    Returns
    -------
    pd.DataFrame
        Raw trip records, unmodified apart from column selection. The cached
        file path is stored in ``df.attrs['source_path']``.
    """
    if taxi_type not in {"yellow", "green", "fhv", "fhvhv"}:
        raise ValueError(f"Unknown taxi_type {taxi_type!r}")
    filename = f"{taxi_type}_tripdata_{year}-{month:02d}.parquet"
    path = fetch_cached(TLC_TRIP_BASE_URL + filename, cache_dir, filename)
    df = pd.read_parquet(path, columns=columns)
    df.attrs["source_path"] = str(path)
    return df


def load_taxi_zone_lookup(cache_dir: Union[str, Path] = "../data/raw/tlc") -> pd.DataFrame:
    """Load the TLC taxi-zone lookup table (LocationID, Borough, Zone, service_zone)."""
    path = fetch_cached(TLC_ZONE_LOOKUP_URL, cache_dir)
    df = pd.read_csv(path)
    expected = {"LocationID", "Borough", "Zone", "service_zone"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Zone lookup is missing columns {sorted(missing)}")
    df.attrs["source_path"] = str(path)
    return df


def _read_dbf(path: Union[str, Path]) -> pd.DataFrame:
    """Minimal dBASE III reader for the attribute table of a shapefile."""
    import struct

    raw = Path(path).read_bytes()
    n_records, header_len, record_len = struct.unpack("<IHH", raw[4:12])
    fields = []
    pos = 32
    while raw[pos] != 0x0D:
        desc = raw[pos:pos + 32]
        name = desc[:11].split(b"\x00", 1)[0].decode("ascii")
        ftype = chr(desc[11])
        flen = desc[16]
        ndec = desc[17]
        fields.append((name, ftype, flen, ndec))
        pos += 32
    rows = []
    for i in range(n_records):
        start = header_len + i * record_len
        rec = raw[start:start + record_len]
        if rec[:1] == b"*":  # deleted record
            continue
        offset = 1
        row = {}
        for name, ftype, flen, ndec in fields:
            text = rec[offset:offset + flen].decode("latin-1").strip()
            offset += flen
            if ftype in ("N", "F"):
                if text == "":
                    row[name] = np.nan
                elif ftype == "N" and ndec == 0:
                    row[name] = int(text)
                else:
                    row[name] = float(text)
            else:
                row[name] = text
        rows.append(row)
    return pd.DataFrame(rows)


def _read_shp_polygons(path: Union[str, Path]) -> List[List[np.ndarray]]:
    """
    Minimal ESRI shapefile reader for Polygon (type 5) geometries.

    Returns one list of rings (each an ``(m, 2)`` array) per record, in file
    order. Any other shape type raises ``ValueError``.
    """
    import struct

    raw = Path(path).read_bytes()
    file_code = struct.unpack(">i", raw[0:4])[0]
    if file_code != 9994:
        raise ValueError(f"{path} is not a shapefile (file code {file_code})")
    shape_type = struct.unpack("<i", raw[32:36])[0]
    if shape_type != 5:
        raise ValueError(f"Only Polygon shapefiles (type 5) are supported, got {shape_type}")
    pos = 100
    shapes = []
    while pos < len(raw):
        _, content_words = struct.unpack(">ii", raw[pos:pos + 8])
        content = raw[pos + 8:pos + 8 + 2 * content_words]
        pos += 8 + 2 * content_words
        rec_type = struct.unpack("<i", content[0:4])[0]
        if rec_type == 0:
            shapes.append([])
            continue
        if rec_type != 5:
            raise ValueError(f"Unsupported record shape type {rec_type}")
        n_parts, n_points = struct.unpack("<ii", content[36:44])
        parts = list(struct.unpack(f"<{n_parts}i", content[44:44 + 4 * n_parts]))
        pts_start = 44 + 4 * n_parts
        pts = np.frombuffer(content[pts_start:pts_start + 16 * n_points],
                            dtype="<f8").reshape(n_points, 2)
        bounds = parts + [n_points]
        shapes.append([pts[bounds[j]:bounds[j + 1]].copy() for j in range(n_parts)])
    return shapes


def _ring_area_centroid(ring: np.ndarray) -> tuple:
    """Signed shoelace area and first moments (area x centroid) of one ring."""
    x, y = ring[:, 0], ring[:, 1]
    x1, y1 = np.roll(x, -1), np.roll(y, -1)
    cross = x * y1 - x1 * y
    a = cross.sum() / 2.0
    mx = ((x + x1) * cross).sum() / 6.0
    my = ((y + y1) * cross).sum() / 6.0
    return a, mx, my


def load_taxi_zone_geometry(cache_dir: Union[str, Path] = "../data/raw/tlc") -> pd.DataFrame:
    """
    Load TLC taxi-zone polygons and compute areas and centroids in kilometres.

    The TLC shapefile is in NAD83 / New York Long Island state plane
    (EPSG:2263, US survey feet), a conformal projection whose scale error over
    New York City is far below 0.1 %, so Euclidean distances between projected
    coordinates are good approximations of ground distances. Coordinates are
    converted to kilometres with the unit factor read from the ``.prj`` file.

    Zones stored as several shapefile records (same ``LocationID``) are merged.
    The polygon centroid is the area-weighted centroid of all rings (holes
    subtract). As a parsing sanity check, every record's planar area is
    compared with the shapefile's legacy ``Shape_Area`` attribute (square
    degrees, converted at 40.7 N). Shoreline zones have smaller planar areas
    than the legacy value (which evidently predates clipping to the
    shoreline), so the check is one-sided: ``ValueError`` is raised unless the
    median ratio planar/legacy is within 2 % of 1 and no record exceeds its
    legacy area by 2 % or more. A wrong unit factor or a parsing error would
    violate both conditions.

    Returns
    -------
    pd.DataFrame
        One row per LocationID with columns ``LocationID``, ``zone``,
        ``borough``, ``area_km2``, ``x_km``, ``y_km`` and ``rings_km`` (list of
        ``(m, 2)`` arrays, for plotting).
    """
    import re
    import zipfile

    zip_path = fetch_cached(TLC_ZONE_SHAPEFILE_URL, cache_dir)
    extract_dir = Path(cache_dir) / "taxi_zones"
    extract_dir.mkdir(parents=True, exist_ok=True)
    wanted = {".shp", ".dbf", ".prj"}
    found = {}
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            base = Path(info.filename).name
            suffix = Path(base).suffix.lower()
            if suffix in wanted and not base.startswith("."):
                if suffix in found:
                    raise ValueError(f"Multiple {suffix} files in {zip_path}")
                target = extract_dir / f"taxi_zones{suffix}"  # never trust archive paths
                target.write_bytes(zf.read(info))
                found[suffix] = target
    if set(found) != wanted:
        raise ValueError(f"{zip_path} lacks {sorted(wanted - set(found))}")

    prj = found[".prj"].read_text()
    if "Long_Island" not in prj and "Long Island" not in prj:
        raise ValueError(f"Unexpected CRS in taxi_zones.prj: {prj[:200]}")
    units = re.findall(r'UNIT\["([^"]+)",\s*([0-9.]+)\]', prj)
    linear = [u for u in units if "foot" in u[0].lower() or "feet" in u[0].lower()]
    if len(linear) != 1:
        raise ValueError(f"Expected one US-foot linear unit in taxi_zones.prj, found {units}")
    km_per_unit = float(linear[0][1]) / 1000.0

    attrs = _read_dbf(found[".dbf"])
    shapes = _read_shp_polygons(found[".shp"])
    if len(attrs) != len(shapes):
        raise ValueError(f"{len(attrs)} attribute rows but {len(shapes)} shapes")

    # The legacy Shape_Area attribute is in square degrees (it predates the
    # reprojection). Converted with local degree lengths at 40.7 N it should
    # agree with the planar area to within a few per cent; this is a sanity
    # check on the binary parsing and the unit factor, not a precise test.
    phi = np.radians(40.7)
    km2_per_deg2 = ((111.41 * np.cos(phi) - 0.09 * np.cos(3 * phi))     # km per degree longitude
                    * (111.13 - 0.56 * np.cos(2 * phi)))                # km per degree latitude
    area_ratios = []
    records = {}
    for (_, row), rings in zip(attrs.iterrows(), shapes):
        loc = int(row["LocationID"])
        a_tot = mx_tot = my_tot = 0.0
        for ring in rings:
            a, mx, my = _ring_area_centroid(ring)
            a_tot += a
            mx_tot += mx
            my_tot += my
        if "Shape_Area" in attrs.columns and row["Shape_Area"] > 0:
            area_ratios.append(abs(a_tot) * km_per_unit ** 2
                               / (row["Shape_Area"] * km2_per_deg2))
        rec = records.setdefault(loc, {"zone": row["zone"], "borough": row["borough"],
                                       "a": 0.0, "mx": 0.0, "my": 0.0, "rings": []})
        rec["a"] += a_tot
        rec["mx"] += mx_tot
        rec["my"] += my_tot
        rec["rings"].extend(rings)

    if len(area_ratios) != len(shapes):
        raise ValueError("Shape_Area attribute missing; cannot sanity-check geometry parsing")
    area_ratios = np.asarray(area_ratios)
    median_ratio = float(np.median(area_ratios))
    max_ratio = float(area_ratios.max())
    if not (0.98 < median_ratio < 1.02 and max_ratio < 1.02):
        raise ValueError(f"Planar areas disagree with Shape_Area: median ratio "
                         f"{median_ratio:.4f}, max ratio {max_ratio:.4f}")

    out = []
    for loc, rec in sorted(records.items()):
        out.append({
            "LocationID": loc,
            "zone": rec["zone"],
            "borough": rec["borough"],
            "area_km2": abs(rec["a"]) * km_per_unit ** 2,
            "x_km": rec["mx"] / rec["a"] * km_per_unit,
            "y_km": rec["my"] / rec["a"] * km_per_unit,
            "rings_km": [r * km_per_unit for r in rec["rings"]],
        })
    df = pd.DataFrame(out)
    df.attrs["source_path"] = str(zip_path)
    df.attrs["crs"] = "NAD83 / New York Long Island (ftUS), EPSG:2263"
    df.attrs["n_shapefile_records"] = len(shapes)
    df.attrs["area_check_median_ratio"] = median_ratio
    df.attrs["area_check_max_ratio"] = max_ratio
    return df
