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
