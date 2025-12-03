import yfinance as yf
import pandas as pd
import requests
import os
import time
from datetime import datetime

DATA_DIR = 'data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

ALPHA_VANTAGE_KEY = 'B8ZDI8TD32Q07F3V'
START_DATE = '2000-01-01'
END_DATE = None  # today

class RawDataCollector:
    def __init__(self, start_date=START_DATE, end_date=END_DATE, data_dir=DATA_DIR):
        self.start_date = start_date
        self.end_date = end_date or datetime.now().strftime('%Y-%m-%d')
        self.data_dir = data_dir

    # ---------- Yahoo Finance ----------
    def collect_yahoo(self, symbols):
        for symbol in symbols:
            print(f"Collecting {symbol} from Yahoo Finance...")
            try:
                df = yf.download(symbol, start=self.start_date, end=self.end_date, progress=False, auto_adjust=True)
                df.to_csv(f'{self.data_dir}/{symbol}_raw.csv')
            except Exception as e:
                print(f"Error collecting {symbol}: {e}")

    # ---------- FRED Economic Data ----------
    def collect_fred(self, fred_series, api_key):
        base_url = 'https://api.stlouisfed.org/fred/series/observations'
        for series_id, name in fred_series.items():
            print(f"Collecting {name} from FRED...")
            try:
                params = {
                    'series_id': series_id,
                    'api_key': api_key,
                    'file_type': 'json',
                    'observation_start': self.start_date,
                    'observation_end': self.end_date
                }
                r = requests.get(base_url, params=params)
                data = r.json()
                df = pd.DataFrame(data['observations'])
                df['date'] = pd.to_datetime(df['date'])
                df['value'] = pd.to_numeric(df['value'], errors='coerce')
                df = df.set_index('date')[["value"]]
                df.columns = [name]
                df.to_csv(f'{self.data_dir}/{name}_fred.csv')
            except Exception as e:
                print(f"Error collecting {name}: {e}")

    # ---------- Alpha Vantage Technical Indicators ----------
    def collect_alpha_vantage(self, symbol, indicators):
        print(f"Collecting Alpha Vantage indicators for {symbol}...")
        base_url = 'https://www.alphavantage.co/query'
        all_data = pd.DataFrame()
        for ind_name, params in indicators.items():
            try:
                request_params = {
                    'function': params['function'],
                    'symbol': symbol,
                    'interval': 'daily',
                    'apikey': ALPHA_VANTAGE_KEY
                }
                if 'time_period' in params:
                    request_params['time_period'] = params['time_period']
                if 'series_type' in params:
                    request_params['series_type'] = params['series_type']
                
                print(f"  Requesting {ind_name}...")
                r = requests.get(base_url, params=request_params)
                data = r.json()
                
                # Debug: Print response keys to understand structure
                print(f"  Response keys: {list(data.keys())}")
                
                # Check for API errors and rate limiting
                if 'Error Message' in data:
                    print(f"  API Error for {ind_name}: {data['Error Message']}")
                    continue
                if 'Note' in data:
                    print(f"  API Note for {ind_name}: {data['Note']}")
                    continue
                if 'Information' in data:
                    print(f"  API Information for {ind_name}: {data['Information']}")
                    continue
                
                technical_key = next((k for k in data.keys() if 'Technical Analysis' in k or 'Time Series' in k), None)
                if not technical_key:
                    print(f"  No technical data found for {ind_name}")
                    continue
                
                df = pd.DataFrame.from_dict(data[technical_key], orient='index')
                df.index = pd.to_datetime(df.index)
                df = df.sort_index()
                df.to_csv(f'{self.data_dir}/technical_{symbol}_{ind_name}.csv')
                print(f"  Saved {ind_name} with {len(df)} records")
                time.sleep(20)  # Increased rate limit to avoid API restrictions
            except Exception as e:
                print(f"Error collecting {ind_name} for {symbol}: {e}")

    # ---------- Sentiment & Market Indicators ----------
    def collect_sentiment(self):
        # Put/Call Ratio - Use CBOE Put/Call Ratio instead
        try:
            print("Collecting Put/Call Ratio...")
            # Try alternative symbols for put/call ratio
            pc_symbols = ['^CPCE', '^VIX', '^TNX']  # Fallback options
            pc_data = None
            for symbol in pc_symbols:
                try:
                    pc_data = yf.download(symbol, start=self.start_date, end=self.end_date, progress=False, auto_adjust=True)
                    if not pc_data.empty:
                        pc_data[['Close']].rename(columns={'Close':'Put_Call_Ratio'}).to_csv(f'{self.data_dir}/put_call_ratio.csv')
                        print(f"Saved Put/Call Ratio using {symbol}")
                        break
                except:
                    continue
            if pc_data is None or pc_data.empty:
                print("Could not collect Put/Call Ratio data")
        except Exception as e:
            print(f"Error collecting Put/Call Ratio: {e}")

        # VIX term structure
        try:
            vix = yf.download(['^VIX','^VIX3M'], start=self.start_date, end=self.end_date, progress=False, auto_adjust=True)[['Close']]
            vix.to_csv(f'{self.data_dir}/vix_term_structure.csv')
        except: pass

        # Market breadth (NYSE)
        try:
            nyse = yf.download('^NYA', start=self.start_date, end=self.end_date, progress=False, auto_adjust=True)
            nyse.to_csv(f'{self.data_dir}/market_breadth.csv')
        except: pass

        # Sector breadth
        sectors = ['XLK','XLF','XLV','XLE','XLI']
        sector_df = pd.DataFrame()
        for s in sectors:
            try:
                data = yf.download(s, start=self.start_date, end=self.end_date, progress=False, auto_adjust=True)[['Close']]
                data.to_csv(f'{self.data_dir}/sector_{s}.csv')
            except: pass

    def run_all(self):
        # Yahoo symbols
        yahoo_symbols = ['QQQ','^GSPC','GOOGL','META']
        self.collect_yahoo(yahoo_symbols)

        # FRED series
        fred_series = {
            'DFF':'Fed_Funds_Rate',
            'UNRATE':'Unemployment_Rate',
            'CPIAUCSL':'CPI',
            'GDP':'GDP'
        }
        self.collect_fred(fred_series, api_key='1b743b037cb6f9fa46f035ab6852d935')

        # Alpha Vantage indicators - Start with just RSI to avoid rate limiting
        indicators = {
            'RSI': {'function':'RSI','time_period':14,'series_type':'close'},
            'MACD':{'function':'MACD','series_type':'close'},
            'SMA_50':{'function':'SMA','time_period':50,'series_type':'close'},
            'SMA_200':{'function':'SMA','time_period':200,'series_type':'close'}
        }
        # Use proper symbols for Alpha Vantage (no ^ prefix)
        alpha_vantage_symbols = ['QQQ', 'SPY']  # SPY instead of ^GSPC
        for symbol in alpha_vantage_symbols:
            self.collect_alpha_vantage(symbol, indicators)

        # Sentiment and market indicators
        self.collect_sentiment()

    def test_apis(self):
        """Test API connections and keys"""
        print("Testing API connections...")
        
        # Test FRED API
        print("\n1. Testing FRED API...")
        try:
            test_url = 'https://api.stlouisfed.org/fred/series/observations'
            params = {
                'series_id': 'DFF',
                'api_key': '1b743b037cb6f9fa46f035ab6852d935',
                'file_type': 'json',
                'limit': 1
            }
            r = requests.get(test_url, params=params)
            if r.status_code == 200:
                print("✓ FRED API key is working")
            else:
                print(f"✗ FRED API error: {r.status_code}")
        except Exception as e:
            print(f"✗ FRED API test failed: {e}")
        
        # Test Alpha Vantage API
        print("\n2. Testing Alpha Vantage API...")
        try:
            test_url = 'https://www.alphavantage.co/query'
            params = {
                'function': 'TIME_SERIES_INTRADAY',
                'symbol': 'IBM',
                'interval': '5min',
                'apikey': ALPHA_VANTAGE_KEY
            }
            r = requests.get(test_url, params=params)
            data = r.json()
            if 'Error Message' in data:
                print(f"✗ Alpha Vantage API error: {data['Error Message']}")
            elif 'Note' in data:
                print(f"⚠ Alpha Vantage API note: {data['Note']}")
            else:
                print("✓ Alpha Vantage API key is working")
        except Exception as e:
            print(f"✗ Alpha Vantage API test failed: {e}")

if __name__ == '__main__':
    collector = RawDataCollector()
    
    # Test APIs first
    collector.test_apis()
    
    # Then run data collection
    print("\n" + "="*50)
    print("Starting data collection...")
    collector.run_all()