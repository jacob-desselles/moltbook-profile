import warnings
warnings.filterwarnings('ignore', message="fatal: bad revision 'HEAD'")
warnings.filterwarnings('ignore', category=UserWarning)
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import threading
import logging
from typing import Dict
import json
import time
import gc
import psutil
import configparser
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates
class DataManager:
    def __init__(self, exchange=None, log_function=None):
        self.price_data = {}
        self.exchange = exchange  # Store exchange instance
        self.lookback_periods = {
            'short': 5,
            'medium': 10,
            'long': 15
        }
        self.log_function = log_function if log_function else print
        self.min_data_points = 20
        self.initialization_complete = {}

    def get_price_data(self, symbol):
        """Get price data for a symbol"""
        try:
            if symbol not in self.price_data:
                self.log_trade(f"No price data available for {symbol}")
                return None
                
            # Return a copy to prevent modification
            return self.price_data[symbol].copy()
            
        except Exception as e:
            self.log_trade(f"Error getting price data for {symbol}: {str(e)}")
            return None

    def log_trade(self, message):
        """Log trade information with proper encoding handling"""
        try:
            # Replace Unicode characters with ASCII alternatives
            message = message.replace('✅', '+')
            message = message.replace('❌', 'X')
            
            # Format with timestamp
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_message = f"{timestamp} - {str(message)}"
            
            # Console output
            print(log_message)
            
            # File logging with encoding specification
            if hasattr(self, 'logger'):
                self.logger.info(message)
            
            # GUI logging
            if hasattr(self, 'log_text'):
                self.root.after(0, lambda: self.log_text.insert(tk.END, f"{log_message}\n"))
                self.root.after(0, lambda: self.log_text.see(tk.END))
                    
        except Exception as e:
            print(f"Logging error: {str(e)}")
            # Fallback logging with just ASCII
            print(f"{timestamp} - {message.encode('ascii', 'replace').decode()}")
    
    def _calculate_indicators(self, symbol: str):
        """Calculate technical indicators using pandas, including RSI"""
        try:
            if symbol not in self.price_data:
                return
            df = self.price_data[symbol]
            if len(df) < 2:
                return
            
            try:
                # Moving averages
                for period in [5, 10, 20]:
                    if len(df) >= period:
                        df[f'sma_{period}'] = df['price'].rolling(window=period).mean()
                        df[f'ema_{period}'] = df['price'].ewm(span=period, adjust=False).mean()
                
                # Volume metrics
                if len(df) >= 5:
                    df['volume_sma_5'] = df['volume'].rolling(window=5).mean()
                    df['volume_ratio'] = df['volume'] / df['volume_sma_5']
                
                # Price momentum and volatility
                df['price_change'] = df['price'].pct_change()
                if len(df) >= 5:
                    df['momentum'] = df['price'].pct_change(periods=5)
                    df['volatility'] = df['price'].rolling(window=5).std()
                
                # Spread
                df['spread'] = (df['ask'] - df['bid']) / df['price'] * 100
                
                # RSI Calculation - Improved version with better error handling
                rsi_period = int(self.rsi_period.get()) if hasattr(self, 'rsi_period') else 14  # Default to 14 if not set
                if len(df) >= rsi_period + 1:  # Need enough data points for RSI
                    # Calculate price changes
                    delta = df['price'].diff()
                    
                    # Separate gains and losses
                    gains = delta.where(delta > 0, 0)
                    losses = -delta.where(delta < 0, 0)
                    
                    # Calculate average gains and losses over the period
                    avg_gain = gains.rolling(window=rsi_period, min_periods=rsi_period).mean()
                    avg_loss = losses.rolling(window=rsi_period, min_periods=rsi_period).mean()
                    
                    # Avoid division by zero - replace zeros with small value
                    avg_loss = avg_loss.replace(0, 0.0001)
                    
                    # Calculate RS and RSI
                    rs = avg_gain / avg_loss
                    rs = rs.replace([np.inf, -np.inf], np.nan)  # Handle infinity
                    
                    # Calculate RSI
                    rsi = 100 - (100 / (1 + rs))
                    
                    # Handle edge cases
                    # If price hasn't changed at all, RSI should be 50 (neutral)
                    if delta.abs().sum() == 0:
                        rsi = pd.Series(50, index=rsi.index)
                    
                    # Fill any remaining NaN values with previous values or 50
                    rsi = rsi.ffill().fillna(50)
                    
                    # Store RSI in DataFrame
                    df[f'rsi_{rsi_period}'] = rsi
                    
                    # Log the last RSI value
                    last_rsi = rsi.iloc[-1]
                    self.log_trade(f"RSI calculated for {symbol} with period {rsi_period}: {last_rsi:.2f}")
                
                # Store updated DataFrame
                self.price_data[symbol] = df
                self.log_trade(f"Calculated indicators for {symbol}, data points: {len(df)}")
            
            except Exception as e:
                self.log_trade(f"Error calculating specific indicator for {symbol}: {str(e)}")
        
        except Exception as e:
            self.log_trade(f"Error in _calculate_indicators for {symbol}: {str(e)}")

    def update_price_data(self, symbol: str, new_data: dict):
        """Update price data with proper data accumulation"""
        try:
            current_timestamp = pd.Timestamp.now()
            
            # Validate incoming data
            required_fields = ['last', 'quoteVolume', 'bid', 'ask']
            if not all(new_data.get(field) is not None for field in required_fields):
                self.log_function(f"Missing required fields for {symbol}")
                return

            try:
                price = float(new_data['last'])
                volume = float(new_data['quoteVolume'])
                bid = float(new_data['bid'])
                ask = float(new_data['ask'])
                
                # Check for duplicate data
                if symbol in self.price_data and not self.price_data[symbol].empty:
                    last_price = self.price_data[symbol]['price'].iloc[-1]
                    last_time = self.price_data[symbol].index[-1]
                    
                    # Skip if price hasn't changed and time difference is small
                    if price == last_price and (current_timestamp - last_time).total_seconds() < 1:
                        self.log_function(f"Skipping duplicate price for {symbol}")
                        return
                    
                if any(val <= 0 for val in [price, volume, bid, ask]):
                    self.log_function(f"Invalid values for {symbol}")
                    return
                    
            except (TypeError, ValueError) as e:
                self.log_function(f"Data conversion error for {symbol}: {str(e)}")
                return

            # Create new data point
            new_data_point = pd.DataFrame({
                'price': [price],
                'volume': [volume],
                'bid': [bid],
                'ask': [ask]
            }, index=[current_timestamp])

            # Initialize or update price data
            if symbol not in self.price_data:
                self.price_data[symbol] = new_data_point
            else:
                # Remove data older than 5 minutes
                cutoff_time = current_timestamp - pd.Timedelta(minutes=5)
                
                # Ensure DataFrame has datetime index
                if not isinstance(self.price_data[symbol].index, pd.DatetimeIndex):
                    try:
                        self.price_data[symbol].index = pd.to_datetime(self.price_data[symbol].index)
                    except:
                        self.price_data[symbol] = new_data_point
                        return
                
                # Concatenate new data with existing data
                self.price_data[symbol] = pd.concat([
                    self.price_data[symbol][self.price_data[symbol].index > cutoff_time],
                    new_data_point
                ]).sort_index()

                # Log data points
                self.log_function(f"Updated {symbol} data: {len(self.price_data[symbol])} points collected")

            # Calculate indicators if we have enough data
            if len(self.price_data[symbol]) >= 5:
                self._calculate_indicators(symbol)
                
            self.log_function(f"Latest price: {price:.8f}")
            
        except Exception as e:
            self.log_function(f"Error updating price data for {symbol}: {str(e)}")
class CryptoScalpingBot:
    def __init__(self):
        # Create root window FIRST
        self.root = tk.Tk()
        self.night_mode = False # Track night mode state
        self.style = ttk.Style()
        self.style.configure('Dark.TEntry',
            fieldbackground='#1a1a1a',
            foreground='white')
        self.style.configure('Dark.TButton',
            background='#1a1a1a',
            foreground='white')
        self.root.title("Vantrex v1.15") 
        self.root.geometry("1200x1000")

        # Initialize basic variables
        self.is_paper_trading = True
        self.paper_balance = 1000.0
        self.start_time = None
        self.timer_running = False
        self.running = False
        self.trading_thread = None
        self.trades = []
        self.active_trades = {}
        self.price_history = {}
        self.cache_timeout = 1
        self.price_cache = {}
        self.total_profit = 0.0
        self.total_fees = 0.0
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.max_price=  tk.StringVar(self.root, value="5.00")

        # Initialize StringVars for GUI inputs
        self.profit_target = tk.StringVar(self.root, value="1.2")
        self.stop_loss = tk.StringVar(self.root, value="0.5")
        self.trailing_stop = tk.StringVar(self.root, value="0.3")
        self.position_size = tk.StringVar(self.root, value="75")
        self.min_volume_entry = tk.StringVar(self.root, value="150")
        self.max_trades_entry = tk.StringVar(self.root, value="3")
        self.top_list_size = tk.StringVar(self.root, value="10")
        self.max_volatility = tk.StringVar(self.root, value="2.0")
        self.consecutive_rises = tk.StringVar(self.root, value="2")
        self.momentum_threshold = tk.StringVar(self.root, value="0.2")
        self.max_spread = tk.StringVar(self.root, value="0.3")
        self.volume_increase = tk.StringVar(self.root, value="10")
        self.price_rise_min = tk.StringVar(self.root, value="0.2")
        self.trailing_activation = tk.StringVar(self.root, value="0.5")
        self.max_position_percent = tk.StringVar(self.root, value="10")
        self.daily_loss_limit = tk.StringVar(self.root, value="5")
        # Add fee structure variables
        #self.maker_fee = tk.StringVar(self.root, value="0.25")  # 0.25% for maker orders
        #self.taker_fee = tk.StringVar(self.root, value="0.40")  # 0.40% for taker orders

        # Add fee structure variables as StringVars (with different names)
        self.maker_fee_var = tk.StringVar(self.root, value="0.25")  # 0.25% for maker orders
        self.taker_fee_var = tk.StringVar(self.root, value="0.40")  # 0.40% for taker orders

        # Fee Structure (Kraken)
        self.maker_fee = 0.0025  # 0.25% for maker orders
        self.taker_fee = 0.004   # 0.40% for taker orders
        self.total_fee_percentage = self.taker_fee * 2  # 0.80% total for entry and exit

        # Trading parameters
        self.min_volume = 50
        self.scan_interval = 1
        self.max_active_trades = 5
        self.price_rise_threshold = 0.01
        self.trailing_stop_pct = 0.05
        self.short_term_lookback = 5
        self.volume_multiplier = 1.2
        self.momentum_threshold_val = -1.0
        self.min_balance_threshold = 20.0

        # Plotting parameters
        self.plot_update_interval = 2000  # milliseconds
        self.chart_timeframe = 50  # number of points to show
        self.price_precision = 8
        self.amount_precision = 8


        # Initialize DataManager
        #self.data_manager = DataManager(log_function=self.log_trade)
        self.log_trade("Testing logger initialization")
        print("Direct print test from __init__")

        # Setup components
        self.setup_logging()
        self.log_trade("Initializing Crypto Scalping Bot...")
        self.config = self.load_config()
        self.init_exchange()
        self.setup_gui()
        self.log_trade("Bot initialization complete.")

        ## Initialize memory monitoring
        self.init_memory_monitor()
        # Scanning parameters
        self.scan_interval = 10  # seconds between scans
        self.scan_timeout = 30   # maximum seconds for a scan
        self.last_scan_time = 0  # track last scan time

    def init_memory_monitor(self):
        """Initialize memory monitoring"""
        self.memory_check_interval = 60  # seconds
        self.last_memory_check = time.time()
        self.memory_threshold = 1000  # MB

    def check_memory_usage(self):
        """Monitor and manage memory usage"""
        try:
            current_time = time.time()
            if current_time - self.last_memory_check < self.memory_check_interval:
                return

            self.last_memory_check = current_time
            process = psutil.Process()
            memory_use = process.memory_info().rss / 1024 / 1024  # Convert to MB

            if memory_use > self.memory_threshold:
                self.log_trade(f"High memory usage detected: {memory_use:.1f} MB")
                self.cleanup_old_data()
                gc.collect()

        except Exception as e:
            self.log_trade(f"Memory check error: {str(e)}")

    def setup_chart(self):
        """Initialize professional trading chart"""
        try:
            # Clear any existing chart
            if hasattr(self, 'fig'):
                plt.close(self.fig)
            if hasattr(self, 'canvas'):
                self.canvas.get_tk_widget().destroy()
            
            # Set colors based on night mode
            bg_color = '#2d2d2d' if self.night_mode else 'white'
            fg_color = 'white' if self.night_mode else 'black'
            grid_color = '#3d3d3d' if self.night_mode else '#cccccc'
            
            # Create figure with theme
            self.fig = Figure(figsize=(8, 4), dpi=100, facecolor=bg_color)
            self.ax = self.fig.add_subplot(111)
            
            # Style the axes and background
            self.ax.set_facecolor(bg_color)
            self.ax.tick_params(axis='both', colors=fg_color)
            self.ax.grid(True, color=grid_color, linestyle='--', alpha=0.5)
            
            # Set spine colors
            for spine in self.ax.spines.values():
                spine.set_color(grid_color)
            
            # Initialize empty plots
            self.price_line, = self.ax.plot([], [], 
                                        color='#4afeb7' if self.night_mode else '#2196F3', 
                                        linewidth=1.5, 
                                        label='Price')
            
            # Formatting
            self.ax.xaxis.set_major_formatter(DateFormatter('%H:%M:%S'))
            self.ax.set_xlabel('Time', color=fg_color, fontsize=10)
            self.ax.set_ylabel('Price Change (%)', color=fg_color, fontsize=10)
            self.ax.set_title('Trade Performance', color=fg_color, pad=20, fontsize=12)
            
            # Legend with theme
            legend = self.ax.legend(facecolor=bg_color, edgecolor=grid_color)
            for text in legend.get_texts():
                text.set_color(fg_color)
            
            # Add to GUI with proper sizing
            self.canvas = FigureCanvasTkAgg(self.fig, master=self.chart_frame)
            self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            # Adjust layout
            self.fig.tight_layout()
            self.canvas.draw()
            
        except Exception as e:
            self.log_trade(f"Error setting up chart: {str(e)}")

    def toggle_market_override(self):
        """Toggle market condition override"""
        override = self.market_override_var.get()
        if override:
            self.log_trade("MARKET OVERRIDE ENABLED: Bot will ignore bearish market conditions")
            messagebox.showinfo("Market Override", "Market condition checks are now disabled. The bot will trade normally regardless of market conditions.")
        else:
            self.log_trade("MARKET OVERRIDE DISABLED: Bot will respect market conditions")
        
    def toggle_night_mode(self):
        """Toggle between light and dark theme"""
        try:
            self.night_mode = not self.night_mode
            
            if self.night_mode:
                # Dark theme colors
                bg_color = '#2d2d2d'        # Dark background
                fg_color = '#ffffff'        # White text
                entry_bg = '#1a1a1a'        # Darker entry background
                button_bg = '#1a1a1a'       # Very dark button background
                text_area_bg = '#1a1a1a'    # Dark text area background
                
                # Create custom dark button style
                self.style.configure('Dark.TButton',
                    background=button_bg,
                    foreground=fg_color,
                    bordercolor=button_bg,
                    darkcolor=button_bg,
                    lightcolor=button_bg,
                    relief='flat')
                
                # Apply dark style to all buttons
                for button in [self.apply_button, self.live_update_button, 
                            self.close_profitable_button, self.close_all_button,
                            self.night_mode_button]:
                    button.configure(style='Dark.TButton')
                
                # Configure other ttk styles
                self.style.configure('TFrame', background=bg_color)
                self.style.configure('TLabelframe', background=bg_color)
                self.style.configure('TLabelframe.Label', foreground=fg_color)
                self.style.configure('TLabel', foreground=fg_color)
                self.style.configure('TNotebook', background=bg_color)
                self.style.configure('TNotebook.Tab', foreground=fg_color)
                self.style.configure('TEntry', 
                    fieldbackground=entry_bg,
                    foreground=fg_color)
                
                # Update text widgets
                text_widgets = [self.log_text, self.history_text, self.trades_text]
                for widget in text_widgets:
                    widget.configure(
                        background=text_area_bg,
                        foreground=fg_color,
                        insertbackground=fg_color,
                        selectbackground='#404040',
                        selectforeground=fg_color
                    )
                
                self.night_mode_button.configure(text="Light Mode")
                
            else:
                # Light theme colors
                bg_color = 'SystemButtonFace'
                fg_color = 'black'
                entry_bg = 'white'
                
                # Create and apply light button style
                self.style.configure('TButton',
                    background=bg_color,
                    foreground=fg_color)
                
                # Reset all buttons to default style
                for button in [self.apply_button, self.live_update_button, 
                            self.close_profitable_button, self.close_all_button,
                            self.night_mode_button]:
                    button.configure(style='TButton')
                
                # Reset other ttk styles
                self.style.configure('TFrame', background=bg_color)
                self.style.configure('TLabelframe', background=bg_color)
                self.style.configure('TLabelframe.Label', foreground=fg_color)
                self.style.configure('TLabel', foreground=fg_color)
                self.style.configure('TNotebook', background=bg_color)
                self.style.configure('TNotebook.Tab', foreground=fg_color)
                self.style.configure('TEntry', 
                    fieldbackground=entry_bg,
                    foreground=fg_color)
                
                # Reset text widgets
                text_widgets = [self.log_text, self.history_text, self.trades_text]
                for widget in text_widgets:
                    widget.configure(
                        background='white',
                        foreground='black',
                        insertbackground='black',
                        selectbackground='#0078D7',
                        selectforeground='white'
                    )
                
                self.night_mode_button.configure(text="Dark Mode")
            
            # Recreate chart with new theme
            self.setup_chart()
            
        except Exception as e:
            self.log_trade(f"Error toggling night mode: {str(e)}")

    def setup_logging(self):
        """Setup logging configuration"""
        try:
            # Create logs directory if it doesn't exist
            if not os.path.exists('logs'):
                os.makedirs('logs')

            # Configure logger
            self.logger = logging.getLogger(__name__)
            self.logger.setLevel(logging.INFO)
            
            # Create file handler
            log_file = f'logs/trading_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.INFO)
            
            # Create console handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            
            # Create formatter
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)
            
            # Add handlers to logger
            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)
            
            self.log_trade("Logging setup complete.")
            
        except Exception as e:
            print(f"Error setting up logging: {str(e)}")
            raise

    def load_config(self):
        """Load or create configuration file with proper structure"""
        self.config = configparser.ConfigParser()
        config_file = 'config.ini'
        
        # Default configuration structure
        defaults = {
            'API_KEYS': {
                'api_key': '',
                'secret': ''  # Using 'secret' instead of 'secret_key' for ccxt compatibility
            },
            'TRADING': {
                'mode': 'paper'
            }
        }

        if not os.path.exists(config_file):
            # Create new config file with defaults
            self.log_trade("Creating new configuration file")
            for section, options in defaults.items():
                self.config[section] = options
            with open(config_file, 'w') as f:
                self.config.write(f)
        else:
            # Load existing config
            self.config.read(config_file)
            # Ensure all required sections exist
            for section, options in defaults.items():
                if section not in self.config:
                    self.config[section] = options
                # Ensure all options exist
                for option, value in options.items():
                    if option not in self.config[section]:
                        self.config[section][option] = value
        
        return self.config

    def initialize_btc_data(self):
        """Initialize BTC/USD market data for market analysis"""
        try:
            self.log_trade("Initializing BTC/USD market data...")
            
            # Fetch BTC/USD ticker
            try:
                btc_ticker = self.exchange.fetch_ticker('BTC/USD')
                if btc_ticker:
                    self.data_manager.update_price_data('BTC/USD', btc_ticker)
                    self.log_trade(f"Initial BTC/USD data point collected: ${float(btc_ticker['last'])}")
                else:
                    self.log_trade("Warning: Could not fetch BTC/USD ticker")
                    return False
            except Exception as e:
                self.log_trade(f"Error fetching BTC/USD ticker: {str(e)}")
                return False
            
            # Try to fetch historical data
            try:
                # Fetch OHLCV data for the past day (24 hourly candles)
                since = int((datetime.now() - timedelta(days=1)).timestamp() * 1000)
                ohlcv = self.exchange.fetch_ohlcv('BTC/USD', timeframe='1h', since=since, limit=24)
                
                # Convert to DataFrame and update price data
                for candle in ohlcv:
                    timestamp, open_price, high, low, close, volume = candle
                    
                    # Create synthetic ticker data
                    synthetic_ticker = {
                        'timestamp': pd.Timestamp(timestamp, unit='ms'),
                        'last': close,
                        'quoteVolume': volume,
                        'bid': close * 0.999,  # Approximate bid
                        'ask': close * 1.001   # Approximate ask
                    }
                    
                    # Update price data
                    self.data_manager.update_price_data('BTC/USD', synthetic_ticker)
                
                # Log success
                df = self.data_manager.get_price_data('BTC/USD')
                if df is not None:
                    self.log_trade(f"Successfully loaded {len(df)} historical data points for BTC/USD")
                
                return True
                
            except Exception as e:
                self.log_trade(f"Error fetching historical BTC/USD data: {str(e)}")
                # Continue with just the single data point
                return True
            
        except Exception as e:
            self.log_trade(f"Error initializing BTC/USD data: {str(e)}")
            return False

    def init_exchange(self):
        """Initialize exchange with retry mechanism"""
        max_retries = 3
        retry_delay = 5  # seconds
        
        for attempt in range(max_retries):
            try:
                self.log_trade(f"Initializing exchange (attempt {attempt + 1}/{max_retries})...")
                
                self.exchange = ccxt.kraken({
                    'enableRateLimit': True,
                    'options': {
                        'adjustForTimeDifference': True,
                    },
                    'timeout': 30000,
                    'rateLimit': 1000,
                })
                
                # Test connection with error handling
                try:
                    self.log_trade("Testing exchange connection...")
                    self.exchange.load_markets()
                    
                    # Verify we can fetch data
                    self.log_trade("Testing ticker fetch...")
                    test_tickers = self.exchange.fetch_tickers()
                    self.log_trade(f"Successfully fetched {len(test_tickers)} tickers")
                    
                    # Initialize DataManager with exchange instance
                    self.data_manager = DataManager(
                        exchange=self.exchange,
                        log_function=self.log_trade
                    )
                    
                    # Initialize BTC/USD market data
                    self.initialize_btc_data()
                    
                    self.log_trade("Exchange initialization successful")
                    return True
                    
                except ccxt.NetworkError as e:
                    if attempt < max_retries - 1:
                        self.log_trade(f"Network error, retrying in {retry_delay} seconds: {str(e)}")
                        time.sleep(retry_delay)
                        continue
                    raise
                    
                except Exception as e:
                    self.log_trade(f"Error testing exchange connection: {str(e)}")
                    raise
                    
            except Exception as e:
                if attempt < max_retries - 1:
                    self.log_trade(f"Initialization failed, retrying in {retry_delay} seconds: {str(e)}")
                    time.sleep(retry_delay)
                else:
                    self.log_trade(f"Failed to initialize exchange after {max_retries} attempts: {str(e)}")
                    raise

        return False

    def fetch_tickers_with_retry(self, max_retries=3, retry_delay=2):
        """Fetch tickers with retry mechanism and ensure fresh data"""
        for attempt in range(max_retries):
            try:
                # Force fresh data fetch
                self.exchange.load_markets(True)  # Force reload
                tickers = self.exchange.fetch_tickers()
                
                # Add current timestamp if missing
                current_time = int(time.time() * 1000)
                for symbol, ticker in tickers.items():
                    if ticker.get('timestamp') is None:
                        ticker['timestamp'] = current_time
                    elif current_time - ticker['timestamp'] > 60000:  # Older than 60 seconds
                        self.log_trade(f"Forcing fresh data fetch for {symbol}")
                        try:
                            fresh_ticker = self.exchange.fetch_ticker(symbol)
                            tickers[symbol].update(fresh_ticker)
                        except Exception as e:
                            self.log_trade(f"Error fetching fresh data for {symbol}: {str(e)}")
                
                return tickers
                
            except Exception as e:
                self.log_trade(f"Error fetching tickers (attempt {attempt + 1}/{max_retries}): {str(e)}")
                time.sleep(retry_delay)
                
        self.log_trade("Failed to fetch tickers after multiple attempts")
        return {}

    def show_api_config(self):
        """Show API configuration window with improved error handling"""
        try:
            config_window = tk.Toplevel(self.root)
            config_window.title("API Configuration")
            config_window.geometry("500x300")
            config_window.transient(self.root)
            config_window.grab_set()
            
            # Get current values if they exist
            current_api_key = ""
            current_secret = ""
            
            if self.config.has_section('API_KEYS'):
                current_api_key = self.config.get('API_KEYS', 'api_key', fallback='')
                current_secret = self.config.get('API_KEYS', 'secret', fallback='')
            
            # Create form
            frame = ttk.Frame(config_window, padding=20)
            frame.pack(fill=tk.BOTH, expand=True)
            
            ttk.Label(frame, text="Kraken API Key:").grid(row=0, column=0, sticky="w", pady=5)
            api_key_entry = ttk.Entry(frame, width=50)
            api_key_entry.insert(0, current_api_key)
            api_key_entry.grid(row=0, column=1, pady=5, padx=5)
            
            ttk.Label(frame, text="Kraken API Secret:").grid(row=1, column=0, sticky="w", pady=5)
            secret_entry = ttk.Entry(frame, width=50, show="*")
            secret_entry.insert(0, current_secret)
            secret_entry.grid(row=1, column=1, pady=5, padx=5)
            
            # Show/hide secret toggle
            show_secret = tk.BooleanVar(value=False)
            
            def toggle_secret_visibility():
                if show_secret.get():
                    secret_entry.config(show="")
                else:
                    secret_entry.config(show="*")
            
            ttk.Checkbutton(frame, text="Show Secret", variable=show_secret, 
                        command=toggle_secret_visibility).grid(row=2, column=1, sticky="w")
            
            # Status label
            status_label = ttk.Label(frame, text="")
            status_label.grid(row=3, column=0, columnspan=2, pady=10)
            
            # Test connection function
            def test_connection():
                api_key = api_key_entry.get().strip()
                secret = secret_entry.get().strip()
                
                if not api_key or not secret:
                    status_label.config(text="Error: API Key and Secret are required", foreground="red")
                    return
                
                status_label.config(text="Testing connection...", foreground="blue")
                config_window.update()
                
                try:
                    test_exchange = ccxt.kraken({
                        'apiKey': api_key,
                        'secret': secret,
                        'enableRateLimit': True
                    })
                    
                    # Test with a lightweight API call
                    test_exchange.fetch_balance()
                    status_label.config(text="Connection successful!", foreground="green")
                    return True
                except ccxt.AuthenticationError:
                    status_label.config(text="Error: Invalid API keys", foreground="red")
                    return False
                except ccxt.NetworkError:
                    status_label.config(text="Error: Network error - check your connection", foreground="red")
                    return False
                except Exception as e:
                    status_label.config(text=f"Error: {str(e)}", foreground="red")
                    return False
            
            # Save function
            def save_config():
                api_key = api_key_entry.get().strip()
                secret = secret_entry.get().strip()
                
                if not api_key or not secret:
                    status_label.config(text="Error: API Key and Secret are required", foreground="red")
                    return
                
                # Test connection before saving
                status_label.config(text="Testing connection before saving...", foreground="blue")
                config_window.update()
                
                if test_connection():
                    # Save to config
                    if not self.config.has_section('API_KEYS'):
                        self.config.add_section('API_KEYS')
                    
                    self.config.set('API_KEYS', 'api_key', api_key)
                    self.config.set('API_KEYS', 'secret', secret)
                    
                    with open('config.ini', 'w') as f:
                        self.config.write(f)
                    
                    status_label.config(text="API keys saved successfully!", foreground="green")
                    self.log_trade("API keys configured and saved")
                    
                    # Close window after short delay
                    config_window.after(1500, config_window.destroy)
            
            # Buttons
            button_frame = ttk.Frame(frame)
            button_frame.grid(row=4, column=0, columnspan=2, pady=20)
            
            ttk.Button(button_frame, text="Test Connection", command=test_connection).pack(side=tk.LEFT, padx=5)
            ttk.Button(button_frame, text="Save", command=save_config).pack(side=tk.LEFT, padx=5)
            ttk.Button(button_frame, text="Cancel", command=config_window.destroy).pack(side=tk.LEFT, padx=5)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to create config window: {str(e)}")

    def test_api_connection(self, api_key, secret_key):
        """Test Kraken API connection with provided keys"""
        try:
            test_exchange = ccxt.kraken({
                'apiKey': api_key,
                'secret': secret_key,
                'enableRateLimit': True
            })
            # Test with a lightweight API call
            test_exchange.fetch_time()
            messagebox.showinfo("Success", "API keys verified successfully!")
            return True
        except ccxt.AuthenticationError:
            messagebox.showerror("Error", "Invalid API keys")
            return False
        except ccxt.NetworkError:
            messagebox.showerror("Error", "Network error - check your connection")
            return False
        except Exception as e:
            messagebox.showerror("Error", f"Connection failed: {str(e)}")
            return False

    def save_api_config(self):
        """Save API keys to config file"""
        try:
            if not self.test_api_connection():  # Verify before saving
                return
                
            self.config['API_KEYS'] = {
                'api_key': self.api_key_entry.get(),
                'secret_key': self.secret_key_entry.get()
            }
            
            with open('config.ini', 'w') as f:
                self.config.write(f)
                
            messagebox.showinfo("Success", "API configuration saved")
            self.init_exchange()  # Reinitialize with new keys
            return True
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save config: {str(e)}")
            return False
        
    def toggle_trading_mode(self):
        """Switch between paper and real trading with proper config handling"""
        try:
            if self.running:
                messagebox.showwarning("Warning", "Please stop the bot before changing mode")
                return

            if not self.is_paper_trading:
                # Switching to paper trading
                self.is_paper_trading = True
                self.mode_var.set("Paper Trading")
                self.update_status("Paper Trading")
                self.balance_label.config(text=f"Paper Balance: ${self.paper_balance:.2f}")
                return

            # Switching to real trading
            if not self.config.has_section('API_KEYS'):
                self.log_trade("API_KEYS section missing in config.ini")
                self.show_api_config()
                return

            api_key = self.config.get('API_KEYS', 'api_key', fallback='')
            secret = self.config.get('API_KEYS', 'secret', fallback='')

            if not api_key or not secret:
                self.log_trade("API keys missing or empty in config.ini")
                if messagebox.askyesno("API Keys Required", 
                                    "Real trading requires API keys. Configure now?"):
                    self.show_api_config()
                return

            # Test the connection with proper error handling
            try:
                self.log_trade("Testing API connection...")
                test_exchange = ccxt.kraken({
                    'apiKey': api_key,
                    'secret': secret,
                    'enableRateLimit': True
                })
                test_exchange.fetch_balance()  # Test API call
                self.log_trade("API connection successful")
            except Exception as e:
                error_msg = str(e)
                self.log_trade(f"API test failed: {error_msg}")
                
                if "apiKey" in error_msg:
                    messagebox.showerror("API Error", 
                                    "API key error. Please check your API keys are correctly formatted.")
                else:
                    messagebox.showerror("Connection Failed", f"API test failed: {error_msg}")
                    
                if messagebox.askyesno("Reconfigure Keys", "Would you like to reconfigure your API keys?"):
                    self.show_api_config()
                return

            # Success - switch to real trading
            self.is_paper_trading = False
            self.mode_var.set("Real Trading")
            self.update_status("Real Trading")
            
            # Reinitialize exchange with verified keys
            self.exchange = ccxt.kraken({
                'apiKey': api_key,
                'secret': secret,
                'enableRateLimit': True,
                'options': {
                    'adjustForTimeDifference': True,
                },
                'timeout': 30000,
                'rateLimit': 1000,
            })
            
            # Update balance display
            try:
                balance = float(self.exchange.fetch_balance()['USD']['free'])
                self.balance_label.config(text=f"Real Balance: ${balance:.2f}")
                self.log_trade(f"Real balance: ${balance:.2f}")
            except Exception as e:
                self.balance_label.config(text="Real Balance: Unavailable")
                self.log_trade(f"Balance fetch error: {str(e)}")

        except Exception as e:
            self.log_trade(f"Mode change error: {str(e)}")
            messagebox.showerror("Error", f"Mode change failed: {str(e)}")

    def verify_api_keys(self):
        """Verify that API keys are valid and have correct permissions"""
        try:
            if not self.is_paper_trading:
                self.log_trade("Verifying API keys...")
                
                # Check if API keys exist
                api_keys = self.config['API_KEYS']
                if not all([api_keys['api_key'], api_keys['secret_key']]):
                    raise ValueError("API keys not configured")
                
                # Test API connection and permissions
                try:
                    # Test balance query
                    self.exchange.fetch_balance()
                    
                    # Test order endpoints with a dummy order (without actually placing it)
                    self.exchange.fetch_open_orders()
                    
                    # Verify active trades against exchange data
                    self.verify_active_trades()
                    
                    self.log_trade("API keys verified successfully")
                    return True
                    
                except Exception as e:
                    error_msg = str(e)
                    if "Permission denied" in error_msg:
                        raise ValueError("Insufficient API key permissions. Please check API key settings.")
                    else:
                        raise ValueError(f"API verification failed: {error_msg}")
                        
        except Exception as e:
            self.log_trade(f"API key verification failed: {str(e)}")
            messagebox.showerror("API Error", f"API key verification failed: {str(e)}")
            self.is_paper_trading = True  # Revert to paper trading
            self.mode_var.set("Paper Trading")
            return False   
        
    def start_bot(self):
        """Start or stop the bot"""
        try:
            if not self.running:
                # START SEQUENCE
                if not self.validate_settings():
                    return
                
                # Initialize tracking
                self.start_time = datetime.now()
                self.running = True
                self.timer_running = True
                self.start_button.config(text="Stop")
                self.update_status(f"Running ({'Paper' if self.is_paper_trading else 'Real'})")
                
                # Start timer
                self.start_timer()
                
                # Start threads
                self.trading_thread = threading.Thread(target=self.run_bot, daemon=True)
                self.trading_thread.start()
                
                # Use the comprehensive version at line 2735
                self.price_monitor_thread = threading.Thread(
                    target=self.monitor_prices_continuously, 
                    daemon=True
                )
                self.price_monitor_thread.start()
                
                self.log_trade("=== BOT STARTED ===")
                
            else:
                # STOP SEQUENCE
                self.running = False
                self.timer_running = False
                self.start_button.config(text="Start")
                self.update_status("Stopping...")
                
                # Close all positions
                closing_thread = threading.Thread(
                    target=self.close_all_positions_on_stop,
                    daemon=True
                )
                closing_thread.start()
                
                # Wait for threads to finish
                if self.trading_thread:
                    self.trading_thread.join(timeout=5)
                if self.price_monitor_thread:
                    self.price_monitor_thread.join(timeout=5)
                
                self.update_status("Stopped")
                self.log_trade("=== BOT STOPPED ===")

                # Final performance report
                self.log_trade(f"""
                Bot Session Summary:
                Total Trades: {self.total_trades}
                Winning Trades: {self.winning_trades}
                Losing Trades: {self.losing_trades}
                Win Rate: {(self.winning_trades/max(1, self.total_trades))*100:.1f}%
                Total Profit: ${self.total_profit:.2f}
                Total Fees: ${self.total_fees:.2f}
                Net Profit: ${(self.total_profit - self.total_fees):.2f}
                Final Balance: ${self.paper_balance:.2f}
                Runtime: {str(datetime.now() - self.start_time)}
                """)

        except Exception as e:
            self.log_trade(f"Toggle error: {str(e)}")
            messagebox.showerror("Error", f"Failed to toggle bot: {str(e)}")
            self.running = False  # Force stop on error
            self.timer_running = False
            self.start_button.config(text="Start")
            self.update_status("Error - Stopped")

    def close_all_positions_on_stop(self):
        """Close all positions when stopping the bot"""
        try:
            if self.active_trades:
                self.log_trade("Closing all positions due to stop signal")
                for trade_id, trade in list(self.active_trades.items()):
                    try:
                        # Get current price with error handling
                        try:
                            current_price = self.exchange.fetch_ticker(trade['symbol'])['last']
                        except:
                            current_price = trade.get('highest_price', trade['entry_price'])
                        
                        self.close_trade(trade_id, trade, current_price, "bot stopped")
                        time.sleep(0.1)  # Brief pause between closures
                        
                    except Exception as e:
                        self.log_trade(f"Error closing trade {trade_id}: {str(e)}")
                        
                self.log_trade("All positions closed")
                
        except Exception as e:
            self.log_trade(f"Error during shutdown: {str(e)}")
    def log_current_settings(self):
        """Log all current settings"""
        try:
            settings = f"""
            Current Settings:
            ----------------
            Trading Mode: {'Paper Trading' if self.is_paper_trading else 'Real Trading'}
            
            Price Range:
            - Min Price: {self.min_price} USD
            - Max Price: {self.max_price} USD
            - Preferred Range: {self.preferred_price_range[0]} - {self.preferred_price_range[1]} USD
            
            Basic Parameters:
            - Profit Target: {self.profit_target.get()}%
            - Stop Loss: {self.stop_loss.get()}%
            - Position Size: {self.position_size.get()} USD
            
            Advanced Settings:
            - Min Volume: {self.min_volume} USD
            - Max Active Trades: {self.max_active_trades}
            - Price Rise Threshold: {self.price_rise_threshold}%
            - Trailing Stop: {self.trailing_stop_pct}%
            - Scan Interval: {self.scan_interval}s
            - Momentum Threshold: {self.momentum_threshold}%
            
            Balance Information:
            - {'Paper' if self.is_paper_trading else 'Real'} Balance: {self.paper_balance if self.is_paper_trading else 'N/A'} USD
            - Min Balance Threshold: {self.min_balance_threshold} USD
            
            Fee Structure:
            - Maker Fee: {self.maker_fee*100:.2f}%
            - Taker Fee: {self.taker_fee*100:.2f}%
            
            Performance:
            - Total Trades: {self.total_trades}
            - Winning Trades: {self.winning_trades}
            - Losing Trades: {self.losing_trades}
            - Total Profit: {self.total_profit:.2f} USD
            """
            self.log_trade(settings)
            
        except Exception as e:
            self.log_trade(f"Error logging current settings: {str(e)}")

    def initialize_fees(self):
        """Initialize fee values from StringVar objects"""
        try:
            # Parse fee values from GUI
            maker_fee = float(self.maker_fee_var.get()) / 100  # Convert from percentage to decimal
            taker_fee = float(self.taker_fee_var.get()) / 100  # Convert from percentage to decimal
            
            # Update instance variables
            self.maker_fee = maker_fee
            self.taker_fee = taker_fee
            self.total_fee_percentage = maker_fee + taker_fee  # Assuming both entry and exit are taker orders
            
            # Update the total fees label
            total_fee_pct = self.total_fee_percentage * 100
            self.total_fees_label.config(text=f"Total Round-Trip Fee: {total_fee_pct:.2f}%")
            
            self.log_trade(f"Fee structure initialized: Maker {maker_fee*100:.2f}%, Taker {taker_fee*100:.2f}%, Total {total_fee_pct:.2f}%")
        except Exception as e:
            self.log_trade(f"Error initializing fees: {str(e)}")

    def update_status(self, status):
        """Update status display with proper GUI handling"""
        try:
            mode = "Paper Trading" if self.is_paper_trading else "Real Trading"
            self.status_label.config(text=f"Status: {status} ({mode})")
            
            # Update GUI immediately
            self.root.update_idletasks()
            
            # Log status change
            self.log_trade(f"Status updated: {status}")
            
            # Color code the status based on content
            if "Error" in status:
                self.status_label.config(foreground="red")
            elif "Warning" in status:
                self.status_label.config(foreground="orange") 
            elif "Running" in status:
                self.status_label.config(foreground="green")
            elif "Stopped" in status:
                self.status_label.config(foreground="black")
                
        except Exception as e:
            self.log_trade(f"Error updating status: {str(e)}")

    def reset_paper_balance(self):
        """Reset paper balance to initial value"""
        self.paper_balance = 1000.0
        self.total_profit = 0.0
        self.total_fees = 0.0
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.update_balance_display()
        self.log_trade("Paper balance reset to $1000.00")

    def setup_gui(self):
        """Setup the GUI with all parameters and dark mode support"""
        try:
            # Main container
            main_container = ttk.Frame(self.root)
            main_container.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
            
            # Configure main window grid
            self.root.grid_rowconfigure(0, weight=1)
            self.root.grid_columnconfigure(0, weight=1)
            
            # Left column (70%) - Controls and Charts
            left_column = ttk.Frame(main_container)
            left_column.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
            
            # Right column (30%) - Metrics and History
            right_column = ttk.Frame(main_container)
            right_column.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")
            
            main_container.grid_columnconfigure(0, weight=7)
            main_container.grid_columnconfigure(1, weight=3)

            # === CONTROLS SECTION ===
            control_frame = ttk.LabelFrame(left_column, text="Controls")
            control_frame.grid(row=0, column=0, sticky="ew", padx=5, pady=5)

            # Trading Mode Frame
            mode_frame = ttk.Frame(control_frame)
            mode_frame.grid(row=0, column=0, padx=5, pady=5, columnspan=2, sticky="ew")

            # Mode controls
            self.mode_var = tk.StringVar(value="Paper Trading")
            ttk.Label(mode_frame, text="Trading Mode:").grid(row=0, column=0, padx=5, sticky="w")
            ttk.Button(mode_frame, textvariable=self.mode_var, 
                    command=self.toggle_trading_mode).grid(row=0, column=1, padx=5, sticky="w")

            self.balance_label = ttk.Label(mode_frame, 
                text=f"{'Paper' if self.is_paper_trading else 'Real'} Balance: ${self.paper_balance:.2f}")
            self.balance_label.grid(row=0, column=2, padx=20)

            self.start_button = ttk.Button(mode_frame, text="Start", command=self.start_bot)
            self.start_button.grid(row=0, column=3, padx=5)

            self.status_label = ttk.Label(mode_frame, 
                text=f"Status: Idle ({'Paper' if self.is_paper_trading else 'Real'} Trading)")
            self.status_label.grid(row=0, column=4, padx=5)

            self.timer_label = ttk.Label(mode_frame, text="Runtime: 00:00:00")
            self.timer_label.grid(row=0, column=5, padx=5)

            ttk.Button(mode_frame, text="API Config", 
                    command=self.show_api_config).grid(row=0, column=6, padx=5)
            
    # Add market override checkbox
            self.market_override_var = tk.BooleanVar(value=False)
            self.market_override_check = ttk.Checkbutton(
                mode_frame, 
                text="Ignore Market Conditions",
                variable=self.market_override_var,
                command=self.toggle_market_override
            )
            self.market_override_check.grid(row=1, column=0, columnspan=3, padx=5, pady=2, sticky="w")

            # Create notebook for parameter organization
            param_notebook = ttk.Notebook(control_frame)
            param_notebook.grid(row=1, column=0, columnspan=2, padx=5, pady=5, sticky="ew")

    
            # Reset paper balance button
            self.reset_balance_button = ttk.Button(
                mode_frame,  # Use mode_frame instead of self.control_frame
                text="Reset Balance",
                command=self.reset_paper_balance
            )
            self.reset_balance_button.grid(row=0, column=7, padx=5)
        

            # === BASIC PARAMETERS TAB ===
            basic_frame = ttk.Frame(param_notebook)
            param_notebook.add(basic_frame, text='Basic Parameters')

            # Entry Conditions Frame
            entry_frame = ttk.LabelFrame(basic_frame, text="Entry Conditions")
            entry_frame.grid(row=0, column=0, padx=5, pady=5, sticky="ew")

            ttk.Label(entry_frame, text="Profit Target (%)").grid(row=0, column=0, padx=5, pady=2, sticky="w")
            self.profit_target = ttk.Entry(entry_frame, width=8)
            self.profit_target.insert(0, "1.2")
            self.profit_target.grid(row=0, column=1, padx=5, pady=2)

            ttk.Label(entry_frame, text="Min Price Rise (%)").grid(row=1, column=0, padx=5, pady=2, sticky="w")
            self.price_rise_min = ttk.Entry(entry_frame, width=8)
            self.price_rise_min.insert(0, "0.1")
            self.price_rise_min.grid(row=1, column=1, padx=5, pady=2)

            ttk.Label(entry_frame, text="Volume Surge (%)").grid(row=2, column=0, padx=5, pady=2, sticky="w")
            self.volume_surge = ttk.Entry(entry_frame, width=8)
            self.volume_surge.insert(0, "120")
            self.volume_surge.grid(row=2, column=1, padx=5, pady=2)

            # Exit Conditions Frame
            exit_frame = ttk.LabelFrame(basic_frame, text="Exit Conditions")
            exit_frame.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

            ttk.Label(exit_frame, text="Stop Loss (%)").grid(row=0, column=0, padx=5, pady=2, sticky="w")
            self.stop_loss = ttk.Entry(exit_frame, width=8)
            self.stop_loss.insert(0, "0.5")
            self.stop_loss.grid(row=0, column=1, padx=5, pady=2)

            ttk.Label(exit_frame, text="Trailing Stop (%)").grid(row=1, column=0, padx=5, pady=2, sticky="w")
            self.trailing_stop = ttk.Entry(exit_frame, width=8)
            self.trailing_stop.insert(0, "0.2")
            self.trailing_stop.grid(row=1, column=1, padx=5, pady=2)

            ttk.Label(exit_frame, text="Trailing Activation (%)").grid(row=2, column=0, padx=5, pady=2, sticky="w")
            self.trailing_activation = ttk.Entry(exit_frame, width=8)
            self.trailing_activation.insert(0, "0.4")
            self.trailing_activation.grid(row=2, column=1, padx=5, pady=2)

            # Risk Management Frame
            risk_frame = ttk.LabelFrame(basic_frame, text="Risk Management")
            risk_frame.grid(row=0, column=2, padx=5, pady=5, sticky="ew")

            ttk.Label(risk_frame, text="Position Size (USD)").grid(row=0, column=0, padx=5, pady=2, sticky="w")
            self.position_size = ttk.Entry(risk_frame, width=8)
            self.position_size.insert(0, "150")
            self.position_size.grid(row=0, column=1, padx=5, pady=2)

            ttk.Label(risk_frame, text="Max Position (% Balance)").grid(row=1, column=0, padx=5, pady=2, sticky="w")
            self.max_position_percent = ttk.Entry(risk_frame, width=8)
            self.max_position_percent.insert(0, "20")
            self.max_position_percent.grid(row=1, column=1, padx=5, pady=2)

            ttk.Label(risk_frame, text="Daily Loss Limit (%)").grid(row=2, column=0, padx=5, pady=2, sticky="w")
            self.daily_loss_limit = ttk.Entry(risk_frame, width=8)
            self.daily_loss_limit.insert(0, "8")
            self.daily_loss_limit.grid(row=2, column=1, padx=5, pady=2)
            # Market Filters Frame
            filters_frame = ttk.LabelFrame(basic_frame, text="Market Filters")
            filters_frame.grid(row=1, column=0, columnspan=3, padx=5, pady=5, sticky="ew")

            # First row of filters
            ttk.Label(filters_frame, text="Min Volume (USD)").grid(row=0, column=0, padx=5, pady=2, sticky="w")
            self.min_volume_entry = ttk.Entry(filters_frame, width=8)
            self.min_volume_entry.insert(0, "300")
            self.min_volume_entry.grid(row=0, column=1, padx=5, pady=2)

            ttk.Label(filters_frame, text="Max Active Trades").grid(row=0, column=2, padx=5, pady=2, sticky="w")
            self.max_trades_entry = ttk.Entry(filters_frame, width=8)
            self.max_trades_entry.insert(0, "5")
            self.max_trades_entry.grid(row=0, column=3, padx=5, pady=2)

            # Second row of filters
            ttk.Label(filters_frame, text="Top Ranked Pairs").grid(row=1, column=0, padx=5, pady=2, sticky="w")
            self.top_list_size = ttk.Entry(filters_frame, width=8)
            self.top_list_size.insert(0, "20")
            self.top_list_size.grid(row=1, column=1, padx=5, pady=2)

            ttk.Label(filters_frame, text="Required Conditions").grid(row=1, column=2, padx=5, pady=2, sticky="w")
            self.required_conditions = ttk.Entry(filters_frame, width=8)
            self.required_conditions.insert(0, "3")
            self.required_conditions.grid(row=1, column=3, padx=5, pady=2)

            ttk.Label(filters_frame, text="Max Spread (%):").grid(row=4, column=0, padx=5, pady=2, sticky="w")
            self.max_spread = ttk.Entry(filters_frame, width=8)
            self.max_spread.insert(0, "0.3")  # Default to 0.3%
            self.max_spread.grid(row=4, column=1, padx=5, pady=2)

            # Third row of filters
            ttk.Label(filters_frame, text="Consecutive Rises").grid(row=2, column=0, padx=5, pady=2, sticky="w")
            self.consecutive_rises = ttk.Entry(filters_frame, width=8)
            self.consecutive_rises.insert(0, "2")
            self.consecutive_rises.grid(row=2, column=1, padx=5, pady=2)

            ttk.Label(filters_frame, text="Volume Increase (%)").grid(row=2, column=2, padx=5, pady=2, sticky="w")
            self.volume_increase = ttk.Entry(filters_frame, width=8)
            self.volume_increase.insert(0, "10")
            self.volume_increase.grid(row=2, column=3, padx=5, pady=2)

            # Fourth row of filters (adding momentum threshold)
            ttk.Label(filters_frame, text="Momentum Min (%)").grid(row=3, column=0, padx=5, pady=2, sticky="w")
            self.momentum_threshold = ttk.Entry(filters_frame, width=8)
            self.momentum_threshold.insert(0, "0.2")
            self.momentum_threshold.grid(row=3, column=1, padx=5, pady=2)

            # === VALIDATION CRITERIA TAB ===
            validation_frame = ttk.Frame(param_notebook)
            param_notebook.add(validation_frame, text='Validation Criteria')

            # Validation Criteria
            ttk.Label(validation_frame, text="Trend Strength (Beta)").grid(row=0, column=0, padx=5, pady=2, sticky="w")
            self.momentum_beta = ttk.Entry(validation_frame, width=8)
            self.momentum_beta.insert(0, "0.4")
            self.momentum_beta.grid(row=0, column=1, padx=5, pady=2)
            ttk.Label(validation_frame, text="Higher = stronger trend required").grid(row=0, column=2, padx=5, pady=2, sticky="w")

            ttk.Label(validation_frame, text="Price Momentum (Alpha)").grid(row=1, column=0, padx=5, pady=2, sticky="w")
            self.price_alpha = ttk.Entry(validation_frame, width=8)
            self.price_alpha.insert(0, "0.03")
            self.price_alpha.grid(row=1, column=1, padx=5, pady=2)
            ttk.Label(validation_frame, text="Higher = faster price movement required").grid(row=1, column=2, padx=5, pady=2, sticky="w")

            ttk.Label(validation_frame, text="Momentum Quality (Theta)").grid(row=2, column=0, padx=5, pady=2, sticky="w")
            self.time_theta = ttk.Entry(validation_frame, width=8)
            self.time_theta.insert(0, "0.25")
            self.time_theta.grid(row=2, column=1, padx=5, pady=2)
            ttk.Label(validation_frame, text="Lower = more stable momentum required").grid(row=2, column=2, padx=5, pady=2, sticky="w")

            ttk.Label(validation_frame, text="Volatility Filter (Vega)").grid(row=3, column=0, padx=5, pady=2, sticky="w")
            self.vol_vega = ttk.Entry(validation_frame, width=8)
            self.vol_vega.insert(0, "0.2")
            self.vol_vega.grid(row=3, column=1, padx=5, pady=2)
            ttk.Label(validation_frame, text="Lower = less volatile markets required").grid(row=3, column=2, padx=5, pady=2, sticky="w")

            ttk.Label(validation_frame, text="Volume Quality (Rho)").grid(row=4, column=0, padx=5, pady=2, sticky="w")
            self.volume_rho = ttk.Entry(validation_frame, width=8)
            self.volume_rho.insert(0, "0.3")
            self.volume_rho.grid(row=4, column=1, padx=5, pady=2)
            ttk.Label(validation_frame, text="Higher = stronger volume required").grid(row=4, column=2, padx=5, pady=2, sticky="w")

            # === ADVANCED PARAMETERS TAB ===
            advanced_frame = ttk.Frame(param_notebook)
            param_notebook.add(advanced_frame, text='Advanced Parameters')

            # Technical Indicators Frame
            indicators_frame = ttk.LabelFrame(advanced_frame, text="Technical Indicators")
            indicators_frame.grid(row=0, column=0, padx=5, pady=5, sticky="ew")

            # RSI Settings
            ttk.Label(indicators_frame, text="RSI Period:").grid(row=0, column=0, padx=5, pady=2, sticky="w")
            self.rsi_period = ttk.Entry(indicators_frame, width=8)
            self.rsi_period.insert(0, "14")
            self.rsi_period.grid(row=0, column=1, padx=5, pady=2)

            ttk.Label(indicators_frame, text="RSI Overbought:").grid(row=1, column=0, padx=5, pady=2, sticky="w")
            self.rsi_overbought = ttk.Entry(indicators_frame, width=8)
            self.rsi_overbought.insert(0, "70")
            self.rsi_overbought.grid(row=1, column=1, padx=5, pady=2)

            ttk.Label(indicators_frame, text="RSI Oversold:").grid(row=2, column=0, padx=5, pady=2, sticky="w")
            self.rsi_oversold = ttk.Entry(indicators_frame, width=8)
            self.rsi_oversold.insert(0, "30")
            self.rsi_oversold.grid(row=2, column=1, padx=5, pady=2)

            # Add Fee Structure Frame
            fee_frame = ttk.LabelFrame(advanced_frame, text="Fee Structure")
            fee_frame.grid(row=1, column=0, padx=5, pady=5, sticky="ew")

            ttk.Label(fee_frame, text="Maker Fee (%):").grid(row=0, column=0, padx=5, pady=2, sticky="w")
            ttk.Entry(fee_frame, textvariable=self.maker_fee_var, width=8).grid(row=0, column=1, padx=5, pady=2)

            ttk.Label(fee_frame, text="Taker Fee (%):").grid(row=1, column=0, padx=5, pady=2, sticky="w")
            ttk.Entry(fee_frame, textvariable=self.taker_fee_var, width=8).grid(row=1, column=1, padx=5, pady=2)

            initial_maker_fee = float(self.maker_fee_var.get()) / 100
            initial_taker_fee = float(self.taker_fee_var.get()) / 100
            initial_total_fee = (initial_maker_fee + initial_taker_fee) * 100 

            # Total fees display
            self.total_fees_label = ttk.Label(fee_frame, text=f"Total Round-Trip Fee: {initial_total_fee:.2f}%")
            self.total_fees_label.grid(row=2, column=0, columnspan=2, padx=5, pady=2, sticky="w")

            # Update button for fees
            ttk.Button(fee_frame, text="Update Fees", command=self.update_fees).grid(row=3, column=0, columnspan=2, padx=5, pady=5)


            # Add tooltips for Advanced Parameters
            self.add_tooltip(self.rsi_period, 
                "Period length for RSI calculation")
            self.add_tooltip(self.rsi_overbought, 
                "RSI level considered overbought")
            self.add_tooltip(self.rsi_oversold, 
                "RSI level considered oversold")

            # Buttons Frame
            button_frame = ttk.Frame(control_frame)
            button_frame.grid(row=2, column=0, columnspan=2, pady=10)

            # Add buttons using ttk
            self.apply_button = ttk.Button(button_frame, 
                text="Apply Conditions",
                command=self.validate_conditions)
            self.apply_button.grid(row=0, column=0, padx=5)

            self.live_update_button = ttk.Button(button_frame,
                text="Live Update",
                command=self.live_update_conditions)
            self.live_update_button.grid(row=0, column=1, padx=5)

            self.close_profitable_button = ttk.Button(button_frame,
                text="Close Profitable",
                command=self.close_profitable_positions)
            self.close_profitable_button.grid(row=0, column=2, padx=5)

            self.close_all_button = ttk.Button(button_frame,
                text="Close All Trades",
                command=self.close_all_positions)
            self.close_all_button.grid(row=0, column=3, padx=5)

            self.verify_trades_button = ttk.Button(
                control_frame, 
                text="Verify Trades", 
                command=self.verify_active_trades
            )
            self.verify_trades_button.grid(row=0, column=8, padx=5)

            # Chart Frame
            self.chart_frame = ttk.LabelFrame(left_column, text="Price Charts")
            self.chart_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
            
            # Trading Log
            log_frame = ttk.LabelFrame(left_column, text="Trading Log")
            log_frame.grid(row=2, column=0, sticky="ew", padx=5, pady=5)
            self.log_text = scrolledtext.ScrolledText(log_frame, height=6)
            self.log_text.pack(fill=tk.BOTH, expand=True)

            # === RIGHT COLUMN CONTENTS ===
            # Performance Metrics Frame
            metrics_frame = ttk.LabelFrame(right_column, text="Performance Metrics")
            metrics_frame.grid(row=0, column=0, sticky="ew", padx=5, pady=5)

            self.total_profit_label = ttk.Label(metrics_frame, text="Total Profit: 0.00 USD")
            self.total_profit_label.pack(anchor="w", padx=5, pady=2)

            self.total_fees_label = ttk.Label(metrics_frame, text="Total Fees: 0.00 USD")
            self.total_fees_label.pack(anchor="w", padx=5, pady=2)

            self.net_profit_label = ttk.Label(metrics_frame, text="Net Profit: 0.00 USD")
            self.net_profit_label.pack(anchor="w", padx=5, pady=2)

            self.win_loss_label = ttk.Label(metrics_frame, text="Win/Loss: 0/0 (0.0%)")
            self.win_loss_label.pack(anchor="w", padx=5, pady=2)

            # Trade History Frame
            history_frame = ttk.LabelFrame(right_column, text="Trade History")
            history_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
            self.history_text = scrolledtext.ScrolledText(history_frame, height=10)
            self.history_text.pack(fill=tk.BOTH, expand=True)

            # Active Trades Frame
            trades_frame = ttk.LabelFrame(right_column, text="Active Trades")
            trades_frame.grid(row=2, column=0, sticky="ew", padx=5, pady=5)
            self.trades_text = scrolledtext.ScrolledText(trades_frame, height=6)
            self.trades_text.pack(fill=tk.BOTH, expand=True)

            # Night Mode Button Frame
            night_mode_frame = ttk.Frame(right_column)
            night_mode_frame.grid(row=3, column=0, sticky="ew", padx=5, pady=5)

            self.night_mode_button = ttk.Button(
                night_mode_frame,
                text="Dark Mode",
                command=self.toggle_night_mode
            )
            self.night_mode_button.pack(fill=tk.X)

            # Configure grid weights
            left_column.grid_rowconfigure(1, weight=3)
            left_column.grid_rowconfigure(2, weight=1)
            right_column.grid_rowconfigure(1, weight=2)
            right_column.grid_rowconfigure(2, weight=1)

            # Set minimum window size
            self.root.minsize(1200, 800)

            # Initialize chart
            self.setup_chart()
            self.initialize_fees()
            
        except Exception as e:
            self.log_trade(f"Error setting up GUI: {str(e)}")
            raise

    def update_fees(self):
        """Update fee structure based on user input"""
        try:
            # Use the StringVar objects directly
            maker_fee_str = self.maker_fee_var.get()
            taker_fee_str = self.taker_fee_var.get()
            
            # Debug: Print the raw values from the StringVar objects
            self.log_trade(f"DEBUG: maker_fee_str = '{maker_fee_str}'")
            self.log_trade(f"DEBUG: taker_fee_str = '{taker_fee_str}'")
            
            # Parse fee values
            maker_fee = float(maker_fee_str) / 100  # Convert from percentage to decimal
            taker_fee = float(taker_fee_str) / 100  # Convert from percentage to decimal
            
            # Debug: Print the converted values
            self.log_trade(f"DEBUG: maker_fee (converted) = {maker_fee}")
            self.log_trade(f"DEBUG: taker_fee (converted) = {taker_fee}")
            
            # Update instance variables
            self.maker_fee = maker_fee
            self.taker_fee = taker_fee
            self.total_fee_percentage = taker_fee + maker_fee  # Assuming both entry and exit are taker orders
            
            # Debug: Print the updated instance variables
            self.log_trade(f"DEBUG: self.maker_fee = {self.maker_fee}")
            self.log_trade(f"DEBUG: self.taker_fee = {self.taker_fee}")
            self.log_trade(f"DEBUG: self.total_fee_percentage = {self.total_fee_percentage}")
            
            # Update the total fees label
            total_fee_pct = self.total_fee_percentage * 100
            self.total_fees_label.config(text=f"Total Round-Trip Fee: {total_fee_pct:.2f}%")
            
            # Log the update
            self.log_trade(f"Fee structure updated: Maker {maker_fee*100:.2f}%, Taker {taker_fee*100:.2f}%, Total {total_fee_pct:.2f}%")
            
            # Update tooltips
            self.add_tooltip(self.profit_target, 
                f"Target profit percentage for trades (min: {total_fee_pct:.2f}% to cover fees)")
            
            # Validate parameters to ensure profit target is above fees
            self.validate_parameters()
            
        except ValueError as e:
            self.log_trade(f"Error updating fees: {str(e)}")

    def setup_tooltips(self):
        """Setup tooltips for all parameters"""
        try:
            # Basic Parameters Tooltips
            self.add_tooltip(self.profit_target, 
                "Target profit percentage for trades (min: 0.8% to cover Kraken's 0.8% total fees)")
            self.add_tooltip(self.price_rise_min, 
                "Minimum price increase required before entering a trade")
            self.add_tooltip(self.volume_surge, 
                "Required volume increase compared to average")
            
            # Exit Conditions Tooltips
            self.add_tooltip(self.stop_loss, 
                "Maximum allowed loss percentage before closing position")
            self.add_tooltip(self.trailing_stop, 
                "Dynamic stop loss that follows price movement")
            self.add_tooltip(self.trailing_activation, 
                "Profit percentage required to activate trailing stop")
                
            # Risk Management Tooltips
            self.add_tooltip(self.position_size, 
                "Base position size in USD for each trade")
            self.add_tooltip(self.max_position_percent, 
                "Maximum position size as percentage of total balance")
            self.add_tooltip(self.daily_loss_limit, 
                "Maximum allowed loss percentage per day")
                
            # Market Filters Tooltips
            self.add_tooltip(self.min_volume_entry, 
                "Minimum 24h trading volume required for trading")
            self.add_tooltip(self.max_trades_entry, 
                "Maximum number of concurrent active trades")
            self.add_tooltip(self.top_list_size, 
                "Number of top pairs to analyze")
            self.add_tooltip(self.required_conditions, 
                "Minimum number of conditions that must be met for entry")
            self.add_tooltip(self.max_spread,
                "Maximum bid-ask spread percentage allowed for trading")
                
            # Validation Criteria Tooltips
            self.add_tooltip(self.momentum_beta, 
                "Trend strength indicator (0.1-1.0). Higher values require stronger trends")
            self.add_tooltip(self.price_alpha, 
                "Price momentum indicator (0.01-0.1). Higher values require faster price movement")
            self.add_tooltip(self.time_theta, 
                "Momentum stability indicator (0.1-1.0). Lower values require more stable momentum")
            self.add_tooltip(self.vol_vega, 
                "Volatility filter (0.1-1.0). Lower values filter out high volatility")
            self.add_tooltip(self.volume_rho, 
                "Volume quality indicator (0.1-1.0). Higher values require stronger volume")
                
            # Technical Indicators Tooltips
            self.add_tooltip(self.rsi_period, 
                "Period for RSI calculation (typical: 14)")
            self.add_tooltip(self.rsi_overbought, 
                "RSI level considered overbought (typical: 70)")
            self.add_tooltip(self.rsi_oversold, 
                "RSI level considered oversold (typical: 30)")
            self.add_tooltip(self.macd_fast, 
                "Fast EMA period for MACD (typical: 12)")
            self.add_tooltip(self.macd_slow, 
                "Slow EMA period for MACD (typical: 26)")
            self.add_tooltip(self.macd_signal, 
                "Signal line period for MACD (typical: 9)")
            
            # Order Book Tooltips
            self.add_tooltip(self.book_depth, 
                "Number of order book levels to analyze for support/resistance")
            self.add_tooltip(self.buy_wall_ratio, 
                "Ratio to identify significant buy walls (>1.0 indicates strong support)")
            self.add_tooltip(self.sell_wall_ratio, 
                "Ratio to identify significant sell walls (>1.0 indicates strong resistance)")
            
            # Fee Structure Tooltips
            self.add_tooltip(self.maker_fee,
                "Fee percentage for maker orders (limit orders that add liquidity)")
            self.add_tooltip(self.taker_fee,
                "Fee percentage for taker orders (market orders that remove liquidity)")
            self.add_tooltip(self.total_fees_label,
                "Total fees for a complete trade (entry + exit)")
            
            # Position Sizing Tooltips
            self.add_tooltip(self.scale_in_levels, 
                "Number of entry points for scaling into position")
            self.add_tooltip(self.level_gap, 
                "Price gap percentage between scale-in levels")
                
        except Exception as e:
            self.log_trade(f"Error setting up tooltips: {str(e)}")

    def add_tooltip(self, widget, text):
        """Add tooltip to widget with improved styling and positioning"""
        try:
            def show_tooltip(event):
                tooltip = tk.Toplevel()
                tooltip.wm_overrideredirect(True)
                
                # Calculate position
                x = event.x_root + 15
                y = event.y_root + 10
                
                # Ensure tooltip stays within screen bounds
                screen_width = self.root.winfo_screenwidth()
                screen_height = self.root.winfo_screenheight()
                
                # Create tooltip with dark theme styling
                label = ttk.Label(tooltip, 
                                text=text,
                                background="#2d2d2d",
                                foreground="white",
                                relief="solid",
                                borderwidth=1,
                                padding=(5, 3))
                label.pack()
                
                # Adjust position if tooltip would go off screen
                tooltip_width = label.winfo_reqwidth()
                tooltip_height = label.winfo_reqheight()
                
                if x + tooltip_width > screen_width:
                    x = screen_width - tooltip_width - 5
                if y + tooltip_height > screen_height:
                    y = screen_height - tooltip_height - 5
                    
                tooltip.wm_geometry(f"+{x}+{y}")
                
                def hide_tooltip():
                    tooltip.destroy()
                    
                widget.tooltip = tooltip
                widget.bind('<Leave>', lambda e: hide_tooltip())
                
            widget.bind('<Enter>', show_tooltip)
            
        except Exception as e:
            self.log_trade(f"Error adding tooltip: {str(e)}")

    def calculate_momentum_intensity(self, df):
        """Calculate momentum intensity (Beta)"""
        try:
            if df is None or len(df) < 5:
                return 0.0
                
            # Calculate momentum over last 5 periods
            price_changes = df['price'].pct_change().dropna()
            if len(price_changes) < 5:
                return 0.0
                
            # Get last 5 price changes
            recent_changes = price_changes.tail(5)
            
            # Calculate average absolute change
            avg_abs_change = recent_changes.abs().mean() * 100
            
            # Calculate direction consistency (all same direction)
            direction_consistency = (recent_changes > 0).all() or (recent_changes < 0).all()
            
            # Boost if consistent direction
            if direction_consistency:
                avg_abs_change *= 1.5
                
            return min(avg_abs_change, 1.0)  # Cap at 1.0
            
        except Exception as e:
            self.log_trade(f"Error calculating momentum intensity: {str(e)}")
            return 0.0
        
    def calculate_momentum_quality(self, df):
        """Calculate momentum quality (Theta)"""
        try:
            if df is None or len(df) < 5:
                return 1.0  # Default to high instability
                
            # Calculate momentum
            momentum = df['price'].pct_change().dropna()
            if len(momentum) < 5:
                return 1.0
                
            # Calculate stability as standard deviation
            stability = momentum.rolling(window=5).std().iloc[-1] * 100
            
            # Normalize to 0-1 range (lower is better)
            normalized_stability = min(stability, 1.0)
            
            return normalized_stability
            
        except Exception as e:
            self.log_trade(f"Error calculating momentum quality: {str(e)}")
            return 1.0  # Default to high instability
    
    def calculate_price_acceleration(self, df):
        """Calculate price acceleration (Alpha)"""
        try:
            if df is None or len(df) < 5:
                return 0.0
                
            # Get price changes
            price_changes = df['price'].pct_change().dropna()
            if len(price_changes) < 5:
                return 0.0
                
            # Calculate acceleration as change in momentum
            momentum_changes = price_changes.diff().dropna()
            if len(momentum_changes) < 3:
                return 0.0
                
            # Get recent momentum changes
            recent_accel = momentum_changes.tail(3)
            
            # Calculate average acceleration
            avg_accel = recent_accel.mean() * 100
            
            # Normalize to 0-1 range
            normalized_accel = min(abs(avg_accel) * 10, 1.0)
            
            return normalized_accel
            
        except Exception as e:
            self.log_trade(f"Error calculating price acceleration: {str(e)}")
            return 0.0

    def calculate_volatility(self, df):
        """Calculate price volatility as percentage"""
        try:
            if df is None or len(df) < 10:
                return 0
                
            # Use close or price column
            price_col = 'close' if 'close' in df.columns else 'price'
            
            if price_col not in df.columns:
                return 0
                
            # Calculate standard deviation of returns
            returns = df[price_col].pct_change().dropna()
            
            # Return annualized volatility
            return returns.std() * 100
                
        except Exception as e:
            self.log_trade(f"Error calculating volatility: {str(e)}")
            return 0

    def update_market_condition_display(self):
        """Update the UI with current market condition information"""
        try:
            # Get current market conditions
            market = self.analyze_market_conditions()
            
            # Format the display text
            if market['state'] == 'bearish':
                condition_text = f"BEARISH (Strength: {market['strength']:.2f})"
                color = "red"
            elif market['state'] == 'bullish':
                condition_text = f"BULLISH (Strength: {market['strength']:.2f})"
                color = "green"
            else:
                condition_text = f"NEUTRAL (Strength: {market['strength']:.2f})"
                color = "yellow"
                
            # Update the label if it exists
            if hasattr(self, 'market_condition_label'):
                self.market_condition_label.config(text=f"Market: {condition_text}", fg=color)
            
            # Log the update
            self.log_trade(f"Market condition updated: {condition_text}")
            
        except Exception as e:
            self.log_trade(f"Error updating market condition display: {str(e)}")

    def calculate_volatility_sensitivity(self, df):
        """Calculate volatility sensitivity (Vega)"""
        try:
            if df is None or len(df) < 5:
                return 1.0  # Default to high volatility
                
            # Calculate returns
            returns = df['price'].pct_change().dropna()
            if len(returns) < 5:
                return 1.0
                
            # Calculate volatility as standard deviation
            volatility = returns.rolling(window=5).std().iloc[-1] * 100
            
            # Normalize to 0-1 range
            normalized_volatility = min(volatility, 1.0)
            
            return normalized_volatility
            
        except Exception as e:
            self.log_trade(f"Error calculating volatility: {str(e)}")
            return 1.0  # Default to high volatility

    def calculate_volume_impact(self, df, current_volume):
        """Calculate volume impact (Rho)"""
        try:
            if df is None or len(df) < 5:
                return 0.0
                
            # Calculate average volume
            avg_volume = df['volume'].mean()
            if avg_volume == 0:
                return 0.0
                
            # Calculate volume ratio
            volume_ratio = current_volume / avg_volume
            
            # Normalize to 0-1 range
            normalized_ratio = min(volume_ratio, 1.0)
            
            return normalized_ratio
            
        except Exception as e:
            self.log_trade(f"Error calculating volume impact: {str(e)}")
            return 0.0

    def calculate_current_momentum(self, df):
        """Calculate current price momentum"""
        try:
            if df is None or len(df) < 2:
                return 0.0
                
            # Calculate percentage change between last two prices
            last_price = df['price'].iloc[-1]
            prev_price = df['price'].iloc[-2]
            
            momentum = ((last_price - prev_price) / prev_price) * 100
            return momentum
            
        except Exception as e:
            self.log_trade(f"Error calculating momentum: {str(e)}")
            return 0.0

    def advanced_checks(self, symbol, df):
        """Advanced market checks including order book analysis and RSI"""
        try:
            # 1. EMA Cross (5/15)
            if 'ema_5' in df.columns and 'ema_15' in df.columns and len(df) >= 2:
                ema_cross = (df['ema_5'].iloc[-2] < df['ema_15'].iloc[-2]) and \
                            (df['ema_5'].iloc[-1] > df['ema_15'].iloc[-1])
                self.log_trade(f"EMA Cross check for {symbol}: {'Bullish' if ema_cross else 'No cross'}")
            else:
                self.log_trade(f"Insufficient EMA data for {symbol}")
                ema_cross = False
            
            # 2. Volume-Weighted Momentum
            if 'price' in df.columns and 'volume' in df.columns and len(df) >= 5:
                vwap = (df['price'] * df['volume']).cumsum() / df['volume'].cumsum()
                vwm = (vwap.iloc[-1] - vwap.iloc[-5]) / vwap.iloc[-5] * 100
                self.log_trade(f"Volume-Weighted Momentum for {symbol}: {vwm:.2f}%")
            else:
                self.log_trade(f"Insufficient data for VWAP calculation for {symbol}")
                vwm = 0
            
            # 3. Order Book Imbalance
            try:
                ob = self.exchange.fetch_order_book(symbol)
                bid_vol = sum(x[1] for x in ob['bids'][:3])
                ask_vol = sum(x[1] for x in ob['asks'][:3])
                ob_ratio = bid_vol / (ask_vol + 0.0001)  # Avoid division by zero
                self.log_trade(f"Order Book Imbalance for {symbol}: {ob_ratio:.2f}")
            except Exception as e:
                self.log_trade(f"Order book fetch error: {str(e)}")
                ob_ratio = 1.0  # Neutral value on error
            
            # 4. RSI Check
            rsi_period = int(self.rsi_period.get()) if hasattr(self, 'rsi_period') else 14
            rsi_column = f'rsi_{rsi_period}'
            
            # Check if RSI column exists and has valid data
            if rsi_column in df.columns:
                rsi_value = df[rsi_column].iloc[-1]
                
                # Check if RSI is a valid number
                if pd.isna(rsi_value):
                    self.log_trade(f"RSI value is NaN for {symbol}, calculating directly")
                    
                    # Calculate RSI directly if it's NaN
                    if len(df) >= rsi_period + 1:
                        delta = df['price'].diff().dropna()
                        gains = delta.where(delta > 0, 0)
                        losses = -delta.where(delta < 0, 0)
                        
                        avg_gain = gains.rolling(window=rsi_period).mean().iloc[-1]
                        avg_loss = losses.rolling(window=rsi_period).mean().iloc[-1]
                        
                        # Avoid division by zero
                        if avg_loss == 0:
                            rsi_value = 100
                        else:
                            rs = avg_gain / avg_loss
                            rsi_value = 100 - (100 / (1 + rs))
                        
                        # Update the DataFrame with the calculated value
                        df.at[df.index[-1], rsi_column] = rsi_value
                        self.price_data[symbol] = df
                        
                        self.log_trade(f"Directly calculated RSI for {symbol}: {rsi_value:.2f}")
                    else:
                        self.log_trade(f"Insufficient data to calculate RSI for {symbol}")
                        return False
                
                rsi_overbought = float(self.rsi_overbought.get()) if hasattr(self, 'rsi_overbought') else 70
                rsi_oversold = float(self.rsi_oversold.get()) if hasattr(self, 'rsi_oversold') else 30
                
                # Avoid overbought conditions (RSI > overbought threshold)
                if rsi_value > rsi_overbought:
                    self.log_trade(f"Rejected {symbol}: RSI indicates overbought condition ({rsi_value:.2f} > {rsi_overbought})")
                    return False
                
                self.log_trade(f"RSI for {symbol}: {rsi_value:.2f} (Overbought: {rsi_overbought}, Oversold: {rsi_oversold})")
            else:
                self.log_trade(f"RSI column {rsi_column} not found in DataFrame for {symbol}")
                return False
            
            self.log_trade(f"Advanced checks passed for {symbol}:")
            self.log_trade(f"- EMA Cross: {ema_cross}")
            self.log_trade(f"- VWAP Momentum: {vwm:.2f}%")
            self.log_trade(f"- Order Book Ratio: {ob_ratio:.2f}")
            self.log_trade(f"- RSI: {rsi_value:.2f} (Overbought: {rsi_overbought}, Oversold: {rsi_oversold})")
            
            return all([
                ema_cross,          # EMA crossover
                vwm > 0.2,          # Min 0.2% VWAP rise
                ob_ratio > 1.5,     # Bids > Asks by 50%
                True                # RSI check already handled above
            ])
        
        except Exception as e:
            self.log_trade(f"Error in advanced checks for {symbol}: {str(e)}")
            return False

    def calculate_volume_increase(self, df):
        """Calculate volume increase percentage"""
        try:
            if df is None or len(df) < 2:
                return 0.0
                
            # Calculate percentage change between last two volumes
            last_volume = df['volume'].iloc[-1]
            prev_volume = df['volume'].iloc[-2]
            
            if prev_volume == 0:
                return 0.0
                
            volume_increase = ((last_volume - prev_volume) / prev_volume) * 100
            return volume_increase
            
        except Exception as e:
            self.log_trade(f"Error calculating volume increase: {str(e)}")
            return 0.0


    def validate_parameters(self):
        """Validate all trading parameters"""
        try:
            # Create a dictionary of parameters to validate
            validation_rules = {
                'profit_target': {
                    'min': self.total_fee_percentage * 100,  # Minimum to cover fees
                    'max': 5.0,
                    'value': float(self.profit_target.get()),
                    'name': 'Profit Target'
                },
                'stop_loss': {
                    'min': 0.3,
                    'max': 2.0,
                    'value': float(self.stop_loss.get()),
                    'name': 'Stop Loss'
                },
                'trailing_stop': {
                    'min': 0.2,
                    'max': 1.0,
                    'value': float(self.trailing_stop.get()),
                    'name': 'Trailing Stop'
                },
                'trailing_activation': {
                    'min': 0.3,
                    'max': 2.0,
                    'value': float(self.trailing_activation.get()),
                    'name': 'Trailing Activation'
                }
            }
            
            # Add optional parameters only if they exist
            if hasattr(self, 'rsi_period'):
                validation_rules['rsi_period'] = {
                    'min': 5,
                    'max': 30,
                    'value': float(self.rsi_period.get()),
                    'name': 'RSI Period'
                }
                
            if hasattr(self, 'book_depth'):
                validation_rules['book_depth'] = {
                    'min': 5,
                    'max': 50,
                    'value': float(self.book_depth.get()),
                    'name': 'Order Book Depth'
                }
                
            if hasattr(self, 'scale_in_levels'):
                validation_rules['scale_in_levels'] = {
                    'min': 1,
                    'max': 5,
                    'value': float(self.scale_in_levels.get()),
                    'name': 'Scale-in Levels'
                }
                
            # Greek parameters - only add if they exist
            greek_params = ['momentum_beta', 'price_alpha', 'time_theta', 'vol_vega', 'volume_rho']
            for param in greek_params:
                if hasattr(self, param):
                    validation_rules[param] = {
                        'min': 0.001,
                        'max': 1.0,
                        'value': float(getattr(self, param).get()),
                        'name': param.replace('_', ' ').title()
                    }
                    
            # RSI thresholds - only add if they exist
            if hasattr(self, 'rsi_overbought') and hasattr(self, 'rsi_oversold'):
                validation_rules['rsi_overbought'] = {
                    'min': 1,
                    'max': 90,
                    'value': float(self.rsi_overbought.get()),
                    'name': 'RSI Overbought'
                }
                validation_rules['rsi_oversold'] = {
                    'min': 1,
                    'max': 90,
                    'value': float(self.rsi_oversold.get()),
                    'name': 'RSI Oversold'
                }

            # Validate each parameter
            for param, rules in validation_rules.items():
                if rules['value'] < rules['min'] or rules['value'] > rules['max']:
                    error_msg = f"{rules['name']} must be between {rules['min']} and {rules['max']}"
                    self.log_trade(error_msg)
                    raise ValueError(error_msg)

            # Cross-Parameter Validation - only if all required parameters exist
            if all(param in validation_rules for param in ['profit_target', 'stop_loss']):
                if validation_rules['stop_loss']['value'] >= validation_rules['profit_target']['value']:
                    raise ValueError("Stop loss must be less than profit target")
            
            if all(param in validation_rules for param in ['profit_target', 'trailing_stop']):
                if validation_rules['trailing_stop']['value'] >= validation_rules['profit_target']['value']:
                    raise ValueError("Trailing stop must be less than profit target")
            
            if all(param in validation_rules for param in ['profit_target', 'trailing_activation']):
                if validation_rules['trailing_activation']['value'] >= validation_rules['profit_target']['value']:
                    raise ValueError("Trailing activation must be less than profit target")
            
            if all(param in validation_rules for param in ['rsi_oversold', 'rsi_overbought']):
                if validation_rules['rsi_oversold']['value'] >= validation_rules['rsi_overbought']['value']:
                    raise ValueError("RSI oversold must be less than RSI overbought")

            # Balance Check
            if self.is_paper_trading:
                available_balance = self.paper_balance
            else:
                available_balance = float(self.exchange.fetch_balance()['USD']['free'])
            
            position_size = float(self.position_size.get())
            
            # Only check max position percent if it exists
            if hasattr(self, 'max_position_percent'):
                max_position_percent = float(self.max_position_percent.get()) / 100
                max_position = available_balance * max_position_percent
                
                if position_size > max_position:
                    raise ValueError(f"Position size exceeds maximum allowed ({max_position:.2f} USD)")

            self.log_trade("All parameters validated successfully")
            return True
                
        except ValueError as e:
            self.log_trade(f"Settings validation failed: {str(e)}")
            raise ValueError(f"Invalid settings: {str(e)}")

    def save_parameters(self):
        """Save current parameters to config file"""
        try:
            config = {
                'trading_parameters': {
                    'profit_target': self.profit_target.get(),
                    'stop_loss': self.stop_loss.get(),
                    'trailing_stop': self.trailing_stop.get(),
                    'trailing_activation': self.trailing_activation.get(),
                    'rsi_period': self.rsi_period.get(),
                    'rsi_overbought': self.rsi_overbought.get(),
                    'rsi_oversold': self.rsi_oversold.get(),
                    'macd_fast': self.macd_fast.get(),
                    'macd_slow': self.macd_slow.get(),
                    'macd_signal': self.macd_signal.get(),
                    'book_depth': self.book_depth.get(),
                    'buy_wall_ratio': self.buy_wall_ratio.get(),
                    'sell_wall_ratio': self.sell_wall_ratio.get(),
                    'scale_in_levels': self.scale_in_levels.get(),
                    'level_gap': self.level_gap.get()
                }
            }
            
            with open('trading_config.json', 'w') as f:
                json.dump(config, f, indent=4)
                
            self.log_trade("Parameters saved successfully")
            
        except Exception as e:
            self.log_trade(f"Error saving parameters: {str(e)}")

    def load_parameters(self):
        """Load parameters from config file"""
        try:
            if not os.path.exists('trading_config.json'):
                self.log_trade("No saved configuration found")
                return
                
            with open('trading_config.json', 'r') as f:
                config = json.load(f)
                
            params = config.get('trading_parameters', {})
            
            # Update GUI elements with loaded values
            for param, value in params.items():
                if hasattr(self, param):
                    widget = getattr(self, param)
                    if isinstance(widget, ttk.Entry):
                        widget.delete(0, tk.END)
                        widget.insert(0, str(value))
                        
            self.log_trade("Parameters loaded successfully")
            
        except Exception as e:
            self.log_trade(f"Error loading parameters: {str(e)}")

    def safe_execute(self, func, *args, **kwargs):
        """Safely execute functions with retry logic"""
        max_retries = 3
        retry_delay = 1

        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if attempt == max_retries - 1:
                    self.log_trade(f"Operation failed after {max_retries} attempts: {str(e)}")
                    raise
                time.sleep(retry_delay)

    def safe_update_gui(self, func):
        """Safely execute GUI updates from any thread"""
        if self.root and self.root.winfo_exists():
            self.root.after(0, func)

    def start_timer(self):
        """Start the runtime timer"""
        self.start_time = datetime.now()
        self.timer_running = True
        self.update_timer()
    def stop_timer(self):
        """Stop the runtime timer"""
        self.timer_running = False
    def update_timer(self):
        """Update the timer display"""
        if self.timer_running and self.start_time:
            try:
                current_time = datetime.now()
                elapsed = current_time - self.start_time
                
                # Convert to hours, minutes, seconds
                hours = int(elapsed.total_seconds() // 3600)
                minutes = int((elapsed.total_seconds() % 3600) // 60)
                seconds = int(elapsed.total_seconds() % 60)
                
                # Update timer label
                self.timer_label.config(text=f"Runtime: {hours:02d}:{minutes:02d}:{seconds:02d}")
                
                # Schedule next update
                self.root.after(1000, self.update_timer)
            except Exception as e:
                self.log_trade(f"Error updating timer: {str(e)}")

    def update_chart(self):
        """Update the price chart with active trades"""
        try:
            self.ax.clear()
            
            if not self.active_trades:
                self.ax.set_title("No Active Trades")
                self.ax.set_xlabel("Time")
                self.ax.set_ylabel("Price Change (%)")
                self.ax.grid(True, alpha=0.3)
                self.canvas.draw()
                return
            
            # Plot each active trade
            for trade_id, trade in self.active_trades.items():
                symbol = trade['symbol']
                
                # Ensure we have price history for this symbol
                if symbol not in self.price_history or len(self.price_history[symbol]) < 2:
                    # If no history, create a starting point
                    if symbol not in self.price_history:
                        self.price_history[symbol] = []
                    
                    # Add current point if we have it
                    if 'current_price' in trade:
                        current_time = datetime.now()
                        entry_time = trade.get('entry_time', current_time - timedelta(seconds=10))
                        
                        # Add entry point
                        self.price_history[symbol].append((entry_time, trade['entry_price']))
                        
                        # Add current point
                        self.price_history[symbol].append((current_time, trade['current_price']))
                        
                        self.log_trade(f"Created initial price history for {symbol} with {len(self.price_history[symbol])} points")
                
                # Now plot if we have data
                if symbol in self.price_history and len(self.price_history[symbol]) >= 2:
                    # Extract times and prices
                    times = [p[0] for p in self.price_history[symbol]]
                    prices = [p[1] for p in self.price_history[symbol]]
                    
                    # Calculate percentage change from entry
                    entry_price = trade['entry_price']
                    price_changes = [(price - entry_price) / entry_price * 100 for price in prices]
                    
                    # Convert times to numeric for plotting
                    time_nums = mdates.date2num(times)
                    
                    # Plot with proper formatting
                    self.ax.plot(time_nums, price_changes, label=f"{symbol} ({price_changes[-1]:.2f}%)")
                    
                    # Format x-axis as times
                    self.ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
                    self.ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            
            # Add horizontal lines for profit target and stop loss
            try:
                profit_target = float(self.profit_target.get())
                stop_loss = float(self.stop_loss.get())
                
                self.ax.axhline(y=profit_target, color='g', linestyle='--', alpha=0.5, label=f"Target: {profit_target}%")
                self.ax.axhline(y=-stop_loss, color='r', linestyle='--', alpha=0.5, label=f"Stop: -{stop_loss}%")
            except:
                pass
            
            self.ax.set_title("Active Trades (% Change from Entry)")
            self.ax.set_xlabel("Time")
            self.ax.set_ylabel("Price Change (%)")
            self.ax.legend(loc='upper left')
            self.ax.grid(True, alpha=0.3)
            
            # Set reasonable y-axis limits
            try:
                profit_target = float(self.profit_target.get())
                stop_loss = float(self.stop_loss.get())
                
                # Set y-axis limits to show a reasonable range around profit target and stop loss
                self.ax.set_ylim(
                    min(-stop_loss * 1.5, -1),  # At least show -1%
                    max(profit_target * 1.5, 1)   # At least show +1%
                )
            except:
                # Default limits if we can't get profit target and stop loss
                self.ax.set_ylim(-2, 2)  # Default to ±2% range
            
            # Format the plot for better readability
            self.fig.tight_layout()
            self.canvas.draw()
            
        except Exception as e:
            self.log_trade(f"Error updating chart: {str(e)}")

    def _update_chart_internal(self):
        """Internal chart update method that runs on main thread"""
        try:
            # Clear the figure
            self.ax.clear()
            
            if not self.active_trades:
                self.ax.set_title("No Active Trades")
                self.canvas.draw()
                return
                
            # Plot each active trade
            for i, (trade_id, trade) in enumerate(self.active_trades.items()):
                if trade['symbol'] in self.price_history:
                    history = self.price_history[trade['symbol']]
                    if len(history) > 0:
                        # Unzip the history data
                        times, prices = zip(*history[-50:])  # Only keep last 50 points
                        
                        # Calculate percentage change from entry
                        entry_price = trade['entry_price']
                        prices_pct = [(p - entry_price) / entry_price * 100 for p in prices]
                        
                        # Plot with distinct colors
                        color = ['blue', 'green', 'red', 'purple', 'orange'][i % 5]
                        self.ax.plot(range(len(prices)), prices_pct, 
                                label=f"{trade['symbol']} ({prices_pct[-1]:.2f}%)",
                                color=color,
                                linewidth=2)
                        
                        # Plot key levels
                        self.ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
                        self.ax.axhline(y=float(self.profit_target.get()), 
                                    color='green', linestyle=':', alpha=0.3)
                        self.ax.axhline(y=-float(self.stop_loss.get()),
                                    color='red', linestyle=':', alpha=0.3)
            
            if self.active_trades:
                self.ax.set_title("Active Trades (% Change from Entry)")
                self.ax.set_xlabel("Time Points")
                self.ax.set_ylabel("Price Change (%)")
                self.ax.legend(loc='upper left')
                self.ax.grid(True, alpha=0.3)
                
                # Set reasonable y-axis limits
                self.ax.set_ylim(
                    min(-float(self.stop_loss.get()) * 1.5, self.ax.get_ylim()[0]),
                    max(float(self.profit_target.get()) * 1.5, self.ax.get_ylim()[1])
                )
            
            # Draw without tight_layout
            self.canvas.draw()
            
        except Exception as e:
            self.log_trade(f"Chart update error: {str(e)}")

    def update_price_history(self, symbol, price):
        try:
            current_time = datetime.now()
            
            if symbol not in self.price_history:
                self.price_history[symbol] = []
                
            # Add new price point
            self.price_history[symbol].append((current_time, price))
            
            # Keep only last 50 points
            if len(self.price_history[symbol]) > 50:
                self.price_history[symbol] = self.price_history[symbol][-50:]
                
            # Force chart update
            self.update_chart()
            
        except Exception as e:
            self.log_trade(f"Error updating price history: {str(e)}")
    def update_all_displays(self):
        """Comprehensive display update system"""
        try:
            # Clear and update trades display
            self.trades_text.delete(1.0, tk.END)
            
            if not self.active_trades:
                self.trades_text.insert(tk.END, "No active trades\n")
            else:
                self.trades_text.insert(tk.END, f"Active Trades: {len(self.active_trades)}\n\n")
                
                for trade_id in list(self.active_trades.keys()):
                    try:
                        current_price = self.get_cached_price(trade['symbol'])['last']
                        profit_percentage = ((current_price - trade['entry_price']) / trade['entry_price']) * 100
                        time_in_trade = (datetime.now() - trade['timestamp']).total_seconds()
                        
                        trade_info = (
                            f"Symbol: {trade['symbol']}\n"
                            f"Entry: {trade['entry_price']:.8f}\n"
                            f"Current: {current_price:.8f}\n"
                            f"P/L: {profit_percentage:.2f}%\n"
                            f"Time: {time_in_trade:.1f}s\n"
                            f"Target: {trade['target_price']:.8f}\n"
                            f"-----------------\n"
                        )
                        
                        self.trades_text.insert(tk.END, trade_info)
                        
                        # Color coding
                        start_idx = self.trades_text.index("end-8c linestart")
                        end_idx = self.trades_text.index("end-1c")
                        
                        if profit_percentage > 0:
                            self.trades_text.tag_add("profit", start_idx, end_idx)
                            self.trades_text.tag_config("profit", foreground="green")
                        else:
                            self.trades_text.tag_add("loss", start_idx, end_idx)
                            self.trades_text.tag_config("loss", foreground="red")
                            
                    except Exception as e:
                        self.log_trade(f"Error updating display for {trade_id}: {str(e)}")
            
            # Update chart
            self.ax.clear()
            
            for trade_id, trade in list(self.active_trades.items()):
                if trade['symbol'] in self.price_history:
                    history = self.price_history[trade['symbol']]
                    if history:
                        times = [point[0] for point in history]
                        prices = [point[1] for point in history]
                        
                        # Calculate percentage change from entry
                        entry_price = trade['entry_price']
                        prices_pct = [(p - entry_price) / entry_price * 100 for p in prices]
                        
                        self.ax.plot(times, prices_pct, 
                                label=f"{trade['symbol']} ({prices_pct[-1]:.2f}%)",
                                linewidth=2)
                        
                        # Plot key levels
                        self.ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
                        self.ax.axhline(y=float(self.profit_target.get()), 
                                    color='green', linestyle=':', alpha=0.5)
            
            if self.active_trades:
                self.ax.set_title("Active Trades (% Change from Entry)")
                self.ax.set_xlabel("Time")
                self.ax.set_ylabel("Price Change (%)")
                self.ax.legend(loc='upper left')
                self.ax.grid(True, alpha=0.3)
            else:
                self.ax.set_title("No Active Trades")
                
            self.fig.tight_layout()
            self.canvas.draw()
            
            # Update metrics
            self.update_metrics()
            self.update_balance_display()
            
            # Force GUI update
            self.root.update_idletasks()
            
        except Exception as e:
            self.log_trade(f"Error in display update: {str(e)}")

    def close_all_positions(self):
        """Manually close all open positions with accurate P/L calculation"""
        try:
            if not self.active_trades:
                self.log_trade("No active trades to close")
                return

            for trade_id, trade in list(self.active_trades.items()):
                try:
                    # Get fresh price data with error checking
                    try:
                        ticker = self.exchange.fetch_ticker(trade['symbol'])
                        current_price = float(ticker['last'])
                    except Exception as e:
                        self.log_trade(f"Error getting price for {trade['symbol']}: {str(e)}")
                        # Use last known price as fallback
                        current_price = float(trade.get('current_price', trade['entry_price']))

                    # Calculate exact P/L
                    entry_price = float(trade['entry_price'])
                    amount = float(trade['amount'])
                    position_size = float(trade['position_size'])
                    
                    # Calculate raw P/L numbers
                    profit_pct = ((current_price - entry_price) / entry_price) * 100
                    gross_profit = (current_price - entry_price) * amount
                    
                    # Calculate fees
                    entry_fee = position_size * self.taker_fee
                    exit_fee = (position_size * (1 + (profit_pct/100))) * self.taker_fee
                    total_fees = entry_fee + exit_fee
                    
                    # Calculate net P/L
                    net_profit = gross_profit - total_fees
                    net_profit_pct = (net_profit / position_size) * 100

                    # Update paper balance
                    if self.is_paper_trading:
                        self.paper_balance += (position_size + net_profit)

                    # Log closure with exact numbers
                    self.log_trade(f"""
                    Closing trade {trade['symbol']}:
                    Entry: ${entry_price:.8f}
                    Exit: ${current_price:.8f}
                    Gross P/L: {profit_pct:.2f}%
                    Fees: ${total_fees:.2f}
                    Net P/L: {net_profit_pct:.2f}%
                    """)

                    # Update trade history with accurate numbers
                    self.update_trade_history(
                        symbol=trade['symbol'],
                        percentage=net_profit_pct,
                        profit=net_profit,
                        is_win=(net_profit > 0)
                    )

                    # Remove from active trades
                    del self.active_trades[trade_id]

                except Exception as e:
                    self.log_trade(f"Error closing trade {trade_id}: {str(e)}")
                    continue

            # Update displays
            self.update_active_trades_display()
            self.update_metrics()
            self.update_balance_display()
            self.log_trade("All positions closed")

        except Exception as e:
            self.log_trade(f"Error in close_all_positions: {str(e)}")

    def close_profitable_positions(self):
        """Manually close only positions that are currently profitable"""
        try:
            if not self.active_trades:
                self.log_trade("No active trades to close")
                return

            for trade_id, trade in list(self.active_trades.items()):
                try:
                    current_price = self.exchange.fetch_ticker(trade['symbol'])['last']
                    profit = (current_price - trade['entry_price']) * trade['amount']
                    
                    if profit > 0:
                        self.close_trade(trade_id, trade, profit, "manual profit take")
                        
                except Exception as e:
                    self.log_trade(f"Error closing trade {trade_id}: {str(e)}")

            self.log_trade("Profitable positions closed manually")
            
        except Exception as e:
            self.log_trade(f"Error closing profitable positions: {str(e)}")

    def update_balance_display(self):
        try:
            if self.is_paper_trading:
                # Calculate allocated balance (sum of position sizes in active trades)
                allocated_balance = sum(float(trade.get('position_size', 0)) for trade in self.active_trades.values())
                
                # Calculate unrealized profit/loss
                unrealized_pnl = 0
                for trade in self.active_trades.values():
                    entry_price = float(trade.get('entry_price', 0))
                    current_price = float(trade.get('current_price', entry_price))
                    position_size = float(trade.get('position_size', 0))
                    
                    if entry_price > 0:
                        price_change = (current_price - entry_price) / entry_price
                        unrealized_pnl += position_size * price_change
                
                # Calculate total balance (paper balance + unrealized P/L)
                total_balance = self.paper_balance
                
                # Update the balance label
                self.balance_label.config(
                    text=f"Paper Balance: ${total_balance:.2f} (Allocated: ${allocated_balance:.2f})")
                
                # Log detailed balance breakdown
                self.log_trade(f"""
                Balance Breakdown:
                Paper Balance: ${self.paper_balance:.2f}
                Allocated in Trades: ${allocated_balance:.2f}
                Unrealized P/L: ${unrealized_pnl:.2f}
                Total Balance: ${total_balance:.2f}
                """)
                
                # Check for low balance
                if total_balance < self.min_balance_threshold:
                    self.balance_label.config(foreground='red')
                elif total_balance < self.min_balance_threshold * 2:
                    self.balance_label.config(foreground='orange')
                else:
                    self.balance_label.config(foreground='black')
            else:
                # Real trading balance
                balance = float(self.exchange.fetch_balance()['USD']['free'])
                self.balance_label.config(text=f"Real Balance: ${balance:.2f}")
                
                # Check for low balance
                if balance < self.min_balance_threshold:
                    self.balance_label.config(foreground='red')
                elif balance < self.min_balance_threshold * 2:
                    self.balance_label.config(foreground='orange')
                else:
                    self.balance_label.config(foreground='black')
                    
        except Exception as e:
            self.log_trade(f"Error updating balance display: {str(e)}")

    def live_update_conditions(self):
        """Apply condition changes without restarting the bot"""
        if not self.validate_conditions():
            return
        
        # Get current settings
        new_conditions = {
            'consecutive_rises': int(self.consecutive_rises.get()),
            'momentum': float(self.momentum_threshold.get()),
            'max_spread': float(self.max_spread.get()),
            'volume_increase': float(self.volume_increase.get()),
            'max_volatility': float(self.max_volatility.get())
        }
        # Update active trades
        for trade_id in list(self.active_trades.keys()):
            try:
                # Tighten trailing stops for existing trades
                self.active_trades[trade_id]['trailing_stop_pct'] = min(
                    self.active_trades[trade_id]['trailing_stop_pct'],
                    float(self.trailing_stop.get()) / 100
                )
            except Exception as e:
                self.log_trade(f"Failed to update trade {trade_id}: {str(e)}")
        
        # Log the changes
        changes = "\n".join([f"{k}: {v}" for k,v in new_conditions.items()])
        self.log_trade(f"""
        [LIVE UPDATE] Applied new conditions:
        {changes}
        Active trades updated with tighter stops
        """)
        
        # Visual feedback
        self.status_label.config(text="Conditions Updated Live", foreground="blue")
        self.root.after(3000, lambda: self.update_status(f"Running ({'Paper' if self.is_paper_trading else 'Real'})"))
    def validate_settings(self):
        try:
            # Validate basic settings
            profit = float(self.profit_target.get())
            position = float(self.position_size.get())
            stop = float(self.stop_loss.get())
            
            # Validate advanced settings
            min_vol = float(self.min_volume_entry.get())
            max_trades = int(self.max_trades_entry.get())
            
            # Validation checks
            if any(x <= 0 for x in [profit, position, stop, min_vol, max_trades]):
                raise ValueError("All values must be positive")
            
            if position < 10:
                raise ValueError("Position size must be at least 10 USD")
            
            if max_trades < 1:
                raise ValueError("Maximum active trades must be at least 1")
            
            # Check balance
            if self.is_paper_trading:
                available_balance = self.paper_balance
            else:
                available_balance = float(self.exchange.fetch_balance()['USD']['free'])
                
            if position > available_balance:
                raise ValueError(f"Position size ({position} USD) exceeds available balance ({available_balance:.2f} USD)")
            
            # Update instance variables
            self.min_volume = min_vol
            self.max_active_trades = max_trades
            
            self.log_trade("Settings validated successfully")
            return True
                
        except ValueError as e:
            self.log_trade(f"Settings validation failed: {str(e)}")
            raise ValueError(f"Invalid settings: {str(e)}")
        
    def monitor_prices_continuously(self):
        """Monitor active trades and manage trailing stops"""
        while self.running:
            try:
                # Skip if no active trades
                if not self.active_trades:
                    time.sleep(1)
                    continue
                    
                # Use the monitor_trades function
                self.monitor_trades()
                
                # Sleep to avoid excessive API calls
                time.sleep(5)  # Check every 5 seconds
                    
            except Exception as e:
                self.log_trade(f"Monitoring error: {str(e)}")
                time.sleep(1)
                
    def validate_conditions(self):
        """Validate all user-configured trading conditions before execution"""
        try:
            conditions = {
                # Core Parameters
                'profit_target': float(self.profit_target.get()),
                'stop_loss': float(self.stop_loss.get()),
                'position_size': float(self.position_size.get()),
                'trailing_stop': float(self.trailing_stop.get()),
                
                # Validation Criteria
                'price_rise_min': float(self.price_rise_min.get()),
                'volume_surge': float(self.volume_surge.get()),
                'trailing_activation': float(self.trailing_activation.get()),
                
                # Risk Management
                'max_position_percent': float(self.max_position_percent.get()),
                'daily_loss_limit': float(self.daily_loss_limit.get()),
                
                # Market Filters
                'min_volume': float(self.min_volume_entry.get()),
                'max_trades': int(self.max_trades_entry.get()),
                'max_volatility': float(self.max_volatility.get()),
                'required_conditions': int(self.required_conditions.get())
            }

            # Core Parameter Validation
            assert 0.05 <= conditions['profit_target'] <= 5.0, "Profit target must be between 0.05% and 5%"
            assert 0.05 <= conditions['stop_loss'] <= 2.0, "Stop loss must be between 0.05% and 2%"
            assert 10 <= conditions['position_size'] <= 1000, "Position size must be between $10 and $1000"
            assert 0.05 <= conditions['trailing_stop'] <= 1.0, "Trailing stop must be between 0.05% and 1%"

            # Entry Condition Validation
            assert 0.1 <= conditions['price_rise_min'] <= 2.0, "Minimum price rise must be between 0.1% and 2%"
            assert 100 <= conditions['volume_surge'] <= 500, "Volume surge must be between 100% and 500%"
            assert 0.1 <= conditions['trailing_activation'] <= 2.0, "Trailing activation must be between 0.1% and 2%"

            # Risk Management Validation
            assert 1 <= conditions['max_position_percent'] <= 20, "Maximum position must be between 1% and 20% of balance"
            assert 1 <= conditions['daily_loss_limit'] <= 10, "Daily loss limit must be between 1% and 10%"

            # Market Filter Validation
            assert 50 <= conditions['min_volume'] <= 1000, "Minimum volume must be between $50 and $1000"
            assert 1 <= conditions['max_trades'] <= 10, "Maximum trades must be between 1 and 10"
            assert 0.1 <= conditions['max_volatility'] <= 5.0, "Maximum volatility must be between 0.1% and 5%"
            assert 1 <= conditions['required_conditions'] <= 5, "Required conditions must be between 1 and 5"

            # Cross-Parameter Validation
            assert conditions['stop_loss'] < conditions['profit_target'], "Stop loss must be less than profit target"
            assert conditions['trailing_stop'] < conditions['profit_target'], "Trailing stop must be less than profit target"
            assert conditions['trailing_activation'] < conditions['profit_target'], "Trailing activation must be less than profit target"
            
            # Balance Check
            if self.is_paper_trading:
                available_balance = self.paper_balance
            else:
                available_balance = float(self.exchange.fetch_balance()['USD']['free'])
                
            max_position = available_balance * (conditions['max_position_percent'] / 100)
            assert conditions['position_size'] <= max_position, f"Position size exceeds maximum allowed ({max_position:.2f} USD)"

            # Validate Current Trading Status
            if not self.is_paper_trading:
                # Additional checks for real trading
                assert self.verify_api_keys(), "API keys not properly configured"
                assert self.verify_exchange_connection(), "Cannot connect to exchange"
                
                # Check if we're within daily loss limit
                daily_loss = self.calculate_daily_loss()
                max_daily_loss = available_balance * (conditions['daily_loss_limit'] / 100)
                assert daily_loss <= max_daily_loss, f"Daily loss limit reached (${daily_loss:.2f}/${max_daily_loss:.2f})"

            # Log successful validation
            self.log_trade(f"""
            Conditions Validated Successfully:
            Profit Target: {conditions['profit_target']}%
            Stop Loss: {conditions['stop_loss']}%
            Position Size: ${conditions['position_size']}
            Max Position: {conditions['max_position_percent']}% of balance
            Volume Requirements: ${conditions['min_volume']}
            Max Active Trades: {conditions['max_trades']}
            Required Conditions: {conditions['required_conditions']}/5
            """)

            return True

        except ValueError as e:
            messagebox.showerror("Input Error", f"Invalid number format: {str(e)}")
            return False
        except AssertionError as e:
            messagebox.showerror("Validation Error", str(e))
            return False
        except Exception as e:
            self.log_trade(f"Validation error: {str(e)}")
            messagebox.showerror("Error", f"Unexpected error during validation: {str(e)}")
            return False

    def verify_api_keys(self):
        """Verify API keys are properly configured"""
        try:
            if not self.config.has_section('API_KEYS'):
                return False
            if not all([self.config.get('API_KEYS', 'api_key'), 
                    self.config.get('API_KEYS', 'secret')]):
                return False
            return True
        except:
            return False

    def verify_exchange_connection(self):
        """Verify connection to exchange"""
        try:
            self.exchange.fetch_time()
            return True
        except:
            return False

    def calculate_daily_loss(self):
        """Calculate total loss for current day"""
        try:
            today_trades = [t for t in self.trades 
                        if t['timestamp'].date() == datetime.now().date()]
            return abs(sum(t['net_profit'] for t in today_trades 
                        if t['net_profit'] < 0))
        except:
            return 0.0

    def format_kraken_symbol(self, base_symbol):
        """Convert standard symbol to Kraken format"""
        try:
            # Only handle special cases
            special_cases = {
                'BTC': 'XXBTZUSD',
                'ETH': 'XETHZUSD'
            }
            
            # If it's a special case, use the mapping
            if base_symbol in special_cases:
                return special_cases[base_symbol]
                
            # For all other symbols, try multiple formats
            possible_formats = [
                f"{base_symbol}/USD",     # Standard format with separator
                f"{base_symbol}USD"       # Format without separator
            ]
            
            # Verify which format exists
            available_pairs = self.exchange.fetch_tickers()
            for pair_format in possible_formats:
                if pair_format in available_pairs:
                    self.log_trade(f"Found valid pair: {pair_format}")
                    return pair_format
                        
            self.log_trade(f"Could not find valid Kraken pair for {base_symbol}")
            # Log available pairs containing the symbol for debugging
            similar_pairs = [p for p in available_pairs.keys() if base_symbol in p]
            if similar_pairs:
                self.log_trade(f"Similar pairs found: {similar_pairs}")
            return None
                
        except Exception as e:
            self.log_trade(f"Error formatting Kraken symbol: {str(e)}")
            return None

    def get_market_pairs(self):
        """Get market pairs with volume filtering"""
        try:
            self.log_trade("Fetching market pairs...")
            
            # Fetch tickers with retry mechanism
            tickers = self.fetch_tickers_with_retry()
            if not tickers:
                self.log_trade("Failed to fetch tickers")
                return []
                
            self.log_trade(f"Fetched {len(tickers)} tickers")
            
            # Filter for USD pairs under $5
            valid_pairs = []
            for symbol, ticker in tickers.items():
                try:
                    # Basic filtering
                    if not symbol.endswith('/USD'):
                        continue
                        
                    # Skip pairs with invalid data
                    if not self.validate_ticker(ticker):
                        continue
                        
                    # Extract price and volume
                    price = float(ticker['last'])
                    volume = float(ticker.get('quoteVolume', 0))
                    
                    # Apply price and volume filters
                    if price <= 0 or price > 5.0:
                        continue
                        
                    min_volume = float(self.min_volume_entry.get()) if hasattr(self, 'min_volume_entry') else 50
                    if volume < min_volume:
                        continue
                    
                    # Process ticker data
                    pair_data = self.process_ticker(symbol, ticker)
                    if pair_data:
                        valid_pairs.append(pair_data)
                        
                except Exception as e:
                    continue
                    
            self.log_trade(f"Found {len(valid_pairs)} valid pairs")
            return valid_pairs
            
        except Exception as e:
            self.log_trade(f"Error getting market pairs: {str(e)}")
            return []

    def scan_opportunities(self):
        """Scan for trading opportunities with improved filtering"""
        try:
            self.log_trade("==== SCANNING FOR OPPORTUNITIES ====")
            trades_executed = False
            
            # Get market pairs with volume filtering
            pairs = self.get_market_pairs()
            
            # Skip if no pairs found
            if not pairs:
                self.log_trade("No valid pairs found")
                return False
                
            # Limit number of pairs to evaluate to avoid excessive API calls
            max_pairs_to_evaluate = 20
            
            # Sort pairs by volume (descending) and take top N
            pairs.sort(key=lambda x: x.get('volume', 0), reverse=True)
            pairs = pairs[:max_pairs_to_evaluate]
            
            # Check if market override is active
            market_override = hasattr(self, 'market_override_var') and self.market_override_var.get()
            
            # If we already have max trades, don't bother scanning
            max_trades = int(self.max_trades_entry.get()) if hasattr(self, 'max_trades_entry') else 5
            if len(self.active_trades) >= max_trades:
                self.log_trade(f"Maximum trades ({max_trades}) already active, skipping scan")
                return False
            
            # Process each pair
            for pair_data in pairs:
                try:
                    symbol = pair_data['symbol']
                    
                    # Skip if already trading this pair
                    if any(trade['symbol'] == symbol for trade in self.active_trades.values()):
                        self.log_trade(f"Already trading {symbol}, skipping")
                        continue
                    
                    # Apply market-based filtering unless override is active
                    if market_override:
                        self.log_trade(f"[OK] Market override ACTIVE - bypassing market-based filtering for {symbol}")
                    else:
                        # Check if market conditions are favorable
                        market_conditions = self.analyze_market_conditions()
                        if market_conditions['state'] == 'bearish':
                            self.log_trade(f"Bearish market detected, skipping {symbol}")
                            continue
                    
                    # Apply smart trade filter (this will be bypassed internally if override is active)
                    self.log_trade(f"Evaluating {symbol}...")
                    if not self.smart_trade_filter(pair_data):
                        self.log_trade(f"Smart filter rejected {symbol}")
                        continue
                        
                    # Execute trade if it passes all filters
                    self.log_trade(f"[OK] {symbol} passed all filters, executing trade...")
                    if self.execute_trade(pair_data):
                        self.log_trade(f"[OK] Successfully executed trade for {symbol}")
                        trades_executed = True
                        
                        # If we've reached max trades, stop scanning
                        if len(self.active_trades) >= max_trades:
                            self.log_trade(f"Maximum trades ({max_trades}) reached, stopping scan")
                            break
                    else:
                        self.log_trade(f"[!] Failed to execute trade for {symbol}")
                
                except Exception as e:
                    self.log_trade(f"Error evaluating {symbol}: {str(e)}")
                    continue
            
            self.log_trade("==== SCAN COMPLETE ====")
            return trades_executed
            
        except Exception as e:
            self.log_trade(f"Error scanning opportunities: {str(e)}")
            return False

    def quick_filter(self, ticker: dict) -> bool:
        """Fast initial filtering of pairs"""
        try:
            if not self.validate_ticker(ticker):
                return False
            
            # Extract basic values
            price = float(ticker['last'])
            volume = float(ticker['quoteVolume']) if 'quoteVolume' in ticker else 0
            
            # Simple, permissive filter that should allow many pairs through
            return (
                price > 0 and                # Valid price
                price <= 5.0 and           # Price under $5
                volume > 0 and               # Any volume
                '/USD' in ticker['symbol']   # USD pair
            )
        except Exception as e:
            return False

    def get_cached_price(self, symbol: str) -> Dict:
        """Get cached price or fetch new one"""
        current_time = time.time()
        
        if (symbol in self.price_cache and 
            current_time - self.price_cache[symbol]['time'] < self.cache_timeout):
            return self.price_cache[symbol]['data']
            
        # Fetch new data
        ticker = self.exchange.fetch_ticker(symbol)
        self.price_cache[symbol] = {
            'data': ticker,
            'time': current_time
        }
        return ticker

    def process_ticker(self, symbol, ticker):
        """Process a single ticker with data collection tracking"""
        try:
            if not self.validate_ticker(ticker, symbol):
                return None
                    
            price = float(ticker['last'])
            volume = float(ticker['quoteVolume'])
            bid = float(ticker['bid'])
            ask = float(ticker['ask'])
                
            # Update price data
            self.data_manager.update_price_data(symbol, ticker)
                
            # Get current data points
            df = self.data_manager.get_price_data(symbol)
            if df is not None:
                data_points = len(df)
                min_required = self.data_manager.min_data_points
                self.log_trade(f"Updated {symbol} data: {data_points}/{min_required} points collected")
                
            spread = (ask - bid) / bid * 100
                
            return {
                'symbol': symbol,
                'ticker': ticker,
                'volume': volume,
                'price': price,
                'spread': spread,
                'data_points': data_points if df is not None else 0
            }
            
        except Exception as e:
            self.log_trade(f"Error processing {symbol}: {str(e)}")
            return None

    def update_displays(self):
        """Batch update displays to reduce GUI overhead"""
        try:
            if not hasattr(self, '_last_update'):
                self._last_update = time.time()
                return

            current_time = time.time()
            if current_time - self._last_update < 1.0:  # Update max once per second
                return

            self._last_update = current_time

            # Update displays in a single batch
            self.root.after(0, self._batch_update_displays)

        except Exception as e:
            self.log_trade(f"Display update error: {str(e)}")

    def _batch_update_displays(self):
        """Perform all display updates in one batch"""
        try:
            # Update active trades display
            if hasattr(self, 'trades_text'):
                self.trades_text.delete(1.0, tk.END)
                if self.active_trades:
                    for trade_id, trade in self.active_trades.items():
                        # ... display trade info ...
                        pass

            # Update metrics
            if hasattr(self, 'total_profit_label'):
                self.total_profit_label.config(
                    text=f"Total Profit: {self.total_profit:.2f} USD")

            # Update chart only if needed
            if time.time() - getattr(self, '_last_chart_update', 0) > 2:
                self.update_chart()
                self._last_chart_update = time.time()

        except Exception as e:
            self.log_trade(f"Batch update error: {str(e)}")

    def cleanup_on_shutdown(self):
        """Clean up resources and close positions before shutdown"""
        try:
            self.log_trade("Performing cleanup before shutdown...")
            
            # Stop all threads
            self.running = False
            
            # Close all active trades
            if self.active_trades:
                self.log_trade("Closing all active trades...")
                for trade_id, trade in list(self.active_trades.items()):
                    try:
                        current_price = self.get_cached_price(trade['symbol'])['last']
                        self.close_trade(trade_id, trade, current_price, "shutdown")
                    except Exception as e:
                        self.log_trade(f"Error closing trade {trade_id}: {str(e)}")
            
            # Save trade history
            try:
                if hasattr(self, 'trades') and self.trades:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"trade_history_{timestamp}.json"
                    with open(filename, 'w') as f:
                        json.dump(self.trades, f, default=str)
                    self.log_trade(f"Trade history saved to {filename}")
            except Exception as e:
                self.log_trade(f"Error saving trade history: {str(e)}")
            
            # Log final performance
            self.log_trade(f"""
            Final Performance Summary:
            Total Trades: {self.total_trades}
            Winning Trades: {self.winning_trades}
            Losing Trades: {self.losing_trades}
            Win Rate: {(self.winning_trades/max(1, self.total_trades))*100:.1f}%
            Total Profit: ${self.total_profit:.2f}
            Total Fees: ${self.total_fees:.2f}
            Net Profit: ${(self.total_profit - self.total_fees):.2f}
            Final Balance: ${self.paper_balance:.2f}
            """)
            
            # Clear data structures
            self.active_trades.clear()
            self.price_history.clear()
            self.price_cache.clear()
            
            # Close any matplotlib figures
            if hasattr(self, 'fig'):
                plt.close(self.fig)
            
            self.log_trade("Cleanup completed successfully")
            
        except Exception as e:
            self.log_trade(f"Error during cleanup: {str(e)}")

    def cleanup_old_data(self):
        """Remove stale data"""
        try:
            now = pd.Timestamp.now()
            cutoff = now - pd.Timedelta(minutes=5)
            
            for symbol in list(self.price_data.keys()):
                if isinstance(self.price_data[symbol], pd.DataFrame):
                    # Remove old data
                    self.price_data[symbol] = self.price_data[symbol][
                        self.price_data[symbol].index > cutoff
                    ]
                    
                    # Remove symbol if no recent data
                    if len(self.price_data[symbol]) == 0:
                        del self.price_data[symbol]
                        
        except Exception as e:
            self.log_trade(f"Error in cleanup: {str(e)}")

    def analyze_opportunity(self, ticker, volume_usd, pair_data):
        """Analyze trading opportunity using advanced metrics, Validation Criteria, and Advanced Parameters"""
        try:
            symbol = pair_data['symbol']
            self.log_trade(f"\nAnalyzing {symbol}...")

            # Force update price data before analysis
            try:
                self.data_manager.update_price_data(symbol, ticker)
                self.log_trade(f"Price data updated for {symbol}")
            except Exception as e:
                self.log_trade(f"Error updating price data: {str(e)}")
                return False

            # Get price data for analysis
            df = self.data_manager.get_price_data(symbol)
            
            # Get min_data_points from DataManager
            min_data_points = self.data_manager.min_data_points
            
            if df is None or len(df) < min_data_points:
                self.log_trade(f"Insufficient data for {symbol}: {len(df) if df is not None else 0}/{min_data_points} points")
                return False

            # Calculate basic metrics for existing checks
            momentum = self.calculate_current_momentum(df)
            volume_increase = self.calculate_volume_increase(df)

            # Validation logic for existing checks
            if momentum == 0.0 and volume_increase == 0.0:
                self.log_trade(f"Rejected {pair_data['symbol']}: Static data detected")
                return False

            # TEMPORARY FIX: Check if market is bearish and log it, but don't let it block trades
            market = self.analyze_market_conditions()
            if market['state'] == 'bearish':
                self.log_trade(f"NOTE: Market is bearish (Strength: {market['strength']:.2f}) but continuing analysis")
            
            # Apply Validation Criteria (Greek parameters) with slightly relaxed thresholds
            # 1. Trend Strength (Beta)
            trend_strength = self.calculate_momentum_intensity(df)
            momentum_beta_threshold = float(self.momentum_beta.get()) * 0.9  # 10% more lenient
            if trend_strength < momentum_beta_threshold:
                self.log_trade(f"Rejected {pair_data['symbol']}: Trend strength below threshold ({trend_strength:.2f} < {momentum_beta_threshold})")
                return False
            self.log_trade(f"✓ Trend Strength (Beta) passed: {trend_strength:.2f} >= {momentum_beta_threshold}")

            # 2. Price Acceleration (Alpha)
            price_acceleration = self.calculate_price_acceleration(df)
            price_alpha_threshold = float(self.price_alpha.get()) * 0.9  # 10% more lenient
            if price_acceleration < price_alpha_threshold:
                self.log_trade(f"Rejected {pair_data['symbol']}: Price acceleration below threshold ({price_acceleration:.2f} < {price_alpha_threshold})")
                return False
            self.log_trade(f"✓ Price Acceleration (Alpha) passed: {price_acceleration:.2f} >= {price_alpha_threshold}")

            # 3. Momentum Quality (Theta)
            momentum_quality = self.calculate_momentum_quality(df)
            momentum_theta_threshold = float(self.momentum_theta.get()) * 1.1  # 10% more lenient
            if momentum_quality > momentum_theta_threshold:
                self.log_trade(f"Rejected {pair_data['symbol']}: Momentum quality above threshold ({momentum_quality:.2f} > {momentum_theta_threshold})")
                return False
            self.log_trade(f"✓ Momentum Quality (Theta) passed: {momentum_quality:.2f} <= {momentum_theta_threshold}")

            # 4. Volatility Filter (Vega)
            volatility = self.calculate_volatility_sensitivity(df)
            vol_vega_threshold = float(self.vol_vega.get()) * 1.1  # 10% more lenient
            if volatility > vol_vega_threshold:
                self.log_trade(f"Rejected {pair_data['symbol']}: Volatility too high ({volatility:.2f} > {vol_vega_threshold})")
                return False
            self.log_trade(f"✓ Volatility Filter (Vega) passed: {volatility:.2f} <= {vol_vega_threshold}")

            # 5. Volume Quality (Rho)
            current_volume = df['volume'].iloc[-1]
            volume_impact = self.calculate_volume_impact(df, current_volume)
            volume_rho_threshold = float(self.volume_rho.get()) * 0.9  # 10% more lenient
            if volume_impact < volume_rho_threshold:
                self.log_trade(f"Rejected {pair_data['symbol']}: Volume impact below threshold ({volume_impact:.2f} < {volume_rho_threshold})")
                return False
            self.log_trade(f"✓ Volume Quality (Rho) passed: {volume_impact:.2f} >= {volume_rho_threshold}")

            # All checks passed
            self.log_trade(f"All criteria passed for {pair_data['symbol']} - OPPORTUNITY DETECTED")
            return True

        except Exception as e:
            self.log_trade(f"Error analyzing opportunity for {pair_data['symbol']}: {str(e)}")
            return False

    def analyze_pairs(self, pairs):
        """Analyze pairs for trading opportunities"""
        try:
            self.log_trade(f"\nAnalyzing {len(pairs)} pairs for trading opportunities...")
            trades_executed = False
            
            # Check if we've reached max trades
            max_trades = int(self.max_trades_entry.get())
            if len(self.active_trades) >= max_trades:
                self.log_trade(f"Maximum trades reached ({len(self.active_trades)}/{max_trades})")
                return False
            
            # Process each pair
            for pair_data in pairs:
                try:
                    symbol = pair_data['symbol']
                    
                    # Skip if already trading this pair
                    if any(trade['symbol'] == symbol for trade in self.active_trades.values()):
                        self.log_trade(f"Already trading {symbol}, skipping")
                        continue
                    
                    # Apply smart trade filter
                    if not self.smart_trade_filter(pair_data):
                        self.log_trade(f"Smart filter rejected {symbol}")
                        continue
                    
                    # Get ticker data
                    ticker = pair_data['ticker']
                    volume = pair_data['volume']
                    
                    # Check trading conditions
                    if self.analyze_opportunity(ticker, volume, pair_data):
                        self.log_trade(f"[TARGET] Trade opportunity found: {symbol}")
                        
                        # Validate and execute trade
                        if self.validate_trade(pair_data, pair_data['price']):
                            if self.execute_trade(pair_data):
                                trades_executed = True
                                self.log_trade(f"Trade executed successfully for {symbol}")
                            else:
                                self.log_trade(f"Failed to execute trade for {symbol}")
                        else:
                            self.log_trade(f"Trade validation failed for {symbol}")
                    
                except Exception as e:
                    self.log_trade(f"Error analyzing {pair_data['symbol']}: {str(e)}")
                    continue
            
            if not trades_executed:
                self.log_trade("No trading opportunities found this scan")
            
            return trades_executed
                
        except Exception as e:
            self.log_trade(f"Error analyzing pairs: {str(e)}")
            return False

    def calculate_trade_parameters(self, entry_price):
        """Calculate optimal trade parameters considering fees"""
        try:
            total_fees = self._total_fee_percentage  # Use the instance variable
            
            # Minimum profit needed to break even
            min_profit = total_fees * 1.2  # Add 20% buffer over fees
            
            # Set profit target higher than minimum needed
            profit_target = max(
                float(self.profit_target.get()) / 100,  # User-set target
                min_profit * 1.5  # Or 1.5x minimum profitable move
            )
            
            # Tighter stop loss to minimize losses
            stop_loss = min(
                float(self.stop_loss.get()) / 100,  # User-set stop
                total_fees * 0.75  # Or 75% of fees
            )
            
            return {
                'profit_target': profit_target,
                'stop_loss': stop_loss,
                'min_profit': min_profit
            }
        except Exception as e:
            self.log_trade(f"Error calculating trade parameters: {str(e)}")
            return {
                'profit_target': float(self.profit_target.get()) / 100,
                'stop_loss': float(self.stop_loss.get()) / 100,
                'min_profit': 0.01  # Default 1%
            }
                
    def execute_trade(self, pair_data):
        """Execute a trade with smart risk management that respects user parameters"""
        try:
            symbol = pair_data['symbol']
            self.log_trade(f"\n==== ATTEMPTING TRADE EXECUTION FOR {symbol} ====")
            
            # Check number of active trades
            max_trades = int(self.max_trades_entry.get()) if hasattr(self, 'max_trades_entry') else 5
            if len(self.active_trades) >= max_trades:
                self.log_trade(f"❌ Maximum trades ({max_trades}) already active")
                return False
                        
            # Get current price with error handling
            try:
                current_price = float(pair_data['ticker']['last'])
                if not current_price or current_price <= 0:
                    self.log_trade(f"❌ Invalid price for {symbol}: {current_price}")
                    return False
                self.log_trade(f"Current price: {current_price}")
            except Exception as e:
                self.log_trade(f"❌ Error getting price: {str(e)}")
                return False

            # Skip smart filter if override is active
            if hasattr(self, 'market_override_var') and self.market_override_var.get():
                self.log_trade(f"✅ Market override ACTIVE - bypassing smart filter")
            else:
                # Apply smart filter before executing
                if not self.smart_trade_filter(pair_data):
                    self.log_trade(f"❌ Smart filter rejected trade for {symbol}")
                    return False
                    
            # Calculate position size
            position_size = float(self.position_size.get())
            self.log_trade(f"Position size: ${position_size}")
            
            # Check if we have enough balance for paper trading
            if self.is_paper_trading:
                if position_size > self.paper_balance:
                    self.log_trade(f"❌ Insufficient paper balance: ${self.paper_balance:.2f} < ${position_size:.2f}")
                    return False
                
                # Deduct position size from paper balance
                self.paper_balance -= position_size
                self.log_trade(f"Deducted ${position_size:.2f} from paper balance. New balance: ${self.paper_balance:.2f}")
            
            # Calculate quantity based on position size and current price
            quantity = position_size / current_price
            self.log_trade(f"Calculated quantity: {quantity}")
            
            # Generate a unique trade ID
            trade_id = f"{symbol.replace('/', '_')}_{int(time.time())}"
            
            # Log the trade execution attempt
            self.log_trade(f"✅ EXECUTING TRADE: {symbol} at ${current_price} for ${position_size}")
            
            # Record the trade with proper timestamp
            current_time = datetime.now()
            self.active_trades[trade_id] = {
                'symbol': symbol,
                'entry_price': current_price,
                'current_price': current_price,  # Initialize current price
                'quantity': quantity,
                'amount': quantity,  # Add amount field (same as quantity)
                'position_size': position_size,
                'entry_time': current_time,
                'timestamp': current_time,  # Ensure timestamp is set
                'highest_price': current_price,
                'highest_profit': 0.0,
                'market_condition': self.analyze_market_conditions()['state'],
                'is_paper': self.is_paper_trading  # Add paper trading flag
            }
            
            # Update displays
            self.update_active_trades_display()
            
            # Log success
            self.log_trade(f"✅ Trade executed successfully for {symbol}")
            self.log_trade(f"==== TRADE EXECUTION COMPLETE FOR {symbol} ====\n")
            return True
            
        except Exception as e:
            self.log_trade(f"❌ Error executing trade: {str(e)}")
            return False

    def close_trade(self, trade_id, trade, current_price, reason):
        """Close a trade with proper balance updates"""
        try:
            self.log_trade(f"Closing trade {trade_id} ({trade['symbol']}) due to {reason}")
            
            symbol = trade['symbol']
            entry_price = float(trade['entry_price'])
            position_size = float(trade['position_size'])
            
            # Calculate profit/loss
            price_change = (current_price - entry_price) / entry_price
            gross_profit = position_size * price_change
            
            # Apply fees (using instance variables)
            entry_fee = position_size * self.taker_fee
            exit_fee = (position_size * (1 + price_change)) * self.taker_fee
            total_fees = entry_fee + exit_fee
            
            # Calculate net profit
            net_profit = gross_profit - total_fees
            
            # Update paper balance - ONLY add back position size + net profit
            if self.is_paper_trading:
                self.paper_balance += position_size + net_profit
                self.log_trade(f"Updated paper balance: ${self.paper_balance:.2f}")
            
            # Update trade statistics
            self.total_profit += gross_profit
            self.total_fees += total_fees
            self.total_trades += 1
            
            if net_profit > 0:
                self.winning_trades += 1
            else:
                self.losing_trades += 1
            
            # Add to trade history
            trade_record = {
                'symbol': symbol,
                'entry_price': entry_price,
                'exit_price': current_price,
                'position_size': position_size,
                'gross_profit': gross_profit,
                'fees': total_fees,
                'net_profit': net_profit,
                'timestamp': datetime.now(),
                'reason': reason,
                'duration': (datetime.now() - trade.get('entry_time', datetime.now())).total_seconds()
            }
            self.trades.append(trade_record)
            
            # Update trade history display
            self.update_trade_history(symbol, price_change*100, net_profit, net_profit > 0)
            
            # Format reason for display
            trade_reason = reason.replace("_", " ").title()
            
            # Log closure details
            self.log_trade(f"""
            Trade Closed: {symbol}
            Reason: {trade_reason.upper()}
            Entry: ${entry_price:.8f}
            Exit: ${current_price:.8f}
            Gross P/L: ${gross_profit:.2f} ({price_change*100:.2f}%)
            Fees: ${total_fees:.2f} ({(total_fees/position_size)*100:.2f}%)
            Net P/L: ${net_profit:.2f} ({(net_profit/position_size)*100:.2f}%)
            Current Balance: ${self.paper_balance:.2f}
            """)
            
            # Remove from active trades
            if trade_id in self.active_trades:
                del self.active_trades[trade_id]
            
            # Update displays - force immediate update
            self.update_metrics()
            self.update_balance_display()
            self.update_active_trades_display()
            
            return True
            
        except Exception as e:
            self.log_trade(f"Error closing trade: {str(e)}")
            return False
    
    def verify_active_trades(self):
        """Automatically verify active trades against Kraken's open positions"""
        if self.is_paper_trading:
            self.log_trade("Trade verification skipped in paper trading mode")
            return
            
        try:
            self.log_trade("Verifying active trades with Kraken...")
            
            # Get open positions from Kraken
            try:
                # Fetch positions
                positions = self.exchange.fetch_positions()
                self.log_trade(f"Found {len(positions)} open positions on Kraken")
                
                # Create lookup by symbol
                kraken_positions = {}
                for position in positions:
                    if float(position.get('contracts', 0)) > 0:
                        kraken_positions[position['symbol']] = position
                        
                # Also check open orders
                open_orders = self.exchange.fetch_open_orders()
                self.log_trade(f"Found {len(open_orders)} open orders on Kraken")
                
                # Add symbols from open orders
                for order in open_orders:
                    if order['side'] == 'buy' and order['status'] == 'open':
                        kraken_positions[order['symbol']] = order
                        
            except Exception as e:
                self.log_trade(f"Error fetching positions: {str(e)}")
                return
                
            # Check each active trade
            for trade_id, trade in list(self.active_trades.items()):
                symbol = trade['symbol']
                
                # Check if position exists on Kraken
                if symbol not in kraken_positions:
                    self.log_trade(f"Warning: Trade {trade_id} for {symbol} not found in Kraken positions")
                    
                    # Instead of handling it here, use the close_trade method
                    # This ensures consistent handling and proper history updates
                    current_price = float(self.exchange.fetch_ticker(symbol)['last'])
                    self.close_trade(trade_id, trade, current_price, "position_not_found")
                else:
                    self.log_trade(f"Verified: {symbol} exists in Kraken positions")
                    
        except Exception as e:
            self.log_trade(f"Error verifying trades: {str(e)}")
        
    def advanced_checks(self, symbol, df):
        """Advanced market checks including order book analysis and RSI"""
        try:
            # 1. EMA Cross (5/15)
            ema_cross = (df['ema_5'].iloc[-2] < df['ema_15'].iloc[-2]) and \
                        (df['ema_5'].iloc[-1] > df['ema_15'].iloc[-1])
            if ema_cross:
                self.log_trade(f"EMA Cross detected for {symbol}: Short-term trend is bullish")
            else:
                self.log_trade(f"No EMA Cross detected for {symbol}")
            
            # 2. Volume-Weighted Momentum
            if 'price' in df.columns and 'volume' in df.columns and len(df) >= 5:
                vwap = (df['price'] * df['volume']).cumsum() / df['volume'].cumsum()
                vwm = (vwap.iloc[-1] - vwap.iloc[-5]) / vwap.iloc[-5] * 100
                self.log_trade(f"Volume-Weighted Momentum for {symbol}: {vwm:.2f}%")
            else:
                self.log_trade(f"Insufficient data for Volume-Weighted Momentum calculation for {symbol}")
                vwm = 0
            
            # 3. Order Book Imbalance
            try:
                ob = self.exchange.fetch_order_book(symbol)
                bid_vol = sum(x[1] for x in ob['bids'][:3])
                ask_vol = sum(x[1] for x in ob['asks'][:3])
                ob_ratio = bid_vol / (ask_vol + 0.0001)  # Avoid division by zero
                self.log_trade(f"Order Book Imbalance for {symbol}: {ob_ratio:.2f}")
            except Exception as e:
                self.log_trade(f"Order book fetch error: {str(e)}")
                ob_ratio = 1.0  # Neutral value on error
            
            # 4. RSI Check
            rsi_period = int(self.rsi_period.get()) if hasattr(self, 'rsi_period') else 14
            rsi_column = f'rsi_{rsi_period}'
            
            ema_cross = (df['ema_5'].iloc[-2] < df['ema_15'].iloc[-2]) and \
                        (df['ema_5'].iloc[-1] > df['ema_15'].iloc[-1])
            
            # 2. Volume-Weighted Momentum
            vwap = (df['price'] * df['volume']).cumsum() / df['volume'].cumsum()
            vwm = (vwap.iloc[-1] - vwap.iloc[-5]) / vwap.iloc[-5] * 100
            
            # 3. Order Book Imbalance
            ob = self.exchange.fetch_order_book(symbol)
            bid_vol = sum(x[1] for x in ob['bids'][:3])
            ask_vol = sum(x[1] for x in ob['asks'][:3])
            ob_ratio = bid_vol / (ask_vol + 0.0001)  # Avoid division by zero
            
            # 4. RSI Check
            rsi_period = int(self.rsi_period.get()) if hasattr(self, 'rsi_period') else 14
            rsi_column = f'rsi_{rsi_period}'
            if rsi_column not in df:
                self.log_trade(f"RSI column {rsi_column} not found in DataFrame for {symbol}")
                return False
            
            rsi_value = df[rsi_column].iloc[-1]
            if pd.isna(rsi_value):
                self.log_trade(f"RSI value is NaN for {symbol}")
                return False
            
            rsi_overbought = float(self.rsi_overbought.get()) if hasattr(self, 'rsi_overbought') else 70
            rsi_oversold = float(self.rsi_oversold.get()) if hasattr(self, 'rsi_oversold') else 30
            
            # Avoid overbought conditions (RSI > overbought threshold)
            if rsi_value > rsi_overbought:
                self.log_trade(f"Rejected {symbol}: RSI indicates overbought condition ({rsi_value:.2f} > {rsi_overbought})")
                return False
            
            # Optionally, prefer oversold conditions for entry (RSI < oversold threshold)
            # This is a design choice; you can enable it if desired
            # if rsi_value > rsi_oversold:
            #     self.log_trade(f"Rejected {symbol}: RSI not in oversold range ({rsi_value:.2f} > {rsi_oversold})")
            #     return False
            
            self.log_trade(f"Advanced checks passed for {symbol}:")
            self.log_trade(f"- EMA Cross: {ema_cross}")
            self.log_trade(f"- VWAP Momentum: {vwm:.2f}%")
            self.log_trade(f"- Order Book Ratio: {ob_ratio:.2f}")
            self.log_trade(f"- RSI: {rsi_value:.2f} (Overbought: {rsi_overbought}, Oversold: {rsi_oversold})")
            
            return all([
                ema_cross,          # EMA crossover
                vwm > 0.2,          # Min 0.2% VWAP rise
                ob_ratio > 1.5,     # Bids > Asks by 50%
                True                # RSI check already handled above
            ])
        
        except Exception as e:
            self.log_trade(f"Error in advanced checks for {symbol}: {str(e)}")
            return False
        
    def update_active_trades_display(self):
        """Update the active trades display in the GUI"""
        try:
            self.trades_text.delete(1.0, tk.END)
            
            if not self.active_trades:
                self.trades_text.insert(tk.END, "No active trades\n")
                return
            
            self.trades_text.insert(tk.END, f"Number of Active Trades: {len(self.active_trades)}\n\n")
            
            for trade_id, trade in list(self.active_trades.items()):
                try:
                    symbol = trade['symbol']
                    entry_price = float(trade['entry_price'])
                    
                    # Get current price safely
                    current_price = trade.get('current_price', entry_price)
                    
                    # Calculate profit percentage
                    profit_percentage = ((current_price - entry_price) / entry_price) * 100
                    
                    # Get highest profit percentage
                    highest_profit = trade.get('highest_profit_percentage', profit_percentage)
                    
                    # Calculate time in trade
                    entry_time = trade.get('entry_time', datetime.now())
                    duration = (datetime.now() - entry_time).total_seconds()
                    
                    # Format last update time - ensure it's set
                    if 'last_update' not in trade:
                        trade['last_update'] = datetime.now().strftime("%H:%M:%S")
                    last_update = trade['last_update']
                    
                    # Ensure we update price history for charting
                    if symbol not in self.price_history:
                        self.price_history[symbol] = []
                    self.price_history[symbol].append((datetime.now(), current_price))
                    if len(self.price_history[symbol]) > 50:
                        self.price_history[symbol] = self.price_history[symbol][-50:]
                    
                    trade_info = (
                        f"Symbol: {symbol}\n"
                        f"Entry: {entry_price:.8f}\n"
                        f"Current: {current_price:.8f}\n"
                        f"P/L: {profit_percentage:.2f}%\n"
                        f"Highest: {highest_profit:.2f}%\n"
                        f"Time: {duration:.1f}s\n"
                        f"Last Update: {last_update}\n"
                        f"-----------------\n"
                    )
                    
                    self.trades_text.insert(tk.END, trade_info)
                    
                    # Color coding
                    start_idx = f"end-{len(trade_info)+1}c linestart"
                    end_idx = "end-1c"
                    
                    if profit_percentage > 0:
                        self.trades_text.tag_add("profit", start_idx, end_idx)
                        self.trades_text.tag_config("profit", foreground="green")
                    else:
                        self.trades_text.tag_add("loss", start_idx, end_idx)
                        self.trades_text.tag_config("loss", foreground="red")
                    
                except Exception as e:
                    self.log_trade(f"Error updating display for trade {trade_id}: {str(e)}")
                    continue
            
            # Force chart update after updating trades display
            self.update_chart()
            
            # Force GUI update
            self.trades_text.see(tk.END)
            self.root.update_idletasks()
            
        except Exception as e:
            self.log_trade(f"Error updating trades display: {str(e)}")

    def validate_trade(self, pair_data, entry_price):
        """Validate a trade before execution"""
        try:
            symbol = pair_data['symbol']
            self.log_trade(f"Validating trade for {symbol} at ${entry_price}")
            
            # Check if we're already trading this symbol
            if any(trade['symbol'] == symbol for trade in self.active_trades.values()):
                self.log_trade(f"Already trading {symbol}")
                return False
            
            # Check if we've reached max trades
            max_trades = int(self.max_trades_entry.get())
            if len(self.active_trades) >= max_trades:
                self.log_trade(f"Maximum trades reached ({len(self.active_trades)}/{max_trades})")
                return False
            
            # Check position size against balance
            position_size = float(self.position_size.get())
            if self.is_paper_trading:
                available_balance = self.paper_balance
            else:
                available_balance = float(self.exchange.fetch_balance()['USD']['free'])
                
            if position_size > available_balance:
                self.log_trade(f"Insufficient balance: ${available_balance:.2f} < ${position_size:.2f}")
                return False
            
            # Check if price is reasonable (not too high)
            if entry_price > 5.0:  # $5 max for low-priced assets
                self.log_trade(f"Price too high: ${entry_price} > $5.00")
                return False
            
            # Check if price is reasonable (not too low)
            if entry_price < 0.00001:  # Minimum price
                self.log_trade(f"Price too low: ${entry_price} < $0.00001")
                return False
            
            # Check if spread is reasonable
            if 'spread' in pair_data and pair_data['spread'] > 2.0:  # 2% max spread
                self.log_trade(f"Spread too high: {pair_data['spread']:.2f}% > 2.00%")
                return False
            
            # All checks passed
            self.log_trade(f"Trade validation passed for {symbol}")
            return True
            
        except Exception as e:
            self.log_trade(f"Error validating trade: {str(e)}")
            return False
    
    def manage_trades(self):
        """Manage trades with smart exit conditions that respect user parameters"""
        try:
            for trade_id, trade in list(self.active_trades.items()):
                try:
                    # Get current price and calculate profit
                    current_price = float(self.exchange.fetch_ticker(trade['symbol'])['last'])
                    entry_price = float(trade['entry_price'])
                    profit_pct = ((current_price - entry_price) / entry_price) * 100
                    
                    # Update highest profit if we have a new high
                    if 'highest_profit' not in trade or profit_pct > trade['highest_profit']:
                        trade['highest_profit'] = profit_pct

                    # Get exit thresholds from trade (these are the smart parameters)
                    stop_loss = trade.get('stop_loss_pct', float(self.stop_loss.get()) / 100) * 100
                    trailing_stop = trade.get('trailing_stop_pct', float(self.trailing_stop.get()) / 100) * 100
                    profit_target = trade.get('profit_target_pct', float(self.profit_target.get()) / 100) * 100
                    trailing_activation = float(self.trailing_activation.get())

                    # Debug log
                    self.log_trade(f"""
                    Trade Check - {trade['symbol']}:
                    Current P/L: {profit_pct:.2f}%
                    Highest: {trade.get('highest_profit', 0):.2f}%
                    Stop Loss: -{stop_loss:.2f}%
                    Profit Target: {profit_target:.2f}%
                    Trailing Stop: {trailing_stop:.2f}%
                    Trailing Activation: {trailing_activation:.2f}%
                    """)

                    # CASE 1: Stop Loss - Immediate exit if loss exceeds stop loss
                    if profit_pct <= -stop_loss:
                        self.log_trade(f"Stop Loss hit on {trade['symbol']} at {profit_pct:.2f}%")
                        self.close_trade(trade_id, trade, current_price, "stop loss")
                        continue

                    # CASE 2: Take Profit - Exit if profit exceeds target
                    if profit_pct >= profit_target:
                        self.log_trade(f"Profit target reached on {trade['symbol']} at {profit_pct:.2f}%")
                        self.close_trade(trade_id, trade, current_price, "take profit")
                        continue
                    
                    # CASE 3: Trailing Stop - Exit if price drops from highest by trailing stop amount
                    highest_profit = trade.get('highest_profit', 0)
                    if highest_profit >= trailing_activation:
                        drop_from_high = highest_profit - profit_pct
                        if drop_from_high >= trailing_stop:
                            self.log_trade(f"Trailing stop triggered on {trade['symbol']} - Drop from {highest_profit:.2f}% to {profit_pct:.2f}%")
                            self.close_trade(trade_id, trade, current_price, "trailing stop")
                            continue
                    
                    # CASE 4: Market Condition Change - Consider early exit in deteriorating markets
                    market_condition = self.analyze_market_conditions()
                    entry_market = trade.get('market_condition', 'neutral')
                    
                    # If market has deteriorated significantly since entry, consider partial exit
                    if entry_market != 'bearish' and market_condition['state'] == 'bearish' and profit_pct > 0:
                        # Market turned bearish and we have profit - consider taking partial profit
                        self.log_trade(f"Market deteriorated to bearish - considering early exit for {trade['symbol']}")
                        
                        # If profit is at least 50% of target, take profit
                        if profit_pct >= (profit_target * 0.5):
                            self.log_trade(f"Taking early profit due to market deterioration: {profit_pct:.2f}%")
                            self.close_trade(trade_id, trade, current_price, "market deterioration")
                            continue
                    
                    # CASE 5: Time-based exit - Close trades that have been open too long
                    max_trade_duration = int(self.max_trade_duration.get()) if hasattr(self, 'max_trade_duration') else 3600  # Default 1 hour
                    duration = (datetime.now() - trade['timestamp']).total_seconds()
                    
                    if duration > max_trade_duration:
                        self.log_trade(f"Max duration reached for {trade['symbol']} ({duration:.0f}s)")
                        self.close_trade(trade_id, trade, current_price, "max duration")
                        continue
                    
                    # Update trade data for monitoring
                    trade['current_price'] = current_price
                    trade['current_profit_percentage'] = profit_pct
                    trade['last_update'] = datetime.now()
                    
                    # Update price history for charting
                    if trade['symbol'] in self.price_history:
                        self.price_history[trade['symbol']].append((datetime.now(), current_price))
                        # Limit history size
                        self.price_history[trade['symbol']] = self.price_history[trade['symbol']][-100:]

                except Exception as e:
                    self.log_trade(f"Error managing trade {trade_id}: {str(e)}")
                    continue
                    
        except Exception as e:
            self.log_trade(f"Error in trade management: {str(e)}")
                

    def analyze_market_conditions(self):
        """Analyze current market conditions without changing user parameters"""
        try:
            # Check for market override
            if hasattr(self, 'market_override_var') and self.market_override_var.get():
                self.log_trade("Market override active - reporting neutral market regardless of conditions")
                return {'state': 'neutral', 'strength': 0}
            
            # Try to fetch BTC/USD ticker if we don't have data
            if not hasattr(self.data_manager, 'price_data') or 'BTC/USD' not in self.data_manager.price_data:
                self.log_trade("WARNING: No BTC/USD price data available, attempting to fetch...")
                try:
                    btc_ticker = self.exchange.fetch_ticker('BTC/USD')
                    if btc_ticker:
                        self.data_manager.update_price_data('BTC/USD', btc_ticker)
                        self.log_trade(f"Fetched BTC/USD price: ${float(btc_ticker['last'])}")
                    else:
                        self.log_trade("WARNING: Could not fetch BTC/USD ticker, defaulting to neutral market")
                        return {'state': 'neutral', 'strength': 0}
                except Exception as e:
                    self.log_trade(f"Error fetching BTC/USD data: {str(e)}")
                    return {'state': 'neutral', 'strength': 0}
            
            # Get price data
            df = self.data_manager.get_price_data('BTC/USD')
            if df is None or len(df) < 5:  # Need at least 5 data points for basic analysis
                self.log_trade(f"WARNING: Insufficient BTC/USD data points ({len(df) if df is not None else 0}), defaulting to neutral market")
                return {'state': 'neutral', 'strength': 0}
                    
            # Calculate indicators
            # Use available data for calculations, adapting to what we have
            try:
                # Calculate short EMA (use available data)
                short_span = min(5, len(df) - 1)
                short_ema = df['price'].ewm(span=short_span).mean().iloc[-1]
                
                # Calculate longer EMA if possible, otherwise use a simple average
                if len(df) >= 20:
                    long_ema = df['price'].ewm(span=20).mean().iloc[-1]
                else:
                    long_span = max(3, len(df) // 2)
                    long_ema = df['price'].ewm(span=long_span).mean().iloc[-1]
                
                # Calculate recent price change
                lookback = min(8, len(df) - 1)
                four_hour_change = (df['price'].iloc[-1] / df['price'].iloc[-lookback] - 1) * 100
                
                # Calculate market strength (-10 to +10 scale)
                ema_diff = ((short_ema / long_ema) - 1) * 100
                strength = (ema_diff + four_hour_change) / 2
                
                # Determine market state - USING MORE NEUTRAL THRESHOLDS
                if strength < -3:  # Changed from -2 to -3
                    state = 'bearish'
                elif strength > 3:  # Changed from 2 to 3
                    state = 'bullish'
                else:
                    state = 'neutral'
                        
                # Log market conditions
                self.log_trade(f"Market conditions: {state.upper()} (Strength: {strength:.2f})")
                
                return {
                    'state': state,
                    'strength': strength,
                    'ema_diff': ema_diff,
                    'price_change': four_hour_change
                }
            except Exception as e:
                self.log_trade(f"Error calculating market indicators: {str(e)}")
                return {'state': 'neutral', 'strength': 0}
                        
        except Exception as e:
            self.log_trade(f"Error analyzing market conditions: {str(e)}")
            return {'state': 'neutral', 'strength': 0}

    def initialize_market_data(self):
        """Initialize market data for BTC/USD to enable market analysis"""
        try:
            self.log_trade("Initializing market data for BTC/USD...")
            
            # Ensure we have an exchange connection
            if not hasattr(self, 'exchange') or self.exchange is None:
                self.log_trade("No exchange connection available for market data")
                return False
            
            # Fetch BTC/USD ticker with retry
            for attempt in range(3):
                try:
                    btc_ticker = self.exchange.fetch_ticker('BTC/USD')
                    if not btc_ticker:
                        self.log_trade(f"Failed to fetch BTC/USD ticker (attempt {attempt+1}/3)")
                        time.sleep(1)
                        continue
                    
                    # Initialize price data for BTC/USD
                    self.data_manager.update_price_data('BTC/USD', btc_ticker)
                    self.log_trade(f"Initial BTC/USD data point collected: ${float(btc_ticker['last'])}")
                    
                    # Fetch historical data to build up the required data points
                    self.log_trade("Fetching historical BTC/USD data...")
                    try:
                        # Fetch OHLCV data for the past day (24 hourly candles)
                        since = int((datetime.now() - timedelta(days=1)).timestamp() * 1000)
                        ohlcv = self.exchange.fetch_ohlcv('BTC/USD', timeframe='1h', since=since, limit=24)
                        
                        # Convert to DataFrame and update price data
                        for candle in ohlcv:
                            timestamp, open_price, high, low, close, volume = candle
                            
                            # Create synthetic ticker data
                            synthetic_ticker = {
                                'timestamp': pd.Timestamp(timestamp, unit='ms'),
                                'last': close,
                                'quoteVolume': volume,
                                'bid': close * 0.999,  # Approximate bid
                                'ask': close * 1.001   # Approximate ask
                            }
                            
                            # Update price data
                            self.data_manager.update_price_data('BTC/USD', synthetic_ticker)
                        
                        # Log success
                        df = self.data_manager.get_price_data('BTC/USD')
                        if df is not None:
                            self.log_trade(f"Successfully loaded {len(df)} historical data points for BTC/USD")
                        
                        return True
                        
                    except Exception as e:
                        self.log_trade(f"Error fetching historical BTC/USD data: {str(e)}")
                        # Continue with just the single data point
                        return True
                    
                except Exception as e:
                    self.log_trade(f"Error fetching BTC/USD ticker (attempt {attempt+1}/3): {str(e)}")
                    time.sleep(1)
            
            self.log_trade("Failed to initialize BTC/USD market data after multiple attempts")
            return False
            
        except Exception as e:
            self.log_trade(f"Error initializing market data: {str(e)}")
            return False

    def smart_trade_filter(self, pair_data):
        """Smart trade filter that respects user parameters but adds market intelligence"""
        try:
            symbol = pair_data['symbol']
            
            # Get symbol data first - needed for all checks
            df = self.data_manager.get_price_data(symbol)
            if df is None or len(df) < 3:  # Need at least 3 data points
                self.log_trade(f"Insufficient data for {symbol}")
                return False
                
            # Check for market override - only bypass market condition check
            market_override = hasattr(self, 'market_override_var') and self.market_override_var.get()
            if not market_override:
                # Only check market conditions if override is not active
                market = self.analyze_market_conditions()
                if market['state'] == 'bearish':
                    self.log_trade(f"Bearish market detected, rejecting {symbol}")
                    return False
            else:
                self.log_trade(f"[OK] Market override ACTIVE - bypassing market condition check for {symbol}")
            
            # Always apply basic price filter
            ticker = pair_data.get('ticker', {})
            price = float(ticker.get('last', 0))
            if price <= 0 or price > 5.0:
                self.log_trade(f"Price filter rejected {symbol}: ${price} (must be >0 and ≤$5)")
                return False
            
            # Always apply volume filter
            volume = float(ticker.get('quoteVolume', 0))
            min_volume = float(self.min_volume_entry.get()) if hasattr(self, 'min_volume_entry') else 50
            if volume < min_volume:
                self.log_trade(f"Volume filter rejected {symbol}: ${volume} < ${min_volume}")
                return False
                
            # Always apply technical indicators and user parameters
            # These checks run regardless of market override
            
            # Check RSI if available
            if 'rsi_14' in df.columns:
                rsi_value = df['rsi_14'].iloc[-1]
                rsi_overbought = float(self.rsi_overbought.get()) if hasattr(self, 'rsi_overbought') else 70
                rsi_oversold = float(self.rsi_oversold.get()) if hasattr(self, 'rsi_oversold') else 30
                
                if rsi_value > rsi_overbought:
                    self.log_trade(f"RSI filter rejected {symbol}: {rsi_value:.2f} > {rsi_overbought}")
                    return False
                    
                self.log_trade(f"✓ RSI check passed: {rsi_value:.2f}")
            
            # Check EMA crossover if available
            if 'ema_5' in df.columns and 'ema_15' in df.columns:
                ema_cross = (df['ema_5'].iloc[-1] > df['ema_15'].iloc[-1])
                if not ema_cross:
                    self.log_trade(f"EMA filter rejected {symbol}: No bullish crossover")
                    return False
                    
                self.log_trade(f"✓ EMA crossover check passed")
            
            # Check momentum
            momentum = self.calculate_current_momentum(df)
            min_momentum = 0.2  # Minimum 0.2% price rise
            if momentum < min_momentum:
                self.log_trade(f"Momentum filter rejected {symbol}: {momentum:.2f}% < {min_momentum}%")
                return False
                
            self.log_trade(f"✓ Momentum check passed: {momentum:.2f}%")
            
            # All filters passed
            self.log_trade(f"All filters passed for {symbol}")
            return True
            
        except Exception as e:
            self.log_trade(f"Error in smart filter for {symbol}: {str(e)}")
            return False

    def check_counter_trend_strength(self, df):
        """Check for counter-trend strength in bear markets"""
        try:
            # Calculate RSI
            if 'rsi' not in df.columns:
                delta = df['close'].diff()
                gain = delta.where(delta > 0, 0)
                loss = -delta.where(delta < 0, 0)
                avg_gain = gain.rolling(window=14).mean()
                avg_loss = loss.rolling(window=14).mean()
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
            else:
                rsi = df['rsi']
                
            # Check for oversold conditions (RSI < 30)
            is_oversold = rsi.iloc[-1] < 30
            
            # Check for bullish divergence
            price_lower = df['close'].iloc[-1] < df['close'].iloc[-3]
            rsi_higher = rsi.iloc[-1] > rsi.iloc[-3]
            
            # Check for volume spike
            volume_spike = df['volume'].iloc[-1] > df['volume'].rolling(window=10).mean().iloc[-1] * 2
            
            # Return True if we have signs of counter-trend strength
            return is_oversold or (price_lower and rsi_higher) or volume_spike
            
        except Exception as e:
            self.log_trade(f"Error checking counter-trend strength: {str(e)}")
            return False

    def analyze_performance(self):
        """Analyze trading performance using pandas"""
        try:
            # Convert trades to DataFrame
            trades_df = pd.DataFrame(self.trades)
            
            if len(trades_df) < 5:  # Need minimum trades
                return
                
            # Calculate key metrics
            metrics = {
                'win_rate': len(trades_df[trades_df['net_profit'] > 0]) / len(trades_df) * 100,
                'avg_profit': trades_df['net_profit'].mean(),
                'avg_duration': trades_df['duration'].mean(),
                'profit_factor': abs(
                    trades_df[trades_df['net_profit'] > 0]['net_profit'].sum() /
                    trades_df[trades_df['net_profit'] < 0]['net_profit'].sum()
                ),
                'sharpe_ratio': (
                    trades_df['net_profit'].mean() / 
                    trades_df['net_profit'].std()
                ) * np.sqrt(252)  # Annualized
            }
            
            # Log performance
            self.log_trade(f"""
            Performance Metrics:
            Win Rate: {metrics['win_rate']:.1f}%
            Avg Profit: ${metrics['avg_profit']:.2f}
            Avg Duration: {metrics['avg_duration']:.1f}s
            Profit Factor: {metrics['profit_factor']:.2f}
            Sharpe Ratio: {metrics['sharpe_ratio']:.2f}
            """)
            
            return metrics
            
        except Exception as e:
            self.log_trade(f"Performance analysis error: {str(e)}")
            return None

    def update_metrics(self):
        try:
            # Calculate win rate
            win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0
            
            # Calculate net profit
            net_profit = self.total_profit - self.total_fees
            
            # Debug logging
            self.log_trade(f"""
            Updating Metrics:
            Winning Trades: {self.winning_trades}
            Losing Trades: {self.losing_trades}
            Total Trades: {self.total_trades}
            Total Profit: ${self.total_profit:.2f}
            Total Fees: ${self.total_fees:.2f}
            Net Profit: ${net_profit:.2f}
            Win Rate: {win_rate:.1f}%
            """)
            
            # Update labels with proper formatting
            self.total_profit_label.config(
                text=f"Total Profit: {self.total_profit:.2f} USD",
                foreground="green" if self.total_profit > 0 else "red"
            )
            
            self.total_fees_label.config(
                text=f"Total Fees: {self.total_fees:.2f} USD"
            )
            
            self.net_profit_label.config(
                text=f"Net Profit: {net_profit:.2f} USD",
                foreground="green" if net_profit > 0 else "red"
            )
            
            self.win_loss_label.config(
                text=f"Win/Loss: {self.winning_trades}/{self.losing_trades} ({win_rate:.1f}%)"
            )
            
            # Force GUI update
            self.root.update_idletasks()
            
        except Exception as e:
            self.log_trade(f"Error updating metrics: {str(e)}")

    def collect_trade_data(self):
        """Initialize trade data collection system"""
        try:
            import pandas as pd
            from datetime import datetime
            import json
            import os

            # Initialize DataFrame if not exists
            if not hasattr(self, 'trade_data'):
                self.trade_data = pd.DataFrame(columns=[
                    'timestamp',
                    'symbol',
                    'entry_price',
                    'exit_price',
                    'volume',
                    'entry_spread',
                    'duration',
                    'profit_loss',
                    'profit_loss_pct',
                    'fees',
                    'net_profit',
                    'entry_rsi',
                    'entry_volume_change',
                    'price_momentum',
                    'market_condition',
                    'success'  # True if profitable after fees
                ])

            # Create data directory if it doesn't exist
            if not os.path.exists('trade_data'):
                os.makedirs('trade_data')

        except Exception as e:
            self.log_trade(f"Error initializing trade data collection: {str(e)}")

    def update_btc_data(self):
        """Update BTC/USD market data for market analysis"""
        try:
            self.log_trade("Updating BTC/USD market data...")
            
            # Skip if no exchange connection
            if not hasattr(self, 'exchange') or self.exchange is None:
                self.log_trade("No exchange connection available for BTC data update")
                return False
            
            # Fetch BTC/USD ticker
            try:
                btc_ticker = self.exchange.fetch_ticker('BTC/USD')
                if not btc_ticker:
                    self.log_trade("Failed to fetch BTC/USD ticker")
                    return False
                
                # Update price data for BTC/USD
                self.data_manager.update_price_data('BTC/USD', btc_ticker)
                
                # Log success
                btc_price = float(btc_ticker['last'])
                self.log_trade(f"Updated BTC/USD price: ${btc_price:.2f}")
                
                # Calculate market indicators if we have enough data
                df = self.data_manager.get_price_data('BTC/USD')
                if df is not None and len(df) >= 5:
                    # Calculate indicators
                    self.data_manager._calculate_indicators('BTC/USD')
                    self.log_trade("BTC/USD indicators updated")
                
                return True
                
            except Exception as e:
                self.log_trade(f"Error fetching BTC/USD data: {str(e)}")
                return False
            
        except Exception as e:
            self.log_trade(f"Error updating BTC/USD data: {str(e)}")
            return False

    def update_trade_history(self, symbol, percentage, profit, is_win=True, status="closed"):
        """Update the trade history with a completed trade"""
        try:
            # Ensure percentage is within reasonable bounds
            if abs(percentage) > 50:  # Sanity check
                self.log_trade(f"Warning: Unusual percentage detected for {symbol}: {percentage}%")
                percentage = min(max(percentage, -50), 50)  # Cap at ±50%

            # Format timestamp
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Format the trade result with status if not normal close
            status_text = ""
            if status != "closed":
                status_text = f" [{status}]"
            
            # Format the trade result
            result = f"[{timestamp}] {symbol}: {percentage:.2f}%, ${profit:.2f}{status_text}\n"
            
            # Store trade in history list FIRST (most important)
            if not hasattr(self, 'trades'):
                self.trades = []
                self.log_trade("Initializing trades list")
                
            trade_record = {
                'timestamp': datetime.now(),
                'symbol': symbol,
                'percentage': percentage,
                'profit': profit,
                'is_win': is_win,
                'status': status,
                'is_paper': self.is_paper_trading
            }
            
            # Add to trades list
            self.trades.append(trade_record)
            self.log_trade(f"Added trade to history list: {symbol} {percentage:.2f}% ${profit:.2f}")
            
            # Keep only last 100 trades
            if len(self.trades) > 100:
                self.trades = self.trades[-100:]
            
            # Update GUI trade history if available
            if hasattr(self, 'history_text'):
                try:
                    # Make sure the history window exists and is initialized
                    if not hasattr(self, 'history_window') or not self.history_window:
                        self.setup_trade_history_display()
                    
                    # Ensure the history text widget exists
                    if hasattr(self, 'history_text'):
                        self.history_text.config(state=tk.NORMAL)
                        self.history_text.insert(tk.END, result)
                        
                        # Color code based on profit/loss
                        line_start = f"{float(self.history_text.index('end-2c').split('.')[0])-1}.0"
                        line_end = f"{float(self.history_text.index('end-1c').split('.')[0])-1}.end"
                        
                        color = "green" if is_win else "red"
                        self.history_text.tag_add(color, line_start, line_end)
                        self.history_text.tag_config("green", foreground="green")
                        self.history_text.tag_config("red", foreground="red")
                        
                        self.history_text.see(tk.END)
                        self.history_text.config(state=tk.DISABLED)
                        
                        # Force GUI update
                        self.history_text.update_idletasks()
                        
                        self.log_trade(f"Updated history text widget with trade: {symbol}")
                    else:
                        self.log_trade("History text widget not available")
                except Exception as e:
                    self.log_trade(f"Error updating trade history display: {str(e)}")
            else:
                self.log_trade("History text widget not found")
                
            # Also save to file for persistence
            try:
                history_file = "trade_history.json"
                with open(history_file, 'w') as f:
                    # Convert datetime objects to strings for JSON serialization
                    serializable_trades = []
                    for t in self.trades:
                        t_copy = t.copy()
                        t_copy['timestamp'] = t_copy['timestamp'].strftime("%Y-%m-%d %H:%M:%S")
                        serializable_trades.append(t_copy)
                        
                    json.dump(serializable_trades, f)
                self.log_trade(f"Trade history saved to {history_file}")
            except Exception as e:
                self.log_trade(f"Error saving trade history to file: {str(e)}")
                
        except Exception as e:
            self.log_trade(f"Critical error in update_trade_history: {str(e)}")
            # Attempt to log basic trade info even if display fails
            self.log_trade(f"Trade completed - {symbol}: {percentage:.2f}% (${profit:.2f})")

    def analyze_patterns(self):
        """Analyze collected trade data for patterns"""
        try:
            if len(self.trade_data) < 100:
                return

            # Basic analysis
            analysis = {
                'overall_win_rate': (self.trade_data['success'].sum() / len(self.trade_data)) * 100,
                'avg_profit_loss': self.trade_data['net_profit'].mean(),
                'avg_duration': self.trade_data['duration'].mean(),
                'best_performing_symbols': self.trade_data.groupby('symbol')['net_profit'].mean().sort_values(ascending=False).head(),
                'optimal_entry_rsi': self.trade_data[self.trade_data['success']]['entry_rsi'].mean(),
                'optimal_volume_change': self.trade_data[self.trade_data['success']]['entry_volume_change'].mean()
            }

            # Time-based analysis
            self.trade_data['hour'] = pd.to_datetime(self.trade_data['timestamp']).dt.hour
            hourly_performance = self.trade_data.groupby('hour')['success'].mean()
            best_hours = hourly_performance.sort_values(ascending=False).head()

            # Pattern analysis
            successful_conditions = self.trade_data[self.trade_data['success']]
            pattern_insights = {
                'avg_entry_spread': successful_conditions['entry_spread'].mean(),
                'avg_momentum': successful_conditions['price_momentum'].mean(),
                'profitable_duration_range': (
                    successful_conditions['duration'].quantile(0.25),
                    successful_conditions['duration'].quantile(0.75)
                )
            }

            # Log insights
            self.log_trade(f"""
            Trade Pattern Analysis:
            Overall Win Rate: {analysis['overall_win_rate']:.2f}%
            Average Profit/Loss: ${analysis['avg_profit_loss']:.2f}
            Average Duration: {analysis['avg_duration']:.1f}s
            
            Best Performing Symbols:
            {analysis['best_performing_symbols'].to_string()}
            
            Optimal Conditions:
            - RSI: {analysis['optimal_entry_rsi']:.1f}
            - Volume Change: {analysis['optimal_volume_change']:.1f}%
            
            Best Trading Hours:
            {best_hours.to_string()}
            
            Pattern Insights:
            - Optimal Spread: {pattern_insights['avg_entry_spread']:.4f}%
            - Optimal Momentum: {pattern_insights['avg_momentum']:.2f}%
            - Profitable Duration: {pattern_insights['profitable_duration_range'][0]:.1f}s - {pattern_insights['profitable_duration_range'][1]:.1f}s
            """)

            # Update trading parameters based on analysis
            self.update_trading_parameters(analysis, pattern_insights)

        except Exception as e:
            self.log_trade(f"Error analyzing patterns: {str(e)}")

    def save_trade_data(self):
        """Save trade data to file"""
        try:
            filename = f'trade_data/trade_history_{datetime.now().strftime("%Y%m%d")}.csv'
            self.trade_data.to_csv(filename, index=False)
            self.log_trade(f"Trade data saved to {filename}")

        except Exception as e:
            self.log_trade(f"Error saving trade data: {str(e)}")

    def update_trading_parameters(self, analysis, pattern_insights):
        """Update trading parameters based on analysis"""
        try:
            # Only update if we have significant data
            if len(self.trade_data) < 100:
                return

            # Update parameters based on successful trades
            if analysis['overall_win_rate'] > 50:
                # Adjust RSI thresholds
                optimal_rsi = analysis['optimal_entry_rsi']
                self.rsi_lower = max(30, optimal_rsi - 10)
                self.rsi_upper = min(70, optimal_rsi + 10)

                # Adjust volume requirements
                optimal_volume_change = analysis['optimal_volume_change']
                self.volume_change_threshold = max(0.05, optimal_volume_change * 0.8)

                # Adjust timing parameters
                optimal_duration = pattern_insights['profitable_duration_range'][1]
                self.max_trade_duration = int(optimal_duration * 1.2)

                self.log_trade(f"""
                Updated Trading Parameters:
                - RSI Range: {self.rsi_lower:.1f} - {self.rsi_upper:.1f}
                - Volume Change Threshold: {self.volume_change_threshold:.2f}%
                - Max Trade Duration: {self.max_trade_duration}s
                """)

        except Exception as e:
            self.log_trade(f"Error updating trading parameters: {str(e)}")    

    def update_trade_history(self, symbol, percentage, profit, is_win=True, status="closed"):
        """Update the trade history with a completed trade"""
        try:
            # Ensure percentage is within reasonable bounds
            if abs(percentage) > 50:  # Sanity check
                self.log_trade(f"Warning: Unusual percentage detected for {symbol}: {percentage}%")
                percentage = min(max(percentage, -50), 50)  # Cap at ±50%

            # Format timestamp - use a shorter format for display
            timestamp = datetime.now().strftime("%H:%M:%S")
            
            # Format the trade result with status if not normal close
            status_text = ""
            if status != "closed":
                status_text = f" [{status}]"
            
            # Format the trade result
            result = f"[{timestamp}] {symbol}: {percentage:.2f}%, ${profit:.2f}{status_text}\n"
            
            # Store trade in history list FIRST (most important)
            if not hasattr(self, 'trades'):
                self.trades = []
                self.log_trade("Initializing trades list")
                
            trade_record = {
                'timestamp': datetime.now(),
                'symbol': symbol,
                'percentage': percentage,
                'profit': profit,
                'is_win': is_win,
                'status': status,
                'is_paper': self.is_paper_trading
            }
            
            # Add to trades list
            self.trades.append(trade_record)
            self.log_trade(f"Added trade to history list: {symbol} {percentage:.2f}% ${profit:.2f}")
            
            # Keep only last 100 trades
            if len(self.trades) > 100:
                self.trades = self.trades[-100:]
            
            # Update GUI trade history if available
            if hasattr(self, 'trades_text'):
                try:
                    # Use trades_text instead of history_text
                    self.trades_text.config(state=tk.NORMAL)
                    self.trades_text.insert(tk.END, result)
                    
                    # Color code based on profit/loss
                    line_start = f"{float(self.trades_text.index('end-2c').split('.')[0])-1}.0"
                    line_end = f"{float(self.trades_text.index('end-1c').split('.')[0])-1}.end"
                    
                    color = "green" if is_win else "red"
                    self.trades_text.tag_add(color, line_start, line_end)
                    self.trades_text.tag_config("green", foreground="green")
                    self.trades_text.tag_config("red", foreground="red")
                    
                    self.trades_text.see(tk.END)
                    self.trades_text.config(state=tk.DISABLED)
                    
                    # Force GUI update
                    self.trades_text.update_idletasks()
                    
                    self.log_trade(f"Updated trades text widget with trade: {symbol}")
                except Exception as e:
                    self.log_trade(f"Error updating trades text display: {str(e)}")
            else:
                self.log_trade("Trades text widget not found")
                    
            # Also save to file for persistence
            try:
                history_file = "trade_history.json"
                with open(history_file, 'w') as f:
                    # Convert datetime objects to strings for JSON serialization
                    serializable_trades = []
                    for t in self.trades:
                        t_copy = t.copy()
                        t_copy['timestamp'] = t_copy['timestamp'].strftime("%Y-%m-%d %H:%M:%S")
                        serializable_trades.append(t_copy)
                        
                    json.dump(serializable_trades, f)
                self.log_trade(f"Trade history saved to {history_file}")
            except Exception as e:
                self.log_trade(f"Error saving trade history to file: {str(e)}")
                
        except Exception as e:
            self.log_trade(f"Critical error in update_trade_history: {str(e)}")
            # Attempt to log basic trade info even if display fails
            self.log_trade(f"Trade completed - {symbol}: {percentage:.2f}% (${profit:.2f})")

    def log_trade(self, message):
        """Log a trade message to both the GUI and the log file with proper encoding"""
        try:
            # Replace Unicode characters with ASCII alternatives
            message = str(message)  # Ensure message is a string
            message = message.replace("✓", "√").replace("√", "v")  # Replace checkmark and square root with ASCII
            message = message.replace("✅", "[OK]")
            message = message.replace("❌", "[X]")
            message = message.replace("⚠️", "[!]")
            message = message.replace("📈", "[UP]")
            message = message.replace("📉", "[DOWN]")
            message = message.replace("\u221a", "v")  # Replace square root symbol
            
            # Get timestamp
            timestamp = datetime.now().strftime("%H:%M:%S")
            
            # Log to console and file
            print(f"{timestamp} - {message}")
            if hasattr(self, 'logger'):
                # Ensure ASCII-only for file logging
                ascii_message = message.encode('ascii', 'replace').decode()
                self.logger.info(ascii_message)
            
            # Log to GUI if available
            if hasattr(self, 'log_text') and self.log_text:
                # Add to log text widget
                self.log_text.insert(tk.END, f"{timestamp} - {message}\n")
                self.log_text.see(tk.END)
                
                # Apply tag for important messages
                if "[OK]" in message:
                    self.log_text.tag_add("success", f"end-{len(message)+12}c", "end-1c")
                    self.log_text.tag_config("success", foreground="green")
                elif "[X]" in message:
                    self.log_text.tag_add("error", f"end-{len(message)+12}c", "end-1c")
                    self.log_text.tag_config("error", foreground="red")
                elif "[!]" in message:
                    self.log_text.tag_add("warning", f"end-{len(message)+12}c", "end-1c")
                    self.log_text.tag_config("warning", foreground="orange")
                
                # Force GUI update
                if hasattr(self, 'root'):
                    self.root.update_idletasks()
                    
        except Exception as e:
            print(f"Logging error: {str(e)}")
            # Fallback logging with just ASCII
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"{timestamp} - {str(message).encode('ascii', 'replace').decode()}")

    def update_trade_history(self, symbol, percentage, profit, is_win=True):
        try:
            # Ensure percentage is within reasonable bounds
            if abs(percentage) > 50:  # Sanity check
                self.log_trade(f"Warning: Unusual percentage detected for {symbol}: {percentage}%")
                percentage = min(max(percentage, -50), 50)  # Cap at ±50%

            # Format the trade result
            result = f"{symbol}: {percentage:.2f}%, ${profit:.2f}\n"
            
            if hasattr(self, 'history_text'):
                self.history_text.insert(tk.END, result)
                
                # Color code based on actual profit/loss
                line_start = f"{float(self.history_text.index('end-2c'))-1:.1f}"
                color = "green" if profit > 0 else "red"
                self.history_text.tag_add(color, f"{line_start} linestart", f"{line_start} lineend")
                self.history_text.tag_config("green", foreground="green")
                self.history_text.tag_config("red", foreground="red")
                
                self.history_text.see(tk.END)
                
        except Exception as e:
            self.log_trade(f"Error updating trade history: {str(e)}")

    def get_min_volume(self):
        """Safely get min volume setting"""
        try:
            return float(self.min_volume_entry.get())
        except:
            return 100.0  # Default value
    def get_market_state(df):
        # 0=Range, 1=Uptrend, -1=Downtrend
        atr = df['high'] - df['low'].rolling(14).mean()
        return 1 if (df['close'][-1] > df['upper_bb'][-1]) else \
            -1 if (df['close'][-1] < df['lower_bb'][-1]) else 0

      
    def validate_ticker(self, ticker, symbol=None):
        """Validate ticker data has required fields"""
        try:
            # If symbol is provided, check it matches
            if symbol and ticker.get('symbol') != symbol:
                return False
                
            # Check required fields
            required_fields = ['last', 'bid', 'ask']
            for field in required_fields:
                if field not in ticker or ticker[field] is None:
                    return False
                    
            # Check values are numeric and positive
            for field in required_fields:
                try:
                    value = float(ticker[field])
                    if value <= 0:
                        return False
                except:
                    return False
                    
            return True
            
        except Exception:
            return False

    def update_active_trades_display(self):
        """Update the active trades display in the GUI"""
        try:
            self.trades_text.delete(1.0, tk.END)
            
            if not self.active_trades:
                self.trades_text.insert(tk.END, "No active trades\n")
                return
            
            self.trades_text.insert(tk.END, f"Number of Active Trades: {len(self.active_trades)}\n\n")
            
            for trade_id, trade in list(self.active_trades.items()):
                try:
                    current_price = trade.get('current_price', trade['entry_price'])
                    profit_percentage = trade.get('current_profit_percentage', 0)
                    duration = (datetime.now() - trade['timestamp']).total_seconds()
                    highest_profit = trade.get('highest_profit_percentage', 0)
                    
                    trade_info = (
                        f"Symbol: {trade['symbol']}\n"
                        f"Entry: {trade['entry_price']:.8f}\n"
                        f"Current: {current_price:.8f}\n"
                        f"P/L: {profit_percentage:.2f}%\n"
                        f"Highest: {highest_profit:.2f}%\n"
                        f"Time: {duration:.1f}s\n"
                        f"Last Update: {trade.get('last_update', 'Never')}\n"
                        f"-----------------\n"
                    )
                    
                    self.trades_text.insert(tk.END, trade_info)
                    
                    # Color coding
                    start_idx = f"end-{len(trade_info)+1}c linestart"
                    end_idx = "end-1c"
                    
                    if profit_percentage > 0:
                        self.trades_text.tag_add("profit", start_idx, end_idx)
                        self.trades_text.tag_config("profit", foreground="green")
                    else:
                        self.trades_text.tag_add("loss", start_idx, end_idx)
                        self.trades_text.tag_config("loss", foreground="red")
                    
                except Exception as e:
                    self.log_trade(f"Error updating display for trade {trade_id}: {str(e)}")
                    continue
            
            # Force GUI update
            self.trades_text.see(tk.END)
            self.root.update_idletasks()
            
        except Exception as e:
            self.log_trade(f"Error updating trades display: {str(e)}")

    def monitor_trades(self):
        """Monitor active trades with proper trailing stop implementation"""
        try:
            # Skip if no active trades
            if not self.active_trades:
                return
                    
            self.log_trade(f"Monitoring {len(self.active_trades)} active trades...")
            
            # Get current trading conditions
            current_conditions = {
                'profit_target': float(self.profit_target.get()) / 100,  # Convert to decimal
                'stop_loss': float(self.stop_loss.get()) / 100,  # Convert to decimal
                'trailing_stop': float(self.trailing_stop.get()) / 100,  # Convert to decimal
                'trailing_activation': float(self.trailing_activation.get()) / 100  # Convert to decimal
            }
            
            # For logging frequency control
            current_time = time.time()
            log_interval = 30  # Log status every 30 seconds
            
            # Monitor each trade
            for trade_id, trade in list(self.active_trades.items()):
                try:
                    symbol = trade['symbol']
                    entry_price = float(trade['entry_price'])
                    entry_time = trade.get('entry_time', datetime.now())
                    time_in_trade = (datetime.now() - entry_time).total_seconds()
                    last_log_time = trade.get('last_log_time', 0)
                    
                    # Skip very new trades (give them at least 10 seconds to develop)
                    if time_in_trade < 10:
                        self.log_trade(f"Trade {symbol} is too new ({time_in_trade:.1f}s), skipping checks")
                        continue
                    
                    # Get current price with error handling
                    try:
                        ticker = self.exchange.fetch_ticker(symbol)
                        current_price = float(ticker['last'])
                        
                        # Update trade with current price
                        self.active_trades[trade_id]['current_price'] = current_price
                        self.active_trades[trade_id]['last_update'] = datetime.now().strftime("%H:%M:%S")
                        
                        # Calculate profit percentage
                        profit_pct = ((current_price - entry_price) / entry_price)  # Decimal form
                        profit_pct_display = profit_pct * 100  # For display
                        
                        # Update trade with current profit percentage
                        self.active_trades[trade_id]['current_profit_percentage'] = profit_pct_display
                        
                        # Update highest price and profit if applicable
                        if current_price > trade.get('highest_price', 0):
                            self.active_trades[trade_id]['highest_price'] = current_price
                            
                        if profit_pct > trade.get('highest_profit', -1):
                            self.active_trades[trade_id]['highest_profit'] = profit_pct
                            self.active_trades[trade_id]['highest_profit_percentage'] = profit_pct_display
                        
                        # Check for take profit
                        if profit_pct >= current_conditions['profit_target']:
                            self.log_trade(f"Take profit triggered for {symbol}: {profit_pct_display:.2f}% >= {current_conditions['profit_target']*100:.2f}%")
                            self.close_trade(trade_id, trade, current_price, "take_profit")
                            continue
                        
                        # Check for stop loss
                        if profit_pct <= -current_conditions['stop_loss']:
                            self.log_trade(f"Stop loss triggered for {symbol}: {profit_pct_display:.2f}% <= -{current_conditions['stop_loss']*100:.2f}%")
                            self.close_trade(trade_id, trade, current_price, "stop_loss")
                            continue
                        
                        # Check for trailing stop
                        highest_profit = trade.get('highest_profit', 0)
                        if highest_profit >= current_conditions['trailing_activation']:
                            # Calculate drop from highest profit
                            drop = highest_profit - profit_pct
                            if drop >= current_conditions['trailing_stop']:
                                self.log_trade(f"Trailing stop triggered for {symbol}: Drop from {highest_profit*100:.2f}% to {profit_pct_display:.2f}% (>{current_conditions['trailing_stop']*100:.2f}%)")
                                self.close_trade(trade_id, trade, current_price, "trailing_stop")
                                continue
                        
                        # Update price history for charting
                        if symbol in self.price_history:
                            self.price_history[symbol].append((datetime.now(), current_price))
                            # Limit history size
                            self.price_history[symbol] = self.price_history[symbol][-100:]
                        
                        # Log status periodically
                        if current_time - last_log_time > log_interval:
                            self.log_trade(f"Trade {symbol}: Entry=${entry_price:.8f}, Current=${current_price:.8f}, P/L={profit_pct_display:.2f}%, Highest={trade.get('highest_profit_percentage', 0):.2f}%")
                            self.active_trades[trade_id]['last_log_time'] = current_time
                    
                    except Exception as e:
                        self.log_trade(f"Error getting price for {symbol}: {str(e)}")
                        continue
                
                except Exception as e:
                    self.log_trade(f"Error monitoring trade {trade_id}: {str(e)}")
                    continue
            
            # Update displays after monitoring
            self.update_active_trades_display()
            self.update_balance_display()
            
        except Exception as e:
            self.log_trade(f"Error in trade monitoring: {str(e)}")

    def run_bot(self):
        """Main bot loop with improved trade monitoring"""
        try:
            self.log_trade("=== BOT STARTED ===")
            self.update_status("Running (Paper)" if self.is_paper_trading else "Running (Real)")
            self.running = True
            
            # Main loop
            while self.running:
                try:
                    # 1. UPDATE MARKET DATA
                    self.update_status("Fetching market data...")
                    self.update_btc_data()  # Always update BTC as market indicator
                    
                    # 2. MONITOR EXISTING TRADES
                    if self.active_trades:
                        self.monitor_trades()
                        
                        # Give trades time to develop before scanning for new ones
                        time.sleep(3)
                    
                    # 3. LOOK FOR NEW TRADES (Entry conditions)
                    self.update_status("Scanning for opportunities...")
                    scan_success = self.scan_opportunities()
                    
                    # 4. UPDATE DISPLAYS
                    self.update_active_trades_display()
                    self.update_chart()
                    self.update_metrics()
                    
                    # 5. WAIT BEFORE NEXT SCAN
                    # Longer wait if we have active trades to give them time to develop
                    wait_time = 5 if self.active_trades else 2
                    time.sleep(wait_time)

                except Exception as e:
                    self.log_trade(f"Error in main loop: {str(e)}")
                    time.sleep(5)  # Longer pause on error
                    continue

        except Exception as e:
            self.log_trade(f"Critical error in run_bot: {str(e)}")
        finally:
            self.running = False
            self.update_status("Stopped")
            self.log_trade("=== BOT STOPPED ===")

    def run(self):
        """Main run method"""
        try:
            startup_message = f"""
            Crypto Scalping Bot Started
            -------------------------
            Mode: {'Paper' if self.is_paper_trading else 'Real'} Trading
            Exchange: Kraken
            Initial Balance: ${self.paper_balance:.2f}
            """
            self.log_trade(startup_message)
            
            # Update mode display
            if hasattr(self, 'mode_var'):
                self.mode_var.set("Paper Trading" if self.is_paper_trading else "Real Trading")
            
            # Start in the main thread
            self.root.mainloop()
            
        except Exception as e:
            self.log_trade(f"Critical error in main loop: {str(e)}")
            try:
                if self.active_trades:
                    self.log_trade("Attempting to close open trades before shutdown...")
                    for trade_id, trade in list(self.active_trades.items()):
                        try:
                            current_price = self.exchange.fetch_ticker(trade['symbol'])['last']
                            self.close_trade(trade_id, trade, current_price, "emergency shutdown")
                        except:
                            continue
            finally:
                self.log_trade("Bot shutdown complete.")
if __name__ == "__main__":
    try:
        bot = CryptoScalpingBot()
        bot.run()
    except Exception as e:
        print(f"Fatal error: {str(e)}")
        logging.error(f"Fatal error: {str(e)}")
