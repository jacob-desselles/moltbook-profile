import warnings
warnings.filterwarnings('ignore', message="fatal: bad revision 'HEAD'")
warnings.filterwarnings('ignore', category=UserWarning)
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import ccxt
import pandas as pd
import numpy as np
from datetime import datetime
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
    def log_trade(self, message):
        """Internal logging method"""
        try:
            if self.log_function:
                self.log_function(message)
            else:
                print(message)
        except Exception as e:
            print(f"Logging error: {str(e)}")

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
                
                # RSI Calculation
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
                    
                    # Calculate RS and RSI
                    rs = avg_gain / avg_loss
                    rs = rs.replace([np.inf, -np.inf], np.nan)  # Avoid division by zero
                    rsi = 100 - (100 / (1 + rs))
                    
                    # Store RSI in DataFrame
                    df[f'rsi_{rsi_period}'] = rsi
                    
                    self.log_trade(f"RSI calculated for {symbol} with period {rsi_period}: {rsi.iloc[-1]:.2f}")
                
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
        super().__init__()
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
        self.root.title("Kairos Trading v1.0") 
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
        """Show API configuration window with proper config handling"""
        try:
            config_window = tk.Toplevel(self.root)
            config_window.title("API Configuration")
            config_window.resizable(False, False)
            
            # Center the window
            window_width = 400
            window_height = 250
            screen_width = self.root.winfo_screenwidth()
            screen_height = self.root.winfo_screenheight()
            x = (screen_width - window_width) // 2
            y = (screen_height - window_height) // 2
            config_window.geometry(f"{window_width}x{window_height}+{x}+{y}")

            ttk.Label(config_window, text="Kraken API Configuration", font=('Arial', 12)).pack(pady=10)

            frame = ttk.Frame(config_window)
            frame.pack(padx=20, pady=10, fill='both', expand=True)

            # API Key
            ttk.Label(frame, text="API Key:").grid(row=0, column=0, sticky='w', pady=5)
            api_key_entry = ttk.Entry(frame, width=40)
            api_key_entry.grid(row=0, column=1, padx=5)
            if self.config.has_option('API_KEYS', 'api_key'):
                api_key_entry.insert(0, self.config.get('API_KEYS', 'api_key'))

            # Secret Key
            ttk.Label(frame, text="Secret Key:").grid(row=1, column=0, sticky='w', pady=5)
            secret_key_entry = ttk.Entry(frame, width=40, show="*")
            secret_key_entry.grid(row=1, column=1, padx=5)
            if self.config.has_option('API_KEYS', 'secret'):
                secret_key_entry.insert(0, self.config.get('API_KEYS', 'secret'))

            def save_config():
                self.config['API_KEYS'] = {
                    'api_key': api_key_entry.get(),
                    'secret': secret_key_entry.get()  # Note: using 'secret' to match ccxt
                }
                with open('config.ini', 'w') as f:
                    self.config.write(f)
                messagebox.showinfo("Success", "API configuration saved")
                config_window.destroy()
                self.init_exchange()

            # Button frame
            button_frame = ttk.Frame(frame)
            button_frame.grid(row=3, column=0, columnspan=2, pady=15)

            ttk.Button(button_frame, text="Test Connection", 
                    command=lambda: self.test_api_connection(api_key_entry.get(), secret_key_entry.get())).pack(side=tk.LEFT, padx=5)
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
                self.show_api_config()
                return

            api_key = self.config.get('API_KEYS', 'api_key', fallback='')
            secret = self.config.get('API_KEYS', 'secret', fallback='')

            if not api_key or not secret:
                if messagebox.askyesno("API Keys Required", 
                                    "Real trading requires API keys. Configure now?"):
                    self.show_api_config()
                return

            # Test the connection
            try:
                test_exchange = ccxt.kraken({
                    'apiKey': api_key,
                    'secret': secret,
                    'enableRateLimit': True
                })
                test_exchange.fetch_balance()  # Test API call
            except Exception as e:
                self.log_trade(f"API test failed: {str(e)}")
                if messagebox.askyesno("Connection Failed", 
                                    f"API test failed: {str(e)}\nReconfigure keys?"):
                    self.show_api_config()
                return

            # Success - switch to real trading
            self.is_paper_trading = False
            self.mode_var.set("Real Trading")
            self.update_status("Real Trading")
            self.init_exchange()  # Reinitialize with verified keys
            
            # Update balance display
            try:
                balance = float(self.exchange.fetch_balance()['USD']['free'])
                self.balance_label.config(text=f"Real Balance: ${balance:.2f}")
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
    def toggle_bot(self):
        """Start/stop the bot with full condition validation"""
        try:
            if not self.running:
                # START SEQUENCE
                if not self.validate_conditions():
                    self.log_trade("Start aborted - invalid conditions")
                    return

                # Verify exchange connection
                try:
                    if not self.is_paper_trading:
                        self.exchange.fetch_balance()  # Test connection
                except Exception as e:
                    messagebox.showerror("Connection Error", f"Exchange error: {str(e)}")
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

            self.start_button = ttk.Button(mode_frame, text="Start", command=self.toggle_bot)
            self.start_button.grid(row=0, column=3, padx=5)

            self.status_label = ttk.Label(mode_frame, 
                text=f"Status: Idle ({'Paper' if self.is_paper_trading else 'Real'} Trading)")
            self.status_label.grid(row=0, column=4, padx=5)

            self.timer_label = ttk.Label(mode_frame, text="Runtime: 00:00:00")
            self.timer_label.grid(row=0, column=5, padx=5)

            ttk.Button(mode_frame, text="API Config", 
                    command=self.show_api_config).grid(row=0, column=6, padx=5)

            # Create notebook for parameter organization
            param_notebook = ttk.Notebook(control_frame)
            param_notebook.grid(row=1, column=0, columnspan=2, padx=5, pady=5, sticky="ew")

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
            
        except Exception as e:
            self.log_trade(f"Error setting up GUI: {str(e)}")
            raise

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
        """Calculate momentum intensity (Beta) - Market sensitivity"""
        try:
            if df is None or len(df) < 5:
                return 0.0
                
            # Ensure DataFrame is properly sorted
            df = df.copy()
            df.sort_index(inplace=True)
            
            # Calculate returns using pct_change
            returns = df['price'].pct_change().fillna(0)
            
            # Calculate short and long momentum using standard periods
            short_momentum = returns.rolling(window=5).mean()
            long_momentum = returns.rolling(window=15).mean()
            
            # Calculate trend strength using the difference between short and long momentum
            trend_strength = abs(short_momentum.iloc[-1] - long_momentum.iloc[-1]) * 100
            
            # Log calculation details
            self.log_trade(f"Momentum calculation details:")
            self.log_trade(f"Recent returns: {returns.tail()}")
            self.log_trade(f"Short momentum: {short_momentum.iloc[-1]:.8f}")
            self.log_trade(f"Long momentum: {long_momentum.iloc[-1]:.8f}")
            self.log_trade(f"Trend strength: {trend_strength:.8f}")
            
            # Normalize to 0-1 range
            result = min(trend_strength, 1.0)
            return max(result, 0.0)
            
        except Exception as e:
            self.log_trade(f"Error in momentum calculation: {str(e)}")
            return 0.0
        
    def calculate_momentum_quality(self, df):
        """Calculate momentum quality (Theta) - Stability of momentum"""
        try:
            if df is None or len(df) < 5:
                self.log_trade("Insufficient data for momentum quality calculation")
                return 1.0  # Default to high instability if data is insufficient

            df = df.copy()
            df.sort_index(inplace=True)

            # Calculate 5-period momentum (percentage change over 5 periods)
            momentum = df['price'].pct_change(periods=5).fillna(0)

            # Calculate stability as the standard deviation of momentum over the last 5 periods
            momentum_stability = momentum.rolling(window=5).std().iloc[-1] * 100

            # Log calculation details
            self.log_trade(f"Momentum quality details:")
            self.log_trade(f"Recent momentum: {momentum.tail()}")
            self.log_trade(f"Momentum stability: {momentum_stability:.8f}")

            # Normalize to 0-1 range
            result = min(momentum_stability, 1.0)
            return max(result, 0.0)

        except Exception as e:
            self.log_trade(f"Error in momentum quality calculation: {str(e)}")
            return 1.0  # Default to high instability on error
    
    def calculate_price_acceleration(self, df):
        """Calculate price acceleration (Alpha)"""
        try:
            if df is None or len(df) < 5:
                return 0.0
                
            df = df.copy()
            df.sort_index(inplace=True)
            
            # Calculate price changes and acceleration
            price_changes = df['price'].pct_change().fillna(0)
            acceleration = price_changes.diff().fillna(0)
            
            # Use recent acceleration (last 5 periods)
            recent_acceleration = acceleration.tail(5).mean() * 100
            
            self.log_trade(f"Price acceleration details:")
            self.log_trade(f"Recent price changes: {price_changes.tail()}")
            self.log_trade(f"Recent acceleration: {recent_acceleration:.8f}")
            
            # Normalize
            result = min(abs(recent_acceleration), 1.0)
            return result
            
        except Exception as e:
            self.log_trade(f"Error in acceleration calculation: {str(e)}")
            return 0.0

    def calculate_volatility_sensitivity(self, df):
        """Calculate volatility sensitivity (Vega)"""
        try:
            if df is None or len(df) < 5:
                return 0.0
                
            df = df.copy()
            df.sort_index(inplace=True)
            
            # Calculate rolling volatility using standard deviation
            returns = df['price'].pct_change().fillna(0)
            volatility = returns.rolling(window=5).std()
            
            # Calculate volatility change
            vol_change = volatility.pct_change().fillna(0)
            recent_vol_change = abs(vol_change.tail(5).mean()) * 100
            
            self.log_trade(f"Volatility details:")
            self.log_trade(f"Recent volatility: {volatility.tail()}")
            self.log_trade(f"Recent vol change: {recent_vol_change:.8f}")
            
            # Normalize
            result = min(recent_vol_change, 1.0)
            return result
            
        except Exception as e:
            self.log_trade(f"Error in volatility calculation: {str(e)}")
            return 0.0

    def calculate_volume_impact(self, df, current_volume):
        """Calculate volume impact (Rho)"""
        try:
            if df is None or len(df) < 5:
                return 0.0
                
            df = df.copy()
            df.sort_index(inplace=True)
            
            # Calculate average volume
            avg_volume = df['volume'].rolling(window=5).mean().fillna(0)
            if avg_volume.iloc[-1] == 0:
                return 0.0
                
            # Calculate volume ratio
            volume_ratio = current_volume / avg_volume.iloc[-1]
            
            # Calculate excess volume
            excess_volume = max(0, volume_ratio - 1)
            
            self.log_trade(f"Volume impact details:")
            self.log_trade(f"Current volume: {current_volume}")
            self.log_trade(f"Average volume: {avg_volume.iloc[-1]}")
            self.log_trade(f"Volume ratio: {volume_ratio:.8f}")
            
            # Normalize
            result = min(excess_volume, 1.0)
            return result
            
        except Exception as e:
            self.log_trade(f"Error in volume impact calculation: {str(e)}")
            return 0.0

    def calculate_current_momentum(self, df):
        """Calculate current momentum percentage"""
        try:
            # Add debug logging
            recent_prices = df['price'].tail(5)
            self.log_trade(f"Recent prices for momentum calc: {recent_prices}")
            momentum = ((recent_prices.iloc[-1] - recent_prices.iloc[0]) / recent_prices.iloc[0]) * 100
            self.log_trade(f"Calculated momentum: {momentum}")
            return momentum
        except Exception as e:
            self.log_trade(f"Error calculating momentum: {str(e)}")
            return 0

    def calculate_volume_increase(self, df):
        """Calculate volume increase percentage"""
        try:
            # Compare current volume to average
            current_volume = df['volume'].iloc[-1]
            avg_volume = df['volume'].tail(10).mean()  # 10-period average
            volume_increase = ((current_volume - avg_volume) / avg_volume) * 100
            return volume_increase
        except Exception as e:
            self.log_trade(f"Error calculating volume increase: {str(e)}")
            return 0
        
    def validate_parameters(self):
        """Validate all trading parameters"""
        try:
            validation_rules = {
                'profit_target': {
                    'min': 0.8,  # Minimum to cover fees
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
                'rsi_period': {
                    'min': 5,
                    'max': 30,
                    'value': float(self.rsi_period.get()),
                    'name': 'RSI Period'
                },
                'book_depth': {
                    'min': 5,
                    'max': 50,
                    'value': float(self.book_depth.get()),
                    'name': 'Order Book Depth'
                },
                'scale_in_levels': {
                    'min': 1,
                    'max': 5,
                    'value': float(self.scale_in_levels.get()),
                    'name': 'Scale-in Levels'
                }
            }

            errors = []
            for param, rules in validation_rules.items():
                if rules['value'] < rules['min'] or rules['value'] > rules['max']:
                    errors.append(
                        f"{rules['name']} must be between {rules['min']} and {rules['max']}"
                    )

            # Cross-parameter validation
            if float(self.stop_loss.get()) >= float(self.profit_target.get()):
                errors.append("Stop Loss must be less than Profit Target")
                
            if float(self.trailing_stop.get()) >= float(self.profit_target.get()):
                errors.append("Trailing Stop must be less than Profit Target")
                
            if float(self.trailing_activation.get()) >= float(self.profit_target.get()):
                errors.append("Trailing Activation must be less than Profit Target")

            if errors:
                error_message = "\n".join(errors)
                messagebox.showerror("Parameter Validation Error", error_message)
                return False

            self.log_trade("All parameters validated successfully")
            return True

        except ValueError as e:
            messagebox.showerror("Input Error", "Please ensure all values are numbers")
            return False
        except Exception as e:
            self.log_trade(f"Parameter validation error: {str(e)}")
            return False

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
        """Update the chart with current price data"""
        try:
            if not hasattr(self, 'fig') or self.fig is None:
                return
                    
            self.ax.clear()
            
            # Theme colors
            bg_color = '#2d2d2d' if self.night_mode else 'white'
            fg_color = 'white' if self.night_mode else 'black'
            grid_color = '#3d3d3d' if self.night_mode else '#cccccc'
            
            self.ax.set_facecolor(bg_color)
            self.ax.tick_params(axis='both', colors=fg_color)
            self.ax.grid(True, color=grid_color, linestyle='--', alpha=0.5)
            
            has_data = False
            
            # Plot each active trade
            for trade_id, trade in self.active_trades.items():
                if trade['symbol'] in self.price_history:
                    history = self.price_history[trade['symbol']]
                    if len(history) > 1:
                        times, prices = zip(*history)
                        
                        # Calculate percentage change from entry
                        entry_price = trade['entry_price']
                        prices_pct = [(p - entry_price) / entry_price * 100 for p in prices]
                        
                        # Plot with timestamp x-axis
                        self.ax.plot(times, prices_pct, 
                                label=f"{trade['symbol']} ({prices_pct[-1]:.2f}%)",
                                linewidth=1.5)
                        
                        has_data = True
            
            if has_data:
                # Format axes
                self.ax.xaxis.set_major_formatter(DateFormatter('%H:%M:%S'))
                self.ax.set_xlabel('Time', color=fg_color)
                self.ax.set_ylabel('Price Change (%)', color=fg_color)
                
                # Add legend
                self.ax.legend(loc='upper left')
                
                # Draw profit target and stop loss lines
                self.ax.axhline(y=float(self.profit_target.get()), 
                            color='green', linestyle=':', alpha=0.5)
                self.ax.axhline(y=-float(self.stop_loss.get()), 
                            color='red', linestyle=':', alpha=0.5)
            
            # Draw
            self.canvas.draw()
            
        except Exception as e:
            self.log_trade(f"Chart update error: {str(e)}")

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
                balance = self.paper_balance
            else:
                balance = float(self.exchange.fetch_balance()['USD']['free'])
            
            self.balance_label.config(
                text=f"{'Paper' if self.is_paper_trading else 'Real'} Balance: ${balance:.2f}")
            
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
        """Enhanced version with dynamic condition checking"""
        while self.running:
            try:
                current_conditions = {
                    'max_spread': float(self.max_spread.get()) / 100,
                    'volume_increase': float(self.volume_increase.get()) / 100
                }
                
                for trade_id, trade in list(self.active_trades.items()):
                    ticker = self.exchange.fetch_ticker(trade['symbol'])
                    
                    # Check if current market violates our updated conditions
                    spread = (ticker['ask'] - ticker['bid']) / ticker['bid']
                    if spread > current_conditions['max_spread'] * 1.5:  # 50% buffer
                        self.close_trade(trade_id, trade, ticker['last'], "spread violation")
                        continue
                        
                    # Add other dynamic checks here...
                    
                time.sleep(0.1)  # Faster monitoring interval
                
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

    def scan_opportunities(self):
        try:
            self.log_trade("\n=== Starting New Market Scan ===")
            scan_start_time = time.time()
            
            # Get all tickers with timeout
            try:
                self.update_status("Fetching market data...")
                all_tickers = self.exchange.fetch_tickers()
                if not all_tickers:
                    self.log_trade("Error: No tickers returned from exchange")
                    return False
                self.log_trade(f"Successfully fetched {len(all_tickers)} total pairs")
            except Exception as e:
                self.log_trade(f"Error fetching tickers: {str(e)}")
                return False

            # Filter USD pairs first
            usd_pairs = {symbol: ticker for symbol, ticker in all_tickers.items() 
                        if symbol.endswith('/USD')}
            self.log_trade(f"Found {len(usd_pairs)} USD pairs")

            # Process valid pairs
            valid_pairs = []
            for symbol, ticker in usd_pairs.items():
                try:
                    # Basic price check first
                    if 'last' not in ticker or ticker['last'] is None:
                        continue
                        
                    price = float(ticker['last'])
                    if not (0.00001 <= price <= 5.0):
                        continue

                    processed = self.process_ticker(symbol, ticker)
                    if processed:
                        valid_pairs.append(processed)
                        
                except Exception as e:
                    self.log_trade(f"Error processing {symbol}: {str(e)}")
                    continue

            # Log results
            if not valid_pairs:
                self.log_trade("No valid pairs found matching criteria")
                return False

            self.log_trade(f"\nFound {len(valid_pairs)} valid pairs under $5")

            # Sort and get top pairs
            valid_pairs.sort(key=lambda x: x['volume'], reverse=True)
            top_size = int(self.top_list_size.get())
            top_pairs = valid_pairs[:top_size]
            
            self.log_trade("\nTop pairs by volume:")
            for pair in top_pairs:
                self.log_trade(
                    f"{pair['symbol']}: ${pair['price']:.6f}, "
                    f"Vol: ${pair['volume']:,.2f}, "
                    f"Spread: {pair['spread']:.2f}%"
                )

            # Log scan duration
            scan_duration = time.time() - scan_start_time
            self.log_trade(f"\nScan completed in {scan_duration:.2f} seconds")
            
            return self.analyze_pairs(top_pairs)

        except Exception as e:
            self.log_trade(f"Critical error in market scan: {str(e)}")
            return False

    def quick_filter(self, ticker: dict) -> bool:
        """Fast initial filtering of pairs"""
        try:
            if not self.validate_ticker(ticker):
                return False
            price = float(ticker['last'])
            volume = float(ticker['quoteVolume'])
            spread = (float(ticker['ask']) - float(ticker['bid'])) / float(ticker['bid'])
            max_spread = float(self.max_spread.get()) / 100  # Convert percentage to decimal
            return (
                price <= 5.0 and                         # Price under $5
                price > 0.00001 and                      # Avoid extremely low prices
                volume >= float(self.min_volume_entry.get()) and  # Minimum volume
                spread < max_spread and                  # Use user-defined max spread
                ticker['bid'] > 0 and ticker['ask'] > 0  # Valid bid/ask
            )
        except:
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
            df = self.data_manager.price_data.get(symbol)
            if df is not None:
                data_points = len(df)
                self.log_trade(f"Updated {symbol} data: {data_points} points collected")
            
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
            self.log_trade(f"\nAnalyzing {pair_data['symbol']}...")

            # Force update price data before analysis
            try:
                self.data_manager.update_price_data(pair_data['symbol'], ticker)
                self.log_trade(f"Price data updated for {pair_data['symbol']}")
            except Exception as e:
                self.log_trade(f"Error updating price data: {str(e)}")
                return False

            # Check data freshness with longer timeout
            df = self.data_manager.price_data.get(pair_data['symbol'])
            if df is not None and len(df) > 0:
                latest_time = df.index[-1]
                current_time = pd.Timestamp.now()
                data_age = (current_time - latest_time).total_seconds()
                if data_age > 120:  
                    self.log_trade(f"Skipping {pair_data['symbol']}: Data too old ({data_age:.1f} seconds)")
                    fresh_ticker = self.exchange.fetch_ticker(pair_data['symbol'])
                    self.data_manager.update_price_data(pair_data['symbol'], fresh_ticker)
                    return False

            if df is None or len(df) < 20:
                self.log_trade(f"Insufficient data for {pair_data['symbol']}")
                return False

            # Calculate basic metrics for existing checks
            momentum = self.calculate_current_momentum(df)
            volume_increase = self.calculate_volume_increase(df)

            # Validation logic for existing checks
            if momentum == 0.0 and volume_increase == 0.0:
                self.log_trade(f"Rejected {pair_data['symbol']}: Static data detected")
                return False

            if momentum < float(self.momentum_threshold.get()):
                self.log_trade(f"Rejected {pair_data['symbol']}: Momentum below threshold ({momentum:.2f}% < {self.momentum_threshold.get()}%)")
                return False

            if volume_increase < float(self.volume_increase.get()):
                self.log_trade(f"Rejected {pair_data['symbol']}: Volume increase below threshold ({volume_increase:.2f}% < {self.volume_increase.get()}%)")
                return False

            # Apply Validation Criteria (from previous implementation)
            # 1. Trend Strength (Beta)
            trend_strength = self.calculate_momentum_intensity(df)
            if trend_strength < float(self.momentum_beta.get()):
                self.log_trade(f"Rejected {pair_data['symbol']}: Trend strength below threshold ({trend_strength:.2f} < {self.momentum_beta.get()})")
                return False
            self.log_trade(f"✓ Trend Strength (Beta) passed: {trend_strength:.2f} >= {self.momentum_beta.get()}")

            # 2. Price Momentum (Alpha)
            price_acceleration = self.calculate_price_acceleration(df)
            if price_acceleration < float(self.price_alpha.get()):
                self.log_trade(f"Rejected {pair_data['symbol']}: Price acceleration below threshold ({price_acceleration:.2f} < {self.price_alpha.get()})")
                return False
            self.log_trade(f"✓ Price Momentum (Alpha) passed: {price_acceleration:.2f} >= {self.price_alpha.get()}")

            # 3. Momentum Quality (Theta)
            momentum_quality = self.calculate_momentum_quality(df)
            if momentum_quality > float(self.time_theta.get()):
                self.log_trade(f"Rejected {pair_data['symbol']}: Momentum quality too unstable ({momentum_quality:.2f} > {self.time_theta.get()})")
                return False
            self.log_trade(f"✓ Momentum Quality (Theta) passed: {momentum_quality:.2f} <= {self.time_theta.get()}")

            # 4. Volatility Filter (Vega)
            volatility = self.calculate_volatility_sensitivity(df)
            if volatility > float(self.vol_vega.get()):
                self.log_trade(f"Rejected {pair_data['symbol']}: Volatility too high ({volatility:.2f} > {self.vol_vega.get()})")
                return False
            self.log_trade(f"✓ Volatility Filter (Vega) passed: {volatility:.2f} <= {self.vol_vega.get()}")

            # 5. Volume Quality (Rho)
            current_volume = df['volume'].iloc[-1]
            volume_impact = self.calculate_volume_impact(df, current_volume)
            if volume_impact < float(self.volume_rho.get()):
                self.log_trade(f"Rejected {pair_data['symbol']}: Volume impact below threshold ({volume_impact:.2f} < {self.volume_rho.get()})")
                return False
            self.log_trade(f"✓ Volume Quality (Rho) passed: {volume_impact:.2f} >= {self.vol_vega.get()}")

            # Apply Advanced Parameters via advanced_checks
            if not self.advanced_checks(pair_data['symbol'], df):
                self.log_trade(f"Rejected {pair_data['symbol']}: Failed advanced checks")
                return False
            self.log_trade(f"✓ Advanced checks passed for {pair_data['symbol']}")

            # Opportunity found
            self.log_trade(f"Opportunity found for {pair_data['symbol']}")
            return True

        except Exception as e:
            self.log_trade(f"Critical error analyzing {pair_data['symbol']}: {str(e)}")
            import traceback
            self.log_trade(f"Traceback: {traceback.format_exc()}")
            return False

    def analyze_pairs(self, valid_pairs):
        """Analyze multiple pairs for trading opportunities"""
        try:
            trades_executed = False
            self.log_trade("\nAnalyzing pairs for trading opportunities...")
            
            for pair_data in valid_pairs:
                try:
                    symbol = pair_data['symbol']
                    
                    # Get DataFrame and ensure we have enough data
                    df = self.data_manager.price_data.get(symbol)
                    if df is None or len(df) < 20:  # Need at least 20 points
                        continue
                    
                    # Calculate metrics
                    momentum = self.calculate_current_momentum(df)
                    volume_increase = self.calculate_volume_increase(df)
                    
                    # Debug logging
                    self.log_trade(f"""
                    Detailed Analysis for {symbol}:
                    Current Price: ${pair_data['price']:.8f}
                    24h Volume: ${pair_data['volume']:,.2f}
                    Momentum: {momentum:.2f}%
                    Volume Change: {volume_increase:.2f}%
                    Spread: {pair_data['spread']:.2f}%
                    Data Points: {len(df)}
                    """)
                    
                    # Check trading conditions
                    conditions_met = []
                    
                    # 1. Momentum check
                    if momentum > float(self.momentum_threshold.get()):
                        conditions_met.append("Momentum")
                        self.log_trade(f"✓ Momentum condition met: {momentum:.2f}%")
                    
                    # 2. Volume check
                    min_volume = float(self.min_volume_entry.get())
                    if pair_data['volume'] >= min_volume:
                        conditions_met.append("Volume")
                        self.log_trade(f"✓ Volume condition met: ${pair_data['volume']:,.2f}")
                    
                    # 3. Spread check
                    max_spread = float(self.max_spread.get())
                    if pair_data['spread'] < max_spread:
                        conditions_met.append("Spread")
                        self.log_trade(f"✓ Spread condition met: {pair_data['spread']:.2f}%")
                    
                    # Check if we have enough conditions met
                    required_conditions = int(self.required_conditions.get())
                    self.log_trade(f"Conditions met ({len(conditions_met)}/{required_conditions}): {', '.join(conditions_met)}")
                    
                    if len(conditions_met) >= required_conditions:
                        self.log_trade(f"[TARGET] Trade opportunity found: {symbol}")
                        
                        # Validate and execute trade
                        if self.validate_trade(pair_data, pair_data['price']):
                            if self.execute_trade(pair_data):
                                trades_executed = True
                                self.log_trade(f"Trade executed successfully for {symbol}")
                            else:
                                self.log_trade(f"Failed to execute trade for {symbol}")
                    
                except Exception as e:
                    self.log_trade(f"Error analyzing {pair_data['symbol']}: {str(e)}")
                    continue
            
            if not trades_executed:
                self.log_trade("No trading opportunities found this scan")
            
            return trades_executed
            
        except Exception as e:
            self.log_trade(f"Error in analyze_pairs: {str(e)}")
            return False

    def calculate_trade_parameters(self, entry_price):
        """Calculate optimal trade parameters considering fees"""
        try:
            total_fees = self.taker_fee * 2  # 0.8%
            
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
            return None
    def validate_trade(self, pair_data, current_price):
        try:
            # Check if price has moved too much since analysis
            new_price = self.exchange.fetch_ticker(pair_data['symbol'])['last']
            price_change = abs((new_price - current_price) / current_price * 100)
            
            if price_change > 0.1:  # More than 0.1% price change
                self.log_trade(f"Price moved too much: {price_change:.2f}%")
                return False
                
            # Check if we have enough balance
            position_size = float(self.position_size.get())
            if self.paper_balance < position_size:
                self.log_trade(f"Insufficient balance: {self.paper_balance:.2f} < {position_size:.2f}")
                return False
                
            # Check if symbol already in active trades
            for trade in self.active_trades.values():
                if trade['symbol'] == pair_data['symbol']:
                    self.log_trade(f"Already trading {pair_data['symbol']}")
                    return False
                    
            return True
            
        except Exception as e:
            self.log_trade(f"Error validating trade: {str(e)}")
            return False    

    def execute_trade(self, pair_data):
        """Execute a trade with proper error handling"""
        try:
            # Check number of active trades
            max_trades = int(self.max_trades_entry.get())
            if len(self.active_trades) >= max_trades:
                self.log_trade(f"Maximum trades ({max_trades}) already active")
                return False
                    
            # Get current price with error handling
            try:
                current_price = pair_data['ticker']['last']
                if not current_price:
                    self.log_trade(f"Invalid price for {pair_data['symbol']}")
                    return False
            except Exception as e:
                self.log_trade(f"Error getting price: {str(e)}")
                return False

            # Calculate risk
            position_size = float(self.position_size.get())

            # Calculate trade parameters
            profit_target = float(self.profit_target.get()) / 100
            trailing_stop = float(self.trailing_stop.get()) / 100
            amount = position_size / current_price

            # Round amount to appropriate precision
            try:
                amount = self.exchange.amount_to_precision(pair_data['symbol'], amount)
                amount = float(amount)
            except:
                amount = round(amount, 8)

            # Check balance
            if self.is_paper_trading:
                available_balance = self.paper_balance
            else:
                balance = self.exchange.fetch_balance()
                available_balance = float(balance['USD']['free'])

            # Calculate entry fee
            entry_fee = position_size * self.taker_fee

            if available_balance < (position_size + entry_fee):
                self.log_trade(f"Insufficient balance. Available: {available_balance:.2f} USD, Required: {position_size + entry_fee:.2f} USD")
                return False

            # Calculate target price
            target_price = current_price * (1 + profit_target)

            # Execute trade
            try:
                if self.is_paper_trading:
                    order_id = f"paper_trade_{datetime.now().timestamp()}"
                    self.paper_balance -= (position_size + entry_fee)  # Deduct position size and entry fee
                else:
                    order = self.exchange.create_market_buy_order(
                        pair_data['symbol'],
                        amount,
                        {'trading_agreement': 'agree'}
                    )
                    order_id = order['id']

                # Create trade info
                trade_info = {
                    'symbol': pair_data['symbol'],
                    'entry_price': current_price,
                    'target_price': target_price,
                    'trailing_stop_pct': trailing_stop,
                    'amount': amount,
                    'position_size': position_size,
                    'timestamp': datetime.now(),
                    'highest_price': current_price,
                    'last_update': datetime.now(),
                    'order_ids': {'entry': order_id},
                    'is_paper': self.is_paper_trading,
                    'entry_fee': entry_fee  # Store entry fee for reference
                }

                # Generate unique trade ID
                trade_id = f"{pair_data['symbol']}-{datetime.now().timestamp()}"
                
                # Store trade info
                self.active_trades[trade_id] = trade_info
                
                # Initialize price history for new trade
                if pair_data['symbol'] not in self.price_history:
                    self.price_history[pair_data['symbol']] = []
                
                # Add initial price point
                self.price_history[pair_data['symbol']].append(
                    (datetime.now(), current_price)
                )

                # Log entry
                self.log_trade(f"""
                [TRADE] {'Paper' if self.is_paper_trading else 'Real'} Position opened:
                Symbol: {pair_data['symbol']}
                Entry Price: {current_price:.8f}
                Amount: {amount}
                Position Size: ${position_size:.2f}
                Target: {target_price:.8f} (+{profit_target*100:.1f}%)
                Trailing Stop: {trailing_stop*100:.1f}%
                Entry Fee: ${entry_fee:.2f}
                Total Fees Accumulated: ${self.total_fees + entry_fee:.2f}
                Balance After Entry: ${self.paper_balance:.2f}
                Order ID: {order_id}
                """)

                # Update displays
                self.update_active_trades_display()
                self.update_balance_display()
                return True

            except Exception as e:
                self.log_trade(f"[ERROR] Failed to place {'paper' if self.is_paper_trading else 'real'} orders: {str(e)}")
                return False

        except Exception as e:
            self.log_trade(f"Error executing trade: {str(e)}")
            return False


    def close_trade(self, trade_id, trade, current_price, reason):
        """Close a trade with accurate P/L calculation"""
        try:
            # Validate inputs
            if not all([trade_id, trade, current_price]):
                self.log_trade("Invalid trade closure parameters")
                return False
                
            # Calculate exact values
            entry_price = trade['entry_price']
            amount = trade['amount']
            position_size = trade['position_size']
            
            # Calculate raw P/L
            gross_profit = (current_price - entry_price) * amount
            gross_profit_percentage = ((current_price - entry_price) / entry_price) * 100
            
            # Calculate fees
            entry_fee = trade.get('entry_fee', position_size * self.taker_fee)  # Use stored entry fee if available
            exit_fee = position_size * self.taker_fee
            total_fees = entry_fee + exit_fee
            
            # Calculate net P/L
            net_profit = gross_profit - total_fees
            net_profit_percentage = (net_profit / position_size) * 100
            
            # Update paper balance
            if self.is_paper_trading:
                self.paper_balance += position_size + net_profit
            
            # Log closure details
            self.log_trade(f"""
            Trade Closed: {trade['symbol']}
            Reason: {reason.upper()}
            Entry: ${entry_price:.8f}
            Exit: ${current_price:.8f}
            Gross P/L: ${gross_profit:.2f} ({gross_profit_percentage:.2f}%)
            Fees: ${total_fees:.2f}
            Net P/L: ${net_profit:.2f} ({net_profit_percentage:.2f}%)
            Current Balance: ${self.paper_balance:.2f}
            """)
            
            # Update trade history
            self.update_trade_history(
                symbol=trade['symbol'],
                percentage=net_profit_percentage,
                profit=net_profit,
                is_win=net_profit > 0
            )
            
            # Remove from active trades
            if trade_id in self.active_trades:
                del self.active_trades[trade_id]
            
            # Update performance metrics
            if net_profit > 0:
                self.winning_trades += 1
            else:
                self.losing_trades += 1
                
            self.total_trades += 1
            self.total_profit += gross_profit  # Store raw profit
            self.total_fees += total_fees  # Accumulate full fees
            
            # Update displays
            self.update_active_trades_display()
            self.update_metrics()
            self.update_balance_display()
            
            return True
            
        except Exception as e:
            self.log_trade(f"Error closing trade: {str(e)}")
            # Emergency cleanup
            if trade_id in self.active_trades:
                del self.active_trades[trade_id]
            return False
        
    def advanced_checks(self, symbol, df):
        """Advanced market checks including order book analysis and RSI"""
        try:
            # 1. EMA Cross (5/15)
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
                    current_price = self.get_cached_price(trade['symbol'])['last']
                    profit = (current_price - trade['entry_price']) * trade['amount']
                    profit_percentage = ((current_price - trade['entry_price']) / trade['entry_price']) * 100
                    duration = (datetime.now() - trade['timestamp']).total_seconds()
                    
                    trade_info = (
                        f"Symbol: {trade['symbol']} | "
                        f"Entry: {trade['entry_price']:.8f} | "
                        f"Current: {current_price:.8f} | "
                        f"P/L: {profit:.2f} USD ({profit_percentage:.3f}%) | "
                        f"Duration: {duration:.1f}s | "
                        f"Target: {trade['target_price']:.8f}\n"
                        f"-----------------\n"
                    )
                    
                    # Color code based on profit/loss
                    self.trades_text.insert(tk.END, trade_info)
                    line_start = f"end-{len(trade_info)+1}c linestart"
                    line_end = "end-1c"
                    
                    if profit > 0:
                        self.trades_text.tag_add("profit", line_start, line_end)
                        self.trades_text.tag_config("profit", foreground="green")
                    else:
                        self.trades_text.tag_add("loss", line_start, line_end)
                        self.trades_text.tag_config("loss", foreground="red")
                    
                except Exception as e:
                    self.log_trade(f"Error updating display for trade {trade_id}: {str(e)}")
                    continue
            
            # Force GUI update
            self.trades_text.see(tk.END)
            self.root.update_idletasks()
            
        except Exception as e:
            self.log_trade(f"Error updating trades display: {str(e)}")

    def manage_trades(self):
        """Manage trades with straightforward exit conditions"""
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

                    # Get exit thresholds
                    stop_loss = float(self.stop_loss.get())
                    trailing_stop = float(self.trailing_stop.get())
                    trailing_activation = float(self.trailing_activation.get())

                    # Debug log
                    self.log_trade(f"""
                    Trade Check - {trade['symbol']}:
                    Current P/L: {profit_pct:.2f}%
                    Highest: {trade['highest_profit']:.2f}%
                    Stop Loss: -{stop_loss:.2f}%
                    Trailing Stop: {trailing_stop:.2f}%
                    Trailing Activation: {trailing_activation:.2f}%
                    """)

                    # CASE 1: Stop Loss - Immediate exit if loss exceeds stop loss
                    if profit_pct <= -stop_loss:
                        self.log_trade(f"Stop Loss hit on {trade['symbol']} at {profit_pct:.2f}%")
                        self.close_trade(trade_id, trade, current_price, "stop loss")
                        continue

                    # CASE 2: Trailing Stop
                    if trade['highest_profit'] >= trailing_activation:
                        drop_from_high = trade['highest_profit'] - profit_pct
                        
                        if drop_from_high >= trailing_stop:
                            self.log_trade(f"""
                            Trailing Stop hit on {trade['symbol']}:
                            Peak profit: {trade['highest_profit']:.2f}%
                            Current profit: {profit_pct:.2f}%
                            Drop from peak: {drop_from_high:.2f}%
                            """)
                            self.close_trade(trade_id, trade, current_price, "trailing stop")
                            continue

                except Exception as e:
                    self.log_trade(f"Error managing trade {trade_id}: {str(e)}")
                    continue

        except Exception as e:
            self.log_trade(f"Error in trade management: {str(e)}")

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

    def update_trade_history(self, symbol, percentage, profit, is_win=True):
        """Update the trade history display"""
        try:
            # Format timestamp
            timestamp = datetime.now().strftime("%H:%M:%S")
            
            # Format the trade result
            result = f"[{timestamp}] {symbol}: {percentage:.2f}%, ${profit:.2f}\n"
            
            # Update GUI trade history if available
            if hasattr(self, 'history_text'):
                try:
                    # Get current number of lines
                    current_lines = int(self.history_text.index('end-1c').split('.')[0])
                    
                    # If more than 100 lines, remove oldest entries
                    if current_lines > 100:
                        self.history_text.delete('1.0', f'{current_lines-100}.0')
                    
                    # Add new trade result
                    self.history_text.insert(tk.END, result)
                    
                    # Color code based on profit/loss
                    line_start = f"{float(self.history_text.index('end-2c'))-1:.1f}"
                    
                    if profit > 0:
                        self.history_text.tag_add("profit", f"{line_start} linestart", f"{line_start} lineend")
                        self.history_text.tag_config("profit", foreground="green")
                    else:
                        self.history_text.tag_add("loss", f"{line_start} linestart", f"{line_start} lineend")
                        self.history_text.tag_config("loss", foreground="red")
                    
                    # Scroll to show latest entry
                    self.history_text.see(tk.END)
                    
                    # Force GUI update
                    self.history_text.update_idletasks()
                    
                except Exception as e:
                    self.log_trade(f"Error updating trade history display: {str(e)}")
            
            # Log the trade
            self.log_trade(f"Trade History Updated - {result.strip()}")
            
        except Exception as e:
            self.log_trade(f"Error in update_trade_history: {str(e)}")

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

    def update_trade_history(self, symbol, percentage, profit, is_win=True):
        """
        Update the trade history display with new trade results
        
        Args:
            symbol (str): Trading pair symbol
            percentage (float): Profit/loss percentage
            profit (float): Actual profit/loss amount in USD
            is_win (bool): Whether trade was profitable
        """
        try:
            # Sanity check on percentage values
            if abs(percentage) > 50:  # Cap at ±50%
                self.log_trade(f"Warning: Unusual percentage detected for {symbol}: {percentage}%")
                percentage = max(min(percentage, 50), -50)
            
            # Format timestamp
            timestamp = datetime.now().strftime("%H:%M:%S")
            
            # Format the trade result
            result = f"[{timestamp}] {symbol}: {percentage:.2f}%, ${profit:.2f}\n"
            
            # Update GUI trade history if available
            if hasattr(self, 'history_text'):
                try:
                    # Get current number of lines
                    current_lines = int(self.history_text.index('end-1c').split('.')[0])
                    
                    # If more than 100 lines, remove oldest entries
                    if current_lines > 100:
                        self.history_text.delete('1.0', f'{current_lines-100}.0')
                    
                    # Add new trade result
                    self.history_text.insert(tk.END, result)
                    
                    # Color code based on actual profit
                    line_start = f"{float(self.history_text.index('end-2c'))-1:.1f}"
                    
                    # Determine color based on actual profit
                    if profit > 0:
                        color = "profit"
                        self.history_text.tag_config("profit", foreground="green")
                    else:
                        color = "loss"
                        self.history_text.tag_config("loss", foreground="red")
                    
                    # Apply color tag
                    self.history_text.tag_add(color, f"{line_start} linestart", f"{line_start} lineend")
                    
                    # Scroll to show latest entry
                    self.history_text.see(tk.END)
                    
                    # Force GUI update
                    self.history_text.update_idletasks()
                    
                except tk.TclError as e:
                    self.log_trade(f"GUI update error in trade history: {str(e)}")
                except Exception as e:
                    self.log_trade(f"Error updating trade history display: {str(e)}")
            
            # Store trade in history list (with cleanup)
            trade_record = {
                'timestamp': datetime.now(),
                'symbol': symbol,
                'percentage': percentage,
                'profit': profit,
                'is_win': is_win
            }
            
            # Add to trades list
            self.trades.append(trade_record)
            
            # Keep only last 100 trades
            if len(self.trades) > 100:
                self.trades = self.trades[-100:]
            
            # Log the trade
            self.log_trade(f"Trade History Updated - {symbol}: {percentage:.2f}% (${profit:.2f})")
            
        except Exception as e:
            self.log_trade(f"Critical error in update_trade_history: {str(e)}")
            # Attempt to log basic trade info even if display fails
            self.log_trade(f"Trade completed - {symbol}: {percentage:.2f}% (${profit:.2f})")

    def log_trade(self, message):
        """Log trade information with proper encoding handling"""
        try:
            # Replace Unicode checkmark with ASCII alternative
            message = message.replace('✓', '+')
            
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

    def validate_ticker(self, ticker, symbol):
        """Validate ticker data safely"""
        try:
            # Log the validation process
            self.log_trade(f"Validating {symbol}")
            
            # Check if required fields exist
            required_fields = ['last', 'quoteVolume', 'bid', 'ask']
            if not all(field in ticker for field in required_fields):
                self.log_trade(f"Missing required fields for {symbol}")
                return False

            # Convert values safely
            try:
                price = float(ticker['last'])
                volume = float(ticker['quoteVolume'])
                bid = float(ticker['bid'])
                ask = float(ticker['ask'])
            except (TypeError, ValueError):
                self.log_trade(f"Invalid numeric values for {symbol}")
                return False

            # Check minimum volume
            min_volume = float(self.min_volume_entry.get())
            if volume < min_volume:
                return False

            # Check price range (0.00001 to 5.0)
            if not (0.00001 <= price <= 5.0):
                return False

            # Check for valid bid/ask
            if bid <= 0 or ask <= 0:
                return False

            # Calculate spread
            spread = (ask - bid) / bid * 100
            if spread > 2.0:  # 2% max spread
                return False

            self.log_trade(f"Valid pair found: {symbol} at ${price:.6f}")
            return True

        except Exception as e:
            self.log_trade(f"Error validating {symbol}: {str(e)}")
            return False

    def monitor_prices_continuously(self):
        """Monitor active trades with real-time updates"""
        while self.running:
            try:
                if self.active_trades:
                    for trade_id, trade in list(self.active_trades.items()):
                        try:
                            # Get fresh price data (force fetch, don't use cache)
                            ticker = self.exchange.fetch_ticker(trade['symbol'])
                            current_price = float(ticker['last'])
                            
                            # Update trade data
                            entry_price = trade['entry_price']
                            profit_percentage = ((current_price - entry_price) / entry_price) * 100
                            
                            # Update trade info
                            trade['current_price'] = current_price
                            trade['current_profit_percentage'] = profit_percentage
                            trade['last_update'] = datetime.now()
                            
                            # Update highest price if applicable
                            if current_price > trade.get('highest_price', entry_price):
                                trade['highest_price'] = current_price
                                trade['highest_profit_percentage'] = profit_percentage
                            
                            # Update price history
                            if trade['symbol'] not in self.price_history:
                                self.price_history[trade['symbol']] = []
                            
                            self.price_history[trade['symbol']].append(
                                (datetime.now(), current_price)
                            )
                            
                            # Keep only recent history
                            self.price_history[trade['symbol']] = self.price_history[trade['symbol']][-50:]
                            
                            # Log updates
                            self.log_trade(f"""
                            Trade Update - {trade['symbol']}:
                            Current Price: {current_price:.8f}
                            Entry Price: {entry_price:.8f}
                            P/L: {profit_percentage:.2f}%
                            Highest: {trade.get('highest_profit_percentage', 0):.2f}%
                            """)
                            
                            # Update displays
                            self.safe_execute(self.update_active_trades_display)
                            self.safe_execute(self.update_chart)
                            
                        except Exception as e:
                            self.log_trade(f"Error monitoring {trade['symbol']}: {str(e)}")
                            continue
                            
                time.sleep(1)  # Update every second
                    
            except Exception as e:
                self.log_trade(f"Price monitoring error: {str(e)}")
                time.sleep(1)

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

    def run_bot(self):
        """Main trading loop with straightforward logic"""
        try:
            # Initial startup checks
            self.log_trade("\n=== BOT STARTED ===")
            self.update_status(f"Running ({'Paper' if self.is_paper_trading else 'Real'})")

            while self.running:
                try:
                    # 1. SCAN MARKET
                    self.update_status("Fetching market data...")
                    tickers = self.fetch_tickers_with_retry()  # Get fresh data
                    
                    # 2. MANAGE ACTIVE TRADES FIRST (Exit conditions)
                    if self.active_trades:
                        for trade_id, trade in list(self.active_trades.items()):
                            try:
                                current_price = float(self.exchange.fetch_ticker(trade['symbol'])['last'])
                                entry_price = float(trade['entry_price'])
                                profit_pct = ((current_price - entry_price) / entry_price) * 100
                                
                                # Update highest profit if new high
                                if profit_pct > trade.get('highest_profit', -999):
                                    trade['highest_profit'] = profit_pct
                                
                                # Get thresholds
                                stop_loss = float(self.stop_loss.get())
                                trailing_stop = float(self.trailing_stop.get())
                                trailing_activation = float(self.trailing_activation.get())
                                highest_profit = trade.get('highest_profit', 0)

                                # Exit conditions in order of priority
                                if profit_pct <= -stop_loss:  # Stop loss hit
                                    self.close_trade(trade_id, trade, current_price, "stop loss")
                                    continue
                                    
                                if highest_profit >= trailing_activation:  # Check trailing stop
                                    drop = highest_profit - profit_pct
                                    if drop >= trailing_stop:
                                        self.close_trade(trade_id, trade, current_price, "trailing stop")
                                        continue

                            except Exception as e:
                                self.log_trade(f"Error managing trade {trade_id}: {str(e)}")

                    # 3. LOOK FOR NEW TRADES (Entry conditions)
                    self.update_status("Scanning for opportunities...")
                    scan_success = self.scan_opportunities()
                    
                    # 4. UPDATE DISPLAYS
                    self.update_active_trades_display()
                    self.update_chart()
                    self.update_metrics()
                    
                    # 5. WAIT BEFORE NEXT SCAN
                    time.sleep(2)  # Pause between cycles

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