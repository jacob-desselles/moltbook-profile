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
        def __init__(self):
            self.price_data = {}

        def update_price_data(self, symbol, ticker):
            """Update price data for a symbol and calculate indicators"""
            try:
                price = float(ticker['last'])
                volume = float(ticker['quoteVolume'])
                timestamp = pd.Timestamp.now()

                # Initialize DataFrame if not exists
                if symbol not in self.price_data:
                    self.price_data[symbol] = pd.DataFrame(columns=['price', 'volume', 'bid', 'ask'])

                # Append new data
                new_data = pd.DataFrame({
                    'price': [price],
                    'volume': [volume],
                    'bid': [float(ticker['bid'])],
                    'ask': [float(ticker['ask'])]
                }, index=[timestamp])
                self.price_data[symbol] = pd.concat([self.price_data[symbol], new_data])

                # Calculate indicators
                self.calculate_indicators(symbol)

            except Exception as e:
                print(f"Error updating price data for {symbol}: {str(e)}")

        def calculate_indicators(self, symbol):
            """Calculate technical indicators for the symbol"""
            try:
                df = self.price_data[symbol]
                if len(df) < 2:
                    return

                # Calculate EMAs
                df['ema_5'] = df['price'].ewm(span=5, adjust=False).mean()
                df['ema_15'] = df['price'].ewm(span=15, adjust=False).mean()

                # Calculate RSI
                rsi_period = 14  # Match GUI default
                delta = df['price'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
                rs = gain / loss
                df['rsi_14'] = 100 - (100 / (1 + rs))

                # Update DataFrame
                self.price_data[symbol] = df

            except Exception as e:
                print(f"Error calculating indicators for {symbol}: {str(e)}")

class CryptoScalpingBot:

    def __init__(self):
        import configparser
        self.config = configparser.ConfigParser()
        self.config['API_KEYS'] = {
            'api_key': '',
            'secret': ''
        }
        self.root = tk.Tk()
        self.root.title("Crypto Scalping Bot")
        self.root.geometry("1400x900")
        self.root.configure(bg="black")
        self.running = False
        self.is_paper_trading = True
        self.paper_balance = 10000.0
        self.min_balance_threshold = 1000.0
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_profit = 0.0
        self.total_fees = 0.0
        self.net_profit = 0.0
        self.wins = 0
        self.losses = 0
        self.active_trades = {}
        self.price_history = {}
        self.price_cache = {}
        self.trades = []
        self.trade_history = []
        self.mock_ticker_state = {
            'BTC/USD': {
                'last': 70000.0,
                'bid': 69950.0,
                'ask': 70050.0,
                'quoteVolume': 1000.0,
                'price_change_24h': 2.5,
                'liquidity': 500000.0,
                'market_cap': 1300000000000.0,
                'volatility_24h': 1.8,
                'trade_count_24h': 10000
            },
            'ETH/USD': {
                'last': 2500.0,
                'bid': 2495.0,
                'ask': 2505.0,
                'quoteVolume': 800.0,
                'price_change_24h': 3.0,
                'liquidity': 300000.0,
                'market_cap': 300000000000.0,
                'volatility_24h': 2.0,
                'trade_count_24h': 8000
            },
            'XRP/USD': {
                'last': 1.2,
                'bid': 1.195,
                'ask': 1.205,
                'quoteVolume': 500.0,
                'price_change_24h': 1.5,
                'liquidity': 100000.0,
                'market_cap': 50000000000.0,
                'volatility_24h': 1.5,
                'trade_count_24h': 5000
            }
        }
        self.log_trade(f"Initialized mock_ticker_state with {len(self.mock_ticker_state)} pairs: {list(self.mock_ticker_state.keys())}")
        self.taker_fee = 0.004
        self.cache_timeout = 2.0
        self.min_volume = 100.0
        self.exchange = ccxt.kraken({
            'enableRateLimit': True,
            'apiKey': self.config.get('API_KEYS', 'api_key', fallback=''),
            'secret': self.config.get('API_KEYS', 'secret', fallback='')
        })
        self.data_manager = DataManager()
        self.setup_gui()
        if not hasattr(self, 'active_trades_label'):
            self.log_trade("Warning: active_trades_label not initialized in setup_gui")
            self.active_trades_label = tk.Label(self.root, text="Number of Active Trades: 0", fg="white", bg="black")
            self.active_trades_label.pack()
        if not hasattr(self, 'trades_text'):
            self.log_trade("Warning: trades_text not initialized in setup_gui")
            self.trades_text = tk.Text(self.root, height=10, width=50, state="disabled", fg="white", bg="black")
            self.trades_text.pack()
        self.log_trade(f"GUI components initialized: active_trades_label={hasattr(self, 'active_trades_label')}, trades_text={hasattr(self, 'trades_text')}, active_trades_frame={hasattr(self, 'active_trades_frame')}")
        self.bot_thread = None
        self.price_monitor_thread = None
        self.start_time = None
        self.timer_running = False
        self.fig, self.ax = plt.subplots(figsize=(8, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        import logging
        logging.basicConfig(
            filename='trading_bot.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        self.load_parameters()
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def stop_bot(self):
        """Stop the bot"""
        if not self.running:
            self.log_trade("Bot is not running")
            return

        self.running = False
        self.status_label.config(text="Stopped")
        self.log_trade("Bot stopped")
        # Do NOT reset self.mock_ticker_state here to preserve mock data


    def init_memory_monitor(self):
        """Initialize memory monitoring"""
        self.memory_check_interval = 60  # seconds
        self.last_memory_check = time.time()
        self.memory_threshold = 1000  # MB


    def run_bot_loop(self):
        try:
            self.log_trade("Starting bot loop...")
            while self.running:
                try:
                    tickers = self.fetch_tickers_with_retry()
                    if not tickers:
                        self.log_trade("No tickers fetched, skipping cycle")
                        time.sleep(5)
                        continue
                    self.log_trade(f"Scanning opportunities with {len(tickers)} tickers...")
                    opportunities = self.scan_opportunities(tickers)
                    if opportunities:
                        self.log_trade(f"Found {len(opportunities)} potential opportunities: {[opp['symbol'] for opp in opportunities]}")
                        self.analyze_pairs(opportunities)
                    self.log_trade("Monitoring active trades...")
                    self.monitor_trades()
                    self.log_trade("Finished scanning cycle. Sleeping for 5 seconds...")
                    time.sleep(5)
                except Exception as e:
                    self.log_trade(f"Error in run_bot_loop cycle: {str(e)}")
                    time.sleep(5)
            self.log_trade("Bot loop stopped")
        except Exception as e:
            self.log_trade(f"Fatal error in run_bot_loop: {str(e)}")
        finally:
            self.running = False
            self.status_label.config(text="Stopped")
            self.log_trade("=== BOT STOPPED ===")

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

    def fetch_tickers_with_retry(self):
        """Return mock ticker data with simulated price movements and additional metrics"""
        try:
            # Check if mock_ticker_state is None or not a dictionary
            if not isinstance(self.mock_ticker_state, dict):
                self.log_trade("Initializing mock ticker state...")
                self.mock_ticker_state = {
                    'BTC/USD': {'last': 70000.0, 'quoteVolume': 1000000, 'bid': 69965.0, 'ask': 70035.0, 'price_change_24h': 1.2, 'liquidity': 5000000, 'market_cap': 1400000000000, 'volatility_24h': 2.0, 'trade_count_24h': 5000},
                    'ETH/USD': {'last': 2500.0, 'quoteVolume': 800000, 'bid': 2498.75, 'ask': 2501.25, 'price_change_24h': 1.5, 'liquidity': 4000000, 'market_cap': 300000000000, 'volatility_24h': 2.5, 'trade_count_24h': 4000},
                    'BNB/USD': {'last': 600.0, 'quoteVolume': 500000, 'bid': 599.40, 'ask': 600.60, 'price_change_24h': 1.0, 'liquidity': 2000000, 'market_cap': 90000000000, 'volatility_24h': 2.0, 'trade_count_24h': 3000},
                    'SOL/USD': {'last': 150.0, 'quoteVolume': 300000, 'bid': 149.85, 'ask': 150.15, 'price_change_24h': 2.0, 'liquidity': 1000000, 'market_cap': 70000000000, 'volatility_24h': 4.0, 'trade_count_24h': 2000},
                    'ADA/USD': {'last': 0.35, 'quoteVolume': 200000, 'bid': 0.3498, 'ask': 0.3502, 'price_change_24h': 1.8, 'liquidity': 800000, 'market_cap': 12000000000, 'volatility_24h': 3.5, 'trade_count_24h': 1500},
                    'DOT/USD': {'last': 4.5, 'quoteVolume': 150000, 'bid': 4.4955, 'ask': 4.5045, 'price_change_24h': 2.2, 'liquidity': 600000, 'market_cap': 6000000000, 'volatility_24h': 4.5, 'trade_count_24h': 1200},
                    'OM/USD': {'last': 0.015, 'quoteVolume': 500, 'bid': 0.01499, 'ask': 0.01501, 'price_change_24h': 2.0, 'liquidity': 1500, 'market_cap': 2000000, 'volatility_24h': 3.0, 'trade_count_24h': 75},
                    'EFT/USD': {'last': 0.022, 'quoteVolume': 300, 'bid': 0.02199, 'ask': 0.02201, 'price_change_24h': 1.5, 'liquidity': 1200, 'market_cap': 1500000, 'volatility_24h': 4.0, 'trade_count_24h': 60},
                    'MEW/USD': {'last': 0.008, 'quoteVolume': 400, 'bid': 0.00799, 'ask': 0.00801, 'price_change_24h': 0.8, 'liquidity': 800, 'market_cap': 500000, 'volatility_24h': 6.0, 'trade_count_24h': 40},
                    'ZRX/USD': {'last': 0.30, 'quoteVolume': 1000, 'bid': 0.2997, 'ask': 0.3003, 'price_change_24h': 3.0, 'liquidity': 2000, 'market_cap': 2500000, 'volatility_24h': 7.0, 'trade_count_24h': 80},
                    'ETH/BTC': {'last': 0.0357, 'quoteVolume': 50000, 'bid': 0.03567, 'ask': 0.03573, 'price_change_24h': 0.5, 'liquidity': 300000, 'market_cap': 300000000000, 'volatility_24h': 1.5, 'trade_count_24h': 1000},
                    'SOL/BTC': {'last': 0.00214, 'quoteVolume': 30000, 'bid': 0.00213, 'ask': 0.00215, 'price_change_24h': 1.0, 'liquidity': 150000, 'market_cap': 70000000000, 'volatility_24h': 2.0, 'trade_count_24h': 800},
                    'ADA/BTC': {'last': 0.000005, 'quoteVolume': 20000, 'bid': 0.00000499, 'ask': 0.00000501, 'price_change_24h': 0.8, 'liquidity': 100000, 'market_cap': 12000000000, 'volatility_24h': 2.5, 'trade_count_24h': 600},
                    'DOGE/USD': {'last': 0.14, 'quoteVolume': 250000, 'bid': 0.13986, 'ask': 0.14014, 'price_change_24h': 5.0, 'liquidity': 500000, 'market_cap': 20000000000, 'volatility_24h': 10.0, 'trade_count_24h': 2500},
                    'SHIB/USD': {'last': 0.000018, 'quoteVolume': 150000, 'bid': 0.00001799, 'ask': 0.00001801, 'price_change_24h': 6.0, 'liquidity': 300000, 'market_cap': 10000000000, 'volatility_24h': 12.0, 'trade_count_24h': 2000},
                    'PEPE/USD': {'last': 0.000009, 'quoteVolume': 100000, 'bid': 0.00000899, 'ask': 0.00000901, 'price_change_24h': 7.0, 'liquidity': 200000, 'market_cap': 4000000000, 'volatility_24h': 15.0, 'trade_count_24h': 1500},
                    'XYZ/USD': {'last': 0.05, 'quoteVolume': 100, 'bid': 0.04995, 'ask': 0.05005, 'price_change_24h': 1.0, 'liquidity': 500, 'market_cap': 1000000, 'volatility_24h': 5.0, 'trade_count_24h': 30},
                    'ABC/USD': {'last': 0.02, 'quoteVolume': 80, 'bid': 0.01999, 'ask': 0.02001, 'price_change_24h': 0.5, 'liquidity': 400, 'market_cap': 800000, 'volatility_24h': 6.0, 'trade_count_24h': 25},
                    'DEF/USD': {'last': 0.01, 'quoteVolume': 50, 'bid': 0.00999, 'ask': 0.01001, 'price_change_24h': 0.3, 'liquidity': 300, 'market_cap': 500000, 'volatility_24h': 7.0, 'trade_count_24h': 20},
                    'LUNA/USD': {'last': 0.40, 'quoteVolume': 200000, 'bid': 0.3996, 'ask': 0.4004, 'price_change_24h': 10.0, 'liquidity': 800000, 'market_cap': 1000000000, 'volatility_24h': 8.0, 'trade_count_24h': 1200},
                    'AVAX/USD': {'last': 30.0, 'quoteVolume': 180000, 'bid': 29.985, 'ask': 30.015, 'price_change_24h': 8.0, 'liquidity': 700000, 'market_cap': 12000000000, 'volatility_24h': 7.0, 'trade_count_24h': 1100},
                    'NEAR/USD': {'last': 5.0, 'quoteVolume': 160000, 'bid': 4.995, 'ask': 5.005, 'price_change_24h': 9.0, 'liquidity': 600000, 'market_cap': 5000000000, 'volatility_24h': 6.5, 'trade_count_24h': 1000},
                    'USDC/USD': {'last': 1.0, 'quoteVolume': 5000000, 'bid': 0.9995, 'ask': 1.0005, 'price_change_24h': 0.1, 'liquidity': 10000000, 'market_cap': 35000000000, 'volatility_24h': 0.2, 'trade_count_24h': 10000},
                    'USDT/USD': {'last': 1.0, 'quoteVolume': 6000000, 'bid': 0.9995, 'ask': 1.0005, 'price_change_24h': 0.05, 'liquidity': 12000000, 'market_cap': 90000000000, 'volatility_24h': 0.1, 'trade_count_24h': 12000},
                    'DAI/USD': {'last': 1.0, 'quoteVolume': 4000000, 'bid': 0.9995, 'ask': 1.0005, 'price_change_24h': 0.08, 'liquidity': 8000000, 'market_cap': 5000000000, 'volatility_24h': 0.15, 'trade_count_24h': 8000},
                    'LINK/USD': {'last': 12.0, 'quoteVolume': 120000, 'bid': 11.988, 'ask': 12.012, 'price_change_24h': 2.5, 'liquidity': 500000, 'market_cap': 7000000000, 'volatility_24h': 4.0, 'trade_count_24h': 900},
                    'MATIC/USD': {'last': 0.50, 'quoteVolume': 110000, 'bid': 0.4995, 'ask': 0.5005, 'price_change_24h': 2.0, 'liquidity': 450000, 'market_cap': 5000000000, 'volatility_24h': 3.8, 'trade_count_24h': 850},
                    'XLM/USD': {'last': 0.10, 'quoteVolume': 100000, 'bid': 0.0999, 'ask': 0.1001, 'price_change_24h': 1.8, 'liquidity': 400000, 'market_cap': 3000000000, 'volatility_24h': 3.5, 'trade_count_24h': 800},
                    'ALGO/USD': {'last': 0.15, 'quoteVolume': 90000, 'bid': 0.14985, 'ask': 0.15015, 'price_change_24h': 2.2, 'liquidity': 350000, 'market_cap': 2000000000, 'volatility_24h': 4.2, 'trade_count_24h': 750},
                    'VET/USD': {'last': 0.025, 'quoteVolume': 80000, 'bid': 0.02498, 'ask': 0.02502, 'price_change_24h': 2.0, 'liquidity': 300000, 'market_cap': 1800000000, 'volatility_24h': 4.5, 'trade_count_24h': 700},
                    'HBAR/USD': {'last': 0.06, 'quoteVolume': 70000, 'bid': 0.05994, 'ask': 0.06006, 'price_change_24h': 1.9, 'liquidity': 250000, 'market_cap': 1500000000, 'volatility_24h': 4.0, 'trade_count_24h': 650}
                }
                self.log_trade(f"Mock ticker state initialized with {len(self.mock_ticker_state)} pairs")

            # Verify that mock_ticker_state is still valid before proceeding
            if not self.mock_ticker_state or not isinstance(self.mock_ticker_state, dict):
                raise ValueError("Mock ticker state is invalid or empty after initialization")

            # Update ticker data with simulated movements
            self.log_trade("Updating mock ticker data...")
            for symbol in self.mock_ticker_state:
                current_price = self.mock_ticker_state[symbol]['last']
                # Simulate price movement with a mix of upward and downward changes
                direction = np.random.choice([-1, 1], p=[0.4, 0.6])  # 60% chance upward, 40% downward
                change_magnitude = np.random.uniform(0.0005, 0.0015)  # 0.05% to 0.15% change
                price_change = current_price * (change_magnitude * direction)
                new_price = round(max(current_price + price_change, 0.00001), 5)
                self.mock_ticker_state[symbol]['last'] = new_price

                # Update bid and ask prices based on new price
                spread_factor = 0.0005  # Spread of 0.1%
                self.mock_ticker_state[symbol]['bid'] = round(new_price * (1 - spread_factor), 5)
                self.mock_ticker_state[symbol]['ask'] = round(new_price * (1 + spread_factor), 5)

                # Update volume with a cap at 10x initial volume
                current_volume = self.mock_ticker_state[symbol]['quoteVolume']
                new_volume = current_volume * np.random.uniform(1.01, 1.03)  # 1% to 3% increase
                initial_volume = self.mock_ticker_state[symbol].get('initial_volume', current_volume)
                self.mock_ticker_state[symbol]['initial_volume'] = initial_volume
                max_volume = initial_volume * 10
                self.mock_ticker_state[symbol]['quoteVolume'] = round(min(new_volume, max_volume), 2)

                # Update other metrics with small random fluctuations
                self.mock_ticker_state[symbol]['price_change_24h'] = round(
                    self.mock_ticker_state[symbol]['price_change_24h'] * np.random.uniform(0.9, 1.1), 2
                )
                self.mock_ticker_state[symbol]['liquidity'] = round(
                    self.mock_ticker_state[symbol]['liquidity'] * np.random.uniform(0.9, 1.1), 2
                )
                self.mock_ticker_state[symbol]['market_cap'] = round(
                    self.mock_ticker_state[symbol]['market_cap'] * np.random.uniform(0.9, 1.1), 2
                )
                self.mock_ticker_state[symbol]['volatility_24h'] = round(
                    self.mock_ticker_state[symbol]['volatility_24h'] * np.random.uniform(0.9, 1.1), 2
                )
                self.mock_ticker_state[symbol]['trade_count_24h'] = round(
                    self.mock_ticker_state[symbol]['trade_count_24h'] * np.random.uniform(0.9, 1.1), 2
                )

            self.log_trade(f"Returning updated tickers with {len(self.mock_ticker_state)} pairs")
            return self.mock_ticker_state

        except Exception as e:
            self.log_trade(f"Error fetching mock tickers: {str(e)}")
            import traceback
            self.log_trade(f"Traceback: {traceback.format_exc()}")
            # Return a default empty dictionary to prevent further errors
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
        try:
            if not self.running:
                self.log_trade("Starting bot via toggle...")
                self.run_bot()
            else:
                self.log_trade("Stopping bot via toggle...")
                self.running = False
                self.status_label.config(text="Stopped")
                self.log_trade("=== BOT STOPPED ===")
        except Exception as e:
            self.log_trade(f"Error in toggle_bot: {str(e)}")
            self.running = False
            self.status_label.config(text="Stopped")

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
        """Setup GUI for paper trading with a tabbed layout similar to the live bot"""
        # Configure the main window
        self.root.configure(bg="#2E2E2E")
        self.root.geometry("1200x800")

        # Create a notebook (tabbed interface)
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True, padx=5, pady=5)

        # Tab 1: Main (Controls and Displays)
        main_tab = ttk.Frame(self.notebook)
        self.notebook.add(main_tab, text="Main")
        
        main_tab.grid_rowconfigure(0, weight=0)
        main_tab.grid_rowconfigure(1, weight=1)
        main_tab.grid_columnconfigure(0, weight=1)

        # Controls Frame (Start/Stop buttons and status)
        control_frame = ttk.LabelFrame(main_tab, text="Controls", padding=5)
        control_frame.grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        
        control_frame.grid_columnconfigure(0, weight=1)
        control_frame.grid_columnconfigure(1, weight=0)
        control_frame.grid_columnconfigure(2, weight=0)

        self.status_label = ttk.Label(control_frame, text="Stopped")
        self.status_label.grid(row=0, column=0, padx=5, pady=2, sticky="w")

        ttk.Button(control_frame, text="Start", command=self.run_bot).grid(row=0, column=1, padx=5, pady=2)
        ttk.Button(control_frame, text="Stop", command=self.stop_bot).grid(row=0, column=2, padx=5, pady=2)

        # Main Display Frame
        display_frame = ttk.LabelFrame(main_tab, text="Trading Dashboard", padding=5)
        display_frame.grid(row=1, column=0, padx=5, pady=5, sticky="nsew")
        
        display_frame.grid_rowconfigure(0, weight=1)
        display_frame.grid_rowconfigure(1, weight=2)
        display_frame.grid_rowconfigure(2, weight=1)
        display_frame.grid_columnconfigure(0, weight=3)
        display_frame.grid_columnconfigure(1, weight=1)

        # Active Trades
        trades_frame = ttk.LabelFrame(display_frame, text="Active Trades", padding=5)
        trades_frame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
        
        trades_frame.grid_rowconfigure(0, weight=0)  # For the label
        trades_frame.grid_rowconfigure(1, weight=1)  # For the text widget
        trades_frame.grid_columnconfigure(0, weight=1)
        trades_frame.grid_columnconfigure(1, weight=0)

        # Add a label to display the number of active trades
        self.active_trades_label = ttk.Label(trades_frame, text="Number of Active Trades: 0")
        self.active_trades_label.grid(row=0, column=0, padx=5, pady=2, sticky="w")

        self.trades_text = tk.Text(trades_frame, height=6, width=80, bg="#1E1E1E", fg="white")
        self.trades_text.grid(row=1, column=0, padx=5, pady=5, sticky="nsew")
        scrollbar_trades = ttk.Scrollbar(trades_frame, orient="vertical", command=self.trades_text.yview)
        scrollbar_trades.grid(row=1, column=1, sticky="ns")
        self.trades_text.config(yscrollcommand=scrollbar_trades.set)

        # Metrics
        metrics_frame = ttk.LabelFrame(display_frame, text="Metrics", padding=5)
        metrics_frame.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")
        
        metrics_frame.grid_rowconfigure(0, weight=1)
        metrics_frame.grid_rowconfigure(1, weight=1)
        metrics_frame.grid_rowconfigure(2, weight=1)
        metrics_frame.grid_rowconfigure(3, weight=1)
        metrics_frame.grid_columnconfigure(0, weight=1)

        self.total_profit_label = ttk.Label(metrics_frame, text="Total Profit: 0.00 USD")
        self.total_profit_label.grid(row=0, column=0, padx=5, pady=2, sticky="w")

        self.total_fees_label = ttk.Label(metrics_frame, text="Total Fees: 0.00 USD")
        self.total_fees_label.grid(row=1, column=0, padx=5, pady=2, sticky="w")

        self.net_profit_label = ttk.Label(metrics_frame, text="Net Profit: 0.00 USD")
        self.net_profit_label.grid(row=2, column=0, padx=5, pady=2, sticky="w")

        self.win_loss_label = ttk.Label(metrics_frame, text="Win/Loss: 0/0 (0%)")
        self.win_loss_label.grid(row=3, column=0, padx=5, pady=2, sticky="w")

        self.paper_balance_label = ttk.Label(metrics_frame, text=f"Paper Balance: {self.paper_balance:.2f} USD")
        self.paper_balance_label.grid(row=4, column=0, sticky="w", padx=5, pady=2)

        # Price Chart
        chart_frame = ttk.LabelFrame(display_frame, text="Price Chart", padding=5)
        chart_frame.grid(row=1, column=0, columnspan=2, padx=5, pady=5, sticky="nsew")
        
        chart_frame.grid_rowconfigure(0, weight=1)
        chart_frame.grid_columnconfigure(0, weight=1)

        self.fig, self.ax = plt.subplots(figsize=(10, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=chart_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, padx=5, pady=5, sticky="nsew")

        # Trade History and Log (combined in a sub-notebook)
        history_log_notebook = ttk.Notebook(display_frame)
        history_log_notebook.grid(row=2, column=0, columnspan=2, padx=5, pady=5, sticky="nsew")

        # Trade History Tab
        history_tab = ttk.Frame(history_log_notebook)
        history_log_notebook.add(history_tab, text="Trade History")
        
        history_tab.grid_rowconfigure(0, weight=1)
        history_tab.grid_columnconfigure(0, weight=1)
        history_tab.grid_columnconfigure(1, weight=0)

        self.history_text = tk.Text(history_tab, height=4, width=80, bg="#1E1E1E", fg="white")
        self.history_text.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
        scrollbar_history = ttk.Scrollbar(history_tab, orient="vertical", command=self.history_text.yview)
        scrollbar_history.grid(row=0, column=1, sticky="ns")
        self.history_text.config(yscrollcommand=scrollbar_history.set)

        # Log Tab
        log_tab = ttk.Frame(history_log_notebook)
        history_log_notebook.add(log_tab, text="Log")
        
        log_tab.grid_rowconfigure(0, weight=1)
        log_tab.grid_columnconfigure(0, weight=1)
        log_tab.grid_columnconfigure(1, weight=0)

        self.log_text = tk.Text(log_tab, height=4, width=80, bg="#1E1E1E", fg="white")
        self.log_text.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
        scrollbar_log = ttk.Scrollbar(log_tab, orient="vertical", command=self.log_text.yview)
        scrollbar_log.grid(row=0, column=1, sticky="ns")
        self.log_text.config(yscrollcommand=scrollbar_log.set)

        # Tab 2: Parameters
        params_tab = ttk.Frame(self.notebook)
        self.notebook.add(params_tab, text="Parameters")
        
        params_tab.grid_rowconfigure(0, weight=1)
        params_tab.grid_rowconfigure(1, weight=1)
        params_tab.grid_columnconfigure(0, weight=1)
        params_tab.grid_columnconfigure(1, weight=1)

        # Basic Parameters Frame
        params_frame = ttk.LabelFrame(params_tab, text="Basic Parameters", padding=5)
        params_frame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")

        ttk.Label(params_frame, text="Profit Target (%):").grid(row=0, column=0, padx=5, pady=2, sticky="w")
        self.profit_target = ttk.Entry(params_frame, width=10)
        self.profit_target.insert(0, "1.5")
        self.profit_target.grid(row=0, column=1, padx=5, pady=2)

        ttk.Label(params_frame, text="Stop Loss (%):").grid(row=1, column=0, padx=5, pady=2, sticky="w")
        self.stop_loss = ttk.Entry(params_frame, width=10)
        self.stop_loss.insert(0, "0.5")
        self.stop_loss.grid(row=1, column=1, padx=5, pady=2)

        ttk.Label(params_frame, text="Trailing Stop (%):").grid(row=2, column=0, padx=5, pady=2, sticky="w")
        self.trailing_stop = ttk.Entry(params_frame, width=10)
        self.trailing_stop.insert(0, "0.3")
        self.trailing_stop.grid(row=2, column=1, padx=5, pady=2)

        ttk.Label(params_frame, text="Trailing Activation (%):").grid(row=3, column=0, padx=5, pady=2, sticky="w")
        self.trailing_activation = ttk.Entry(params_frame, width=10)
        self.trailing_activation.insert(0, "0.7")
        self.trailing_activation.grid(row=3, column=1, padx=5, pady=2)

        ttk.Label(params_frame, text="Position Size (USD):").grid(row=4, column=0, padx=5, pady=2, sticky="w")
        self.position_size = ttk.Entry(params_frame, width=10)
        self.position_size.insert(0, "50")
        self.position_size.grid(row=4, column=1, padx=5, pady=2)

        # Validation Criteria Frame
        validation_frame = ttk.LabelFrame(params_tab, text="Validation Criteria", padding=5)
        validation_frame.grid(row=1, column=0, padx=5, pady=5, sticky="nsew")
        
        ttk.Label(validation_frame, text="Trend Strength (Beta):").grid(row=0, column=0, padx=5, pady=2, sticky="w")
        self.momentum_beta = ttk.Entry(validation_frame, width=10)
        self.momentum_beta.insert(0, "0.3")
        self.momentum_beta.grid(row=0, column=1, padx=5, pady=2)

        ttk.Label(validation_frame, text="Price Momentum (Alpha):").grid(row=1, column=0, padx=5, pady=2, sticky="w")
        self.price_alpha = ttk.Entry(validation_frame, width=10)
        self.price_alpha.insert(0, "0.05")
        self.price_alpha.grid(row=1, column=1, padx=5, pady=2)

        ttk.Label(validation_frame, text="Momentum Quality (Theta):").grid(row=2, column=0, padx=5, pady=2, sticky="w")
        self.time_theta = ttk.Entry(validation_frame, width=10)
        self.time_theta.insert(0, "1.0")  # Relaxed from 0.3 to 1.0
        self.time_theta.grid(row=2, column=1, padx=5, pady=2)

        ttk.Label(validation_frame, text="Volatility Filter (Vega):").grid(row=3, column=0, padx=5, pady=2, sticky="w")
        self.vol_vega = ttk.Entry(validation_frame, width=10)
        self.vol_vega.insert(0, "1.0")  # Relaxed from 0.3 to 1.0
        self.vol_vega.grid(row=3, column=1, padx=5, pady=2)

        ttk.Label(validation_frame, text="Volume Quality (Rho):").grid(row=4, column=0, padx=5, pady=2, sticky="w")
        self.volume_rho = ttk.Entry(validation_frame, width=10)
        self.volume_rho.insert(0, "0.0")  # Relaxed from 0.2 to 0.0 (allow negative volume increase)
        self.volume_rho.grid(row=4, column=1, padx=5, pady=2)

        # Technical Indicators Frame
        indicators_frame = ttk.LabelFrame(params_tab, text="Technical Indicators", padding=5)
        indicators_frame.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")

        ttk.Label(indicators_frame, text="RSI Period:").grid(row=0, column=0, padx=5, pady=2, sticky="w")
        self.rsi_period = ttk.Entry(indicators_frame, width=10)
        self.rsi_period.insert(0, "14")
        self.rsi_period.grid(row=0, column=1, padx=5, pady=2)

        ttk.Label(indicators_frame, text="RSI Overbought:").grid(row=1, column=0, padx=5, pady=2, sticky="w")
        self.rsi_overbought = ttk.Entry(indicators_frame, width=10)
        self.rsi_overbought.insert(0, "75")
        self.rsi_overbought.grid(row=1, column=1, padx=5, pady=2)

        ttk.Label(indicators_frame, text="RSI Oversold:").grid(row=2, column=0, padx=5, pady=2, sticky="w")
        self.rsi_oversold = ttk.Entry(indicators_frame, width=10)
        self.rsi_oversold.insert(0, "30")
        self.rsi_oversold.grid(row=2, column=1, padx=5, pady=2)

        # Market Filters Frame (Updated with additional parameters)
        filters_frame = ttk.LabelFrame(params_tab, text="Market Filters", padding=5)
        filters_frame.grid(row=1, column=1, padx=5, pady=5, sticky="nsew")

        row = 0
        ttk.Label(filters_frame, text="Min Volume (USD):").grid(row=row, column=0, padx=5, pady=2, sticky="w")
        self.min_volume_entry = ttk.Entry(filters_frame, width=10)
        self.min_volume_entry.insert(0, "200")
        self.min_volume_entry.grid(row=row, column=1, padx=5, pady=2)
        row += 1

        ttk.Label(filters_frame, text="Max Trades:").grid(row=row, column=0, padx=5, pady=2, sticky="w")
        self.max_trades_entry = ttk.Entry(filters_frame, width=10)
        self.max_trades_entry.insert(0, "3")
        self.max_trades_entry.grid(row=row, column=1, padx=5, pady=2)
        row += 1

        ttk.Label(filters_frame, text="Top List Size:").grid(row=row, column=0, padx=5, pady=2, sticky="w")
        self.top_list_size = ttk.Entry(filters_frame, width=10)  # Fixed width typo (was width=1)
        self.top_list_size.insert(0, "10")
        self.top_list_size.grid(row=row, column=1, padx=5, pady=2)
        row += 1

        # New Market Filter Conditions
        ttk.Label(filters_frame, text="Min Price Change (%):").grid(row=row, column=0, padx=5, pady=2, sticky="w")
        self.min_price_change = ttk.Entry(filters_frame, width=10)
        self.min_price_change.insert(0, "1.0")  # Default: 1% price change in last 24h
        self.min_price_change.grid(row=row, column=1, padx=5, pady=2)
        row += 1

        ttk.Label(filters_frame, text="Min Liquidity (USD):").grid(row=row, column=0, padx=5, pady=2, sticky="w")
        self.min_liquidity = ttk.Entry(filters_frame, width=10)
        self.min_liquidity.insert(0, "1000")  # Default: $1000 liquidity
        self.min_liquidity.grid(row=row, column=1, padx=5, pady=2)
        row += 1

        ttk.Label(filters_frame, text="Min Market Cap (USD):").grid(row=row, column=0, padx=5, pady=2, sticky="w")
        self.min_market_cap = ttk.Entry(filters_frame, width=10)
        self.min_market_cap.insert(0, "1000000")  # Default: $1M market cap
        self.min_market_cap.grid(row=row, column=1, padx=5, pady=2)
        row += 1

        ttk.Label(filters_frame, text="Max Volatility (%):").grid(row=row, column=0, padx=5, pady=2, sticky="w")
        self.max_volatility = ttk.Entry(filters_frame, width=10)
        self.max_volatility.insert(0, "5.0")  # Default: 5% volatility
        self.max_volatility.grid(row=row, column=1, padx=5, pady=2)
        row += 1

        ttk.Label(filters_frame, text="Min Trade Count:").grid(row=row, column=0, padx=5, pady=2, sticky="w")
        self.min_trade_count = ttk.Entry(filters_frame, width=10)
        self.min_trade_count.insert(0, "50")  # Default: 50 trades in last 24h
        self.min_trade_count.grid(row=row, column=1, padx=5, pady=2)
        row += 1

        ttk.Label(filters_frame, text="Max Spread (%):").grid(row=row, column=0, padx=5, pady=2, sticky="w")
        self.max_spread = ttk.Entry(filters_frame, width=10)
        self.max_spread.insert(0, "0.1")
        self.max_spread.grid(row=row, column=1, padx=5, pady=2)
        row += 1

        ttk.Label(filters_frame, text="Required Conditions:").grid(row=row, column=0, padx=5, pady=2, sticky="w")
        self.required_conditions = ttk.Entry(filters_frame, width=10)
        self.required_conditions.insert(0, "3")  # Default: Must meet 3 out of the 6 conditions
        self.required_conditions.grid(row=row, column=1, padx=5, pady=2)

        # Tab 3: Performance (Placeholder)
        performance_tab = ttk.Frame(self.notebook)
        self.notebook.add(performance_tab, text="Performance")
        
        performance_tab.grid_rowconfigure(0, weight=1)
        performance_tab.grid_columnconfigure(0, weight=1)

        performance_label = ttk.Label(performance_tab, text="Performance metrics and detailed charts coming soon!")
        performance_label.grid(row=0, column=0, padx=5, pady=5)

        # Bind window close event to cleanup
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def on_closing(self):
        """Handle window closing event"""
        self.stop_bot()
        self.root.destroy()

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

    def safe_update_gui(self, update_function):
        try:
            self.log_trade(f"Calling safe_update_gui with function: {update_function.__name__}")
            if self.root:
                self.root.after(0, update_function)
            else:
                self.log_trade("Error: Tkinter root not available for GUI update")
        except Exception as e:
            self.log_trade(f"Error in safe_update_gui: {str(e)}")

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
        """Update the price chart with the latest data for all active trades"""
        try:
            # Clear the current chart
            self.ax.clear()

            # Plot price change percentage for each active trade
            for symbol in self.active_trades:
                if symbol in self.price_history and self.price_history[symbol]:
                    # Get the price history for this symbol
                    timestamps, prices = zip(*self.price_history[symbol])
                    
                    # Calculate price change percentage relative to the first price
                    prices = list(prices)
                    if prices:
                        base_price = prices[0]
                        price_changes = [(price - base_price) / base_price * 100 for price in prices]
                        
                        # Plot the price change percentage
                        self.ax.plot(timestamps, price_changes, label=f"{symbol} ({price_changes[-1]:.2f}%)")

            # Set chart labels and legend
            self.ax.set_xlabel('Time')
            self.ax.set_ylabel('Price Change (%)')
            self.ax.legend()
            self.ax.grid(True)

            # Rotate x-axis labels for better readability
            self.ax.tick_params(axis='x', rotation=45)

            # Redraw the chart
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

    def monitor_trades(self):
        try:
            self.log_trade(f"Monitoring {len(self.active_trades)} active trades")
            for trade_id, trade in list(self.active_trades.items()):
                try:
                    # Check trade duration
                    trade_age = (pd.Timestamp.now() - trade['timestamp']).total_seconds()
                    if trade_age < 10:  # Minimum 10 seconds before checking exit conditions
                        self.log_trade(f"Trade {trade_id} too new ({trade_age:.1f}s), skipping checks")
                        continue
                    symbol = trade['symbol']
                    current_price = self.get_current_price(symbol)
                    entry_price = trade['entry_price']
                    profit_loss = (current_price - entry_price) / entry_price * 100
                    trade['current_profit'] = profit_loss
                    trade['highest_profit'] = max(trade.get('highest_profit', 0), profit_loss)
                    profit_target = float(self.profit_target.get())
                    stop_loss = float(self.stop_loss.get())
                    trailing_activation = float(self.trailing_activation.get())
                    trailing_stop = float(self.trailing_stop.get())
                    self.log_trade(f"Checking {symbol}: P/L={profit_loss:.2f}%, Highest={trade['highest_profit']:.2f}%")
                    if profit_loss >= profit_target:
                        self.close_trade(trade_id, trade, current_price, "Profit Target")
                    elif profit_loss <= -stop_loss:
                        self.close_trade(trade_id, trade, current_price, "Stop Loss")
                    elif trade['highest_profit'] >= trailing_activation and (trade['highest_profit'] - profit_loss) >= trailing_stop:
                        self.close_trade(trade_id, trade, current_price, "Trailing Stop")
                except Exception as e:
                    self.log_trade(f"Error monitoring {trade_id}: {str(e)}")
            self.safe_update_gui(self.update_active_trades_display)
        except Exception as e:
            self.log_trade(f"Error in monitor_trades: {str(e)}")

    def monitor_prices_continuously(self):
        """Monitor active trades and manage trailing stops (using mock data)"""
        while self.running:
            try:
                for trade_id, trade in list(self.active_trades.items()):
                    if not self.running:
                        break

                    # Get current price (using mock ticker data)
                    ticker = self.fetch_tickers_with_retry().get(trade['symbol'], {})
                    current_price = ticker.get('last')
                    if not current_price:
                        self.log_trade(f"No price data for {trade['symbol']}")
                        continue

                    # Update price history for charting
                    if trade['symbol'] in self.price_history:
                        self.price_history[trade['symbol']].append((datetime.now(), current_price))
                        # Limit price history to last 100 entries
                        self.price_history[trade['symbol']] = self.price_history[trade['symbol']][-100:]

                    # Check profit target
                    profit_percentage = ((current_price - trade['entry_price']) / trade['entry_price']) * 100
                    if profit_percentage >= float(self.profit_target.get()):
                        self.close_trade(trade_id, trade, current_price, "profit target reached")
                        continue

                    # Check stop loss
                    loss_percentage = ((trade['entry_price'] - current_price) / trade['entry_price']) * 100
                    if loss_percentage >= float(self.stop_loss.get()):
                        self.close_trade(trade_id, trade, current_price, "stop loss hit")
                        continue

                    # Update trailing stop
                    if current_price > trade['highest_price']:
                        trade['highest_price'] = current_price

                    trailing_stop_pct = float(self.trailing_stop.get()) / 100
                    trailing_activation = float(self.trailing_activation.get()) / 100
                    if profit_percentage >= (trailing_activation * 100):
                        drop_from_high = ((trade['highest_price'] - current_price) / trade['highest_price'])
                        if drop_from_high >= trailing_stop_pct:
                            self.close_trade(trade_id, trade, current_price, "trailing stop triggered")
                            continue

                # Update GUI displays
                self.update_active_trades_display()
                self.update_metrics()
                self.update_balance_display()
                self.update_chart()

                # Sleep to avoid overloading
                time.sleep(0.1)

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

    def scan_opportunities(self, tickers):
        """Scan for trading opportunities using mock data with updated market filters"""
        try:
            pairs = []
            seen_symbols = set()  # Track symbols to avoid duplicates
            for symbol, ticker in tickers.items():
                # Skip if we've already processed this symbol
                if symbol in seen_symbols:
                    self.log_trade(f"Skipping duplicate symbol {symbol} in tickers")
                    continue

                price = float(ticker['last'])
                volume = float(ticker['quoteVolume'])
                # Initial filters: price <= $5, price > $0.00001, volume >= min_volume
                if not (price <= 5.0 and price > 0.00001 and volume >= float(self.min_volume_entry.get())):
                    self.log_trade(f"Rejected {symbol}: Failed initial filters (Price: {price:.5f}, Volume: {volume:.2f})")
                    continue

                # Market Filter Conditions
                conditions_met = 0
                required_conditions = int(self.required_conditions.get())
                self.log_trade(f"\nChecking market filters for {symbol}...")

                # Condition 1: Min Price Change
                price_change = float(ticker['price_change_24h'])
                min_price_change = float(self.min_price_change.get())
                if price_change >= min_price_change:
                    conditions_met += 1
                    self.log_trade(f"Price Change check passed: {price_change:.2f}% >= {min_price_change}%")
                else:
                    self.log_trade(f"Price Change check failed: {price_change:.2f}% < {min_price_change}%")

                # Condition 2: Min Liquidity
                liquidity = float(ticker['liquidity'])
                min_liquidity = float(self.min_liquidity.get())
                if liquidity >= min_liquidity:
                    conditions_met += 1
                    self.log_trade(f"Liquidity check passed: ${liquidity:.2f} >= ${min_liquidity}")
                else:
                    self.log_trade(f"Liquidity check failed: ${liquidity:.2f} < ${min_liquidity}")

                # Condition 3: Min Market Cap
                market_cap = float(ticker['market_cap'])
                min_market_cap = float(self.min_market_cap.get())
                if market_cap >= min_market_cap:
                    conditions_met += 1
                    self.log_trade(f"Market Cap check passed: ${market_cap:.2f} >= ${min_market_cap}")
                else:
                    self.log_trade(f"Market Cap check failed: ${market_cap:.2f} < ${min_market_cap}")

                # Condition 4: Max Volatility
                volatility = float(ticker['volatility_24h'])
                max_volatility = float(self.max_volatility.get())
                if volatility <= max_volatility:
                    conditions_met += 1
                    self.log_trade(f"Volatility check passed: {volatility:.2f}% <= {max_volatility}%")
                else:
                    self.log_trade(f"Volatility check failed: {volatility:.2f}% > {max_volatility}%")

                # Condition 5: Min Trade Count
                trade_count = float(ticker['trade_count_24h'])
                min_trade_count = float(self.min_trade_count.get())
                if trade_count >= min_trade_count:
                    conditions_met += 1
                    self.log_trade(f"Trade Count check passed: {trade_count:.0f} >= {min_trade_count}")
                else:
                    self.log_trade(f"Trade Count check failed: {trade_count:.0f} < {min_trade_count}")

                # Condition 6: Max Spread
                spread = (float(ticker['ask']) - float(ticker['bid'])) / float(ticker['bid'])
                max_spread = float(self.max_spread.get()) / 100
                if spread < max_spread and ticker['bid'] > 0 and ticker['ask'] > 0:
                    conditions_met += 1
                    self.log_trade(f"Spread check passed: {spread*100:.2f}% < {max_spread*100}%")
                else:
                    self.log_trade(f"Spread check failed: {spread*100:.2f}% >= {max_spread*100}%")

                # Check if enough conditions are met
                if conditions_met < required_conditions:
                    self.log_trade(f"Rejected {symbol}: Only {conditions_met}/{required_conditions} market conditions met")
                    continue

                self.log_trade(f"Market filters passed for {symbol}: {conditions_met}/{required_conditions} conditions met")
                pairs.append({
                    'symbol': symbol,
                    'ticker': ticker,
                    'spread': spread * 100
                })
                seen_symbols.add(symbol)  # Add symbol to seen set

            # Sort by volume and limit to top_list_size
            sorted_pairs = sorted(pairs, key=lambda x: x['ticker']['quoteVolume'], reverse=True)[:int(self.top_list_size.get())]
            self.log_trade(f"Returning {len(sorted_pairs)} valid pairs: {[p['symbol'] for p in sorted_pairs]}")
            return sorted_pairs

        except Exception as e:
            self.log_trade(f"Error scanning opportunities: {str(e)}")
            return []

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

            # Final performance log
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

            self.log_trade("Cleanup completed successfully")
            
        except Exception as e:
            self.log_trade(f"Error during cleanup: {str(e)}")
    def on_closing(self):
        """Handle window close event to ensure proper shutdown."""
        self.log_trade("Window closing, initiating shutdown...")
        
        # Stop all threads
        self.running = False  # Stop the bot loop and price monitoring threads
        self.timer_running = False  # Stop the timer if running
        
        # Perform cleanup using the existing cleanup_on_shutdown method
        self.cleanup_on_shutdown()
        
        # Destroy the Tkinter window and exit
        try:
            self.root.destroy()
        except Exception as e:
            self.log_trade(f"Error destroying window: {str(e)}")
        finally:
            import sys
            sys.exit(0)  # Forcefully exit the program to ensure no threads linger

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
        """Analyze trading opportunity using DataManager"""
        try:
            self.log_trade(f"\nAnalyzing {pair_data['symbol']}...")

            # Update price data in DataManager
            self.data_manager.update_price_data(pair_data['symbol'], ticker)

            # Get the DataFrame for the symbol
            df = self.data_manager.price_data.get(pair_data['symbol'])
            if df is None or df.empty:
                self.log_trade(f"No data available for {pair_data['symbol']}")
                return False

            self.log_trade(f"Data points for {pair_data['symbol']}: {len(df)}")

            # Validation Criteria Checks
            conditions_met = 0
            required_conditions = int(self.required_conditions.get())
            self.log_trade(f"Required conditions: {required_conditions}")

            # Momentum (Beta)
            momentum = self.calculate_current_momentum(df)
            momentum_beta = float(self.momentum_beta.get())
            if momentum >= momentum_beta:
                conditions_met += 1
                self.log_trade(f"Momentum check passed: {momentum:.2f} >= {momentum_beta}")
            else:
                self.log_trade(f"Momentum check failed: {momentum:.2f} < {momentum_beta}")

            # Price Momentum (Alpha)
            price_momentum = self.calculate_price_momentum(df)
            price_alpha = float(self.price_alpha.get())
            if price_momentum >= price_alpha:
                conditions_met += 1
                self.log_trade(f"Price momentum check passed: {price_momentum:.2f} >= {price_alpha}")
            else:
                self.log_trade(f"Price momentum check failed: {price_momentum:.2f} < {price_alpha}")

            # Momentum Quality (Theta)
            momentum_quality = self.calculate_momentum_quality(df)
            time_theta = float(self.time_theta.get())
            if momentum_quality <= time_theta:
                conditions_met += 1
                self.log_trade(f"Momentum quality check passed: {momentum_quality:.2f} <= {time_theta}")
            else:
                self.log_trade(f"Momentum quality check failed: {momentum_quality:.2f} > {time_theta}")

            # Volatility (Vega)
            volatility = self.calculate_volatility_sensitivity(df)
            vol_vega = float(self.vol_vega.get())
            if volatility <= vol_vega:
                conditions_met += 1
                self.log_trade(f"Volatility check passed: {volatility:.2f} <= {vol_vega}")
            else:
                self.log_trade(f"Volatility check failed: {volatility:.2f} > {vol_vega}")

            # Volume (Rho)
            volume_increase = self.calculate_volume_increase(df)
            volume_rho = float(self.volume_rho.get())
            if volume_increase >= volume_rho:
                conditions_met += 1
                self.log_trade(f"Volume check passed: {volume_increase:.2f} >= {volume_rho}")
            else:
                self.log_trade(f"Volume check failed: {volume_increase:.2f} < {volume_rho}")

            # Check if enough conditions are met
            if conditions_met < required_conditions:
                self.log_trade(f"Rejected {pair_data['symbol']}: Only {conditions_met}/{required_conditions} conditions met")
                return False

            # Advanced Checks (e.g., RSI, EMA Cross)
            if not self.advanced_checks(pair_data['symbol'], df):
                self.log_trade(f"Rejected {pair_data['symbol']}: Advanced checks failed")
                return False

            self.log_trade(f"Opportunity found for {pair_data['symbol']}")
            return True

        except Exception as e:
            self.log_trade(f"Critical error analyzing {pair_data['symbol']}: {str(e)}")
            return False

    def analyze_pairs(self, valid_pairs):
        """Analyze multiple pairs for trading opportunities"""
        try:
            trades_executed = False
            self.log_trade("\nAnalyzing pairs for trading opportunities...")
            self.log_trade(f"Processing {len(valid_pairs)} valid pairs: {[pair['symbol'] for pair in valid_pairs]}")

            try:
                max_trades = int(self.max_trades_entry.get())
            except (ValueError, AttributeError, tk.TclError) as e:
                self.log_trade(f"Error retrieving max_trades from GUI: {str(e)}. Defaulting to 3.")
                max_trades = 3

            self.log_trade(f"Max trades limit: {max_trades}, Current active trades: {len(self.active_trades)}")
            self.log_trade(f"Active trades keys: {list(self.active_trades.keys())}")


            for pair_data in valid_pairs:
                # Check max_trades limit *before* processing each pair
                if len(self.active_trades) >= max_trades:
                    self.log_trade(f"Stopping analysis: Max trades ({max_trades}) reached...")
                    break # Exit the loop immediately if limit is reached

                try:
                    symbol = pair_data['symbol']
                    ticker = pair_data['ticker'] # Get the ticker data for the update

                    # --- Add DataManager Update Here ---
                    self.log_trade(f"Updating DataManager with latest ticker for {symbol}...")
                    self.data_manager.update_price_data(symbol, ticker)
                    # --- End DataManager Update ---


                    # Ensure symbol isn't already active
                    if symbol in [trade['symbol'] for trade in self.active_trades.values()]:
                        self.log_trade(f"Skipping {symbol}: Already in active trades.")
                        continue

                    # Get DataFrame and ensure we have enough data
                    df = self.data_manager.price_data.get(symbol)
                    min_data_points = 20 # Minimum points needed for analysis (adjust if needed)

                    # Check if DataFrame exists and has enough rows
                    if df is None or len(df) < min_data_points:
                        self.log_trade(f"Skipping {symbol}: Insufficient data ({len(df) if df is not None else 0}/{min_data_points} points available in DataManager)")
                        # Log current data points count to track progress
                        continue # Wait for more data in subsequent cycles


                    # --- If we reach here, we have enough data ---
                    self.log_trade(f"Sufficient data ({len(df)} points) found for {symbol}. Proceeding with analysis.")

                    # --- Your existing analysis logic using df ---
                    momentum = self.calculate_current_momentum(df)
                    volume_increase = self.calculate_volume_increase(df) # Ensure this uses df correctly
                    current_price = df['price'].iloc[-1] # Use latest price from df
                    volume_24h = float(ticker.get('quoteVolume', 0)) # Use 24h vol from ticker
                    spread = pair_data.get('spread', 100) # Use spread from scan result

                    self.log_trade(f"""
                    Detailed Analysis for {symbol}:
                    Current Price: ${current_price:.8f} (from DataFrame)
                    24h Volume: ${volume_24h:,.2f} (from Ticker)
                    Momentum: {momentum:.2f}% (from DataFrame)
                    Volume Change: {volume_increase:.2f}% (from DataFrame)
                    Spread: {spread:.2f}% (from Scan)
                    Data Points: {len(df)}
                    """)

                    # (Rest of the condition checks: Use calculated metrics from df)
                    conditions_met = []
                    try: required_conditions = int(self.required_conditions.get())
                    except: required_conditions = 3
                    try: momentum_threshold = float(self.momentum_beta.get()) # Using momentum_beta now
                    except: momentum_threshold = 0.3 # Default for beta
                    try: min_volume_gui = float(self.min_volume_entry.get())
                    except: min_volume_gui = 200.0
                    try: max_spread = float(self.max_spread.get())
                    except: max_spread = 0.1 # Default spread %

                    # 1. Momentum check (Beta)
                    if momentum >= momentum_threshold: # Compare calculated momentum to threshold
                        conditions_met.append("Momentum")
                        self.log_trade(f"✓ Momentum condition met: {momentum:.2f}% >= {momentum_threshold}%")
                    else:
                        self.log_trade(f"✗ Momentum condition failed: {momentum:.2f}% < {momentum_threshold}%")


                    # 2. Volume check (using 24h volume from ticker)
                    if volume_24h >= min_volume_gui:
                        conditions_met.append("Volume")
                        self.log_trade(f"✓ Volume condition met (24h): ${volume_24h:,.2f} >= ${min_volume_gui:,.2f}")
                    else:
                        self.log_trade(f"✗ Volume condition failed (24h): ${volume_24h:,.2f} < ${min_volume_gui:,.2f}")


                    # 3. Spread check (using spread from pair_data)
                    if spread < max_spread:
                        conditions_met.append("Spread")
                        self.log_trade(f"✓ Spread condition met: {spread:.2f}% < {max_spread:.2f}%")
                    else:
                        self.log_trade(f"✗ Spread condition failed: {spread:.2f}% >= {max_spread:.2f}%")


                    # (Include other validation criteria checks: Alpha, Theta, Vega, Rho using df)
                    # Example: price_momentum = self.calculate_price_momentum(df) -> check against price_alpha
                    # Example: momentum_quality = self.calculate_momentum_quality(df) -> check against time_theta
                    # ... add these checks here and append to conditions_met if passed ...


                    # --- Final check and trade execution ---
                    self.log_trade(f"Conditions met ({len(conditions_met)}/{required_conditions}): {', '.join(conditions_met)}")

                    if len(conditions_met) >= required_conditions:
                        self.log_trade(f"[TARGET] Trade opportunity found: {symbol}")

                        # Validate and Execute trade using the current_price from df
                        trade_data_for_execution = {
                            'symbol': symbol,
                            'ticker': ticker, # Pass ticker if needed by validation/execution
                            'entry_price': current_price # Pass price confirmed from df
                        }
                        if self.validate_trade(pair_data, current_price): # Validate using df price
                            if self.execute_trade(trade_data_for_execution):
                                trades_executed = True
                                self.log_trade(f"Trade executed successfully for {symbol}")
                                if len(self.active_trades) >= max_trades:
                                    self.log_trade(f"Max trades ({max_trades}) reached after opening trade for {symbol}")
                                    break
                            else:
                                self.log_trade(f"Failed to execute trade for {symbol}")
                        else:
                            self.log_trade(f"Trade validation failed for {symbol}")
                    else:
                        self.log_trade(f"Conditions not met for {symbol}")


                except Exception as e:
                    self.log_trade(f"Error analyzing {pair_data.get('symbol', 'N/A')}: {str(e)}")
                    import traceback
                    self.log_trade(traceback.format_exc()) # Log full traceback
                    continue # Continue to the next pair

            if not trades_executed:
                self.log_trade("No new trading opportunities met criteria this scan")
            else:
                self.log_trade(f"Finished scan cycle. Active trades: {len(self.active_trades)}")

            return trades_executed

        except Exception as e:
            self.log_trade(f"Critical Error in analyze_pairs: {str(e)}")
            import traceback
            self.log_trade(traceback.format_exc())
            return False

        
    def calculate_price_momentum(self, df):
        """Calculate price momentum as percentage change over recent data points"""
        try:
            if len(df) < 2:
                self.log_trade("Not enough data points for price momentum calculation")
                return 0.0

            # Use the last 5 data points (or fewer if not enough data)
            recent_prices = df['price'].tail(5)
            self.log_trade(f"Recent prices for price momentum calc: {recent_prices}")

            if len(recent_prices) < 2:
                self.log_trade("Not enough recent prices for price momentum calculation")
                return 0.0

            # Calculate percentage change: (latest - earliest) / earliest * 100
            earliest_price = recent_prices.iloc[0]
            latest_price = recent_prices.iloc[-1]
            price_momentum = ((latest_price - earliest_price) / earliest_price) * 100
            self.log_trade(f"Calculated price momentum: {price_momentum:.2f}%")
            return price_momentum

        except Exception as e:
            self.log_trade(f"Error calculating price momentum: {str(e)}")
            return 0.0

    def calculate_current_momentum(self, df):
        """Calculate current momentum as percentage change over recent data points"""
        try:
            if len(df) < 2:
                self.log_trade("Not enough data points for momentum calculation")
                return 0.0

            recent_prices = df['price'].tail(5)
            self.log_trade(f"Recent prices for momentum calc: {recent_prices}")

            if len(recent_prices) < 2:
                self.log_trade("Not enough recent prices for momentum calculation")
                return 0.0

            earliest_price = recent_prices.iloc[0]
            latest_price = recent_prices.iloc[-1]
            momentum = ((latest_price - earliest_price) / earliest_price) * 100
            self.log_trade(f"Calculated momentum: {momentum}")
            return momentum

        except Exception as e:
            self.log_trade(f"Error calculating momentum: {str(e)}")
            return 0.0

    def calculate_momentum_quality(self, df):
        """Calculate momentum quality (Theta) as the standard deviation of recent price changes"""
        try:
            if len(df) < 5:
                self.log_trade("Not enough data points for momentum quality calculation")
                return float('inf')  # High value to fail the check if insufficient data

            recent_prices = df['price'].tail(10)
            price_changes = recent_prices.pct_change().dropna() * 100  # Percentage changes
            momentum_quality = price_changes.std()  # Standard deviation of price changes
            self.log_trade(f"Calculated momentum quality: {momentum_quality:.2f}%")
            return momentum_quality if not pd.isna(momentum_quality) else float('inf')

        except Exception as e:
            self.log_trade(f"Error calculating momentum quality: {str(e)}")
            return float('inf')

    def calculate_volatility_sensitivity(self, df):
        """Calculate volatility (Vega) as the standard deviation of recent prices"""
        try:
            if len(df) < 5:
                self.log_trade("Not enough data points for volatility calculation")
                return float('inf')  # High value to fail the check if insufficient data

            recent_prices = df['price'].tail(10)
            volatility = recent_prices.pct_change().std() * 100  # Standard deviation of percentage changes
            self.log_trade(f"Calculated volatility: {volatility:.2f}%")
            return volatility if not pd.isna(volatility) else float('inf')

        except Exception as e:
            self.log_trade(f"Error calculating volatility: {str(e)}")
            return float('inf')

    def calculate_volume_increase(self, df):
        """Calculate volume increase (Rho) as percentage change in recent volume"""
        try:
            if len(df) < 2:
                self.log_trade("Not enough data points for volume increase calculation")
                return 0.0

            recent_volumes = df['volume'].tail(5)
            self.log_trade(f"Recent volumes for volume increase calc: {recent_volumes}")

            if len(recent_volumes) < 2:
                self.log_trade("Not enough recent volumes for volume increase calculation")
                return 0.0

            earliest_volume = recent_volumes.iloc[0]
            latest_volume = recent_volumes.iloc[-1]
            volume_increase = ((latest_volume - earliest_volume) / earliest_volume) * 100
            self.log_trade(f"Calculated volume increase: {volume_increase:.2f}%")
            return volume_increase

        except Exception as e:
            self.log_trade(f"Error calculating volume increase: {str(e)}")
            return 0.0

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


    # Modify execute_trade to accept price if needed, or ensure it fetches fresh price
    def execute_trade(self, trade_data): # Accept dict now
        try:
            symbol = trade_data['symbol']
            self.log_trade(f"Attempting to execute trade for {symbol}. Current active trades: {len(self.active_trades)}")

            # --- Ensure max trades not exceeded RIGHT BEFORE adding ---
            try:
                max_trades = int(self.max_trades_entry.get())
            except: max_trades = 3 # Default safely
            if len(self.active_trades) >= max_trades:
                self.log_trade(f"Cannot execute trade for {symbol}: Max trades ({max_trades}) limit reached just before execution.")
                return False
            # --- End Check ---

            # Use provided entry price or fetch fresh if necessary
            entry_price = trade_data.get('entry_price')
            if entry_price is None:
                # Fetch fresh price if not provided (adapt your fetching logic)
                # Example: ticker = self.fetch_tickers_with_retry().get(symbol, {})
                # entry_price = float(ticker.get('last', 0))
                # For mock data:
                ticker = self.mock_ticker_state.get(symbol, {})
                entry_price = float(ticker.get('last', 0))
                if entry_price == 0:
                    self.log_trade(f"Cannot execute trade for {symbol}: Failed to get valid entry price.")
                    return False

            position_size = float(self.position_size.get()) # Get fresh value

            # Check balance again just before committing
            if self.is_paper_trading:
                if self.paper_balance < position_size:
                    self.log_trade(f"Cannot execute trade for {symbol}: Insufficient paper balance ({self.paper_balance:.2f} < {position_size:.2f})")
                    return False
                self.paper_balance -= position_size # Deduct balance only if trade proceeds
            # else: # Add real balance check if needed
                # pass

            trade_id = f"{symbol}_{pd.Timestamp.now().strftime('%Y%m%d%H%M%S%f')}" # More unique ID
            trade = {
                'symbol': symbol,
                'entry_price': entry_price,
                'position_size': position_size,
                'timestamp': pd.Timestamp.now(),
                'current_profit': 0.0,
                'highest_profit': 0.0, # Initialize highest profit correctly
                # Add other relevant info if needed
            }

            # Add to active trades
            self.active_trades[trade_id] = trade

            # Add to chart trades if used
            if not hasattr(self, 'chart_trades'):
                self.chart_trades = {}
            self.chart_trades[trade_id] = trade

            # Initialize price history for the new trade
            if symbol not in self.price_history:
                self.price_history[symbol] = [] # Or initialize DataFrame as needed
            # Add initial entry point to history
            self.price_history[symbol].append((trade['timestamp'], entry_price))


            self.log_trade(f"[TRADE OPENED] Paper Position: {symbol} at {entry_price:.8f}, Size: ${position_size:.2f}, ID: {trade_id}. New Balance: ${self.paper_balance:.2f}")

            # Use safe_update_gui for GUI updates
            self.safe_update_gui(self.update_active_trades_display)
            self.safe_update_gui(self.update_metrics) # Update metrics which includes balance
            self.safe_update_gui(self.update_chart) # Update chart

            return True

        except Exception as e:
            self.log_trade(f"Error executing trade for {trade_data.get('symbol', 'N/A')}: {str(e)}")
            import traceback
            self.log_trade(traceback.format_exc())
            # Rollback balance if deduction happened before error
            # if self.is_paper_trading and 'position_size' in locals() and trade_id not in self.active_trades:
            #     self.paper_balance += position_size
            #     self.log_trade(f"Rolled back paper balance deduction for failed trade {symbol}")
            return False

    def close_trade(self, trade_id, trade, exit_price, reason):
        try:
            if trade_id not in self.active_trades:
                self.log_trade(f"Cannot close trade {trade_id}: Not found")
                return
            symbol = trade['symbol']
            entry_price = trade['entry_price']
            position_size = trade['position_size']
            quantity = position_size / entry_price
            profit_loss = (exit_price - entry_price) * quantity
            fees = position_size * 0.001 * 2
            net_profit = profit_loss - fees
            self.paper_balance += position_size + profit_loss
            self.total_profit += profit_loss
            self.total_fees += fees
            self.net_profit += net_profit
            if profit_loss > 0:
                self.wins += 1
            else:
                self.losses += 1
            trade_record = {
                'symbol': symbol,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'profit_loss': profit_loss,
                'fees': fees,
                'reason': reason,
                'timestamp': trade['timestamp'],
                'exit_time': pd.Timestamp.now()
            }
            self.trade_history.append(trade_record)
            del self.active_trades[trade_id]
            if hasattr(self, 'chart_trades') and trade_id in self.chart_trades:
                del self.chart_trades[trade_id]
            self.log_trade(f"[TRADE CLOSED] {symbol}: ID={trade_id}, P/L={profit_loss:.2f} USD, Reason={reason}")
            self.safe_update_gui(self.update_active_trades_display)
            self.update_metrics()
        except Exception as e:
            self.log_trade(f"Error closing trade {trade_id}: {str(e)}")

        
    def advanced_checks(self, symbol, df):
        """Perform advanced checks using technical indicators"""
        try:
            # Ensure enough data for indicators
            if len(df) < 15:
                self.log_trade(f"Not enough data for advanced checks on {symbol}: {len(df)} data points")
                return False

            # Log RSI for informational purposes (check is disabled)
            latest_rsi = df['rsi_14'].iloc[-1]
            rsi_overbought = float(self.rsi_overbought.get())
            rsi_oversold = float(self.rsi_oversold.get())
            self.log_trade(f"RSI for {symbol}: {latest_rsi:.2f} (Overbought: {rsi_overbought}, Oversold: {rsi_oversold})")

            # Check EMA (relaxed to allow trading if EMA_5 is above EMA_15)
            ema_5 = df['ema_5'].iloc[-1]
            ema_15 = df['ema_15'].iloc[-1]
            ema_5_prev = df['ema_5'].iloc[-2]
            ema_15_prev = df['ema_15'].iloc[-2]
            self.log_trade(f"EMA values for {symbol}: EMA_5={ema_5:.5f}, EMA_15={ema_15:.5f}, EMA_5_prev={ema_5_prev:.5f}, EMA_15_prev={ema_15_prev:.5f}")
            
            # Temporarily allow trading regardless of EMA for testing
            if True:  # Replace with `if ema_5 > ema_15:` once confirmed working
                self.log_trade(f"EMA check passed for {symbol}: Bypassing EMA condition for testing")
                return True
            self.log_trade(f"EMA check failed for {symbol}: EMA_5 {ema_5:.5f} <= EMA_15 {ema_15:.5f}")
            return False

        except Exception as e:
            self.log_trade(f"Error in advanced checks for {symbol}: {str(e)}")
            return False
        
    # Force correct update_active_trades_display method
    def update_active_trades_display(self):
        try:
            self.log_trade("DEBUG: Executing correct update_active_trades_gui method")
            self.log_trade(f"Updating active trades display. Current active trades: {len(self.active_trades)}")
            if not hasattr(self, 'trades_text') or not hasattr(self, 'active_trades_label'):
                self.log_trade("Error: trades_text or active_trades_label not initialized")
                return
            self.trades_text.config(state="normal")
            self.trades_text.delete("1.0", tk.END)
            self.active_trades_label.config(text=f"Number of Active Trades: {len(self.active_trades)}")
            for trade_id, trade in self.active_trades.items():
                try:
                    symbol = trade['symbol']
                    entry_price = trade['entry_price']
                    current_price = self.get_current_price(symbol)
                    profit_loss = (current_price - entry_price) / entry_price * 100 if entry_price != 0 else 0
                    trade['current_profit'] = profit_loss
                    trade_info = f"{trade_id}: {symbol}, {profit_loss:.2f}% (Entry: {entry_price:.5f}, Current: {current_price:.5f})\n"
                    self.trades_text.insert(tk.END, trade_info)
                except Exception as e:
                    self.log_trade(f"Error updating display for trade {trade_id}: {str(e)}")
            self.trades_text.config(state="disabled")
            self.log_trade("Active trades display updated successfully")
        except Exception as e:
            self.log_trade(f"Error updating active trades display: {str(e)}")

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
        """Update the GUI with the latest performance metrics"""
        try:
            # Update Total Profit
            self.total_profit_label.config(text=f"Total Profit: {self.total_profit:.2f} USD")
            
            # Update Total Fees
            self.total_fees_label.config(text=f"Total Fees: {self.total_fees:.2f} USD")
            
            # Update Net Profit
            self.net_profit_label.config(text=f"Net Profit: {self.net_profit:.2f} USD")
            
            # Update Win/Loss Ratio
            total_trades = self.wins + self.losses
            win_loss_ratio = (self.wins / total_trades * 100) if total_trades > 0 else 0
            self.win_loss_label.config(text=f"Win/Loss: {self.wins}/{self.losses} ({win_loss_ratio:.1f}%)")
            
            # Update Paper Balance
            self.log_trade(f"Updating GUI paper balance label to: {self.paper_balance:.2f} USD")
            self.paper_balance_label.config(text=f"Paper Balance: {self.paper_balance:.2f} USD")
            
            # Log the metrics update
            self.log_trade(f"Metrics Updated: Total Profit={self.total_profit:.2f}, Net Profit={self.net_profit:.2f}, Paper Balance={self.paper_balance:.2f}, Wins={self.wins}, Losses={self.losses}")

        except Exception as e:
            self.log_trade(f"Error updating metrics display: {str(e)}")

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
        try:
            self.log_trade("DEBUG: Executing correct update_active_trades_gui method") # Original had typo: _gui
            self.log_trade(f"Updating active trades display. Current active trades: {len(self.active_trades)}")
            # Ensure the necessary GUI elements exist before proceeding
            if not hasattr(self, 'trades_text') or not hasattr(self, 'active_trades_label'):
                self.log_trade("Error: trades_text or active_trades_label not initialized properly in setup_gui")
                # Attempt to find or re-create them if possible, or simply return
                # For now, just log and return to prevent further errors
                return

            # Safely update the text widget
            if self.trades_text:
                self.trades_text.config(state="normal")
                self.trades_text.delete("1.0", tk.END)
            else:
                self.log_trade("Error: trades_text widget does not exist.")
                return # Exit if the widget is missing

            # Safely update the label widget
            if self.active_trades_label:
                self.active_trades_label.config(text=f"Number of Active Trades: {len(self.active_trades)}")
            else:
                self.log_trade("Error: active_trades_label widget does not exist.")
                # Optionally handle the missing label here

            # Iterate through active trades and update the text display
            for trade_id, trade in self.active_trades.items():
                try:
                    symbol = trade['symbol']
                    entry_price = trade['entry_price']
                    # Use a helper to get current price safely, maybe from cache or mock data
                    # Assuming get_current_price exists and handles errors/mock data
                    current_price = self.get_current_price(symbol) # You need to implement/verify get_current_price

                    profit_loss = (current_price - entry_price) / entry_price * 100 if entry_price != 0 else 0
                    trade['current_profit'] = profit_loss # Update the trade dict as well

                    trade_info = f"{trade_id}: {symbol}, {profit_loss:.2f}% (Entry: {entry_price:.5f}, Current: {current_price:.5f})\n"
                    if self.trades_text:
                        self.trades_text.insert(tk.END, trade_info)
                except Exception as e:
                    self.log_trade(f"Error updating display for trade {trade_id}: {str(e)}")

            # Disable the text widget again after updates
            if self.trades_text:
                self.trades_text.config(state="disabled")

            self.log_trade("Active trades display updated successfully")

        except tk.TclError as e:
            # Handle cases where GUI elements might be destroyed or inaccessible
            self.log_trade(f"GUI Error updating active trades display: {str(e)}")
        except Exception as e:
            # Catch any other unexpected errors during the update
            self.log_trade(f"General Error updating active trades display: {str(e)}")


    # DELETE THIS VERSION (Lines ~1671-1692 in your file)
    # def update_active_trades_display(self):
    #    # ... (code that uses self.active_trades_frame) ...
    #    pass # Delete this entire method definition

    # You will also need a method to get the current price, adapting your existing logic:
    def get_current_price(self, symbol):
        # Example using mock data - adapt if using live data cache
        if symbol in self.mock_ticker_state:
            return float(self.mock_ticker_state[symbol]['last'])
        else:
            # Fallback or error handling if symbol not found
            self.log_trade(f"Warning: Price not found for {symbol} in mock_ticker_state")
            # Return a default/dummy value or raise an error
            return 0.0 # Or handle appropriately

    def run_bot(self):
        try:
            if self.running:
                self.log_trade("Bot is already running")
                return
            self.running = True
            self.log_trade("=== BOT STARTED ===")
            self.status_label.config(text="Running (Paper)")
            # Initialize price history with mock data
            for symbol in self.mock_ticker_state:
                if symbol not in self.price_history:
                    self.price_history[symbol] = pd.DataFrame(columns=['timestamp', 'price', 'volume'])
                    # Simulate 20 data points
                    for i in range(20):
                        timestamp = pd.Timestamp.now() - pd.Timedelta(seconds=(20 - i) * 5)
                        price = self.mock_ticker_state[symbol]['last'] * (1 + np.random.normal(0, 0.001))
                        volume = self.mock_ticker_state[symbol]['quoteVolume'] * (1 + np.random.normal(0, 0.1))
                        self.price_history[symbol] = pd.concat([
                            self.price_history[symbol],
                            pd.DataFrame({
                                'timestamp': [timestamp],
                                'price': [price],
                                'volume': [volume]
                            })
                        ], ignore_index=True)
                self.log_trade(f"Initialized price_history for {symbol} with {len(self.price_history[symbol])} data points")
            self.bot_thread = threading.Thread(target=self.run_bot_loop)
            self.bot_thread.daemon = True
            self.bot_thread.start()
            self.log_trade("Bot thread started")
        except Exception as e:
            self.log_trade(f"Error in run_bot: {str(e)}")
            self.running = False
            self.status_label.config(text="Stopped")
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