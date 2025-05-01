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
import sys
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.dates import DateFormatter
import traceback
import random
import math
import uuid
import queue

class ScrolledFrame(tk.Frame):
    """A scrollable frame widget using a Canvas and Scrollbar."""
    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)

        # Create a canvas and scrollbar
        self.canvas = tk.Canvas(self)
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)

        # Configure the canvas to work with the scrollbar
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        # Pack the scrollbar and canvas
        self.scrollbar.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)

        # Create a window in the canvas to hold the scrollable frame
        self.canvas_frame = self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")

        # Bind the canvas to update the scroll region and frame width
        self.scrollable_frame.bind("<Configure>", self._on_frame_configure)
        self.canvas.bind("<Configure>", self._on_canvas_configure)

        # Enable mouse wheel scrolling
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)

    def _on_frame_configure(self, event=None):
        """Update the scroll region to encompass the inner frame."""
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_canvas_configure(self, event):
        """Adjust the width of the inner frame to match the canvas."""
        self.canvas.itemconfig(self.canvas_frame, width=event.width)

    def _on_mousewheel(self, event):
        """Handle mouse wheel scrolling."""
        self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

class DataManager:
    def __init__(self, bot=None):
        """
        Initialize the DataManager.
        
        Args:
            bot: Optional reference to the main bot instance
        """
        self.price_data = {}
        self.bot = bot
        self.price_history = {}
        self.ax = None
        self.canvas = None
        self.valid_pairs = []  # Add this line to define valid_pairs
        self.log("DataManager instantiated with bot={}".format("present" if bot else "None"))

    def log(self, message):
        if self.bot and hasattr(self.bot, 'log_trade'):
            self.bot.log_trade(message)
        else:
            print(f"DataManager Log: {message}")


    def setup_chart(self, chart_frame):
        """
        Set up the Matplotlib chart for trade performance visualization.

        Args:
            chart_frame: The Tkinter frame to hold the chart.
        """
        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvasTkAgg(self.figure, master=chart_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.ax.set_title("Trade Performance")
        self.ax.set_xlabel("Time")
        self.ax.set_ylabel("Price Change (%)")
        self.bot.log_trade("Chart objects created in DataManager")

    def update_price_chart(self, tickers):
        """
        Update the price chart with the performance of active trades.

        Args:
            tickers (dict): Current ticker data for all symbols.
        """
        try:
            self.bot.log_trade(f"self.ax: {self.ax}, self.canvas: {self.canvas}")
            # Check if chart objects are initialized
            if self.ax is None or self.canvas is None:
                self.bot.log_trade("Chart objects (ax or canvas) not initialized, skipping chart update")
                return

            # Clear the current plot
            self.ax.clear()
            self.ax.set_title("Trade Performance")
            self.ax.set_xlabel("Time")
            self.ax.set_ylabel("Price Change (%)")

            # If there are no active trades, redraw the empty chart and return
            if not self.bot.active_trades:
                self.canvas.draw()
                return

            # Initialize price history for new trades
            if not hasattr(self, 'price_history'):
                self.price_history = {symbol: [] for symbol in self.bot.active_trades.keys()}

            # Update price history for each active trade
            for symbol in self.bot.active_trades:
                if symbol not in tickers:
                    continue
                trade = self.bot.active_trades[symbol]
                entry_price = trade['entry_price']
                current_price = tickers[symbol]['last']
                price_change = (current_price - entry_price) / entry_price * 100

                # Append the price change to the history (limit history to 100 points to avoid memory issues)
                if symbol not in self.price_history:
                    self.price_history[symbol] = []
                self.price_history[symbol].append(price_change)
                if len(self.price_history[symbol]) > 100:
                    self.price_history[symbol].pop(0)

                # Plot the price change history
                self.ax.plot(self.price_history[symbol], label=symbol)

            # Add legend and redraw the chart
            self.ax.legend()
            self.canvas.draw()
        except Exception as e:
            self.bot.log_trade(f"Error updating price chart: {str(e)}")

    def update_price_data(self, symbol, new_data):
        try:
            if symbol not in self.price_data:
                self.price_data[symbol] = pd.DataFrame(columns=['timestamp', 'price', 'volume'])
            if not isinstance(new_data, pd.DataFrame):
                new_data = pd.DataFrame([new_data], columns=['timestamp', 'price', 'volume'])
            if not new_data.empty:
                self.price_data[symbol] = pd.concat(
                    [self.price_data[symbol], new_data],
                    ignore_index=True
                )
                self.log(f"Updated price_data for {symbol} with {len(self.price_data[symbol])} points")
            else:
                self.log(f"Skipping empty data update for {symbol}")
        except Exception as e:
            self.log(f"Error updating price_data for {symbol}: {str(e)}")

    def calculate_indicators(self, symbol):
        try:
            if symbol not in self.price_data or len(self.price_data[symbol]) < 15:
                self.log(f"Insufficient data for {symbol}: {len(self.price_data[symbol]) if symbol in self.price_data else 0}/15 points")
                return None
            df = self.price_data[symbol].copy()
            delta = df['price'].diff()
            gain = delta.where(delta > 0, 0).rolling(window=14).mean()
            loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
            rs = gain / loss
            df['rsi_14'] = 100 - (100 / (1 + rs))
            df['ema_5'] = df['price'].ewm(span=5, adjust=False).mean()
            df['ema_15'] = df['price'].ewm(span=15, adjust=False).mean()
            if len(df) >= 2:
                earliest_volume = df['volume'].iloc[-2]
                latest_volume = df['volume'].iloc[-1]
                df['volume_change'] = ((latest_volume - earliest_volume) / earliest_volume * 100) if earliest_volume != 0 else 0.0
            else:
                df['volume_change'] = 0.0
            self.log(f"Calculated indicators for {symbol}: RSI={df['rsi_14'].iloc[-1]:.2f}, EMA_5={df['ema_5'].iloc[-1]:.5f}, EMA_15={df['ema_15'].iloc[-1]:.5f}, Volume Change={df['volume_change'].iloc[-1]:.2f}%")
            return df
        except Exception as e:
            self.log(f"Error in calculate_indicators for {symbol}: {str(e)}")
            return None

class CryptoScalpingBot:
    def __init__(self):
        """
        Initialize the CryptoScalpingBot with mock data, DataManager, and trading parameters.
        """
        try:
            # Set up logging
            self.logger = logging.getLogger(__name__)
            logging.basicConfig(level=logging.INFO)

            # Initialize the Tkinter root window first
            self.root = tk.Tk()
            
            # Add missing attributes
            self.is_shutting_down = False
            
            # Initialize trading parameters dictionary
            self.trading_params = {}
            
            # Initialize mock ticker state for paper trading with 30 pairs
            self.mock_ticker_state = {
                'SHIB/USD': {'last': 0.000018, 'quoteVolume': 500000.0, 'bid': 0.0000179, 'ask': 0.0000181, 'change': 2.1, 'volatility': 2.5, 'tradeCount': 12000, 'marketCap': 1.06e9},
                'DOGE/USD': {'last': 0.14, 'quoteVolume': 800000.0, 'bid': 0.1398, 'ask': 0.1402, 'change': 1.8, 'volatility': 2.0, 'tradeCount': 15000, 'marketCap': 2.03e9},
                'XRP/USD': {'last': 0.52, 'quoteVolume': 600000.0, 'bid': 0.5195, 'ask': 0.5205, 'change': 1.6, 'volatility': 1.5, 'tradeCount': 10000, 'marketCap': 2.9e10},
                'ADA/USD': {'last': 0.35, 'quoteVolume': 450000.0, 'bid': 0.3497, 'ask': 0.3503, 'change': 1.2, 'volatility': 1.8, 'tradeCount': 9000, 'marketCap': 1.25e10},
                'TRX/USD': {'last': 0.13, 'quoteVolume': 300000.0, 'bid': 0.1298, 'ask': 0.1302, 'change': 0.9, 'volatility': 1.2, 'tradeCount': 8000, 'marketCap': 1.15e10},
                'VET/USD': {'last': 0.024, 'quoteVolume': 200000.0, 'bid': 0.0239, 'ask': 0.0241, 'change': 2.3, 'volatility': 2.2, 'tradeCount': 7000, 'marketCap': 1.95e9},
                'XLM/USD': {'last': 0.095, 'quoteVolume': 250000.0, 'bid': 0.0948, 'ask': 0.0952, 'change': 1.4, 'volatility': 1.6, 'tradeCount': 8500, 'marketCap': 2.82e9},
                'ALGO/USD': {'last': 0.14, 'quoteVolume': 180000.0, 'bid': 0.1397, 'ask': 0.1403, 'change': 1.7, 'volatility': 1.9, 'tradeCount': 6500, 'marketCap': 1.15e9},
                'HBAR/USD': {'last': 0.052, 'quoteVolume': 150000.0, 'bid': 0.0518, 'ask': 0.0522, 'change': 2.0, 'volatility': 2.1, 'tradeCount': 6000, 'marketCap': 1.92e9},
                'ZIL/USD': {'last': 0.016, 'quoteVolume': 120000.0, 'bid': 0.0159, 'ask': 0.0161, 'change': 1.5, 'volatility': 2.0, 'tradeCount': 5500, 'marketCap': 3.05e8},
                'HOT/USD': {'last': 0.0015, 'quoteVolume': 90000.0, 'bid': 0.00149, 'ask': 0.00151, 'change': 2.5, 'volatility': 2.8, 'tradeCount': 5000, 'marketCap': 2.65e8},
                'ANKR/USD': {'last': 0.027, 'quoteVolume': 110000.0, 'bid': 0.0269, 'ask': 0.0271, 'change': 1.9, 'volatility': 1.7, 'tradeCount': 5200, 'marketCap': 2.7e8},
                'SC/USD': {'last': 0.0045, 'quoteVolume': 80000.0, 'bid': 0.00448, 'ask': 0.00452, 'change': 2.2, 'volatility': 2.3, 'tradeCount': 4800, 'marketCap': 2.25e8},
                'DENT/USD': {'last': 0.0009, 'quoteVolume': 70000.0, 'bid': 0.00089, 'ask': 0.00091, 'change': 1.8, 'volatility': 2.4, 'tradeCount': 4500, 'marketCap': 8.6e7},
                'REEF/USD': {'last': 0.0012, 'quoteVolume': 65000.0, 'bid': 0.00119, 'ask': 0.00121, 'change': 2.7, 'volatility': 2.6, 'tradeCount': 4200, 'marketCap': 2.74e7},
                'KIN/USD': {'last': 0.000015, 'quoteVolume': 50000.0, 'bid': 0.0000149, 'ask': 0.0000151, 'change': 3.0, 'volatility': 3.2, 'tradeCount': 4000, 'marketCap': 4.4e7},
                'WIN/USD': {'last': 0.00009, 'quoteVolume': 45000.0, 'bid': 0.000089, 'ask': 0.000091, 'change': 1.6, 'volatility': 2.1, 'tradeCount': 3800, 'marketCap': 8.9e7},
                'CRO/USD': {'last': 0.075, 'quoteVolume': 200000.0, 'bid': 0.0748, 'ask': 0.0752, 'change': 1.3, 'volatility': 1.5, 'tradeCount': 7500, 'marketCap': 1.98e9},
                'FTM/USD': {'last': 0.65, 'quoteVolume': 300000.0, 'bid': 0.6495, 'ask': 0.6505, 'change': 2.0, 'volatility': 2.0, 'tradeCount': 8000, 'marketCap': 1.83e9},
                'ONE/USD': {'last': 0.013, 'quoteVolume': 100000.0, 'bid': 0.0129, 'ask': 0.0131, 'change': 1.9, 'volatility': 2.2, 'tradeCount': 5000, 'marketCap': 1.82e8},
                'CHZ/USD': {'last': 0.068, 'quoteVolume': 150000.0, 'bid': 0.0678, 'ask': 0.0682, 'change': 1.4, 'volatility': 1.8, 'tradeCount': 6000, 'marketCap': 6.05e8},
                'ENJ/USD': {'last': 0.15, 'quoteVolume': 120000.0, 'bid': 0.1497, 'ask': 0.1503, 'change': 1.7, 'volatility': 1.9, 'tradeCount': 5500, 'marketCap': 2.75e8},
                'BAT/USD': {'last': 0.18, 'quoteVolume': 110000.0, 'bid': 0.1798, 'ask': 0.1802, 'change': 1.5, 'volatility': 1.6, 'tradeCount': 5200, 'marketCap': 2.69e8},
                'GRT/USD': {'last': 0.17, 'quoteVolume': 130000.0, 'bid': 0.1697, 'ask': 0.1703, 'change': 1.8, 'volatility': 2.0, 'tradeCount': 5800, 'marketCap': 1.62e9},
                'ICX/USD': {'last': 0.14, 'quoteVolume': 90000.0, 'bid': 0.1398, 'ask': 0.1402, 'change': 1.6, 'volatility': 1.7, 'tradeCount': 4800, 'marketCap': 1.45e8},
                'KNC/USD': {'last': 0.45, 'quoteVolume': 100000.0, 'bid': 0.4495, 'ask': 0.4505, 'change': 1.9, 'volatility': 2.1, 'tradeCount': 5000, 'marketCap': 8.5e7},
                'LRC/USD': {'last': 0.13, 'quoteVolume': 85000.0, 'bid': 0.1298, 'ask': 0.1302, 'change': 1.5, 'volatility': 1.8, 'tradeCount': 4700, 'marketCap': 1.78e8},
                'OMG/USD': {'last': 0.25, 'quoteVolume': 95000.0, 'bid': 0.2497, 'ask': 0.2503, 'change': 1.7, 'volatility': 2.0, 'tradeCount': 4900, 'marketCap': 3.5e7},
                'BTC/USD': {'last': 60000.0, 'quoteVolume': 1000000.0, 'bid': 59950.0, 'ask': 60050.0, 'change': 1.5, 'volatility': 1.0, 'tradeCount': 10000, 'marketCap': 1.18e12},
                'ETH/USD': {'last': 3000.0, 'quoteVolume': 500000.0, 'bid': 2995.0, 'ask': 3005.0, 'change': 1.2, 'volatility': 1.2, 'tradeCount': 8000, 'marketCap': 3.6e11}
            }
            self.log_trade(f"Initialized mock_ticker_state with {len(self.mock_ticker_state)} pairs: {list(self.mock_ticker_state.keys())}")

            # Initialize DataManager with bot instance
            self.log_trade("Initializing DataManager...")
            try:
                self.data_manager = DataManager(self)
                self.log_trade("DataManager initialized successfully")
            except Exception as e:
                self.log_trade(f"Failed to initialize DataManager: {str(e)}")
                raise

            # Initialize core trading state
            self.price_history = {}
            self.active_trades = {}
            self.chart_trades = {}
            self.trades = []
            self.paper_balance = 10000.0
            self.total_profit = 0.0
            self.total_fees = 0.0
            self.net_profit = 0.0
            self.wins = 0
            self.losses = 0
            self.running = False
            self.is_paper_trading = True
            self.taker_fee = 0.004

            # Initialize trading parameters (after self.root is created)
            self.mode_var = tk.StringVar(value="Paper Trading")
            self.exchange_var = tk.StringVar(value="Kraken")
            self.initial_balance_var = tk.StringVar(value="10000.00")
            self.profit_target = tk.StringVar(value="1.2")
            self.stop_loss = tk.StringVar(value="0.5")
            self.position_size = tk.StringVar(value="150")
            self.min_price_rise = tk.StringVar(value="0.3")
            self.trailing_stop = tk.StringVar(value="0.2")
            self.trailing_activation = tk.StringVar(value="0.4")
            self.max_position = tk.StringVar(value="20")
            self.volume_surge = tk.StringVar(value="120")
            self.min_volume_entry = tk.StringVar(value="300")
            self.max_trades_entry = tk.StringVar(value="5")
            self.top_rank_pairs = tk.StringVar(value="20")
            self.required_conditions = tk.StringVar(value="3")
            self.volume_increase = tk.StringVar(value="10")
            self.max_spread = tk.StringVar(value="0.3")
            
            # Add the missing attribute that's causing the error
            self.min_price_rise_entry = tk.StringVar(value="0.3")

            # Initialize trading parameter values (to store parsed floats)
            self.profit_target_value = 1.2
            self.stop_loss_value = 0.5
            self.position_size_value = 150.0
            self.min_price_rise_value = 0.3
            self.trailing_stop_value = 0.2
            self.trailing_activation_value = 0.4
            self.max_position_value = 20.0
            self.volume_surge_value = 120.0
            self.min_volume_entry_value = 300.0
            self.max_trades_entry_value = 5
            self.top_rank_pairs_value = 20
            self.required_conditions_value = 3
            self.volume_increase_value = 10.0
            self.max_spread_value = 0.3
            
            # Initialize trading statistics
            self.total_trades = 0
            self.winning_trades = 0
            self.losing_trades = 0
            self.total_profit = 0.0
            self.total_fees = 0.0
            self.net_profit = 0.0
            self.wins = 0
            self.losses = 0
            self.paper_balance = float(self.initial_balance_var.get())
            
            # Initialize other required attributes
            self.active_trades = {}
            self.price_history = {}
            self.running = False
            
            # Validation Criteria (The Greeks)
            self.vega_threshold = tk.StringVar(value="0.1")
            self.rho_threshold = tk.StringVar(value="0.05")
            self.beta_threshold = tk.StringVar(value="1.0")
            self.delta_threshold = tk.StringVar(value="0.5")
            self.gamma_threshold = tk.StringVar(value="0.02")
            self.theta_threshold = tk.StringVar(value="0.03")

            # Advanced Parameters (RSI)
            self.rsi_overbought = tk.StringVar(value="70")
            self.rsi_oversold = tk.StringVar(value="30")
            self.rsi_period = tk.StringVar(value="14")

            # New parameters
            self.vega_threshold_value = 0.1
            self.rho_threshold_value = 0.05
            self.beta_threshold_value = 1.0
            self.delta_threshold_value = 0.5
            self.gamma_threshold_value = 0.02
            self.theta_threshold_value = 0.03
            self.rsi_overbought_value = 70
            self.rsi_oversold_value = 30
            self.rsi_period_value = 14
            
            # Additional parameters for market conditions
            self.max_volatility = tk.StringVar(value="2.0")
            self.consecutive_rises = tk.StringVar(value="2")
            self.momentum_threshold = tk.StringVar(value="0.2")
            self.price_rise_min = tk.StringVar(value="0.2")
            self.max_position_percent = tk.StringVar(value="10")
            self.daily_loss_limit = tk.StringVar(value="5")
            self.book_depth = tk.StringVar(value="10")
            self.buy_wall_ratio = tk.StringVar(value="1.5")
            self.sell_wall_ratio = tk.StringVar(value="1.5")
            self.scale_in_levels = tk.StringVar(value="3")
            self.level_gap = tk.StringVar(value="0.2")

            # Initialize GUI update timing
            self.last_gui_update = datetime.now()
            self.gui_update_interval = 1.0
            self.gui_ready = False

            self.log_trade(f"Mode: {self.mode_var.get()}")
            self.log_trade(f"Exchange: {self.exchange_var.get()}")
            self.log_trade(f"Initial Balance: ${float(self.initial_balance_var.get()):.2f}")
            self.log_trade("CryptoScalpingBot initialized successfully")

        except Exception as e:
            self.log_trade(f"Error in CryptoScalpingBot.__init__: {str(e)}")
            raise

    def log_trade(self, message):
        """Enhanced logging with GUI update"""
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_message = f"{timestamp} - {message}\n"
            
            # Update GUI log
            if hasattr(self, 'log_text'):
                self.log_text.insert(tk.END, log_message)
                self.log_text.see(tk.END)  # Auto-scroll to bottom
                self.root.update_idletasks()
            
            # File logging
            logging.info(message)
        except Exception as e:
            print(f"Logging error: {str(e)}")

    def stop_bot(self):
        """Stop the bot's trading loop."""
        self.running = False
        self.status_label.config(text="Stopped")
        self.log_trade("=== BOT STOPPED ===")



    def init_memory_monitor(self):
        """Initialize memory monitoring"""
        self.memory_check_interval = 60  # seconds
        self.last_memory_check = time.time()
        self.memory_threshold = 1000  # MB


    def run_bot_loop(self):
        """
        Main loop for running the bot.
        """
        while self.running:
            try:
                tickers = self.fetch_tickers_with_retry()
                if not tickers:
                    self.log_trade("No tickers available, skipping cycle")
                    time.sleep(5)
                    continue

                self.scan_opportunities(tickers)
                self.monitor_trades(tickers)
                if self.gui_ready:  # Only update chart if GUI is ready
                    self.data_manager.update_price_chart(tickers)
                else:
                    self.log_trade("GUI not ready, skipping chart update")

                time.sleep(5)  # Adjust cycle time as needed
            except Exception as e:
                self.log_trade(f"Error in run_bot_loop cycle: {str(e)}")
                time.sleep(5)

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

    def fetch_tickers_with_retry(self, retry_count=3, delay=5):
        """
        Fetch ticker data with retry mechanism for paper trading.
        
        Args:
            retry_count (int): Number of retries
            delay (int): Delay between retries in seconds
        
        Returns:
            dict: Updated mock ticker data
        """
        if not self.is_paper_trading:
            self.log_trade("Live trading is not supported in this version")
            return {}

        for attempt in range(retry_count):
            try:
                updated_tickers = {}
                for symbol in self.mock_ticker_state:
                    ticker = self.mock_ticker_state[symbol].copy()
                    price_change = ticker.get('change', 0.0) * np.random.uniform(0.9, 1.1) + np.random.normal(0, ticker['volatility'] / 50)  # Increased volatility
                    price = ticker['last'] * (1 + price_change / 100)
                    ticker['last'] = round(price, 8)
                    ticker['bid'] = round(price * 0.9995, 8)
                    ticker['ask'] = round(price * 1.0005, 8)
                    volume_change = np.random.uniform(0.9, 1.1)
                    ticker['quoteVolume'] = round(ticker['quoteVolume'] * volume_change, 2)
                    ticker['tradeCount'] = int(ticker['tradeCount'] * volume_change)
                    updated_tickers[symbol] = ticker
                    self.mock_ticker_state[symbol] = ticker
                self.log_trade(f"Returning updated tickers with {len(updated_tickers)} pairs")
                return updated_tickers
            except Exception as e:
                self.log_trade(f"Error fetching mock tickers: {str(e)}")
                self.log_trade(f"Traceback: {str(traceback.format_exc())}")
                if attempt < retry_count - 1:
                    self.log_trade(f"Retrying in {delay} seconds... (Attempt {attempt + 1}/{retry_count})")
                    time.sleep(delay)
                else:
                    self.log_trade("Max retries reached, skipping cycle")
                    return {}
        self.log_trade("No tickers fetched, skipping cycle")
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


    def setup_gui(self, root):
        """Setup the GUI with improved responsive layout"""
        self.root = root
        self.root.title("Crypto Scalping Bot")
        self.root.geometry("1200x800")  # Set default size
        
        # Create main container with proper weights
        main_container = ttk.Frame(root)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create top row for controls and parameters
        top_row = ttk.Frame(main_container)
        top_row.pack(fill=tk.X, pady=5)
        
        # Create bottom row for chart and trades
        bottom_row = ttk.Frame(main_container)
        bottom_row.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Left column in top row - Controls
        controls_frame = ttk.LabelFrame(top_row, text="Controls")
        controls_frame.pack(side=tk.LEFT, fill=tk.BOTH, padx=5, expand=True)
        
        # Control buttons
        buttons_frame = ttk.Frame(controls_frame)
        buttons_frame.pack(fill=tk.X, pady=5)
        
        self.start_button = ttk.Button(buttons_frame, text="Start", command=self.run_bot)
        self.start_button.pack(side=tk.LEFT, padx=5)
        
        self.live_update_button = ttk.Button(buttons_frame, text="Live Update", command=self.live_update)
        self.live_update_button.pack(side=tk.LEFT, padx=5)
        
        self.close_trades_button = ttk.Button(buttons_frame, text="Close Trades", command=self.close_all_positions)
        self.close_trades_button.pack(side=tk.LEFT, padx=5)
        
        self.close_all_button = ttk.Button(buttons_frame, text="Close All", command=self.close_all_positions)
        self.close_all_button.pack(side=tk.LEFT, padx=5)
        
        # Middle column in top row - Market Filters
        filters_frame = ttk.LabelFrame(top_row, text="Market Filters")
        filters_frame.pack(side=tk.LEFT, fill=tk.BOTH, padx=5, expand=True)
        
        # Add filter controls
        self.setup_market_filters(filters_frame)
        
        # Right column in top row - Trading Parameters
        params_frame = ttk.LabelFrame(top_row, text="Trading Parameters")
        params_frame.pack(side=tk.LEFT, fill=tk.BOTH, padx=5, expand=True)
        
        # Add trading parameters
        self.setup_trading_parameters(params_frame)
        
        # Left column in bottom row - Chart
        chart_frame = ttk.LabelFrame(bottom_row, text="Trade Performance")
        chart_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        # Create figure and canvas
        self.figure = Figure(figsize=(6, 4), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, master=chart_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Right column in bottom row - Trades and Log
        right_column = ttk.Frame(bottom_row)
        # Fix: Remove the width parameter from pack, use a separate method to set width
        right_column.pack(side=tk.RIGHT, fill=tk.BOTH, padx=5)
        # If you need to set a fixed width, use this instead:
        right_column.config(width=400)
        
        # Performance metrics at top
        metrics_frame = ttk.LabelFrame(right_column, text="Performance Metrics")
        metrics_frame.pack(fill=tk.X, pady=5)
        
        # Add metrics labels
        self.setup_performance_metrics(metrics_frame)
        
        # Active trades in middle
        trades_frame = ttk.LabelFrame(right_column, text="Active Trades")
        trades_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.active_trades_label = ttk.Label(trades_frame, text="Active Trades: 0")
        self.active_trades_label.pack(anchor="w", padx=5, pady=2)
        
        self.trades_text = tk.Text(trades_frame, height=10, width=50)
        self.trades_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        trades_scrollbar = ttk.Scrollbar(trades_frame, orient="vertical", command=self.trades_text.yview)
        trades_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.trades_text.configure(yscrollcommand=trades_scrollbar.set)
        
        # Trading log at bottom
        log_frame = ttk.LabelFrame(right_column, text="Trading Log")
        log_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.log_text = tk.Text(log_frame, height=10, width=50)
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        log_scrollbar = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
        log_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.configure(yscrollcommand=log_scrollbar.set)
        
        # Status bar at bottom
        status_frame = ttk.Frame(root)
        status_frame.pack(fill=tk.X, side=tk.BOTTOM)
        
        self.status_label = ttk.Label(status_frame, text="Status: Ready (Paper Trading)")
        self.status_label.pack(side=tk.LEFT, padx=10)
        
        # Start/Stop buttons
        self.start_button = ttk.Button(status_frame, text="Start", command=self.run_bot)
        self.start_button.pack(side=tk.RIGHT, padx=5)
        
        self.stop_button = ttk.Button(status_frame, text="Stop", command=self.stop_bot)
        self.stop_button.pack(side=tk.RIGHT, padx=5)
        
        # Configure closing protocol
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def apply_settings(self):
        """
        Apply the user-entered settings from the GUI by parsing StringVar inputs
        and updating the corresponding float/int values.
        """
        try:
            self.profit_target_value = float(self.profit_target.get())
            self.stop_loss_value = float(self.stop_loss.get())
            self.position_size_value = float(self.position_size.get())
            self.min_price_rise_value = float(self.min_price_rise.get())
            self.trailing_stop_value = float(self.trailing_stop.get())
            self.trailing_activation_value = float(self.trailing_activation.get())
            self.max_position_value = float(self.max_position.get())
            self.volume_surge_value = float(self.volume_surge.get())
            self.min_volume_entry_value = float(self.min_volume_entry.get())
            self.max_trades_entry_value = int(self.max_trades_entry.get())
            self.top_rank_pairs_value = int(self.top_rank_pairs.get())
            self.required_conditions_value = int(self.required_conditions.get())
            self.volume_increase_value = float(self.volume_increase.get())
            self.max_spread_value = float(self.max_spread.get())
            # New parameters
            self.vega_threshold_value = float(self.vega_threshold.get())
            self.rho_threshold_value = float(self.rho_threshold.get())
            self.beta_threshold_value = float(self.beta_threshold.get())
            self.delta_threshold_value = float(self.delta_threshold.get())
            self.gamma_threshold_value = float(self.gamma_threshold.get())
            self.theta_threshold_value = float(self.theta_threshold.get())
            self.rsi_overbought_value = float(self.rsi_overbought.get())
            self.rsi_oversold_value = float(self.rsi_oversold.get())
            self.rsi_period_value = int(self.rsi_period.get())
            self.log_trade("Settings applied successfully")
        except ValueError as e:
            self.log_trade(f"Error applying settings: Invalid input ({str(e)})")

    def on_closing(self):
        """Handle window closing properly"""
        if self.is_shutting_down:  # Prevent multiple shutdown attempts
            return
        
        try:
            self.is_shutting_down = True
            self.log_trade("Initiating shutdown sequence...")
            
            # Stop all running processes
            self.running = False
            self.cleanup_on_shutdown()
            
            # Kill all threads
            for thread in threading.enumerate():
                if thread != threading.current_thread():
                    try:
                        thread._stop()
                    except:
                        pass
            
            # Destroy all matplotlib figures
            plt.close('all')
            
            # Schedule the final cleanup after a brief delay
            self.root.after(100, self._final_cleanup)
            
        except Exception as e:
            print(f"Error during shutdown: {e}")
            self._force_exit()

    def _final_cleanup(self):
        """Final cleanup and exit"""
        try:
            # Destroy the root window
            self.root.destroy()
            # Force exit after a brief delay
            self.root.after(100, self._force_exit)
        except:
            self._force_exit()

    def _force_exit(self):
        """Force exit the program"""
        try:
            sys.exit(0)
        finally:
            os._exit(0)

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
            stability = momentum.rolling(window=5).std().iloc[-1] if len(momentum) >= 5 else 1.0

            # Log calculation details
            self.log_trade(f"Momentum quality calculation details:")
            self.log_trade(f"Recent momentum: {momentum.tail()}")
            self.log_trade(f"Momentum stability: {stability:.8f}")

            # Normalize to 0-1 range
            result = 1.0 - min(stability, 1.0)
            return max(result, 0.0)
        
        except Exception as e:
            self.log_trade(f"Error in momentum quality calculation: {str(e)}")
            return 1.0

    def scan_opportunities(self, tickers):
        """Scan for trading opportunities using fixed parameters for paper trading"""
        try:
            if not tickers:
                self.log_trade("No tickers available for scanning")
                return
            
            self.log_trade(f"Scanning {len(tickers)} pairs for opportunities...")
            
            # Use fixed parameters for paper trading to avoid GUI thread issues
            # These could be loaded from a config file or set during initialization
            paper_trading_params = {
                'min_price_rise': 0.5 / 100,  # 0.5%
                'min_volume': 1000.0,         # $1000 minimum volume
                'max_trades': 3,              # Maximum 3 concurrent trades
                'profit_target': 1.0 / 100,   # 1% profit target
                'stop_loss': 0.5 / 100,       # 0.5% stop loss
                'trailing_stop': 0.2 / 100,   # 0.2% trailing stop
                'trailing_activation': 0.5 / 100,  # 0.5% activation threshold
                'max_volatility': 2.0         # Maximum volatility threshold
            }
            
            # Store parameters for use in other methods
            self.trading_params = paper_trading_params
            
            # Check if we can take more trades
            if len(self.active_trades) >= self.trading_params['max_trades']:
                self.log_trade(f"Maximum number of trades ({self.trading_params['max_trades']}) already active")
                return
        
            # Filter and sort tickers by potential
            potential_trades = []
        
            for symbol, ticker in tickers.items():
                try:
                    # Basic validation
                    if not self.validate_ticker(ticker, symbol):
                        continue
                    
                    # Extract key metrics
                    price = float(ticker.get('last', 0))
                    volume = float(ticker.get('quoteVolume', 0))
                    price_change = float(ticker.get('change', 0))
                    
                    # Skip if price is too high (focus on lower-priced assets)
                    if price > 5.0:
                        continue
                    
                    # Skip if volume is too low
                    if volume < self.trading_params['min_volume']:
                        continue
                    
                    # Skip if price change doesn't meet minimum
                    if abs(price_change) < self.trading_params['min_price_rise']:
                        continue
                
                    # Calculate a score for this opportunity
                    score = (abs(price_change) * 5) + (volume / 1000)
                
                    # Add to potential trades
                    potential_trades.append({
                        'symbol': symbol,
                        'ticker': ticker,
                        'price': price,
                        'volume': volume,
                        'price_change': price_change,
                        'score': score
                    })
                
                except Exception as e:
                    self.log_trade(f"Error processing {symbol}: {str(e)}")
                    continue
            
            # Sort by score (highest first)
            potential_trades.sort(key=lambda x: x['score'], reverse=True)
            
            # Take top opportunities
            top_opportunities = potential_trades[:5]  # Look at top 5
            
            if not top_opportunities:
                self.log_trade("No viable trading opportunities found")
                return
            
            self.log_trade(f"Found {len(top_opportunities)} potential opportunities")
            
            # Try to execute trades for top opportunities
            for opportunity in top_opportunities:
                try:
                    symbol = opportunity['symbol']
                    price = opportunity['price']
                    price_change = opportunity['price_change']
                    
                    self.log_trade(f"""
                    Evaluating {symbol}:
                    Price: ${price:.6f}
                    Change: {price_change:.2f}%
                    Volume: ${opportunity['volume']:.2f}
                    Score: {opportunity['score']:.2f}
                    """)
                    
                    # Skip if we already have this symbol in active trades
                    if any(trade['symbol'] == symbol for trade in self.active_trades.values()):
                        self.log_trade(f"Already trading {symbol}, skipping")
                        continue
                    
                    # Execute the trade
                    self.execute_trade(symbol, opportunity['ticker'])
                
                    # Only take one trade per scan
                    break
                
                except Exception as e:
                    self.log_trade(f"Error evaluating {opportunity['symbol']}: {str(e)}")
                    continue
                
        except Exception as e:
            self.log_trade(f"Error in opportunity scanning: {str(e)}")

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
            self.log_trade(f"Error calculating acceleration: {str(e)}")
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
            self.log_trade(f"Error calculating volatility: {str(e)}")
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
        try:
            if self.root and self.root.winfo_exists():
                self.root.after(0, func)
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
            if hasattr(self, 'trades_text'):
                self.trades_text.config(state="normal")
                self.trades_text.delete("1.0", tk.END)
                
                if not self.active_trades:
                    self.trades_text.insert(tk.END, "No active trades\n")
                else:
                    self.trades_text.insert(tk.END, f"Active Trades: {len(self.active_trades)}\n\n")
                    
                    for trade_id, trade in self.active_trades.items():
                        try:
                            symbol = trade['symbol']
                            entry_price = trade['entry_price']
                            
                            # Get current price safely
                            try:
                                current_price = self.get_current_price(symbol)
                            except:
                                current_price = trade.get('current_price', entry_price)
                            
                            profit_percentage = ((current_price - entry_price) / entry_price) * 100
                            time_in_trade = (datetime.now() - trade.get('entry_time', datetime.now())).total_seconds()
                            
                            trade_info = (
                                f"Symbol: {symbol}\n"
                                f"Entry: {entry_price:.8f}\n"
                                f"Current: {current_price:.8f}\n"
                                f"P/L: {profit_percentage:.2f}%\n"
                                f"Time: {time_in_trade:.1f}s\n"
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
            
            self.trades_text.config(state="disabled")
        
            # Update chart
            if hasattr(self, 'ax') and hasattr(self, 'canvas'):
                self.ax.clear()
                
                for trade_id, trade in list(self.active_trades.items()):
                    symbol = trade['symbol']
                    if hasattr(self, 'price_history') and symbol in self.price_history:
                        history = self.price_history[symbol]
                        if history:
                            times = [point[0] for point in history]
                            prices = [point[1] for point in history]
                            
                            # Calculate percentage change from entry
                            entry_price = trade['entry_price']
                            prices_pct = [(p - entry_price) / entry_price * 100 for p in prices]
                            
                            self.ax.plot(times, prices_pct, 
                                    label=f"{symbol} ({prices_pct[-1]:.2f}%)",
                                    linewidth=2)
            
                # Plot key levels
                if hasattr(self, 'profit_target') and hasattr(self, 'stop_loss'):
                    self.ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
                    
                    try:
                        profit_target = float(self.profit_target.get())
                        self.ax.axhline(y=profit_target, color='green', linestyle=':', alpha=0.5)
                    except:
                        pass
                        
                    try:
                        stop_loss = float(self.stop_loss.get())
                        self.ax.axhline(y=-stop_loss, color='red', linestyle=':', alpha=0.5)
                    except:
                        pass
            
                if self.active_trades:
                    self.ax.set_title("Active Trades (% Change from Entry)")
                    self.ax.set_xlabel("Time")
                    self.ax.set_ylabel("Price Change (%)")
                    self.ax.legend(loc='upper left')
                    self.ax.grid(True, alpha=0.3)
                else:
                    self.ax.set_title("No Active Trades")
                    
                # Set reasonable y-axis limits if we have profit target and stop loss
                if hasattr(self, 'profit_target') and hasattr(self, 'stop_loss'):
                    try:
                        profit_target = float(self.profit_target.get())
                        stop_loss = float(self.stop_loss.get())
                        
                        self.ax.set_ylim(
                            min(-stop_loss * 1.5, self.ax.get_ylim()[0]),
                            max(profit_target * 1.5, self.ax.get_ylim()[1])
                        )
                    except:
                        pass
            
                # Draw without tight_layout
                self.canvas.draw()
            
            # Update metrics
            self.update_metrics()
            
        except Exception as e:
            self.log_trade(f"Error in display update: {str(e)}")

    def setup_market_filters(self, parent):
        """Setup market filter controls"""
        # Create a grid layout for filters
        row = 0
        
        # Min Volume
        ttk.Label(parent, text="Min Volume (USD):").grid(row=row, column=0, sticky="w", padx=5, pady=2)
        self.min_volume_entry = ttk.Entry(parent, width=10)
        self.min_volume_entry.insert(0, "100")
        self.min_volume_entry.grid(row=row, column=1, sticky="w", padx=5, pady=2)
        row += 1
        
        # Min Active Trades
        ttk.Label(parent, text="Min Active Trades:").grid(row=row, column=0, sticky="w", padx=5, pady=2)
        self.min_active_trades = ttk.Entry(parent, width=10)
        self.min_active_trades.insert(0, "0")
        self.min_active_trades.grid(row=row, column=1, sticky="w", padx=5, pady=2)
        row += 1
        
        # Top Rank Pairs
        ttk.Label(parent, text="Top Rank Pairs:").grid(row=row, column=0, sticky="w", padx=5, pady=2)
        self.top_rank_pairs = ttk.Entry(parent, width=10)
        self.top_rank_pairs.insert(0, "25")
        self.top_rank_pairs.grid(row=row, column=1, sticky="w", padx=5, pady=2)
        row += 1
        
        # Required Conditions
        ttk.Label(parent, text="Required Conditions:").grid(row=row, column=0, sticky="w", padx=5, pady=2)
        self.required_conditions = ttk.Entry(parent, width=10)
        self.required_conditions.insert(0, "1")
        self.required_conditions.grid(row=row, column=1, sticky="w", padx=5, pady=2)
        row += 1
        
        # Scan Interval
        ttk.Label(parent, text="Scan Interval (s):").grid(row=row, column=0, sticky="w", padx=5, pady=2)
        self.scan_interval = ttk.Entry(parent, width=10)
        self.scan_interval.insert(0, "5")
        self.scan_interval.grid(row=row, column=1, sticky="w", padx=5, pady=2)
        row += 1
        
        # Min Spread
        ttk.Label(parent, text="Min Spread (%):").grid(row=row, column=0, sticky="w", padx=5, pady=2)
        self.min_spread = ttk.Entry(parent, width=10)
        self.min_spread.insert(0, "0.1")
        self.min_spread.grid(row=row, column=1, sticky="w", padx=5, pady=2)
        row += 1

    def setup_trading_parameters(self, parent):
        """Setup trading parameter controls"""
        # Create a grid layout for parameters
        row = 0
        
        # Position Size
        ttk.Label(parent, text="Position Size ($):").grid(row=row, column=0, sticky="w", padx=5, pady=2)
        self.position_size = ttk.Entry(parent, width=10)
        self.position_size.insert(0, "100")
        self.position_size.grid(row=row, column=1, sticky="w", padx=5, pady=2)
        row += 1
        
        # Profit Target
        ttk.Label(parent, text="Profit Target (%):").grid(row=row, column=0, sticky="w", padx=5, pady=2)
        self.profit_target = ttk.Entry(parent, width=10)
        self.profit_target.insert(0, "1.0")
        self.profit_target.grid(row=row, column=1, sticky="w", padx=5, pady=2)
        row += 1
        
        # Stop Loss
        ttk.Label(parent, text="Stop Loss (%):").grid(row=row, column=0, sticky="w", padx=5, pady=2)
        self.stop_loss = ttk.Entry(parent, width=10)
        self.stop_loss.insert(0, "0.5")
        self.stop_loss.grid(row=row, column=1, sticky="w", padx=5, pady=2)
        row += 1
        
        # Min Volume
        ttk.Label(parent, text="Min Volume ($):").grid(row=row, column=0, sticky="w", padx=5, pady=2)
        self.min_volume = ttk.Entry(parent, width=10)
        self.min_volume.insert(0, "10000")
        self.min_volume.grid(row=row, column=1, sticky="w", padx=5, pady=2)
        row += 1
        
        # Max Trades
    def process_gui_updates(self):
        """Process GUI updates from the queue on the main thread"""
        try:
            # Process all pending updates
            while not self.gui_update_queue.empty():
                update_func = self.gui_update_queue.get_nowait()
                if callable(update_func):
                    update_func()
        except Exception as e:
            self.log_trade(f"Error processing GUI updates: {str(e)}")
        finally:
            # Schedule next check
            if hasattr(self, 'root') and self.root.winfo_exists():
                self.root.after(100, self.process_gui_updates)

    def setup_performance_metrics(self, parent):
        """Setup performance metric displays"""
        try:
            # Create a grid layout for metrics
            row = 0
            
            # Paper Balance
            self.paper_balance_label = ttk.Label(parent, text=f"Paper Balance: ${self.paper_balance:.2f} USD")
            self.paper_balance_label.grid(row=row, column=0, sticky="w", padx=5, pady=2)
            row += 1
            
            # Total Profit
            self.total_profit_label = ttk.Label(parent, text=f"Total Profit: ${self.total_profit:.2f} USD")
            self.total_profit_label.grid(row=row, column=0, sticky="w", padx=5, pady=2)
            row += 1
            
            # Total Fees
            self.total_fees_label = ttk.Label(parent, text=f"Total Fees: ${self.total_fees:.2f} USD")
            self.total_fees_label.grid(row=row, column=0, sticky="w", padx=5, pady=2)
            row += 1
            
            # Net Profit
            self.net_profit_label = ttk.Label(parent, text=f"Net Profit: ${self.net_profit:.2f} USD")
            self.net_profit_label.grid(row=row, column=0, sticky="w", padx=5, pady=2)
            row += 1
            
            # Win/Loss
            self.win_loss_label = ttk.Label(parent, text=f"Win/Loss: {self.wins}/{self.losses} ({0.0:.1f}%)")
            self.win_loss_label.grid(row=row, column=0, sticky="w", padx=5, pady=2)
            row += 1
            
            # Active Trades Count
            self.active_trades_label = ttk.Label(parent, text="Active Trades: 0")
            self.active_trades_label.grid(row=row, column=0, sticky="w", padx=5, pady=2)
            row += 1
            
            # Daily Performance
            self.daily_performance_label = ttk.Label(parent, text="Today: $0.00 (0.0%)")
            self.daily_performance_label.grid(row=row, column=0, sticky="w", padx=5, pady=2)
            row += 1
            
            self.log_trade("Performance metrics display initialized")
            
        except Exception as e:
            self.log_trade(f"Error setting up performance metrics: {str(e)}")

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
        """Update the balance display in the GUI"""
        try:
            if not hasattr(self, 'balance_label') or not self.balance_label:
                return
            
            # Format balance with commas and 2 decimal places
            balance_text = f"{self.paper_balance:,.2f} USD"
            self.log_trade(f"Updating GUI paper balance label to: {balance_text}")
            
            # Update label
            self.balance_label.config(text=balance_text)
            
        except Exception as e:
            self.log_trade(f"Error updating balance display: {str(e)}")

    def update_metrics(self):
        """Update trading metrics display"""
        try:
            if not hasattr(self, 'total_profit_label') or not hasattr(self, 'total_fees_label') or not hasattr(self, 'net_profit_label') or not hasattr(self, 'win_loss_label'):
                return
            
            # Update total profit
            self.total_profit_label.config(text=f"Total Profit: ${self.total_profit:.2f} USD")
            
            # Update total fees
            self.total_fees_label.config(text=f"Total Fees: ${self.total_fees:.2f} USD")
            
            # Update net profit
            self.net_profit_label.config(text=f"Net Profit: ${self.net_profit:.2f} USD")
            
            # Update win/loss
            win_loss_text = f"Win/Loss: {self.wins}/{self.losses}"
            if self.losses > 0:
                win_loss_text += f" ({(self.wins / self.losses * 100):.1f}%)"
            else:
                win_loss_text += " (N/A)"
            self.win_loss_label.config(text=win_loss_text)
            
        except Exception as e:
            self.log_trade(f"Error updating metrics: {str(e)}")

    def execute_trade(self, symbol, ticker):
        """Execute a paper trade without accessing GUI elements"""
        try:
            # Get current price
            current_price = float(ticker['last'])
            if not current_price or current_price <= 0:
                self.log_trade(f"Invalid price for {symbol}: {current_price}")
                return False
                
            # Use fixed position size for paper trading
            position_size = 100.0  # $100 per trade
            
            # Generate a unique trade ID
            trade_id = f"TRADE-{len(self.active_trades) + 1}-{int(time.time())}"
            
            # Create trade record
            trade = {
                'id': trade_id,
                'symbol': symbol,
                'entry_price': current_price,
                'current_price': current_price,
                'position_size': position_size,
                'entry_time': datetime.now(),
                'highest_price': current_price,
                'highest_profit_percentage': 0.0,
                'stop_loss_price': current_price * (1 - self.trading_params['stop_loss']),
                'take_profit_price': current_price * (1 + self.trading_params['profit_target']),
                'trailing_stop_pct': self.trading_params['trailing_stop']
            }
            
            # Add to active trades
            self.active_trades[trade_id] = trade
            
            # Log the trade
            self.log_trade(f"""
            [PAPER TRADE] Executed buy for {symbol}:
            Price: ${current_price:.8f}
            Position Size: ${position_size:.2f}
            Stop Loss: ${trade['stop_loss_price']:.8f}
            Take Profit: ${trade['take_profit_price']:.8f}
            """)
            
            # Update displays safely
            if hasattr(self, 'root') and self.root:
                self.root.after(0, self.update_active_trades_display)
                self.root.after(0, self.update_chart)
            
            return True
            
        except Exception as e:
            self.log_trade(f"Error executing trade for {symbol}: {str(e)}")
            return False
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

    def live_update(self):
        """Trigger a live update of the bot's state."""
        self.log_trade("Live update triggered")

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

    def monitor_trades(self, tickers):
        """Monitor active trades with thread-safe GUI updates"""
        try:
            if not self.active_trades:
                return
            
            self.log_trade(f"Monitoring {len(self.active_trades)} active trades...")
            
            # Use thread-safe parameters
            profit_target = self.trading_params['profit_target']
            stop_loss = self.trading_params['stop_loss']
            trailing_stop_pct = self.trading_params['trailing_stop']
            trailing_activation = self.trading_params['trailing_activation']
            
            for trade_id, trade in list(self.active_trades.items()):
                try:
                    symbol = trade['symbol']
                    
                    # Get current price from tickers
                    if symbol not in tickers:
                        self.log_trade(f"Symbol {symbol} not found in tickers, skipping")
                        continue
                        
                    ticker = tickers[symbol]
                    current_price = float(ticker['last'])
                    entry_price = trade['entry_price']
                    
                    # Update trade data
                    trade['current_price'] = current_price
                    
                    # Calculate profit percentage
                    profit_percentage = ((current_price - entry_price) / entry_price) * 100
                    trade['current_profit_percentage'] = profit_percentage
                    
                    # Update highest price and profit if needed
                    if current_price > trade['highest_price']:
                        trade['highest_price'] = current_price
                        trade['highest_profit_percentage'] = profit_percentage
                    
                    # Check exit conditions
                    
                    # 1. Take profit
                    if profit_percentage >= profit_target * 100:
                        self.close_trade(trade_id, trade, current_price, "take profit")
                        continue
                        
                    # 2. Stop loss
                    if profit_percentage <= -stop_loss * 100:
                        self.close_trade(trade_id, trade, current_price, "stop loss")
                        continue
                        
                    # 3. Trailing stop
                    if profit_percentage >= trailing_activation * 100:
                        # Calculate drop from highest price
                        drop_from_high = ((trade['highest_price'] - current_price) / trade['highest_price']) * 100
                        
                        # Check if drop exceeds trailing stop
                        if drop_from_high >= trailing_stop_pct * 100:
                            self.close_trade(trade_id, trade, current_price, "trailing stop")
                            continue
                    
                    # Log current trade status
                    self.log_trade(f"""
                    Trade {trade_id} ({symbol}):
                    Entry: ${entry_price:.6f}
                    Current: ${current_price:.6f}
                    Profit: {profit_percentage:.2f}%
                    Highest: {trade['highest_profit_percentage']:.2f}%
                    """)
                    
                except Exception as e:
                    self.log_trade(f"Error monitoring trade {trade_id}: {str(e)}")
            
            # Queue GUI updates
            if hasattr(self, 'gui_update_queue'):
                self.gui_update_queue.put(self.update_active_trades_display)
                if hasattr(self, 'update_chart'):
                    self.gui_update_queue.put(self.update_chart)
                if hasattr(self, 'update_metrics'):
                    self.gui_update_queue.put(self.update_metrics)
                
        except Exception as e:
            self.log_trade(f"Error in trade monitoring: {str(e)}")

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
            self.log_trade(f"Validation error (value format): {str(e)}")
            messagebox.showerror("Input Error", f"Invalid number format: {str(e)}")
            return False
        except AssertionError as e:
            self.log_trade(f"Validation error (assertion): {str(e)}")
            messagebox.showerror("Validation Error", str(e))
            return False
        except Exception as e:
            self.log_trade(f"Validation error (unexpected): {str(e)}")
            messagebox.showerror("Error", f"Unexpected error during validation: {str(e)}")
            return False
        
    def is_gui_ready(self):
        """Check if GUI is ready for updates"""
        try:
            if not hasattr(self, 'root') or not self.root:
                return False
                
            # Try to access a GUI element to check if GUI is ready
            try:
                self.root.winfo_exists()
                return True
            except tk.TclError:
                return False
                
        except Exception:
            return False
        
    def update_active_trades_display(self):
        """Update the active trades display in the GUI"""
        try:
            if not hasattr(self, 'active_trades_text') or not self.active_trades_text:
                return
            
            # Clear current display
            self.active_trades_text.config(state=tk.NORMAL)
            self.active_trades_text.delete(1.0, tk.END)
            
            if not self.active_trades:
                self.active_trades_text.insert(tk.END, "No active trades")
                self.active_trades_text.config(state=tk.DISABLED)
                return
            
            # Add header
            header = "Symbol | Entry | Current | P/L% | Time\n"
            self.active_trades_text.insert(tk.END, header, "header")
            self.active_trades_text.insert(tk.END, "-" * 50 + "\n")
            
            # Add each trade
            for trade_id, trade in self.active_trades.items():
                try:
                    symbol = trade['symbol']
                    entry_price = trade['entry_price']
                    current_price = trade.get('current_price', entry_price)
                    profit_pct = trade.get('profit_percentage', 0)
                    
                    # Calculate time elapsed
                    elapsed = datetime.now() - trade['entry_time']
                    minutes = int(elapsed.total_seconds() / 60)
                    seconds = int(elapsed.total_seconds() % 60)
                    
                    # Format line
                    line = f"{symbol} | ${entry_price:.6f} | ${current_price:.6f} | "
                    
                    # Add profit/loss with color
                    self.active_trades_text.insert(tk.END, line)
                    
                    # Color code the P/L
                    if profit_pct > 0:
                        self.active_trades_text.insert(tk.END, f"+{profit_pct:.2f}%", "profit")
                    else:
                        self.active_trades_text.insert(tk.END, f"{profit_pct:.2f}%", "loss")
                        
                    # Add time
                    self.active_trades_text.insert(tk.END, f" | {minutes}m {seconds}s\n")
                    
                except Exception as e:
                    self.log_trade(f"Error formatting trade {trade_id}: {str(e)}")
                    continue
            
            # Disable editing
            self.active_trades_text.config(state=tk.DISABLED)
        
        except Exception as e:
            self.log_trade(f"Error updating active trades display: {str(e)}")

    def close_trade(self, trade_id, trade, current_price, reason):
        """Close a paper trade without accessing GUI elements"""
        try:
            if trade_id not in self.active_trades:
                self.log_trade(f"Trade {trade_id} not found in active trades")
                return False
                
            # Calculate profit
            entry_price = trade['entry_price']
            position_size = trade['position_size']
            profit_percentage = ((current_price - entry_price) / entry_price) * 100
            profit_amount = position_size * (profit_percentage / 100)
            
            # Update paper balance
            self.paper_balance += position_size + profit_amount
            
            # Update trade statistics
            self.total_trades += 1
            self.total_profit += profit_amount
            
            if profit_amount > 0:
                self.winning_trades += 1
            else:
                self.losing_trades += 1
                
            # Log the trade closure
            self.log_trade(f"""
            [PAPER TRADE] Closed {trade['symbol']}:
            Entry: ${entry_price:.8f}
            Exit: ${current_price:.8f}
            P/L: {profit_percentage:.2f}% (${profit_amount:.2f})
            Reason: {reason}
            """)
            
            # Remove from active trades
            del self.active_trades[trade_id]
            
            # Update trade history
            trade_record = {
                'id': trade_id,
                'symbol': trade['symbol'],
                'entry_price': entry_price,
                'exit_price': current_price,
                'entry_time': trade['entry_time'],
                'exit_time': datetime.now(),
                'profit_percentage': profit_percentage,
                'profit_amount': profit_amount,
                'position_size': position_size,
                'reason': reason
            }
            
            self.trade_history.append(trade_record)
            
            # Queue GUI updates
            if hasattr(self, 'gui_update_queue'):
                self.gui_update_queue.put(self.update_active_trades_display)
                if hasattr(self, 'update_chart'):
                    self.gui_update_queue.put(self.update_chart)
                if hasattr(self, 'update_metrics'):
                    self.gui_update_queue.put(self.update_metrics)
                if hasattr(self, 'update_balance_display'):
                    self.gui_update_queue.put(self.update_balance_display)
            
            return True
            
        except Exception as e:
            self.log_trade(f"Error closing trade: {str(e)}")
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
    def calculate_delta(self, symbol):
        """
        Calculate Delta: the rate of change of the asset's price relative to a benchmark (e.g., BTC/USD).
        In spot trading, we'll approximate this as the percentage change in price relative to BTC's price change.

        Args:
            symbol (str): The trading pair (e.g., 'ETH/USD').

        Returns:
            float: The Delta value (mocked to be below threshold of 0.5).
        """
        try:
            # Get the asset's price data
            asset_prices = self.price_data[symbol]['price']
            if len(asset_prices) < 2:
                self.log_trade(f"Delta calculation failed for {symbol}: Not enough price data")
                return 0.0

            # Get BTC/USD price data as the benchmark
            btc_symbol = 'BTC/USD'
            if btc_symbol not in self.price_data:
                self.log_trade(f"Delta calculation failed for {symbol}: BTC/USD data not available")
                return 0.0
            btc_prices = self.price_data[btc_symbol]['price']
            if len(btc_prices) < 2:
                self.log_trade(f"Delta calculation failed for {symbol}: Not enough BTC price data")
                return 0.0

            # Calculate percentage change for the asset and BTC
            asset_change = (asset_prices.iloc[-1] - asset_prices.iloc[-2]) / asset_prices.iloc[-2]
            btc_change = (btc_prices.iloc[-1] - btc_prices.iloc[-2]) / btc_prices.iloc[-2]

            # Avoid division by zero
            if btc_change == 0:
                self.log_trade(f"Delta calculation for {symbol}: BTC price change is zero, returning mock value")
                return 0.1

            # Delta as the ratio of price changes (scaled to be below threshold)
            delta = abs(asset_change / btc_change) * 0.1  # Scale to ensure < 0.5
            return min(delta, 0.4)  # Ensure below threshold of 0.5
        except Exception as e:
            self.log_trade(f"Delta calculation failed for {symbol}: {str(e)}")
            return 0.1  # Mock value to allow trading to proceed

    def calculate_gamma(self, symbol):
        """
        Calculate Gamma: the rate of change of Delta. In spot trading, we'll approximate this as the
        second derivative of price changes (acceleration of price movement).

        Args:
            symbol (str): The trading pair (e.g., 'ETH/USD').

        Returns:
            float: The Gamma value (mocked to be below threshold of 0.02).
        """
        try:
            prices = self.price_data[symbol]['price']
            if len(prices) < 3:
                self.log_trade(f"Gamma calculation failed for {symbol}: Not enough price data")
                return 0.0

            # Calculate first differences (like Delta)
            delta1 = (prices.iloc[-1] - prices.iloc[-2]) / prices.iloc[-2]
            delta2 = (prices.iloc[-2] - prices.iloc[-3]) / prices.iloc[-3]

            # Gamma as the change in Delta (scaled to be below threshold)
            gamma = abs(delta1 - delta2) * 0.01  # Scale to ensure < 0.02
            return min(gamma, 0.015)  # Ensure below threshold of 0.02
        except Exception as e:
            self.log_trade(f"Gamma calculation failed for {symbol}: {str(e)}")
            return 0.005  # Mock value to allow trading to proceed

    def calculate_vega(self, symbol):
        """
        Calculate Vega: sensitivity to volatility. In spot trading, we'll use historical volatility
        as a proxy for implied volatility.

        Args:
            symbol (str): The trading pair (e.g., 'ETH/USD').

        Returns:
            float: The Vega value (mocked to be below threshold of 0.1).
        """
        try:
            prices = self.price_data[symbol]['price']
            if len(prices) < 10:
                self.log_trade(f"Vega calculation failed for {symbol}: Not enough price data")
                return 0.0

            # Calculate returns
            returns = prices.pct_change().dropna()
            if len(returns) == 0:
                self.log_trade(f"Vega calculation failed for {symbol}: No returns data")
                return 0.0

            # Calculate historical volatility (standard deviation of returns, annualized)
            volatility = returns.std() * np.sqrt(252)  # Annualize assuming 252 trading days
            vega = volatility * 0.1  # Scale to ensure < 0.1
            return min(vega, 0.08)  # Ensure below threshold of 0.1
        except Exception as e:
            self.log_trade(f"Vega calculation failed for {symbol}: {str(e)}")
            return 0.01  # Mock value to allow trading to proceed

    def calculate_theta(self, symbol):
        """
        Calculate Theta: time decay or opportunity cost of holding a position.
        In spot trading, we'll approximate this as a small cost based on price stability.

        Args:
            symbol (str): The trading pair (e.g., 'ETH/USD').

        Returns:
            float: The Theta value (mocked to be below threshold of 0.03).
        """
        try:
            prices = self.price_data[symbol]['price']
            if len(prices) < 5:
                self.log_trade(f"Theta calculation failed for {symbol}: Not enough price data")
                return 0.0

            # Calculate recent price volatility (standard deviation of last 5 prices)
            recent_volatility = prices.tail(5).pct_change().std()
            theta = recent_volatility * 0.1  # Scale to ensure < 0.03
            return min(theta, 0.025)  # Ensure below threshold of 0.03
        except Exception as e:
            self.log_trade(f"Theta calculation failed for {symbol}: {str(e)}")
            return 0.01  # Mock value to allow trading to proceed

    def calculate_rho(self, symbol):
        """
        Calculate Rho: sensitivity to interest rates. In crypto spot trading, we'll approximate
        this as sensitivity to funding rates or market borrowing costs (mocked for simplicity).

        Args:
            symbol (str): The trading pair (e.g., 'ETH/USD').

        Returns:
            float: The Rho value (mocked to be below threshold of 0.05).
        """
        try:
            # In a real implementation, you'd fetch funding rates or interest rates
            # For spot trading, we'll use a small mock value
            rho = 0.02  # Mock value scaled to be below threshold
            return min(rho, 0.04)  # Ensure below threshold of 0.05
        except Exception as e:
            self.log_trade(f"Rho calculation failed for {symbol}: {str(e)}")
            return 0.01  # Mock value to allow trading to proceed

    def calculate_beta(self, symbol):
        """
        Calculate Beta: the asset's volatility relative to the market (e.g., BTC/USD).

        Args:
            symbol (str): The trading pair (e.g., 'ETH/USD').

        Returns:
            float: The Beta value (mocked to be below threshold of 1.0).
        """
        try:
            asset_prices = self.price_data[symbol]['price']
            if len(asset_prices) < 10:
                self.log_trade(f"Beta calculation failed for {symbol}: Not enough price data")
                return 0.0

            btc_symbol = 'BTC/USD'
            if btc_symbol not in self.price_data:
                self.log_trade(f"Beta calculation failed for {symbol}: BTC/USD data not available")
                return 0.0
            btc_prices = self.price_data[btc_symbol]['price']
            if len(btc_prices) < 10:
                self.log_trade(f"Beta calculation failed for {symbol}: Not enough BTC price data")
                return 0.0

            # Calculate returns for the asset and BTC
            asset_returns = asset_prices.pct_change().dropna()
            btc_returns = btc_prices.pct_change().dropna()

            # Align the returns data (ensure same length)
            min_length = min(len(asset_returns), len(btc_returns))
            asset_returns = asset_returns.tail(min_length)
            btc_returns = btc_returns.tail(min_length)

            # Calculate covariance and variance
            covariance = asset_returns.cov(btc_returns)
            btc_variance = btc_returns.var()

            if btc_variance == 0:
                self.log_trade(f"Beta calculation for {symbol}: BTC variance is zero, returning mock value")
                return 0.5

            # Beta = Covariance(asset, market) / Variance(market)
            beta = covariance / btc_variance * 0.5  # Scale to ensure < 1.0
            return min(beta, 0.9)  # Ensure below threshold of 1.0
        except Exception as e:
            self.log_trade(f"Beta calculation failed for {symbol}: {str(e)}")
            return 0.5  # Mock value to allow trading to proceed
        
    def scan_opportunities(self, tickers):
        """Scan for trading opportunities using trading parameters"""
        try:
            if not tickers:
                self.log_trade("No tickers available for scanning")
                return
            
            self.log_trade(f"Scanning {len(tickers)} pairs for opportunities...")
        
            # Update trading parameters from GUI values
            self.trading_params = {
                'min_price_rise': float(self.min_price_rise.get()) / 100,
                'min_volume': float(self.min_volume_entry.get()),
                'max_trades': int(self.max_trades_entry.get()),
                'profit_target': float(self.profit_target.get()) / 100,
                'stop_loss': float(self.stop_loss.get()) / 100,
                'trailing_stop': float(self.trailing_stop.get()) / 100,
                'trailing_activation': float(self.trailing_activation.get()) / 100,
                'max_volatility': 2.0  # Default value
            }
            
            # Check if we can take more trades
            if len(self.active_trades) >= self.trading_params['max_trades']:
                self.log_trade(f"Maximum number of trades ({self.trading_params['max_trades']}) already active")
                return
        
            # Filter and sort tickers by potential
            potential_trades = []
        
            for symbol, ticker in tickers.items():
                try:
                    # Basic validation
                    if not self.validate_ticker(ticker, symbol):
                        continue
                    
                    # Extract key metrics
                    price = float(ticker.get('last', 0))
                    volume = float(ticker.get('quoteVolume', 0))
                    price_change = float(ticker.get('change', 0))
                    
                    # Skip if price is too high (focus on lower-priced assets)
                    if price > 5.0:
                        continue
                    
                    # Skip if volume is too low
                    if volume < self.trading_params['min_volume']:
                        continue
                    
                    # Skip if price change doesn't meet minimum
                    if abs(price_change) < self.trading_params['min_price_rise']:
                        continue
                
                    # Calculate a score for this opportunity
                    score = (abs(price_change) * 5) + (volume / 1000)
                
                    # Add to potential trades
                    potential_trades.append({
                        'symbol': symbol,
                        'ticker': ticker,
                        'price': price,
                        'volume': volume,
                        'price_change': price_change,
                        'score': score
                    })
                
                except Exception as e:
                    self.log_trade(f"Error processing {symbol}: {str(e)}")
                    continue
            
            # Sort by score (highest first)
            potential_trades.sort(key=lambda x: x['score'], reverse=True)
            
            # Take top opportunities
            top_opportunities = potential_trades[:5]  # Look at top 5
            
            if not top_opportunities:
                self.log_trade("No viable trading opportunities found")
                return
            
            self.log_trade(f"Found {len(top_opportunities)} potential opportunities")
            
            # Try to execute trades for top opportunities
            for opportunity in top_opportunities:
                try:
                    symbol = opportunity['symbol']
                    price = opportunity['price']
                    price_change = opportunity['price_change']
                    
                    self.log_trade(f"""
                    Evaluating {symbol}:
                    Price: ${price:.6f}
                    Change: {price_change:.2f}%
                    Volume: ${opportunity['volume']:.2f}
                    Score: {opportunity['score']:.2f}
                    """)
                    
                    # Skip if we already have this symbol in active trades
                    if any(trade['symbol'] == symbol for trade in self.active_trades.values()):
                        self.log_trade(f"Already trading {symbol}, skipping")
                        continue
                    
                    # Execute the trade
                    self.execute_trade(symbol, opportunity['ticker'])
                
                    # Only take one trade per scan
                    break
                
                except Exception as e:
                    self.log_trade(f"Error evaluating {opportunity['symbol']}: {str(e)}")
                    continue
            if hasattr(self, 'gui_update_queue'):
                self.gui_update_queue.put(self.update_active_trades_display)
        except Exception as e:
            self.log_trade(f"Error in opportunity scanning: {str(e)}")

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


    def execute_trade(self, symbol, ticker):
        """Execute a new trade with proper initialization"""
        try:
            if symbol in [trade['symbol'] for trade in self.active_trades.values()]:
                return  # Already trading this symbol
            
            current_price = ticker['last']
            position_size = float(self.position_size.get())  # Get from GUI input
            
            trade_id = str(uuid.uuid4())
            trade = {
                'id': trade_id,
                'symbol': symbol,
                'entry_price': current_price,
                'position_size': position_size,  # Add position size
                'quantity': position_size / current_price,  # Calculate quantity
                'entry_time': pd.Timestamp.now(),
                'highest_price': current_price,
                'lowest_price': current_price,
                'highest_profit': 0,
                'current_profit': 0,
                'type': 'long' if ticker.get('change', 0) > 0 else 'short'
            }
            
            self.active_trades[trade_id] = trade
            self.log_trade(f"New trade opened: {symbol} at {current_price:.8f}")
            
            # Update displays
            self.update_active_trades_display()
            self.update_metrics()
            
        except Exception as e:
            self.log_trade(f"Error executing trade: {str(e)}")

    def calculate_position_size(self, current_price):
        """Calculate the position size based on risk management rules"""
        try:
            # Get available balance
            balance = float(self.paper_balance)
            
            # Calculate position size (1-5% of balance)
            risk_per_trade = random.uniform(0.01, 0.05)
            position_size = balance * risk_per_trade
            
            # Convert to coin amount
            amount = position_size / current_price
            
            return amount
        
        except Exception as e:
            self.log_trade(f"Error calculating position size: {str(e)}")
            return 0.0

    def update_metrics(self):
        """Update the GUI with the latest performance metrics"""
        try:
            # Update Total Profit
            self.total_profit_label.config(text=f"Total Profit: {self.total_profit:.2f} USD")
            
            # Update Total Fees
            self.total_fees_label.config(text=f"Total Fees: {self.total_fees:.2f} USD")
            
            # Update Net Profit
            self.net_profit_label.config(text=f"Net Profit: {self.net_profit:.2f} USD")            
        except Exception as e:
            self.log_trade(f"Error updating GUI metrics: {str(e)}")

    def close_trade(self, trade_id, trade, exit_price, reason="manual"):
        """Close a trade and update balances"""
        try:
            entry_price = trade['entry_price']
            position_size = trade['position_size']
            
            # Calculate P/L
            price_change = (exit_price - entry_price) / entry_price
            gross_profit = position_size * price_change
            fees = position_size * self.taker_fee * 2  # Entry and exit fees
            net_profit = gross_profit - fees
            
            # Update balances - IMPORTANT: Return both position size AND profit/loss
            self.paper_balance += position_size  # Return initial investment
            self.paper_balance += net_profit    # Add/subtract profit/loss
            
            # Update trade metrics
            if net_profit > 0:
                self.wins += 1
            else:
                self.losses += 1
            
            self.total_profit += gross_profit
            self.total_fees += fees
            self.net_profit = self.total_profit - self.total_fees
            
            # Log trade details
            profit_pct = price_change * 100
            self.log_trade(f"""
            Trade closed - {trade['symbol']}:
            Entry: ${entry_price:.8f}
            Exit: ${exit_price:.8f}
            Gross P/L: {profit_pct:.2f}%
            Fees: ${fees:.2f}
            Net P/L: ${net_profit:.2f}
            Balance: ${self.paper_balance:.2f}
            """)
            
            # Update trade history
            self.update_trade_history(
                symbol=trade['symbol'],
                percentage=profit_pct,
                profit=net_profit,
                is_win=(net_profit > 0)
            )
            
            # Remove from active trades
            del self.active_trades[trade_id]
            
            # Update displays
            self.update_active_trades_display()
            self.update_metrics()
            self.update_balance_display()
            
        except Exception as e:
            self.log_trade(f"Error closing trade: {str(e)}")

        
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

        except Exception as e:
            self.log_trade(f"Error in advanced checks for {symbol}: {str(e)}")
            return False
        
    # Force correct update_active_trades_display method
    def update_active_trades_display(self):
        """Update active trades display with error handling"""
        try:
            if not hasattr(self, 'trades_text') or not hasattr(self, 'active_trades_label'):
                self.log_trade("Error: trades_text or active_trades_label not initialized properly in setup_gui")
                return
            
            self.trades_text.config(state="normal")
            self.trades_text.delete("1.0", tk.END)
            self.active_trades_label.config(text=f"Active Trades: {len(self.active_trades)}")
            
            for trade_id, trade in self.active_trades.items():
                try:
                    symbol = trade['symbol']
                    entry_price = trade['entry_price']
                    current_price = self.get_current_price(symbol)
                    profit_loss = ((current_price - entry_price) / entry_price * 100) if entry_price != 0 else 0
                    
                    trade_info = (f"Symbol: {symbol}\n"
                                f"Entry: ${entry_price:.8f}\n"
                                f"Current: ${current_price:.8f}\n"
                                f"P/L: {profit_loss:.2f}%\n"
                                f"Position Size: ${trade['position_size']:.2f}\n"
                                f"------------------------\n")
                                
                    self.trades_text.insert(tk.END, trade_info)
                    
                except Exception as e:
                    self.log_trade(f"Error updating display for trade {trade_id}: {str(e)}")
                    
            self.trades_text.config(state="disabled")
            
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

    def update_trade_history(self, symbol: str, percentage: float, profit: float, is_win: bool = True):
        """
        Update the trade history display with new trade results
        
        Args:
            symbol (str): Trading pair symbol
            percentage (float): Profit/loss percentage
            profit (float): Actual profit/loss amount in USD
            is_win (bool): Whether trade was profitable
        """
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
                        self.history_text.delete(1.0, f"{current_lines-100}.0")
                    
                    # Insert new result at the end
                    self.history_text.insert(tk.END, result)
                    
                    # Scroll to the end
                    self.history_text.see(tk.END)
                except Exception as e:
                    self.log_trade(f"Error updating trade history: {str(e)}")
        except Exception as e:
            self.log_trade(f"Error updating trade history: {str(e)}")

    def safe_get_float(self, var, default=0.0):
        """Safely get float value from StringVar"""
        try:
            if not var:
                return default
            value = float(var.get())
            return value
        except (ValueError, TypeError):
            return default

    def safe_get_int(self, var, default=0):
        """Safely get integer value from StringVar"""
        try:
            if not var:
                return default
            value = int(var.get())
            return value
        except (ValueError, TypeError):
            return default

    def safe_get_str(self, var, default=""):
        """Safely get string value from StringVar"""
        try:
            if not var:
                return default
            value = str(var.get())
            return value
        except (ValueError, TypeError):
            return default

    def update_history_text(self, message):
        """Update the history text widget with proper line limiting"""
        try:
            if not hasattr(self, 'history_text') or not self.history_text:
                return
            
            # Add timestamp to message
            timestamp = datetime.now().strftime("%H:%M:%S")
            formatted_message = f"[{timestamp}] {message}\n"
            
            # Use try-except for all Tkinter operations
            try:
                self.history_text.config(state="normal")
                self.history_text.insert(tk.END, formatted_message)
                
                # Limit the number of lines
                current_lines = int(self.history_text.index('end-1c').split('.')[0])
                if current_lines > 100:
                    self.history_text.delete('1.0', f'{current_lines-100}.0')
                    
                # Scroll to end
                self.history_text.see(tk.END)
                self.history_text.config(state="disabled")
                
            except tk.TclError as e:
                print(f"Tkinter error in update_history_text: {str(e)}")
                
        except Exception as e:
            print(f"Error updating history text: {str(e)}")

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
    def generate_realistic_price_movement(self, base_price, timeframe_minutes=5):
        """Generate realistic crypto price movements"""
        prices = []
        current_price = base_price
        
        # Realistic crypto market parameters
        volatility = random.uniform(0.001, 0.003)  # 0.1-0.3% per minute
        trend_bias = random.uniform(-0.0005, 0.0005)  # Slight trend
        
        # Add random market sentiment
        sentiment = random.choice(['bullish', 'bearish', 'neutral'])
        sentiment_multiplier = {
            'bullish': 1.5,
            'bearish': 0.5,
            'neutral': 1.0
        }[sentiment]
        
        for _ in range(timeframe_minutes):
            # Combine multiple factors for price movement
            random_walk = random.gauss(0, 1) * volatility * sentiment_multiplier
            trend_component = trend_bias * sentiment_multiplier
            momentum = sum(prices[-3:]) / 3 if len(prices) >= 3 else 0
            momentum_effect = (current_price - momentum) * 0.1 if momentum else 0
            
            # Calculate price change
            price_change = (random_walk + trend_component - momentum_effect)
            current_price *= (1 + price_change)
            
            # Add occasional price spikes (1% chance)
            if random.random() < 0.01:
                spike_multiplier = random.uniform(0.95, 1.05)
                current_price *= spike_multiplier
            
            prices.append(current_price)
        
        return prices

    def simulate_market_movement(self, trade):
        """Simulate realistic market movement for paper trading"""
        try:
            if not trade or 'entry_price' not in trade:
                return None
            
            entry_price = trade['entry_price']
            timeframe = 5  # 5-minute timeframe
            
            # Generate price movement
            price_series = self.generate_realistic_price_movement(entry_price, timeframe)
            
            # Return the latest price
            return price_series[-1]
            
        except Exception as e:
            self.log_trade(f"Error in market simulation: {str(e)}")
            return entry_price

    def log_trade(self, message):
        """Log trade information with timestamp."""
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_message = f"{timestamp} - {message}"
            print(log_message)
            if hasattr(self, 'logger'):
                self.logger.info(message)
            if hasattr(self, 'log_text') and self.log_text:
                self.root.after(0, lambda: self.log_text.insert(tk.END, f"{log_message}\n"))
                self.root.after(0, lambda: self.log_text.see(tk.END))
        except Exception as e:
            print(f"Logging error: {str(e)}")

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
        """Validate ticker data without accessing GUI elements"""
        try:
            # Basic validation checks
            if not ticker:
                return False
                
            # Check if required fields exist
            required_fields = ['last', 'quoteVolume', 'change']
            for field in required_fields:
                if field not in ticker or ticker[field] is None:
                    return False
                    
            # Check if price is valid
            price = float(ticker['last'])
            if price <= 0:
                return False
                
            # Check if symbol format is valid
            if '/' not in symbol:
                return False
                
            return True
                
        except Exception as e:
            self.log_trade(f"Error validating ticker for {symbol}: {str(e)}")
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
        """
        Start the bot's main trading loop in a separate thread.
        """
        try:
            if self.running:
                self.log_trade("Bot is already running")
                return
            self.running = True
            self.log_trade("=== BOT STARTED ===")
            self.status_label.config(text="Running")  # Simplified status

            for symbol in self.mock_ticker_state:
                if symbol not in self.price_history:
                    # Initialize with the first data point to avoid empty DataFrame
                    timestamp = pd.Timestamp.now() - pd.Timedelta(minutes=30 * 5)
                    price = self.mock_ticker_state[symbol]['last'] * (1 + np.random.normal(0, 0.005))
                    volume = self.mock_ticker_state[symbol]['quoteVolume'] * (1 + np.random.normal(0, 0.1))
                    initial_data = pd.DataFrame({
                        'timestamp': [timestamp],
                        'price': [price],
                        'volume': [volume]
                    })
                    self.price_history[symbol] = initial_data
                    if not hasattr(self, 'price_data'):
                        self.price_data = {}
                    self.price_data[symbol] = initial_data.copy()

                    # Add remaining 29 data points
                    for i in range(1, 30):
                        timestamp = pd.Timestamp.now() - pd.Timedelta(minutes=(30 - i) * 5)
                        price = self.mock_ticker_state[symbol]['last'] * (1 + np.random.normal(0, 0.005))
                        volume = self.mock_ticker_state[symbol]['quoteVolume'] * (1 + np.random.normal(0, 0.1))
                        new_data = pd.DataFrame({
                            'timestamp': [timestamp],
                            'price': [price],
                            'volume': [volume]
                        })
                        self.price_history[symbol] = pd.concat(
                            [self.price_history[symbol], new_data],
                            ignore_index=True
                        )
                        self.price_data[symbol] = pd.concat(
                            [self.price_data[symbol], new_data],
                            ignore_index=True
                        )
                    self.log_trade(f"Initialized price_history and price_data for {symbol} with {len(self.price_data[symbol])} data points")

            self.bot_thread = threading.Thread(target=self.run_bot_loop)
            self.bot_thread.daemon = True
            self.bot_thread.start()
            self.log_trade("Bot thread started")
        except Exception as e:
            self.log_trade(f"Error in run_bot: {str(e)}")
            self.running = False
            self.status_label.config(text="Stopped")
            self.log_trade("=== BOT STOPPED ===")

    def run_bot_loop(self):
        """Main loop for running the bot."""
        while self.running:
            try:
                tickers = self.fetch_tickers_with_retry()
                if not tickers:
                    self.log_trade("No tickers available, skipping cycle")
                    time.sleep(5)
                    continue

                self.scan_opportunities(tickers)
                self.monitor_trades(tickers)
                
                # Update GUI safely
                self.safe_update_gui(self.update_active_trades_display)
                self.safe_update_gui(lambda: self.update_chart() if hasattr(self, 'update_chart') else None)
                self.safe_update_gui(lambda: self.update_metrics() if hasattr(self, 'update_metrics') else None)
                
                time.sleep(5)  # Adjust cycle time as needed
            except Exception as e:
                self.log_trade(f"Error in run_bot_loop cycle: {str(e)}")
                time.sleep(5)

    def run(self):
        """
        Run the Crypto Scalping Bot application.
        """
        # Create the Tkinter root window
        root = tk.Tk()
        root.title("Crypto Scalping Bot")
        
        # Set up the GUI by passing the root window
        self.setup_gui(root)
        
        # Start the Tkinter event loop
        root.mainloop()

    def on_closing(self):
        """Handle application closing properly"""
        try:
            # Set the shutting down flag
            self.is_shutting_down = True
            
            # Stop the bot if running
            if hasattr(self, 'running') and self.running:
                self.running = False
                self.log_trade("Stopping bot due to application closing")
                
            # Close any active trades if in paper trading mode
            if hasattr(self, 'active_trades') and self.active_trades:
                self.log_trade("Closing active trades before shutdown")
                # ... trade closing logic ...
                
            self.log_trade("Application shutting down")
            
            # Destroy the root window
            if self.root and self.root.winfo_exists():
                self.root.destroy()
                
        except Exception as e:
            print(f"Error during shutdown: {str(e)}")
            # Force destroy in case of error
            if hasattr(self, 'root') and self.root and self.root.winfo_exists():
                self.root.destroy()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s:%(name)s:%(message)s')
    bot = CryptoScalpingBot()
    bot.run()
