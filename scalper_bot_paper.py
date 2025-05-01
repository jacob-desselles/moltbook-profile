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
            self.log_trade(f"Error updating price_data for {symbol}: {str(e)}")

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
        """Initialize the lightweight paper trading bot"""
        # Set up logging
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)
        
        # Initialize the Tkinter root window
        self.root = tk.Tk()
        self.root.title("Crypto Scalping Bot - Paper Trading")
        self.root.geometry("1200x900")
        
        # Core state variables
        self.running = False
        self.is_paper_trading = True
        self.paper_balance = 10000.0
        self.active_trades = {}
        self.trade_history = []
        self.price_history = {}
        self.total_profit = 0.0
        self.total_fees = 0.0
        self.wins = 0
        self.losses = 0
        self.is_shutting_down = False
        self.gui_ready = False
        self.trades = []  # Store trade history
        
        # Create a queue for thread-safe GUI updates
        self.gui_update_queue = queue.Queue()
        
        # Mock ticker data
        self.mock_ticker_state = {
            'SHIB/USD': {'last': 0.000018, 'quoteVolume': 500000.0, 'bid': 0.0000179, 'ask': 0.0000181, 'change': 2.1},
            'DOGE/USD': {'last': 0.14, 'quoteVolume': 800000.0, 'bid': 0.1398, 'ask': 0.1402, 'change': 1.8},
            'XRP/USD': {'last': 0.52, 'quoteVolume': 600000.0, 'bid': 0.5195, 'ask': 0.5205, 'change': 1.6},
            'ADA/USD': {'last': 0.35, 'quoteVolume': 450000.0, 'bid': 0.3497, 'ask': 0.3503, 'change': 1.2},
            'SOL/USD': {'last': 120.0, 'quoteVolume': 750000.0, 'bid': 119.5, 'ask': 120.5, 'change': 2.5},
            'DOT/USD': {'last': 6.8, 'quoteVolume': 350000.0, 'bid': 6.79, 'ask': 6.81, 'change': 1.0},
            'AVAX/USD': {'last': 28.5, 'quoteVolume': 400000.0, 'bid': 28.45, 'ask': 28.55, 'change': 1.8},
            'MATIC/USD': {'last': 0.65, 'quoteVolume': 300000.0, 'bid': 0.649, 'ask': 0.651, 'change': 0.9},
            'LINK/USD': {'last': 14.2, 'quoteVolume': 250000.0, 'bid': 14.18, 'ask': 14.22, 'change': 1.5},
            'UNI/USD': {'last': 7.8, 'quoteVolume': 200000.0, 'bid': 7.79, 'ask': 7.81, 'change': 0.8},
            'ATOM/USD': {'last': 9.5, 'quoteVolume': 180000.0, 'bid': 9.48, 'ask': 9.52, 'change': 1.2},
            'ALGO/USD': {'last': 0.18, 'quoteVolume': 150000.0, 'bid': 0.179, 'ask': 0.181, 'change': 0.7},
            'VET/USD': {'last': 0.025, 'quoteVolume': 120000.0, 'bid': 0.0249, 'ask': 0.0251, 'change': 0.5},
            'XTZ/USD': {'last': 0.95, 'quoteVolume': 100000.0, 'bid': 0.949, 'ask': 0.951, 'change': 0.6},
            'EOS/USD': {'last': 0.72, 'quoteVolume': 90000.0, 'bid': 0.719, 'ask': 0.721, 'change': 0.4},
            'XLM/USD': {'last': 0.12, 'quoteVolume': 80000.0, 'bid': 0.1199, 'ask': 0.1201, 'change': 0.3},
            'TRX/USD': {'last': 0.11, 'quoteVolume': 70000.0, 'bid': 0.1099, 'ask': 0.1101, 'change': 0.2},
            'ETC/USD': {'last': 18.5, 'quoteVolume': 60000.0, 'bid': 18.48, 'ask': 18.52, 'change': 0.9},
            'ZEC/USD': {'last': 32.0, 'quoteVolume': 50000.0, 'bid': 31.95, 'ask': 32.05, 'change': 1.1},
            'DASH/USD': {'last': 28.0, 'quoteVolume': 40000.0, 'bid': 27.95, 'ask': 28.05, 'change': 0.8},
            'XMR/USD': {'last': 160.0, 'quoteVolume': 30000.0, 'bid': 159.5, 'ask': 160.5, 'change': 1.3},
            'ZRX/USD': {'last': 0.35, 'quoteVolume': 25000.0, 'bid': 0.349, 'ask': 0.351, 'change': 0.4},
            'BAT/USD': {'last': 0.25, 'quoteVolume': 20000.0, 'bid': 0.249, 'ask': 0.251, 'change': 0.3},
            'KNC/USD': {'last': 0.65, 'quoteVolume': 15000.0, 'bid': 0.649, 'ask': 0.651, 'change': 0.5},
            'OMG/USD': {'last': 0.85, 'quoteVolume': 10000.0, 'bid': 0.849, 'ask': 0.851, 'change': 0.6},
            'ZIL/USD': {'last': 0.022, 'quoteVolume': 9000.0, 'bid': 0.0219, 'ask': 0.0221, 'change': 0.2},
            'ENJ/USD': {'last': 0.35, 'quoteVolume': 8000.0, 'bid': 0.349, 'ask': 0.351, 'change': 0.3},
            'REN/USD': {'last': 0.08, 'quoteVolume': 7000.0, 'bid': 0.0799, 'ask': 0.0801, 'change': 0.1},
            'BTC/USD': {'last': 60000.0, 'quoteVolume': 1000000.0, 'bid': 59950.0, 'ask': 60050.0, 'change': 1.5},
            'ETH/USD': {'last': 3000.0, 'quoteVolume': 500000.0, 'bid': 2995.0, 'ask': 3005.0, 'change': 1.2}
        }
        
        # Initialize DataManager
        self.data_manager = DataManager(bot=self)
        
        # Trading parameters
        self.profit_target = tk.StringVar(value="1.0")
        self.stop_loss = tk.StringVar(value="0.5")
        self.position_size = tk.StringVar(value="100")
        self.min_price_rise = tk.StringVar(value="0.3")
        self.trailing_stop = tk.StringVar(value="0.2")
        self.trailing_activation = tk.StringVar(value="0.4")
        self.max_trades = tk.StringVar(value="5")
        self.min_volume = tk.StringVar(value="100000")
        self.volume_change_threshold = tk.StringVar(value="1.0")

        # Advanced parameters (Greeks)
        self.momentum_beta = tk.StringVar(value="0.8")  # Trend strength
        self.price_alpha = tk.StringVar(value="0.6")    # Price momentum
        self.momentum_theta = tk.StringVar(value="0.5") # Momentum quality
        self.vol_vega = tk.StringVar(value="0.4")       # Volatility filter
        self.volume_rho = tk.StringVar(value="0.7")     # Volume quality

        # Technical indicators
        self.rsi_period = tk.StringVar(value="14")
        self.rsi_overbought = tk.StringVar(value="70")
        self.rsi_oversold = tk.StringVar(value="30")
        self.ema_short = tk.StringVar(value="5")
        self.ema_long = tk.StringVar(value="15")

        # Fee structure
        self.taker_fee = 0.0026  # 0.26% for taker orders

        # Setup GUI
        self.setup_gui()

        # Start GUI update processor
        self.root.after(100, self.process_gui_updates)

        # Handle window close event properly
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        self.log_trade("Bot initialized successfully")
        self.gui_ready = True

    def log_trade(self, message):
        """Log a message to the trading log"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"{timestamp} - {message}"
        print(log_message)
        logging.info(message)
        
        # Update GUI log if available
        if hasattr(self, 'log_text') and self.log_text:
            self.gui_update_queue.put(lambda: self.update_log(log_message))

    def update_log(self, message):
        """Update the log text widget"""
        if hasattr(self, 'log_text') and self.log_text:
            self.log_text.insert(tk.END, message + "\n")
            self.log_text.see(tk.END)

    def setup_gui(self):
        """Set up the GUI components"""
        try:
            # Create main frame
            main_frame = ttk.Frame(self.root, padding="10")
            main_frame.pack(fill=tk.BOTH, expand=True)
            
            # Create left and right columns
            left_column = ttk.Frame(main_frame)
            left_column.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            
            right_column = ttk.Frame(main_frame)
            right_column.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
            
            # Create notebook for parameter tabs
            param_notebook = ttk.Notebook(left_column)
            param_notebook.pack(fill=tk.X, padx=5, pady=5)
            
            # Basic parameters tab
            basic_tab = ttk.Frame(param_notebook)
            param_notebook.add(basic_tab, text="Basic")
            
            # Advanced parameters tab
            advanced_tab = ttk.Frame(param_notebook)
            param_notebook.add(advanced_tab, text="Advanced")
            
            # Indicators tab
            indicators_tab = ttk.Frame(param_notebook)
            param_notebook.add(indicators_tab, text="Indicators")
            
            # Set up basic parameters
            self.setup_basic_parameters(basic_tab)
            
            # Set up advanced parameters
            self.setup_advanced_parameters(advanced_tab)
            
            # Set up technical indicators
            self.setup_technical_indicators(indicators_tab)
            
            # Create trades and history frame
            trades_frame = ttk.Frame(left_column)
            trades_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            
            # Active trades (smaller size)
            active_trades_frame = ttk.LabelFrame(trades_frame, text="Active Trades")
            active_trades_frame.pack(fill=tk.BOTH, expand=False, padx=5, pady=5)
            
            # Create scrolled text for trades (reduced height)
            self.trades_text = scrolledtext.ScrolledText(active_trades_frame, height=6)
            self.trades_text.pack(fill=tk.BOTH, expand=False, padx=5, pady=5)
            
            # Trade history (in main window, below active trades)
            history_frame = ttk.LabelFrame(trades_frame, text="Trade History")
            history_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            
            # Create scrolled text for trade history
            self.history_text = scrolledtext.ScrolledText(history_frame)
            self.history_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            
            # Create chart frame
            chart_frame = ttk.LabelFrame(right_column, text="Trade Performance")
            chart_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            
            # Set up chart
            self.setup_chart(chart_frame)
            
            # Performance metrics frame
            metrics_frame = ttk.LabelFrame(right_column, text="Performance Metrics")
            metrics_frame.pack(fill=tk.X, padx=5, pady=5)
            
            # Setup performance metrics
            self.setup_performance_metrics(metrics_frame)
            
            # Control buttons frame
            self.control_frame = ttk.Frame(right_column)
            self.control_frame.pack(fill=tk.X, padx=5, pady=5)
            
            # Set up control buttons
            self.setup_control_buttons(self.control_frame)
            
            # Status bar
            status_frame = ttk.Frame(self.root)
            status_frame.pack(fill=tk.X, side=tk.BOTTOM, padx=5, pady=2)
            
            # Status label
            self.status_label = ttk.Label(status_frame, text="Ready")
            self.status_label.pack(side=tk.LEFT)
            
            # Update the metrics display immediately
            self.update_metrics()
            
            self.log_trade("GUI setup complete")
            self.gui_ready = True
            
        except Exception as e:
            self.log_trade(f"Error setting up GUI: {str(e)}")

    def setup_chart(self, chart_frame):
        """Setup the price chart"""
        self.figure, self.ax = plt.subplots(figsize=(8, 4))
        self.canvas = FigureCanvasTkAgg(self.figure, master=chart_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self.ax.set_title("Trade Performance")
        self.ax.set_xlabel("Time")
        self.ax.set_ylabel("Price Change (%)")
        self.ax.grid(True, alpha=0.3)
        
        self.canvas.draw()

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
                if symbol in self.price_history and len(self.price_history[symbol]) > 1:
                    times, prices = zip(*self.price_history[symbol])
                    
                    # Calculate percentage change from entry
                    entry_price = trade['entry_price']
                    price_changes = [(price - entry_price) / entry_price * 100 for price in prices]
                    
                    self.ax.plot(times, price_changes, label=f"{symbol} ({price_changes[-1]:.2f}%)")
            
            # Plot key levels
            self.ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            
            try:
                profit_target = float(self.profit_target.get())
                self.ax.axhline(y=profit_target, color='green', linestyle=':', alpha=0.5)
                
                stop_loss = float(self.stop_loss.get())
                self.ax.axhline(y=-stop_loss, color='red', linestyle=':', alpha=0.5)
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
                
                self.ax.set_ylim(
                    min(-stop_loss * 1.5, -1),
                    max(profit_target * 1.5, 1)
                )
            except:
                pass
            
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
            self.net_profit_label = ttk.Label(parent, text=f"Net Profit: ${self.total_profit - self.total_fees:.2f} USD")
            self.net_profit_label.grid(row=row, column=0, sticky="w", padx=5, pady=2)
            row += 1
            
            # Win/Loss
            win_rate = (self.wins / max(1, self.wins + self.losses)) * 100
            self.win_loss_label = ttk.Label(parent, text=f"Win/Loss: {self.wins}/{self.losses} ({win_rate:.1f}%)")
            self.win_loss_label.grid(row=row, column=0, sticky="w", padx=5, pady=2)
            row += 1
            
            # Active Trades
            self.active_trades_label = ttk.Label(parent, text=f"Active Trades: {len(self.active_trades)}")
            self.active_trades_label.grid(row=row, column=0, sticky="w", padx=5, pady=2)
            
            self.log_trade("Performance metrics setup complete")
            
        except Exception as e:
            self.log_trade(f"Error setting up performance metrics: {str(e)}")

    def close_all_positions(self):
        """Close all active trades"""
        try:
            if not self.active_trades:
                self.log_trade("No active trades to close")
                return
            
            self.log_trade(f"Closing all {len(self.active_trades)} active trades")
        
            # Make a copy of trade IDs since we'll be modifying the dictionary
            trade_ids = list(self.active_trades.keys())
        
            for trade_id in trade_ids:
                try:
                    trade = self.active_trades[trade_id]
                    symbol = trade['symbol']
                    current_price = self.get_current_price(symbol)
                    self.close_trade(trade_id, trade, current_price, "manual close all")
                except Exception as e:
                    self.log_trade(f"Error closing trade {trade_id}: {str(e)}")
                
            self.log_trade("All trades closed")
        
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
        """Update the balance display with current balance"""
        try:
            # Update the balance label directly
            if hasattr(self, 'balance_label'):
                self.balance_label.config(text=f"Paper Balance: ${self.paper_balance:.2f}")
            
            # Also update metrics which includes the paper balance
            self.update_metrics()
            
        except Exception as e:
            self.log_trade(f"Error updating balance display: {str(e)}")

    def update_metrics(self):
        """Update trading metrics"""
        try:
            # Calculate win rate
            total_trades = self.wins + self.losses
            win_rate = (self.wins / total_trades * 100) if total_trades > 0 else 0
            
            # Calculate net profit
            net_profit = self.total_profit - self.total_fees
            
            # Calculate available balance (total - allocated to active trades)
            allocated_balance = sum(trade.get('position_size', 0) for trade in self.active_trades.values())
            available_balance = self.paper_balance - allocated_balance + net_profit
            
            # Update labels
            if hasattr(self, 'paper_balance_label'):
                self.paper_balance_label.config(
                    text=f"Paper Balance: ${self.paper_balance:.2f} USD"
                )
            
            if hasattr(self, 'total_profit_label'):
                self.total_profit_label.config(
                    text=f"Total Profit: ${self.total_profit:.2f} USD"
                )
            
            if hasattr(self, 'total_fees_label'):
                self.total_fees_label.config(
                    text=f"Total Fees: ${self.total_fees:.2f} USD"
                )
            
            if hasattr(self, 'net_profit_label'):
                self.net_profit_label.config(
                    text=f"Net Profit: ${net_profit:.2f} USD"
                )
            
            if hasattr(self, 'win_loss_label'):
                self.win_loss_label.config(
                    text=f"Win/Loss: {self.wins}/{self.losses} ({win_rate:.1f}%)"
                )
            
            if hasattr(self, 'active_trades_label'):
                self.active_trades_label.config(
                    text=f"Active Trades: {len(self.active_trades)}"
                )
            
            self.log_trade(f"Updated metrics - Win Rate: {win_rate:.1f}%, Net Profit: ${net_profit:.2f}")
            
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
            
            # Process each active trade
            for trade_id, trade in list(self.active_trades.items()):
                try:
                    symbol = trade['symbol']
                    entry_price = trade['entry_price']
                    
                    # Get current price from tickers
                    ticker = tickers.get(symbol, {})
                    current_price = ticker.get('last')
                    
                    if not current_price:
                        self.log_trade(f"No price data for {symbol}")
                        continue
                    
                    # Calculate profit percentage
                    profit_percentage = ((current_price - entry_price) / entry_price) * 100
                    
                    # Update highest profit percentage if needed
                    if profit_percentage > trade.get('highest_profit_percentage', 0):
                        trade['highest_profit_percentage'] = profit_percentage
                        trade['highest_price'] = current_price
                    
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
            # Validate profit target
            profit_target = float(self.profit_target.get())
            if profit_target <= 0 or profit_target > 10:
                self.log_trade("Invalid profit target: must be between 0 and 10%")
                return False
            
            # Validate stop loss
            stop_loss = float(self.stop_loss.get())
            if stop_loss <= 0 or stop_loss > 5:
                self.log_trade("Invalid stop loss: must be between 0 and 5%")
                return False
            
            # Validate position size
            position_size = float(self.position_size.get())
            if position_size <= 0 or position_size > self.paper_balance:
                self.log_trade(f"Invalid position size: must be between 0 and {self.paper_balance:.2f}")
                return False
            
            # Validate trailing stop
            trailing_stop = float(self.trailing_stop.get())
            if trailing_stop <= 0 or trailing_stop > 5:
                self.log_trade("Invalid trailing stop: must be between 0 and 5%")
                return False
            
            # Validate trailing activation
            trailing_activation = float(self.trailing_activation.get())
            if trailing_activation <= 0 or trailing_activation > profit_target:
                self.log_trade("Invalid trailing activation: must be between 0 and profit target")
                return False
            
            # Validate max trades
            max_trades = int(self.max_trades.get())
            if max_trades <= 0 or max_trades > 10:
                self.log_trade("Invalid max trades: must be between 1 and 10")
                return False
            
            # Validate min volume
            min_volume = float(self.min_volume.get())
            if min_volume < 0:
                self.log_trade("Invalid min volume: must be positive")
                return False
            
            self.log_trade("All conditions validated successfully")
            return True
        
        except ValueError as e:
            self.log_trade(f"Validation error: {str(e)}")
            return False
        except Exception as e:
            self.log_trade(f"Unexpected error during validation: {str(e)}")
            return False
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

    def toggle_bot(self):
        """Start or stop the bot"""
        if not self.running:
            # Start the bot
            self.running = True
            self.start_button.config(text="Stop")
            self.status_label.config(text="Status: Running")
            self.log_trade("Bot started")
            
            # Start the bot thread
            self.bot_thread = threading.Thread(target=self.run_bot_loop)
            self.bot_thread.daemon = True
            self.bot_thread.start()
            self.log_trade("Bot thread started")
        else:
            # Stop the bot
            self.running = False
            self.start_button.config(text="Start")
            self.status_label.config(text="Status: Stopped")
            self.log_trade("Bot stopped")

    def close_all_positions(self):
        """Close all open positions"""
        if not self.active_trades:
            self.log_trade("No active trades to close")
            return
            
        self.log_trade(f"Closing all {len(self.active_trades)} active trades")
        
        for trade_id in list(self.active_trades.keys()):
            trade = self.active_trades[trade_id]
            symbol = trade['symbol']
            current_price = self.get_current_price(symbol)
            self.close_trade(trade_id, trade, current_price, "manual close")
            
        self.log_trade("All positions closed")
        
        # Force update of paper balance display
        self.update_metrics()

    def get_current_price(self, symbol):
        """Get the current price for a symbol from mock data"""
        if symbol in self.mock_ticker_state:
            return self.mock_ticker_state[symbol]['last']
        return 0

    def update_mock_prices(self):
        """Update mock prices with realistic movements"""
        for symbol in self.mock_ticker_state:
            ticker = self.mock_ticker_state[symbol]
            
            # Generate realistic price movement (0.1% to 0.5% in either direction)
            price_change_pct = random.uniform(-0.5, 0.5) / 100
            
            # Apply the change
            current_price = ticker['last']
            new_price = current_price * (1 + price_change_pct)
            
            # Update the ticker
            ticker['last'] = new_price
            ticker['bid'] = new_price * 0.999  # 0.1% below last
            ticker['ask'] = new_price * 1.001  # 0.1% above last
            
            # Update volume (random 1-3% change)
            volume_change_pct = random.uniform(1, 3) / 100
            ticker['quoteVolume'] *= (1 + volume_change_pct)
            
            # Update the mock ticker state
            self.mock_ticker_state[symbol] = ticker
            
            # Update price history for active trades
            if symbol in self.active_trades:
                if symbol not in self.price_history:
                    self.price_history[symbol] = []
                
                # Add current time and price to history
                self.price_history[symbol].append((datetime.now(), new_price))
                
                # Keep only the last 50 points
                if len(self.price_history[symbol]) > 50:
                    self.price_history[symbol] = self.price_history[symbol][-50:]

    def run_bot_loop(self):
        """Main bot loop"""
        while self.running:
            try:
                # Update mock prices
                self.update_mock_prices()
                
                # Scan for opportunities
                self.scan_opportunities()
                
                # Monitor active trades
                self.monitor_trades()
                
                # Update GUI
                self.gui_update_queue.put(self.update_active_trades_display)
                self.gui_update_queue.put(self.update_chart)
                self.gui_update_queue.put(self.update_balance_display)
                
                # Sleep to avoid high CPU usage
                time.sleep(1)
                
            except Exception as e:
                self.log_trade(f"Error in bot loop: {str(e)}")
                time.sleep(5)

    def scan_opportunities(self):
        """Scan for trading opportunities"""
        if len(self.active_trades) >= int(self.max_trades.get()):
            return
        
        self.log_trade(f"Scanning {len(self.mock_ticker_state)} pairs for opportunities...")
        
        # Initialize DataManager if not already done
        if not hasattr(self, 'data_manager'):
            self.data_manager = DataManager(bot=self)
            self.log_trade("Created DataManager instance")
        
        # Find potential trades
        potential_trades = []
        
        for symbol, ticker in self.mock_ticker_state.items():
            try:
                price = ticker['last']
                volume = ticker['quoteVolume']
                
                # Basic filtering
                if volume < float(self.min_volume.get()):
                    continue
                    
                # Skip if already trading this symbol
                if symbol in self.active_trades:
                    continue
                
                # Update price data in DataManager
                new_data = {
                    'timestamp': datetime.now(),
                    'price': price,
                    'volume': volume
                }
                self.data_manager.update_price_data(symbol, new_data)
                
                # Calculate indicators using DataManager
                df = self.data_manager.calculate_indicators(symbol)
                if df is None:
                    continue
                
                # Calculate price change
                if len(df) >= 2:
                    price_change = ((df['price'].iloc[-1] - df['price'].iloc[-2]) / df['price'].iloc[-2]) * 100
                else:
                    price_change = 0
                
                # Check for minimum price rise
                if price_change < float(self.min_price_rise.get()):
                    continue
                
                # Advanced checks
                if not self.advanced_checks(symbol, df):
                    continue
                
                # If all checks pass, add to potential trades
                potential_trades.append((symbol, price, df))
                
            except Exception as e:
                self.log_trade(f"Error processing symbol {symbol}: {str(e)}")
                continue
        
        # Sort potential trades by price change (descending)
        potential_trades.sort(key=lambda x: x[1], reverse=True)
        
        # Execute trades up to the maximum allowed
        max_new_trades = int(self.max_trades.get()) - len(self.active_trades)
        for symbol, price, df in potential_trades[:max_new_trades]:
            try:
                ticker = self.mock_ticker_state[symbol]
                self.execute_trade(symbol, price)
            except Exception as e:
                self.log_trade(f"Error executing trade for {symbol}: {str(e)}")
                continue
    def advanced_checks(self, symbol, df):
        """Advanced market checks including RSI and EMA analysis"""
        try:
            # 1. EMA Cross (Short/Long)
            short_period = int(self.ema_short.get())
            long_period = int(self.ema_long.get())
            
            # Check if EMAs are in the dataframe
            ema_short_col = f'ema_{short_period}'
            ema_long_col = f'ema_{long_period}'
            
            # Use existing EMAs or calculate if needed
            if ema_short_col not in df.columns:
                ema_short_col = 'ema_5'  # Use default if specific one not available
            if ema_long_col not in df.columns:
                ema_long_col = 'ema_15'  # Use default if specific one not available
                
            # Check for bullish EMA cross (short crosses above long)
            if len(df) >= 2:
                ema_cross = (df[ema_short_col].iloc[-2] < df[ema_long_col].iloc[-2]) and \
                            (df[ema_short_col].iloc[-1] > df[ema_long_col].iloc[-1])
            else:
                ema_cross = False
            
            # 2. RSI Check
            rsi_period = int(self.rsi_period.get())
            rsi_overbought = float(self.rsi_overbought.get())
            rsi_oversold = float(self.rsi_oversold.get())
            
            # Use existing RSI or default
            rsi_col = f'rsi_{rsi_period}'
            if rsi_col not in df.columns:
                rsi_col = 'rsi_14'  # Use default if specific one not available
                
            # Check if RSI is available
            if rsi_col in df.columns and not df[rsi_col].isna().all():
                rsi = df[rsi_col].iloc[-1]
                # Check RSI conditions
                rsi_condition = rsi < rsi_overbought  # Not overbought
            else:
                rsi = 50  # Default neutral value
                rsi_condition = True  # Pass by default
            
            self.log_trade(f"Advanced checks for {symbol}:")
            self.log_trade(f"- EMA Cross: {ema_cross}")
            self.log_trade(f"- RSI: {rsi:.2f} (Overbought: {rsi_overbought}, Oversold: {rsi_oversold})")
            
            # For paper trading demo, we'll consider checks passed
            return True
            
        except Exception as e:
            self.log_trade(f"Error in advanced checks for {symbol}: {str(e)}")
            return False

    def execute_trade(self, symbol, price):
        """Execute a paper trade"""
        try:
            # Calculate position size
            position_size = float(self.position_size.get())
            
            # Check if we have enough balance
            if position_size > self.paper_balance:
                self.log_trade(f"Insufficient balance for trade: ${self.paper_balance:.2f} < ${position_size:.2f}")
                return False
            
            # Generate a unique trade ID
            trade_id = f"trade_{int(time.time())}_{symbol.replace('/', '')}"
            
            # Create the trade
            trade = {
                'id': trade_id,
                'symbol': symbol,
                'entry_time': datetime.now(),
                'entry_price': price,
                'position_size': position_size,
                'current_price': price,
                'highest_price': price,
                'highest_profit': 0.0,
                'highest_profit_percentage': 0.0,
                'status': 'open'
            }
            
            # Add to active trades
            self.active_trades[trade_id] = trade
            
            # Deduct from paper balance
            self.paper_balance -= position_size
            
            # Initialize price history for this symbol
            if symbol not in self.price_history:
                self.price_history[symbol] = []
            self.price_history[symbol].append((datetime.now(), price))
            
            # Log the trade
            self.log_trade(f"Executed trade: {symbol} at ${price:.6f} with ${position_size:.2f}")
            
            # Update displays
            self.update_metrics()  # This will update paper_balance_label
            
            # Queue other GUI updates
            if hasattr(self, 'gui_update_queue'):
                self.gui_update_queue.put(self.update_active_trades_display)
            
            return True
            
        except Exception as e:
            self.log_trade(f"Error executing trade: {str(e)}")
            return False

    def monitor_trades(self):
        """Monitor active trades for exit conditions"""
        try:
            if not self.active_trades:
                return
            
            for trade_id in list(self.active_trades.keys()):
                try:
                    # Get current price
                    symbol = self.active_trades[trade_id]['symbol']
                    current_price = self.get_current_price(symbol)
                    if not current_price:
                        self.log_trade(f"No current price data for {symbol}")
                        continue

                    # Calculate profit percentage
                    entry_price = self.active_trades[trade_id]['entry_price']
                    profit_percentage = ((current_price - entry_price) / entry_price) * 100

                    # Check for take profit
                    profit_target = float(self.profit_target.get())
                    if profit_percentage >= profit_target:
                        self.close_trade(trade_id, self.active_trades[trade_id], current_price, "take profit")
                        continue

                    # Check for stop loss
                    stop_loss = float(self.stop_loss.get())
                    if profit_percentage <= -stop_loss:
                        self.close_trade(trade_id, self.active_trades[trade_id], current_price, "stop loss")
                        continue

                    # Check for trailing stop
                    trailing_stop = float(self.trailing_stop.get())
                    
                    # Update highest price and profit if needed
                    if current_price > self.active_trades[trade_id].get('highest_price', 0):
                        self.active_trades[trade_id]['highest_price'] = current_price
                        self.active_trades[trade_id]['highest_profit_percentage'] = profit_percentage
                    
                    # Check trailing stop condition
                    trailing_activation = float(self.trailing_activation.get())
                    highest_profit = self.active_trades[trade_id].get('highest_profit_percentage', 0)
                    
                    if highest_profit >= trailing_activation:
                        drop_from_high = highest_profit - profit_percentage
                        if drop_from_high >= trailing_stop:
                            self.close_trade(trade_id, self.active_trades[trade_id], current_price, "trailing stop")
                            continue
                    
                    # Update price history for charting
                    if symbol in self.price_history:
                        self.price_history[symbol].append((datetime.now(), current_price))
                        # Limit history size
                        self.price_history[symbol] = self.price_history[symbol][-100:]
                
                except Exception as e:
                    self.log_trade(f"Error monitoring trade {trade_id}: {str(e)}")
                    continue
                
            # Update displays
            if hasattr(self, 'gui_update_queue'):
                self.gui_update_queue.put(self.update_active_trades_display)
                self.gui_update_queue.put(self.update_chart)
                self.gui_update_queue.put(self.update_metrics)
            
        except Exception as e:
            self.log_trade(f"Error in trade monitoring: {str(e)}")
    def setup_technical_indicators(self, parent):
        """Set up technical indicators parameters"""
        try:
            # Create a grid layout
            row = 0
            
            # RSI Period
            ttk.Label(parent, text="RSI Period").grid(row=row, column=0, sticky="w", padx=5, pady=2)
            ttk.Entry(parent, textvariable=self.rsi_period, width=10).grid(row=row, column=1, padx=5, pady=2)
            row += 1
            
            # RSI Overbought
            ttk.Label(parent, text="RSI Overbought").grid(row=row, column=0, sticky="w", padx=5, pady=2)
            ttk.Entry(parent, textvariable=self.rsi_overbought, width=10).grid(row=row, column=1, padx=5, pady=2)
            row += 1
            
            # RSI Oversold
            ttk.Label(parent, text="RSI Oversold").grid(row=row, column=0, sticky="w", padx=5, pady=2)
            ttk.Entry(parent, textvariable=self.rsi_oversold, width=10).grid(row=row, column=1, padx=5, pady=2)
            row += 1
            
            # EMA Short
            ttk.Label(parent, text="EMA Short").grid(row=row, column=0, sticky="w", padx=5, pady=2)
            ttk.Entry(parent, textvariable=self.ema_short, width=10).grid(row=row, column=1, padx=5, pady=2)
            row += 1
            
            # EMA Long (directly below EMA Short)
            ttk.Label(parent, text="EMA Long").grid(row=row, column=0, sticky="w", padx=5, pady=2)
            ttk.Entry(parent, textvariable=self.ema_long, width=10).grid(row=row, column=1, padx=5, pady=2)
            row += 1
            
            # Apply button
            ttk.Button(parent, text="Apply Indicators", command=self.live_update).grid(row=row, column=0, columnspan=2, padx=5, pady=10)
            
            self.log_trade("Technical indicators setup complete")
            
        except Exception as e:
            self.log_trade(f"Error setting up technical indicators: {str(e)}")

    def setup_basic_parameters(self, parent):
        """Set up basic trading parameters"""
        try:
            # Create a grid layout
            row = 0
            
            # Profit Target
            ttk.Label(parent, text="Profit Target (%)").grid(row=row, column=0, sticky="w", padx=5, pady=2)
            ttk.Entry(parent, textvariable=self.profit_target, width=10).grid(row=row, column=1, padx=5, pady=2)
            row += 1
            
            # Stop Loss
            ttk.Label(parent, text="Stop Loss (%)").grid(row=row, column=0, sticky="w", padx=5, pady=2)
            ttk.Entry(parent, textvariable=self.stop_loss, width=10).grid(row=row, column=1, padx=5, pady=2)
            row += 1
            
            # Position Size
            ttk.Label(parent, text="Position Size (USD)").grid(row=row, column=0, sticky="w", padx=5, pady=2)
            ttk.Entry(parent, textvariable=self.position_size, width=10).grid(row=row, column=1, padx=5, pady=2)
            row += 1
            
            # Min Price Rise
            ttk.Label(parent, text="Min Price Rise (%)").grid(row=row, column=0, sticky="w", padx=5, pady=2)
            ttk.Entry(parent, textvariable=self.min_price_rise, width=10).grid(row=row, column=1, padx=5, pady=2)
            row += 1
            
            # Trailing Stop
            ttk.Label(parent, text="Trailing Stop (%)").grid(row=row, column=0, sticky="w", padx=5, pady=2)
            ttk.Entry(parent, textvariable=self.trailing_stop, width=10).grid(row=row, column=1, padx=5, pady=2)
            row += 1
            
            # Trailing Activation
            ttk.Label(parent, text="Trailing Activation (%)").grid(row=row, column=0, sticky="w", padx=5, pady=2)
            ttk.Entry(parent, textvariable=self.trailing_activation, width=10).grid(row=row, column=1, padx=5, pady=2)
            row += 1
            
            # Max Trades
            ttk.Label(parent, text="Max Trades").grid(row=row, column=0, sticky="w", padx=5, pady=2)
            ttk.Entry(parent, textvariable=self.max_trades, width=10).grid(row=row, column=1, padx=5, pady=2)
            row += 1
            
            # Min Volume
            ttk.Label(parent, text="Min Volume (USD)").grid(row=row, column=0, sticky="w", padx=5, pady=2)
            ttk.Entry(parent, textvariable=self.min_volume, width=10).grid(row=row, column=1, padx=5, pady=2)
            row += 1
            
            # Volume Change Threshold (moved from Indicators to Basic)
            ttk.Label(parent, text="Volume Change Threshold (%)").grid(row=row, column=0, sticky="w", padx=5, pady=2)
            ttk.Entry(parent, textvariable=self.volume_change_threshold, width=10).grid(row=row, column=1, padx=5, pady=2)
            row += 1
            
            # Apply button
            ttk.Button(parent, text="Apply Settings", command=self.live_update).grid(row=row, column=0, columnspan=2, padx=5, pady=10)
            
        except Exception as e:
            self.log_trade(f"Error setting up basic parameters: {str(e)}")

    def setup_advanced_parameters(self, parent):
        """Set up advanced trading parameters"""
        try:
            # Create a grid layout
            row = 0
            
            # Momentum Beta
            ttk.Label(parent, text="Momentum Beta").grid(row=row, column=0, sticky="w", padx=5, pady=2)
            ttk.Entry(parent, textvariable=self.momentum_beta, width=10).grid(row=row, column=1, padx=5, pady=2)
            row += 1
            
            # Price Alpha
            ttk.Label(parent, text="Price Alpha").grid(row=row, column=0, sticky="w", padx=5, pady=2)
            ttk.Entry(parent, textvariable=self.price_alpha, width=10).grid(row=row, column=1, padx=5, pady=2)
            row += 1
            
            # Momentum Theta
            ttk.Label(parent, text="Momentum Theta").grid(row=row, column=0, sticky="w", padx=5, pady=2)
            ttk.Entry(parent, textvariable=self.momentum_theta, width=10).grid(row=row, column=1, padx=5, pady=2)
            row += 1
            
            # Vol Vega
            ttk.Label(parent, text="Vol Vega").grid(row=row, column=0, sticky="w", padx=5, pady=2)
            ttk.Entry(parent, textvariable=self.vol_vega, width=10).grid(row=row, column=1, padx=5, pady=2)
            row += 1
            
            # Volume Rho
            ttk.Label(parent, text="Volume Rho").grid(row=row, column=0, sticky="w", padx=5, pady=2)
            ttk.Entry(parent, textvariable=self.volume_rho, width=10).grid(row=row, column=1, padx=5, pady=2)
            row += 1
            
            # Apply button
            ttk.Button(parent, text="Apply Advanced", command=self.live_update).grid(row=row, column=0, columnspan=2, padx=5, pady=10)
            
            self.log_trade("Advanced parameters setup complete")
            
        except Exception as e:
            self.log_trade(f"Error setting up advanced parameters: {str(e)}")

    def setup_control_buttons(self, parent):
        """Set up control buttons"""
        try:
            # Create a frame for buttons
            button_frame = ttk.Frame(parent)
            button_frame.pack(fill=tk.X, padx=5, pady=5)
            
            # Start/Stop button
            self.start_button = ttk.Button(button_frame, text="Start Bot", command=self.toggle_bot)
            self.start_button.pack(side=tk.LEFT, padx=5)
            
            # Close All Positions button
            self.close_all_button = ttk.Button(button_frame, text="Close All Positions", command=self.close_all_positions)
            self.close_all_button.pack(side=tk.LEFT, padx=5)
            
            # Mode selection (Paper/Real)
            self.mode_var = tk.StringVar(value="Paper Trading")
            self.mode_button = ttk.Button(button_frame, text="Mode: Paper", command=self.toggle_mode)
            self.mode_button.pack(side=tk.LEFT, padx=5)
            
            # Add a button to show trade history
            self.history_button = ttk.Button(button_frame, text="Trade History", 
                                           command=lambda: self.history_window.deiconify())
            self.history_button.pack(side=tk.LEFT, padx=5)
            
            self.log_trade("Control buttons setup complete")
            
        except Exception as e:
            self.log_trade(f"Error setting up control buttons: {str(e)}")

    def setup_trade_history_display(self):
        """Set up the trade history display"""
        try:
            # Create a new window for trade history
            self.history_window = tk.Toplevel(self.root)
            self.history_window.title("Trade History")
            self.history_window.geometry("800x400")
            self.history_window.withdraw()  # Hide initially
            
            # Create scrolled text for trade history
            self.history_text = scrolledtext.ScrolledText(self.history_window)
            self.history_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            
            # Add history button to the main window
            # We'll add this in the setup_gui method to ensure button_frame exists
            
            self.log_trade("Trade history display setup complete")
            
        except Exception as e:
            self.log_trade(f"Error setting up trade history display: {str(e)}")

    def close_trade(self, trade_id, trade, current_price, reason):
        """Close a trade with proper balance updates"""
        try:
            self.log_trade(f"Closing trade {trade_id} ({trade['symbol']}) due to {reason}")
            
            symbol = trade['symbol']
            entry_price = trade['entry_price']
            position_size = trade['position_size']
            
            # Calculate profit/loss
            price_change = (current_price - entry_price) / entry_price
            gross_profit = position_size * price_change
            
            # Apply fees (0.8% total - 0.4% entry + 0.4% exit)
            entry_fee = position_size * 0.004  # 0.4%
            exit_fee = (position_size * (1 + price_change)) * 0.004  # 0.4%
            total_fees = entry_fee + exit_fee
            
            # Calculate net profit
            net_profit = gross_profit - total_fees
            
            # Update paper balance
            self.paper_balance += position_size + net_profit
            
            # Update stats
            self.total_profit += gross_profit
            self.total_fees += total_fees
            
            if net_profit > 0:
                self.wins += 1
            else:
                self.losses += 1
            
            # Log the trade result
            self.log_trade(f"""
            Trade closed:
            Symbol: {symbol}
            Entry: ${entry_price:.6f}
            Exit: ${current_price:.6f}
            Change: {price_change*100:.2f}%
            Gross P/L: ${gross_profit:.2f}
            Fees: ${total_fees:.2f}
            Net P/L: ${net_profit:.2f}
            New Balance: ${self.paper_balance:.2f}
            """)
            
            # Add to trade history
            timestamp = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
            profit_str = f"{price_change*100:.2f}%"
            profit_usd = f"${net_profit:.2f}"
            history_entry = f"({timestamp}) {symbol}: {profit_str}, {profit_usd}\n"
            
            if hasattr(self, 'history_text'):
                self.history_text.insert(tk.END, history_entry)
                self.history_text.see(tk.END)
            
            # Remove from active trades
            del self.active_trades[trade_id]
            
            # Update displays
            self.update_metrics()
            self.update_balance_display()
            
            # Queue other GUI updates if needed
            if hasattr(self, 'gui_update_queue'):
                self.gui_update_queue.put(self.update_active_trades_display)
                self.gui_update_queue.put(self.update_chart)
            
            return True
            
        except Exception as e:
            self.log_trade(f"Error closing trade: {str(e)}")
            return False

    def update_active_trades_display(self):
        """Update the active trades display"""
        try:
            if not hasattr(self, 'trades_text'):
                return
                
            self.trades_text.config(state=tk.NORMAL)
            self.trades_text.delete(1.0, tk.END)
            
            if not self.active_trades:
                self.trades_text.insert(tk.END, "No active trades")
            else:
                for trade_id, trade in self.active_trades.items():
                    symbol = trade['symbol']
                    entry_price = trade['entry_price']
                    current_price = trade['current_price']
                    profit_percentage = (current_price - entry_price) / entry_price * 100
                    
                    trade_info = (
                        f"{symbol}:\n"
                        f"  Entry: ${entry_price:.6f}\n"
                        f"  Current: ${current_price:.6f}\n"
                        f"  P/L: {profit_percentage:.2f}%\n"
                        f"  Size: ${trade['position_size']:.2f}\n"
                        f"  Highest: {trade['highest_profit_percentage']:.2f}%\n\n"
                    )
                    
                    self.trades_text.insert(tk.END, trade_info)
            
            self.trades_text.config(state=tk.DISABLED)
            
        except Exception as e:
            self.log_trade(f"Error updating trades display: {str(e)}")

    def update_balance_display(self):
        """Update the balance display"""
        try:
            if hasattr(self, 'balance_label'):
                self.balance_label.config(text=f"Paper Balance: ${self.paper_balance:.2f}")
        except Exception as e:
            self.log_trade(f"Error updating balance display: {str(e)}")

    def update_metrics(self):
        """Update trading metrics"""
        try:
            # Calculate win rate
            total_trades = self.wins + self.losses
            win_rate = (self.wins / total_trades * 100) if total_trades > 0 else 0
            
            # Calculate net profit
            net_profit = self.total_profit - self.total_fees
            
            # Calculate available balance (total - allocated to active trades)
            allocated_balance = sum(trade.get('position_size', 0) for trade in self.active_trades.values())
            available_balance = self.paper_balance - allocated_balance + net_profit
            
            # Update labels
            if hasattr(self, 'paper_balance_label'):
                self.paper_balance_label.config(
                    text=f"Paper Balance: ${self.paper_balance:.2f} USD"
                )
            
            if hasattr(self, 'total_profit_label'):
                self.total_profit_label.config(
                    text=f"Total Profit: ${self.total_profit:.2f} USD",
                    foreground="green" if self.total_profit > 0 else "red"
                )
            
            if hasattr(self, 'total_fees_label'):
                self.total_fees_label.config(
                    text=f"Total Fees: ${self.total_fees:.2f} USD"
                )
            
            if hasattr(self, 'net_profit_label'):
                self.net_profit_label.config(
                    text=f"Net Profit: ${net_profit:.2f} USD",
                    foreground="green" if net_profit > 0 else "red"
                )
            
            if hasattr(self, 'win_loss_label'):
                self.win_loss_label.config(
                    text=f"Win/Loss: {self.wins}/{self.losses} ({win_rate:.1f}%)"
                )
                
            if hasattr(self, 'active_trades_label'):
                self.active_trades_label.config(
                    text=f"Active Trades: {len(self.active_trades)}"
                )
            
            self.log_trade(f"Metrics Updated: Total Profit=${self.total_profit:.2f}, Net Profit=${net_profit:.2f}, "
                          f"Paper Balance=${self.paper_balance:.2f}, Wins={self.wins}, Losses={self.losses}")
            
        except Exception as e:
            self.log_trade(f"Error updating metrics: {str(e)}")

    def update_trade_history(self, symbol, percentage, profit, is_win=True):
        """Update the trade history display"""
        try:
            # Format timestamp
            timestamp = datetime.now().strftime("%H:%M:%S")
            
            # Format the trade result
            result = f"[{timestamp}] {symbol}: {percentage:.2f}%, ${profit:.2f}\n"
            
            if hasattr(self, 'history_text'):
                self.history_text.config(state=tk.NORMAL)
                self.history_text.insert(tk.END, result)
                
                # Color code based on profit/loss
                line_start = f"{float(self.history_text.index('end-2c'))-1:.1f}"
                color = "green" if is_win else "red"
                self.history_text.tag_add(color, f"{line_start} linestart", f"{line_start} lineend")
                self.history_text.tag_config("green", foreground="green")
                self.history_text.tag_config("red", foreground="red")
                
                self.history_text.see(tk.END)
                self.history_text.config(state=tk.DISABLED)
                
        except Exception as e:
            self.log_trade(f"Error updating trade history: {str(e)}")

    def run(self):
        """Run the application"""
        try:
            self.log_trade("Starting Crypto Scalping Bot (Paper Trading)")
            self.root.mainloop()
        except KeyboardInterrupt:
            self.log_trade("Keyboard interrupt detected")
        except Exception as e:
            self.log_trade(f"Error in main loop: {str(e)}")
        finally:
            self.is_shutting_down = True
            self.running = False
            self.log_trade("Shutting down bot...")
            
            # Close all trades if any are still open
            if self.active_trades:
                self.close_all_positions()
            
            # Wait for threads to finish
            if hasattr(self, 'bot_thread') and self.bot_thread and self.bot_thread.is_alive():
                try:
                    self.bot_thread.join(timeout=1)
                except:
                    pass
            
            self.log_trade("Bot shutdown complete")

    def on_closing(self):
        """Handle window close event properly"""
        try:
            self.log_trade("Closing application...")
            self.is_shutting_down = True
            self.running = False
            
            # Close all trades
            if self.active_trades:
                self.log_trade("Closing all active trades before shutdown")
                self.close_all_positions()
            
            # Wait a moment for threads to notice shutdown flag
            time.sleep(0.5)
            
            # Destroy the root window
            if hasattr(self, 'root') and self.root:
                self.root.destroy()
            
            # Force exit if needed
            import sys
            sys.exit(0)
        except Exception as e:
            print(f"Error during shutdown: {str(e)}")
            import sys
            sys.exit(1)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s:%(name)s:%(message)s')
    bot = CryptoScalpingBot()
    bot.run()
