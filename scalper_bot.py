import warnings
warnings.filterwarnings('ignore', message="fatal: bad revision 'HEAD'")
warnings.filterwarnings('ignore', category=UserWarning)
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, simpledialog
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
import traceback
import psutil
import configparser
import uuid
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates
class DataManager:
    def __init__(self, exchange=None, log_function=None, bot=None):
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
        self.bot = bot  # Store reference to the main bot
        
        # Default EMA periods if not available from bot
        self.ema_short = 8
        self.ema_long = 21
        self.rsi_period = 14

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
        """Calculate technical indicators using pandas, including RSI and MACD"""
        try:
            if symbol not in self.price_data:
                return
            df = self.price_data[symbol]
            if len(df) < 2:
                return
            
            try:
                # Moving averages
                for period in [5, 15]:  # Calculate the EMAs needed for trading decisions
                    if len(df) >= period:
                        df[f'ema_{period}'] = df['price'].ewm(span=period, adjust=False).mean()
                
                # Use custom EMA periods if set
                short_period = 8  # Default
                long_period = 21  # Default
                
                # Try to get values from bot if available
                if hasattr(self, 'bot') and self.bot is not None:
                    if hasattr(self.bot, 'ema_short') and hasattr(self.bot.ema_short, 'get'):
                        try:
                            short_period = int(self.bot.ema_short.get())
                        except:
                            pass
                    if hasattr(self.bot, 'ema_long') and hasattr(self.bot.ema_long, 'get'):
                        try:
                            long_period = int(self.bot.ema_long.get())
                        except:
                            pass
                # Use local values if bot reference not available
                elif hasattr(self, 'ema_short'):
                    short_period = self.ema_short
                    long_period = self.ema_long
                
                if len(df) >= short_period:
                    df[f'ema_{short_period}'] = df['price'].ewm(span=short_period, adjust=False).mean()
                if len(df) >= long_period:
                    df[f'ema_{long_period}'] = df['price'].ewm(span=long_period, adjust=False).mean()
                
                # Volume metrics
                if len(df) >= 5:
                    df['volume_sma_5'] = df['volume'].rolling(window=5).mean()
                    df['volume_ratio'] = df['volume'] / df['volume_sma_5']
                
                # Price momentum and volatility
                df['price_change'] = df['price'].pct_change()
                if len(df) >= 5:
                    df['momentum'] = df['price'].pct_change(periods=5)
                    df['volatility'] = df['price'].rolling(window=5).std()
                
                # RSI Calculation
                rsi_period = 14  # Default to 14 if not set
                if hasattr(self, 'rsi_period') and self.rsi_period is not None:
                    try:
                        rsi_period = int(self.rsi_period.get())
                    except:
                        pass
                    
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
                
                # MACD Calculation
                if hasattr(self, 'macd_fast') and hasattr(self, 'macd_slow') and hasattr(self, 'macd_signal'):
                    try:
                        fast_period = int(self.macd_fast.get())
                        slow_period = int(self.macd_slow.get())
                        signal_period = int(self.macd_signal.get())
                        
                        if len(df) >= slow_period + signal_period:
                            # Calculate MACD line
                            fast_ema = df['price'].ewm(span=fast_period, adjust=False).mean()
                            slow_ema = df['price'].ewm(span=slow_period, adjust=False).mean()
                            df['macd_line'] = fast_ema - slow_ema
                            
                            # Calculate signal line
                            df['macd_signal_line'] = df['macd_line'].ewm(span=signal_period, adjust=False).mean()
                            
                            # Calculate histogram
                            df['macd_histogram'] = df['macd_line'] - df['macd_signal_line']
                            
                            self.log_trade(f"Calculated MACD for {symbol} with periods {fast_period}/{slow_period}/{signal_period}")
                    except Exception as e:
                        self.log_trade(f"Error calculating MACD: {str(e)}")
                
                # Store updated DataFrame
                self.price_data[symbol] = df
                
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
            if not all(field in new_data for field in required_fields):
                self.log_trade(f"Missing required fields for {symbol}")
                return

            try:
                price = float(new_data['last'])
                volume = float(new_data['quoteVolume'])
                bid = float(new_data['bid'])
                ask = float(new_data['ask'])
                
                # Create new data point
                new_row = {
                    'timestamp': current_timestamp,
                    'price': price,
                    'volume': volume,
                    'bid': bid,
                    'ask': ask
                }
                
                # Initialize DataFrame if it doesn't exist
                if symbol not in self.price_data or self.price_data[symbol] is None:
                    self.price_data[symbol] = pd.DataFrame([new_row])
                else:
                    # Append new data
                    self.price_data[symbol] = pd.concat([self.price_data[symbol], 
                                                    pd.DataFrame([new_row])], 
                                                    ignore_index=True)
                
                # Calculate indicators after updating data
                self._calculate_indicators(symbol)
                
            except Exception as e:
                self.log_trade(f"Error processing data for {symbol}: {str(e)}")
                
        except Exception as e:
            self.log_trade(f"Error in update_price_data: {str(e)}")

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
        self.root.title("Vantrex v1.70") 
        self.root.geometry("1200x1000")
        
        # Initialize metrics
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_profit = 0.0
        self.total_fees = 0.0

        # Add trend strength parameters
        self.use_trend_filter = tk.BooleanVar(self.root, value=True)
        self.trend_strength_min = tk.StringVar(self.root, value="15")
        
        # Initialize state variables
        self.running = False
        self.is_scanning = False
        self.is_paper_trading = True
        self.paper_balance = 1000.0
        self.active_trades = {}
        
        # Add max_active_trades attribute
        self.max_active_trades = 2  # Default value
        
        # Initialize basic variables
        self.start_time = None
        self.timer_running = False
        self.trading_thread = None
        self.trades = []
        self.trade_history = []
        self.price_history = {}
        self.cache_timeout = 1
        self.price_cache = {}
        self.max_price = tk.StringVar(self.root, value="5.00")

        # Initialize StringVars for GUI inputs with values from screenshot
        self.profit_target = tk.StringVar(self.root, value="1.5")
        self.stop_loss = tk.StringVar(self.root, value="0.8")
        self.trailing_stop = tk.StringVar(self.root, value="0.3")
        self.trailing_activation = tk.StringVar(self.root, value="0.5")
        self.position_size = tk.StringVar(self.root, value="50")
        self.min_volume_entry = tk.StringVar(self.root, value="200")
        self.max_trades_entry = tk.StringVar(self.root, value="2")
        self.top_list_size = tk.StringVar(self.root, value="20")
        self.max_volatility = tk.StringVar(self.root, value="1.5")
        self.consecutive_rises = tk.StringVar(self.root, value="2")
        self.momentum_threshold = tk.StringVar(self.root, value="0.2")
        self.max_spread = tk.StringVar(self.root, value="0.2")
        self.volume_increase = tk.StringVar(self.root, value="15")
        self.volume_surge = tk.StringVar(self.root, value="15")
        self.price_rise_min = tk.StringVar(self.root, value="0.3")
        self.max_position_percent = tk.StringVar(self.root, value="5")
        self.daily_loss_limit = tk.StringVar(self.root, value="3")
        self.required_conditions = tk.StringVar(self.root, value="3")
        self.volume_change_threshold = tk.StringVar(self.root, value="1.0")

        # Add fee structure variables as StringVars
        self.maker_fee_var = tk.StringVar(self.root, value="0.25")  # 0.25% for maker orders
        self.taker_fee_var = tk.StringVar(self.root, value="0.40")  # 0.40% for taker orders

        # Fee Structure (Kraken)
        self.maker_fee = 0.0025  # 0.25% for maker orders
        self.taker_fee = 0.004   # 0.40% for taker orders
        self.total_fee_percentage = self.taker_fee * 2  # 0.80% total for entry and exit

        # Add a flag to track if we're using limit orders
        self.using_limit_orders = False

        # Support/Resistance parameters
        self.use_support_resistance = tk.BooleanVar(self.root, value=True)
        self.sr_lookback = tk.StringVar(self.root, value="50")
        self.sr_threshold = tk.StringVar(self.root, value="0.2")
        
        # Candlestick pattern parameters
        self.use_candlestick_patterns = tk.BooleanVar(self.root, value=True)
        self.pattern_confidence_min = tk.StringVar(self.root, value="70")
        
        # Volume profile parameters
        self.use_volume_profile = tk.BooleanVar(self.root, value=True)
        self.volume_quality_min = tk.StringVar(self.root, value="60")

        # Initialize technical indicator StringVars
        self.rsi_period = tk.StringVar(self.root, value="14")
        self.rsi_overbought = tk.StringVar(self.root, value="75")
        self.rsi_oversold = tk.StringVar(self.root, value="30")
        self.macd_fast = tk.StringVar(self.root, value="10")
        self.macd_slow = tk.StringVar(self.root, value="24")
        self.macd_signal = tk.StringVar(self.root, value="7")
        self.ema_short = tk.StringVar(self.root, value="8")
        self.ema_long = tk.StringVar(self.root, value="21")

        # Initialize Greek parameters
        self.momentum_beta = tk.StringVar(value="0.1")
        self.price_alpha = tk.StringVar(value="0.1")
        self.momentum_theta = tk.StringVar(value="0.1")
        self.vol_vega = tk.StringVar(value="0.1")
        self.volume_rho = tk.StringVar(value="0.1")

        # Plotting parameters
        self.plot_update_interval = 2000  # milliseconds
        self.chart_timeframe = 50  # number of points to show
        self.price_precision = 8
        self.amount_precision = 8
        self.gross_profit = 0.0


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

        # Add volume_surge variable to match GUI
        self.volume_surge = tk.StringVar(self.root, value="10")

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
        """Setup the price chart with proper styling"""
        try:
            # Create figure and axis with the right background color
            bg_color = '#121212' if self.night_mode else '#f0f0f0'
            fg_color = '#ffffff' if self.night_mode else '#000000'
            grid_color = '#333333' if self.night_mode else '#cccccc'
            
            self.fig = Figure(figsize=(5, 4), dpi=100, facecolor=bg_color)
            self.ax = self.fig.add_subplot(111, facecolor=bg_color)
            
            # Set up grid and zero line
            self.ax.grid(True, linestyle='--', alpha=0.3, color=grid_color)
            self.ax.axhline(y=0, color=grid_color, linestyle='-', alpha=0.5)
            
            # Set up axis colors
            self.ax.tick_params(colors=fg_color)
            for spine in self.ax.spines.values():
                spine.set_color(fg_color)
                
            # Formatting
            self.ax.xaxis.set_major_formatter(DateFormatter('%H:%M:%S'))
            self.ax.set_xlabel('Time', color=fg_color, fontsize=10)
            self.ax.set_ylabel('Price Change (%)', color=fg_color, fontsize=10)
            self.ax.set_title('Trade Performance', color=fg_color, pad=20, fontsize=12)
            
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
        """Toggle between light and dark theme with improved contrast"""
        try:
            self.night_mode = not self.night_mode
            
            if self.night_mode:
                # Dark theme colors with better contrast
                bg_color = '#121212'        # Much darker background
                frame_bg = '#1a1a1a'        # Slightly lighter frame background
                fg_color = '#ffffff'        # White text
                entry_bg = '#3a3a3a'        # Lighter entry background for better readability
                button_bg = '#333333'       # Dark button background
                button_fg = '#e0e0e0'       # Light gray text for buttons (not pure white)
                text_area_bg = '#0f0f0f'    # Very dark text area background
                
                # Configure root window
                self.root.configure(background=bg_color)
                
                # Configure ttk styles for better contrast
                self.style.configure('TButton',
                    background=button_bg,
                    foreground=button_fg)
                    
                self.style.map('TButton',
                    background=[('active', '#444444')],
                    foreground=[('active', '#ffffff')])
                
                # Configure entry style with better contrast
                self.style.configure('TEntry',
                    fieldbackground=entry_bg,
                    foreground=fg_color)
                    
                # Configure frame style
                self.style.configure('TFrame',
                    background=frame_bg)
                    
                # Configure labelframe style
                self.style.configure('TLabelframe',
                    background=frame_bg)
                    
                self.style.configure('TLabelframe.Label',
                    background=frame_bg,
                    foreground=fg_color)
                    
                # Configure notebook style
                self.style.configure('TNotebook',
                    background=bg_color)
                    
                self.style.configure('TNotebook.Tab',
                    background=button_bg,
                    foreground=button_fg)
                
                # Configure label style
                self.style.configure('TLabel',
                    background=frame_bg,
                    foreground=fg_color)
                    
                # Configure checkbutton style
                self.style.configure('TCheckbutton',
                    background=frame_bg,
                    foreground=fg_color)
                    
                self.style.map('TCheckbutton',
                    background=[('active', frame_bg)],
                    foreground=[('active', fg_color)])
                
                # Apply dark style to all widgets
                for widget in self.root.winfo_children():
                    self._apply_dark_style_recursive(widget, bg_color, fg_color, entry_bg, button_bg, text_area_bg, frame_bg, button_fg)
                
                # Update chart colors if they exist
                if hasattr(self, 'fig') and hasattr(self, 'ax'):
                    self.fig.set_facecolor(bg_color)
                    self.ax.set_facecolor(bg_color)
                    self.ax.tick_params(colors=fg_color)
                    for spine in self.ax.spines.values():
                        spine.set_color(fg_color)
                    self.ax.title.set_color(fg_color)
                    self.ax.xaxis.label.set_color(fg_color)
                    self.ax.yaxis.label.set_color(fg_color)
                    self.canvas.draw()
                
                # Update night mode button text
                if hasattr(self, 'night_mode_button'):
                    self.night_mode_button.configure(text="Light Mode")
                
            else:
                # Light theme colors
                bg_color = '#f0f0f0'        # Light background
                frame_bg = '#f5f5f5'        # Slightly lighter frame background
                fg_color = '#000000'        # Black text
                entry_bg = '#ffffff'        # White entry background
                button_bg = '#e0e0e0'       # Light button background
                button_fg = '#000000'       # Black text for buttons
                text_area_bg = '#ffffff'    # White text area background
                
                # Configure root window
                self.root.configure(background=bg_color)
                
                # Reset ttk styles
                self.style.configure('TButton', 
                    background=button_bg, 
                    foreground=button_fg)
                    
                self.style.map('TButton',
                    background=[('active', '#d0d0d0')],
                    foreground=[('active', '#000000')])
                
                # Reset entry style
                self.style.configure('TEntry', 
                    fieldbackground=entry_bg, 
                    foreground=fg_color)
                    
                # Reset frame style
                self.style.configure('TFrame',
                    background=frame_bg)
                    
                # Reset labelframe style
                self.style.configure('TLabelframe',
                    background=frame_bg)
                    
                self.style.configure('TLabelframe.Label',
                    background=frame_bg,
                    foreground=fg_color)
                    
                # Reset notebook style
                self.style.configure('TNotebook',
                    background=bg_color)
                    
                self.style.configure('TNotebook.Tab',
                    background=button_bg,
                    foreground=button_fg)
                    
                # Reset label style
                self.style.configure('TLabel',
                    background=frame_bg,
                    foreground=fg_color)
                    
                # Reset checkbutton style
                self.style.configure('TCheckbutton',
                    background=frame_bg,
                    foreground=fg_color)
                    
                self.style.map('TCheckbutton',
                    background=[('active', frame_bg)],
                    foreground=[('active', fg_color)])
                
                # Apply light style to all widgets
                for widget in self.root.winfo_children():
                    self._apply_light_style_recursive(widget, bg_color, fg_color, entry_bg, button_bg, text_area_bg, frame_bg, button_fg)
                
                # Update chart colors if they exist
                if hasattr(self, 'fig') and hasattr(self, 'ax'):
                    self.fig.set_facecolor(bg_color)
                    self.ax.set_facecolor(bg_color)
                    self.ax.tick_params(colors=fg_color)
                    for spine in self.ax.spines.values():
                        spine.set_color(fg_color)
                    self.ax.title.set_color(fg_color)
                    self.ax.xaxis.label.set_color(fg_color)
                    self.ax.yaxis.label.set_color(fg_color)
                    self.canvas.draw()
                
                # Update night mode button text
                if hasattr(self, 'night_mode_button'):
                    self.night_mode_button.configure(text="Dark Mode")
            
            # Force redraw
            self.root.update_idletasks()
            
        except Exception as e:
            self.log_trade(f"Error toggling night mode: {str(e)}")

    def _apply_dark_style_recursive(self, widget, bg_color, fg_color, entry_bg, button_bg, text_area_bg, frame_bg, button_fg):
        """Apply dark style to widget and all its children recursively with improved contrast"""
        try:
            widget_class = widget.winfo_class()
            
            if widget_class in ('Frame', 'Labelframe', 'TFrame', 'TLabelframe'):
                # For frames, configure background
                if hasattr(widget, 'configure'):
                    try:
                        widget.configure(background=frame_bg)
                    except:
                        pass
                        
            elif widget_class in ('Entry', 'TEntry'):
                # For entry widgets, configure with high contrast
                if hasattr(widget, 'configure'):
                    try:
                        # Use a lighter background for entry fields to improve readability
                        widget.configure(
                            background='#3a3a3a',  # Lighter than the frame background
                            foreground='#ffffff',  # White text for maximum contrast
                            insertbackground='#ffffff',  # White cursor
                            highlightbackground=bg_color,
                            highlightcolor='#ffffff'
                        )
                    except:
                        pass
                        
            elif widget_class in ('Button', 'TButton'):
                # For buttons, configure with better contrast
                if hasattr(widget, 'configure'):
                    try:
                        widget.configure(
                            background=button_bg,
                            foreground='#e0e0e0',  # Light gray text for better readability
                            highlightbackground=bg_color,
                            activebackground='#444444',  # Darker when pressed
                            activeforeground='#ffffff'   # White text when pressed
                        )
                    except:
                        pass
                        
            elif widget_class in ('Label', 'TLabel'):
                # For labels, configure text color
                if hasattr(widget, 'configure'):
                    try:
                        widget.configure(
                            background=frame_bg,
                            foreground=fg_color
                        )
                    except:
                        pass
                        
            elif widget_class == 'Text':
                # For text widgets, configure with dark theme
                if hasattr(widget, 'configure'):
                    try:
                        widget.configure(
                            background=text_area_bg,
                            foreground=fg_color,
                            insertbackground=fg_color,
                            selectbackground='#404040',
                            selectforeground=fg_color
                        )
                    except:
                        pass
                        
            elif widget_class in ('Listbox', 'TCombobox'):
                # For list widgets, configure with dark theme
                if hasattr(widget, 'configure'):
                    try:
                        widget.configure(
                            background=entry_bg,
                            foreground=fg_color,
                            selectbackground='#404040',
                            selectforeground=fg_color
                        )
                    except:
                        pass
                        
            elif widget_class == 'Canvas':
                # For canvas widgets
                if hasattr(widget, 'configure'):
                    try:
                        widget.configure(
                            background=bg_color,
                            highlightbackground=frame_bg
                        )
                    except:
                        pass
                        
            elif widget_class in ('Notebook', 'TNotebook'):
                # For notebook widgets
                if hasattr(widget, 'configure'):
                    try:
                        widget.configure(
                            background=bg_color
                        )
                    except:
                        pass
                        
            elif widget_class == 'Checkbutton':
                # For checkbuttons
                if hasattr(widget, 'configure'):
                    try:
                        widget.configure(
                            background=frame_bg,
                            foreground=fg_color,
                            selectcolor='#3a3a3a',  # Dark background for selected state
                            activebackground=frame_bg
                        )
                    except:
                        pass
            
            # Apply to all children recursively
            for child in widget.winfo_children():
                self._apply_dark_style_recursive(child, bg_color, fg_color, entry_bg, button_bg, text_area_bg, frame_bg, button_fg)
                
        except Exception as e:
            self.log_trade(f"Error applying dark style to {widget}: {str(e)}")

    def _apply_light_style_recursive(self, widget, bg_color, fg_color, entry_bg, button_bg, text_area_bg, frame_bg, button_fg):
        """Apply light style to widget and all its children recursively"""
        try:
            widget_class = widget.winfo_class()
            
            if widget_class in ('Frame', 'Labelframe', 'TFrame', 'TLabelframe'):
                if hasattr(widget, 'configure'):
                    try:
                        widget.configure(background=frame_bg)
                    except:
                        pass
                        
            elif widget_class in ('Entry', 'TEntry'):
                if hasattr(widget, 'configure'):
                    try:
                        widget.configure(
                            background=entry_bg,
                            foreground=fg_color,
                            insertbackground=fg_color,
                            highlightbackground=bg_color,
                            highlightcolor=fg_color,
                            readonlybackground='#f0f0f0'  # Light gray for readonly fields
                        )
                    except:
                        pass
                        
            elif widget_class in ('Button', 'TButton'):
                if hasattr(widget, 'configure'):
                    try:
                        widget.configure(
                            background=button_bg,
                            foreground=button_fg,
                            highlightbackground=bg_color,
                            activebackground='#d0d0d0',  # Slightly darker when pressed
                            activeforeground='#000000'   # Black text when pressed
                        )
                    except:
                        pass
                        
            elif widget_class in ('Label', 'TLabel'):
                if hasattr(widget, 'configure'):
                    try:
                        widget.configure(
                            background=frame_bg,
                            foreground=fg_color
                        )
                    except:
                        pass
                        
            elif widget_class == 'Text':
                if hasattr(widget, 'configure'):
                    try:
                        widget.configure(
                            background=text_area_bg,
                            foreground=fg_color,
                            insertbackground=fg_color,
                            selectbackground='#0078D7',
                            selectforeground='white'
                        )
                    except:
                        pass
                        
            elif widget_class in ('Listbox', 'TCombobox'):
                if hasattr(widget, 'configure'):
                    try:
                        widget.configure(
                            background=entry_bg,
                            foreground=fg_color,
                            selectbackground='#0078D7',
                            selectforeground='white'
                        )
                    except:
                        pass
                        
            elif widget_class == 'Canvas':
                # For canvas widgets
                if hasattr(widget, 'configure'):
                    try:
                        widget.configure(
                            background=bg_color,
                            highlightbackground=frame_bg
                        )
                    except:
                        pass
                        
            elif widget_class in ('Notebook', 'TNotebook'):
                # For notebook widgets
                if hasattr(widget, 'configure'):
                    try:
                        widget.configure(
                            background=bg_color
                        )
                    except:
                        pass
                        
            elif widget_class == 'Checkbutton':
                # For checkbuttons
                if hasattr(widget, 'configure'):
                    try:
                        widget.configure(
                            background=frame_bg,
                            foreground=fg_color,
                            selectcolor='white',  # White background for selected state
                            activebackground=frame_bg
                        )
                    except:
                        pass
            
            # Apply to all children recursively
            for child in widget.winfo_children():
                self._apply_light_style_recursive(child, bg_color, fg_color, entry_bg, button_bg, text_area_bg, frame_bg, button_fg)
                
        except Exception as e:
            self.log_trade(f"Error applying light style to {widget}: {str(e)}")

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

    def setup_history_panel(self):
        """Set up the trade history panel with proper styling"""
        try:
            # Create a frame for the history panel
            history_frame = ttk.LabelFrame(self.right_frame, text="Trade History")
            history_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            
            # Create a text widget for displaying trade history
            self.history_text = tk.Text(history_frame, height=10, width=40, wrap=tk.WORD)
            self.history_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            
            # Add scrollbar
            history_scrollbar = ttk.Scrollbar(self.history_text, command=self.history_text.yview)
            history_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            self.history_text.config(yscrollcommand=history_scrollbar.set)
            
            # Configure text widget for read-only
            self.history_text.config(state=tk.DISABLED)
            
            # Apply initial styling based on night mode
            if self.night_mode:
                self.history_text.configure(
                    background='#0f0f0f',
                    foreground='white',
                    insertbackground='white',
                    selectbackground='#404040',
                    selectforeground='white'
                )
            
            # Load trade history
            self.load_trade_history()
            
        except Exception as e:
            self.log_trade(f"Error setting up history panel: {str(e)}")

    def load_trade_history(self):
        """Initialize empty trade history"""
        try:
            # Initialize empty trades list
            self.trades = []
            
            # Clear history widget if it exists
            if hasattr(self, 'history_text'):
                self.history_text.config(state=tk.NORMAL)
                self.history_text.delete(1.0, tk.END)
                self.history_text.insert(tk.END, "Trade history initialized\n")
                self.history_text.config(state=tk.DISABLED)
                
        except Exception as e:
            self.log_trade(f"Error initializing trade history: {str(e)}")

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
                # Fetch more OHLCV data - increase from 24 to 100 candles
                since = int((datetime.now() - timedelta(days=5)).timestamp() * 1000)  # 5 days instead of 1
                ohlcv = self.exchange.fetch_ohlcv('BTC/USD', timeframe='1h', since=since, limit=100)
                
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
                    
                    # Initialize DataManager with exchange instance and bot reference
                    self.data_manager = DataManager(
                        exchange=self.exchange,
                        log_function=self.log_trade,
                        bot=self  # Pass reference to the bot
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
                self.mode_button.config(text="Mode: Paper")  # Update button text
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
            self.mode_button.config(text="Mode: Real")  # Update button text
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
            self.log_trade(f"Error toggling trading mode: {str(e)}")
            messagebox.showerror("Error", f"Failed to change trading mode: {str(e)}")

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
        """Start the trading bot with proper initialization"""
        try:
            if self.running:
                self.log_trade("Bot is already running")
                return
                    
            # Verify paper balance before starting
            if self.is_paper_trading:
                self.verify_paper_balance()
                    
            # Set running flag
            self.running = True
            self.start_time = datetime.now()
                
            # Update status
            self.update_status("Starting...")
            self.log_trade("=== BOT STARTED ===")
                
            # Force update fee calculations before starting
            self.force_update_fee_calculations()
                
            # Start the trading thread
            self.trading_thread = threading.Thread(target=self.run_bot)
            self.trading_thread.daemon = True
            self.trading_thread.start()
                
            # Update UI - change start button to show "Stop" and make it call stop_bot
            if hasattr(self, 'start_button'):
                self.start_button.config(text="Stop", command=self.stop_bot)
                
            # Start the timer
            if hasattr(self, 'start_timer'):
                self.start_timer()
                
        except Exception as e:
            self.log_trade(f"Error starting bot: {str(e)}")
            self.running = False

    def stop_bot(self):
        """Stop the trading bot with proper cleanup"""
        try:
            if not self.running:
                self.log_trade("Bot is not running")
                return
                
            # Update status
            self.update_status("Stopping...")
            self.log_trade("=== BOT STOPPING ===")
            
            # Close all positions
            self.close_all_positions_on_stop()
            
            # Set running flag to false AFTER closing positions
            self.running = False
            
            # Stop the timer if it exists
            if hasattr(self, 'stop_timer'):
                self.stop_timer()
            
            # Update UI - change button back to "Start" and make it call start_bot
            if hasattr(self, 'start_button'):
                self.start_button.config(text="Start", command=self.start_bot)
            
            # Wait for trading thread to finish if it exists
            if hasattr(self, 'trading_thread') and self.trading_thread and self.trading_thread.is_alive():
                self.log_trade("Waiting for trading thread to finish...")
                # Don't join with timeout as it might block the GUI
                # Just let it finish naturally as running=False will stop the loop
            
            self.update_status("Stopped")
            self.log_trade("=== BOT STOPPED ===")
            
        except Exception as e:
            self.log_trade(f"Error stopping bot: {str(e)}")

    def close_all_positions_on_stop(self):
        """Close all positions when stopping the bot"""
        try:
            if not self.active_trades:
                self.log_trade("No active trades to close")
                return
                
            self.log_trade(f"Closing all {len(self.active_trades)} positions due to stop signal")
            
            # Make a copy of the trade IDs since we'll be modifying the dictionary
            trade_ids = list(self.active_trades.keys())
            
            for trade_id in trade_ids:
                try:
                    trade = self.active_trades[trade_id]
                    
                    # Get current price with error handling
                    try:
                        current_price = self.exchange.fetch_ticker(trade['symbol'])['last']
                    except Exception as e:
                        self.log_trade(f"Error fetching price for {trade['symbol']}: {str(e)}")
                        # Use fallback price
                        current_price = trade.get('highest_price', trade['entry_price'])
                    
                    # Close the trade
                    self.close_trade(trade_id, trade, current_price, "bot stopped")
                    
                    # Brief pause between closures to avoid rate limits
                    time.sleep(0.1)
                    
                except Exception as e:
                    self.log_trade(f"Error closing trade {trade_id}: {str(e)}")
            
            # Verify all trades were closed
            if self.active_trades:
                self.log_trade(f"WARNING: {len(self.active_trades)} trades could not be closed")
            else:
                self.log_trade("All positions closed successfully")
                
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
            
            # Calculate total round-trip fee based on order type
            use_limit_orders = self.use_limit_orders_var.get() if hasattr(self, 'use_limit_orders_var') else False
            
            if use_limit_orders:
                # Using limit orders (maker fees)
                self.total_fee_percentage = self.maker_fee * 2  # Entry and exit
            else:
                # Using market orders (taker fees)
                self.total_fee_percentage = self.taker_fee * 2  # Entry and exit
            
            # Debug: Print the updated instance variables
            self.log_trade(f"DEBUG: self.maker_fee = {self.maker_fee}")
            self.log_trade(f"DEBUG: self.taker_fee = {self.taker_fee}")
            self.log_trade(f"DEBUG: self.total_fee_percentage = {self.total_fee_percentage}")
            self.log_trade(f"DEBUG: using limit orders = {use_limit_orders}")
            
            # Update the total fees label
            total_fee_pct = self.total_fee_percentage * 100
            fee_type = "maker" if use_limit_orders else "taker"
            if hasattr(self, 'total_fees_label'):
                self.total_fees_label.config(text=f"Total Round-Trip Fee: {total_fee_pct:.2f}%")
            
            # Log the update
            self.log_trade(f"Fee structure updated: Maker {maker_fee*100:.2f}%, Taker {taker_fee*100:.2f}%, " +
                        f"Total ({fee_type}*2) {total_fee_pct:.2f}%")
            
            # Update tooltips
            if hasattr(self, 'profit_target'):
                self.add_tooltip(self.profit_target, 
                    f"Target profit percentage for trades (min: {total_fee_pct:.2f}% to cover fees)")
            
            return True
        except Exception as e:
            self.log_trade(f"Error initializing fees: {str(e)}")
            return False

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
        """Reset the paper balance to 1000 USD"""
        try:
            # Calculate allocated funds
            allocated = sum(trade.get('position_size', 0) for trade in self.active_trades.values())
            
            # Reset paper balance
            old_balance = self.paper_balance
            self.paper_balance = 1000.0 - allocated
            
            # Log the reset
            self.log_trade(f"Paper balance reset: ${old_balance:.2f} -> ${self.paper_balance:.2f}")
            
            # Update displays
            self.update_balance_display()
            
        except Exception as e:
            self.log_trade(f"Error resetting paper balance: {str(e)}")

    def initialize_pair_data(self, symbol):
        """Initialize historical data for a trading pair"""
        try:
            self.log_trade(f"Initializing historical data for {symbol}...")
            
            # Fetch initial ticker
            try:
                ticker = self.exchange.fetch_ticker(symbol)
                if not ticker:
                    self.log_trade(f"Could not fetch ticker for {symbol}")
                    return False
                    
                self.data_manager.update_price_data(symbol, ticker)
            except Exception as e:
                self.log_trade(f"Error fetching ticker for {symbol}: {str(e)}")
                return False
                
            # Fetch historical OHLCV data
            try:
                since = int((datetime.now() - timedelta(days=3)).timestamp() * 1000)
                ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe='15m', since=since, limit=100)
                
                if not ohlcv or len(ohlcv) < 20:
                    self.log_trade(f"Insufficient OHLCV data for {symbol}: {len(ohlcv) if ohlcv else 0} candles")
                    return False
                    
                # Process each candle
                for candle in ohlcv:
                    timestamp, open_price, high, low, close, volume = candle
                    
                    # Create synthetic ticker
                    synthetic_ticker = {
                        'timestamp': pd.Timestamp(timestamp, unit='ms'),
                        'last': close,
                        'quoteVolume': volume,
                        'bid': close * 0.999,
                        'ask': close * 1.001
                    }
                    
                    # Update price data
                    self.data_manager.update_price_data(symbol, synthetic_ticker)
            except Exception as e:
                self.log_trade(f"Error fetching historical data for {symbol}: {str(e)}")
                return False
            
            # Verify we have enough data
            try:
                df = self.data_manager.get_price_data(symbol)
                if df is not None and len(df) >= 20:  # Ensure we have enough data
                    self.log_trade(f"Successfully loaded {len(df)} data points for {symbol}")
                    return True
                else:
                    self.log_trade(f"Insufficient historical data for {symbol}: {len(df) if df is not None else 0} points")
                    return False
            except Exception as e:
                self.log_trade(f"Error verifying data for {symbol}: {str(e)}")
                return False
                
        except Exception as e:
            self.log_trade(f"Error initializing data for {symbol}: {str(e)}")
            return False

    def setup_limit_order_settings(self, parent_frame):
        """Setup limit order settings UI"""
        limit_frame = ttk.LabelFrame(parent_frame, text="Limit Order Settings")
        limit_frame.pack(fill="x", padx=5, pady=5)
        
        # Create a grid layout for the settings
        row = 0
        
        # Timeout for limit orders
        ttk.Label(limit_frame, text="Timeout (seconds):").grid(row=row, column=0, padx=5, pady=2, sticky="w")
        self.limit_order_timeout = tk.StringVar(value="300")  # 5 minutes default
        timeout_entry = ttk.Entry(limit_frame, width=8, textvariable=self.limit_order_timeout)
        timeout_entry.grid(row=row, column=1, padx=5, pady=2, sticky="w")
        self.add_tooltip(timeout_entry, "Cancel limit orders after this many seconds")
        row += 1
        
        # Max price difference
        ttk.Label(limit_frame, text="Max Price Diff (%):").grid(row=row, column=0, padx=5, pady=2, sticky="w")
        self.limit_price_diff = tk.StringVar(value="0.5")  # 0.5% default
        price_diff_entry = ttk.Entry(limit_frame, width=8, textvariable=self.limit_price_diff)
        price_diff_entry.grid(row=row, column=1, padx=5, pady=2, sticky="w")
        self.add_tooltip(price_diff_entry, "Cancel if price moves this % away from limit price")
        row += 1
        
        # Action when limit order is canceled
        ttk.Label(limit_frame, text="On Cancel:").grid(row=row, column=0, padx=5, pady=2, sticky="w")
        self.limit_order_action = tk.StringVar(value="market")
        actions = ttk.Combobox(limit_frame, width=10, textvariable=self.limit_order_action)
        actions['values'] = ("market", "adjust", "cancel")
        actions.grid(row=row, column=1, padx=5, pady=2, sticky="w")
        self.add_tooltip(actions, "Action when limit order is canceled: market (replace with market order), adjust (adjust limit price), cancel (cancel trade)")
        row += 1
        
        # Add a separator
        ttk.Separator(limit_frame, orient="horizontal").grid(row=row, column=0, columnspan=2, sticky="ew", pady=5)
        row += 1
        
        # Add explanation text
        explanation = ttk.Label(limit_frame, text="These settings control how unfilled limit orders are handled:", 
                            wraplength=350, justify="left")
        explanation.grid(row=row, column=0, columnspan=2, sticky="w", padx=5, pady=2)
        row += 1
        
        # Add bullet points for each action
        market_text = "• Market: Replace with market order at current price"
        adjust_text = "• Adjust: Create new limit order at current price"
        cancel_text = "• Cancel: Cancel the trade entirely"
        
        ttk.Label(limit_frame, text=market_text, wraplength=350, justify="left").grid(
            row=row, column=0, columnspan=2, sticky="w", padx=5, pady=1)
        row += 1
        
        ttk.Label(limit_frame, text=adjust_text, wraplength=350, justify="left").grid(
            row=row, column=0, columnspan=2, sticky="w", padx=5, pady=1)
        row += 1
        
        ttk.Label(limit_frame, text=cancel_text, wraplength=350, justify="left").grid(
            row=row, column=0, columnspan=2, sticky="w", padx=5, pady=1)

    def create_parameter_entry(self, parent, row, label_text, variable, default_value, tooltip_text=None):
        """Create a parameter entry with proper styling and optional tooltip"""
        try:
            # Create label
            ttk.Label(parent, text=label_text).grid(row=row, column=0, padx=5, pady=2, sticky="w")
            
            # Create entry with custom style for better visibility
            entry = ttk.Entry(parent, width=8, textvariable=variable, style='Param.TEntry')
            entry.grid(row=row, column=1, padx=5, pady=2)
            
            # Set default value if variable is empty
            if not variable.get():
                variable.set(default_value)
            
            # Add tooltip if provided
            if tooltip_text:
                self.add_tooltip(entry, tooltip_text)
                
            return entry
            
        except Exception as e:
            self.log_trade(f"Error creating parameter entry: {str(e)}")
            return None

    def toggle_order_type(self):
        """Toggle between limit and market orders and update fee calculations"""
        try:
            # Get the current state of the checkbox
            use_limit_orders = self.use_limit_orders_var.get()
            
            # Update the fee calculation
            if use_limit_orders:
                # Using limit orders (maker fees)
                self.total_fee_percentage = self.maker_fee * 2  # Entry and exit
                order_type = "Limit"
                fee_type = "maker"
                fee_pct = self.maker_fee * 100
            else:
                # Using market orders (taker fees)
                self.total_fee_percentage = self.taker_fee * 2  # Entry and exit
                order_type = "Market"
                fee_type = "taker"
                fee_pct = self.taker_fee * 100
            
            # Update the total fees label
            total_fee_pct = self.total_fee_percentage * 100
            if hasattr(self, 'total_fees_label'):
                self.total_fees_label.config(text=f"Total Round-Trip Fee: {total_fee_pct:.2f}%")
            
            # Log the change
            self.log_trade(f"Order type set to {order_type} orders ({fee_type} fee: {fee_pct:.2f}%)")
            self.log_trade(f"Total round-trip fee: {total_fee_pct:.2f}%")
            
            # Update tooltips
            if hasattr(self, 'profit_target_entry'):
                self.add_tooltip(self.profit_target_entry, 
                    f"Target profit percentage for trades (min: {total_fee_pct:.2f}% to cover fees)")
            
            # Show/hide limit order settings based on selection
            if hasattr(self, 'limit_frame'):
                if use_limit_orders:
                    self.limit_frame.grid(row=1, column=2, padx=5, pady=5, sticky="nw")
                else:
                    self.limit_frame.grid_forget()
            
        except Exception as e:
            self.log_trade(f"Error toggling order type: {str(e)}")

    def setup_gui(self):
        """Set up the GUI components"""
        try:
            # Create main frames
            control_frame = ttk.Frame(self.root)
            control_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)
            
            # Add control buttons at the top
            buttons_frame = ttk.Frame(control_frame)
            buttons_frame.pack(fill=tk.X, padx=5, pady=5)
            
            # First row of buttons
            first_row = ttk.Frame(buttons_frame)
            first_row.pack(fill=tk.X, pady=2)
            
            # Start/Stop button
            self.start_button = ttk.Button(first_row, text="Start", command=self.start_bot)
            self.start_button.pack(side=tk.LEFT, padx=5)
            
            # Mode button (Paper/Real)
            self.mode_var = tk.StringVar(value="Paper Trading")
            self.mode_button = ttk.Button(first_row, text="Mode: Paper", command=self.toggle_trading_mode)
            self.mode_button.pack(side=tk.LEFT, padx=5)
            
            # Add timer label
            self.timer_label = ttk.Label(first_row, text="Runtime: 00:00:00")
            self.timer_label.pack(side=tk.RIGHT, padx=10)
            
            # Close All Positions button
            self.close_all_button = ttk.Button(first_row, text="Close All Positions", command=self.close_all_positions)
            self.close_all_button.pack(side=tk.LEFT, padx=5)
            
            # Close Profitable button
            self.close_profitable_button = ttk.Button(first_row, text="Close Profitable", command=self.close_profitable_positions)
            self.close_profitable_button.pack(side=tk.LEFT, padx=5)
            
            # Market override checkbox
            self.market_override_var = tk.BooleanVar(value=False)
            self.market_override_check = ttk.Checkbutton(
                first_row, 
                text="Override Market Conditions", 
                variable=self.market_override_var,
                command=self.toggle_market_override
            )
            self.market_override_check.pack(side=tk.LEFT, padx=5)
            
            # Second row of buttons
            second_row = ttk.Frame(buttons_frame)
            second_row.pack(fill=tk.X, pady=2)
            
            # Force Trade button
            self.force_trade_button = ttk.Button(second_row, text="Force Trade", command=self.force_trade)
            self.force_trade_button.pack(side=tk.LEFT, padx=5)

            # API Keys button
            self.api_keys_button = ttk.Button(second_row, text="API Keys", command=self.show_api_config)
            self.api_keys_button.pack(side=tk.LEFT, padx=5)

            # Verify API button
            self.verify_api_button = ttk.Button(second_row, text="Verify API", command=self.verify_api_keys)
            self.verify_api_button.pack(side=tk.LEFT, padx=5)

            # Apply Conditions button
            self.apply_conditions_button = ttk.Button(second_row, text="Apply Conditions", command=self.live_update_conditions)
            self.apply_conditions_button.pack(side=tk.LEFT, padx=5)

            # Night Mode button in the control panel
            self.night_mode_button_top = ttk.Button(second_row, text="Dark Mode", command=self.toggle_night_mode)
            self.night_mode_button_top.pack(side=tk.LEFT, padx=5)
            
            # Create main parameter frame
            param_frame = ttk.Frame(control_frame)
            param_frame.pack(fill=tk.X, padx=5, pady=5)
            
            # Create a 3-column layout for parameters with equal width
            left_params = ttk.Frame(param_frame)
            left_params.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
            
            middle_params = ttk.Frame(param_frame)
            middle_params.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
            
            right_params = ttk.Frame(param_frame)
            right_params.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

            # === ENTRY CONDITIONS (Left Column) ===
            entry_frame = ttk.LabelFrame(left_params, text="Entry Conditions")
            entry_frame.pack(fill=tk.X, padx=5, pady=5)
            
            profit_target_row = ttk.Frame(entry_frame)
            profit_target_row.pack(fill=tk.X, pady=2)
            ttk.Label(profit_target_row, text="Profit Target (%)").pack(side=tk.LEFT, padx=5)
            ttk.Entry(profit_target_row, textvariable=self.profit_target).pack(side=tk.RIGHT, padx=5)
            
            price_rise_row = ttk.Frame(entry_frame)
            price_rise_row.pack(fill=tk.X, pady=2)
            ttk.Label(price_rise_row, text="Min Price Rise (%)").pack(side=tk.LEFT, padx=5)
            ttk.Entry(price_rise_row, textvariable=self.price_rise_min).pack(side=tk.RIGHT, padx=5)
            
            volume_surge_row = ttk.Frame(entry_frame)
            volume_surge_row.pack(fill=tk.X, pady=2)
            ttk.Label(volume_surge_row, text="Volume Surge (%)").pack(side=tk.LEFT, padx=5)
            ttk.Entry(volume_surge_row, textvariable=self.volume_surge).pack(side=tk.RIGHT, padx=5)
            
            min_volume_row = ttk.Frame(entry_frame)
            min_volume_row.pack(fill=tk.X, pady=2)
            ttk.Label(min_volume_row, text="Min Volume ($)").pack(side=tk.LEFT, padx=5)
            ttk.Entry(min_volume_row, textvariable=self.min_volume_entry).pack(side=tk.RIGHT, padx=5)
            
            # === EXIT CONDITIONS (Left Column) ===
            exit_frame = ttk.LabelFrame(left_params, text="Exit Conditions")
            exit_frame.pack(fill=tk.X, padx=5, pady=5)
            
            stop_loss_row = ttk.Frame(exit_frame)
            stop_loss_row.pack(fill=tk.X, pady=2)
            ttk.Label(stop_loss_row, text="Stop Loss (%)").pack(side=tk.LEFT, padx=5)
            ttk.Entry(stop_loss_row, textvariable=self.stop_loss).pack(side=tk.RIGHT, padx=5)
            
            trailing_stop_row = ttk.Frame(exit_frame)
            trailing_stop_row.pack(fill=tk.X, pady=2)
            ttk.Label(trailing_stop_row, text="Trailing Stop (%)").pack(side=tk.LEFT, padx=5)
            ttk.Entry(trailing_stop_row, textvariable=self.trailing_stop).pack(side=tk.RIGHT, padx=5)

            trailing_activation_row = ttk.Frame(exit_frame)
            trailing_activation_row.pack(fill=tk.X, pady=2)
            ttk.Label(trailing_activation_row, text="Trailing Activation (%)").pack(side=tk.LEFT, padx=5)
            ttk.Entry(trailing_activation_row, textvariable=self.trailing_activation).pack(side=tk.RIGHT, padx=5)
            
            # === VALIDATION CRITERIA (Left Column) ===
            validation_frame = ttk.LabelFrame(left_params, text="Validation Criteria")
            validation_frame.pack(fill=tk.X, padx=5, pady=5)
            
            # Create a frame for the validation criteria with two columns
            validation_grid = ttk.Frame(validation_frame)
            validation_grid.pack(fill=tk.X, pady=2)
            
            # Left column of validation criteria
            validation_left = ttk.Frame(validation_grid)
            validation_left.pack(side=tk.LEFT, fill=tk.X, expand=True)
            
            # Right column of validation criteria
            validation_right = ttk.Frame(validation_grid)
            validation_right.pack(side=tk.LEFT, fill=tk.X, expand=True)
            
            # Momentum Beta (Left column)
            beta_row = ttk.Frame(validation_left)
            beta_row.pack(fill=tk.X, pady=2)
            ttk.Label(beta_row, text="Momentum Beta").pack(side=tk.LEFT, padx=5)
            ttk.Entry(beta_row, textvariable=self.momentum_beta, width=8).pack(side=tk.RIGHT, padx=5)
            
            # Price Alpha (Left column)
            alpha_row = ttk.Frame(validation_left)
            alpha_row.pack(fill=tk.X, pady=2)
            ttk.Label(alpha_row, text="Price Alpha").pack(side=tk.LEFT, padx=5)
            ttk.Entry(alpha_row, textvariable=self.price_alpha, width=8).pack(side=tk.RIGHT, padx=5)
            
            # Time Theta (Left column)
            theta_row = ttk.Frame(validation_left)
            theta_row.pack(fill=tk.X, pady=2)
            ttk.Label(theta_row, text="Time Theta").pack(side=tk.LEFT, padx=5)
            ttk.Entry(theta_row, textvariable=self.momentum_theta, width=8).pack(side=tk.RIGHT, padx=5)
            
            # Vol Vega (Right column)
            vega_row = ttk.Frame(validation_right)
            vega_row.pack(fill=tk.X, pady=2)
            ttk.Label(vega_row, text="Vol Vega").pack(side=tk.LEFT, padx=5)
            ttk.Entry(vega_row, textvariable=self.vol_vega, width=8).pack(side=tk.RIGHT, padx=5)
            
            # Volume Rho (Right column)
            rho_row = ttk.Frame(validation_right)
            rho_row.pack(fill=tk.X, pady=2)
            ttk.Label(rho_row, text="Volume Rho").pack(side=tk.LEFT, padx=5)
            ttk.Entry(rho_row, textvariable=self.volume_rho, width=8).pack(side=tk.RIGHT, padx=5)

            # === RISK MANAGEMENT (Middle Column) ===
            risk_frame = ttk.LabelFrame(middle_params, text="Risk Management")
            risk_frame.pack(fill=tk.X, padx=5, pady=5)
            
            position_size_row = ttk.Frame(risk_frame)
            position_size_row.pack(fill=tk.X, pady=2)
            ttk.Label(position_size_row, text="Position Size ($)").pack(side=tk.LEFT, padx=5)
            ttk.Entry(position_size_row, textvariable=self.position_size).pack(side=tk.RIGHT, padx=5)
            
            max_trades_row = ttk.Frame(risk_frame)
            max_trades_row.pack(fill=tk.X, pady=2)
            ttk.Label(max_trades_row, text="Max Trades").pack(side=tk.LEFT, padx=5)
            ttk.Entry(max_trades_row, textvariable=self.max_trades_entry).pack(side=tk.RIGHT, padx=5)
            
            # Add missing risk management parameters
            max_position_row = ttk.Frame(risk_frame)
            max_position_row.pack(fill=tk.X, pady=2)
            ttk.Label(max_position_row, text="Max Position (% Balance)").pack(side=tk.LEFT, padx=5)
            ttk.Entry(max_position_row, textvariable=self.max_position_percent).pack(side=tk.RIGHT, padx=5)
            
            daily_loss_row = ttk.Frame(risk_frame)
            daily_loss_row.pack(fill=tk.X, pady=2)
            ttk.Label(daily_loss_row, text="Daily Loss Limit (%)").pack(side=tk.LEFT, padx=5)
            ttk.Entry(daily_loss_row, textvariable=self.daily_loss_limit).pack(side=tk.RIGHT, padx=5)
            
            # === MARKET FILTERS (Middle Column) ===
            filters_frame = ttk.LabelFrame(middle_params, text="Market Filters")
            filters_frame.pack(fill=tk.X, padx=5, pady=5)
            
            top_list_row = ttk.Frame(filters_frame)
            top_list_row.pack(fill=tk.X, pady=2)
            ttk.Label(top_list_row, text="Top List Size").pack(side=tk.LEFT, padx=5)
            ttk.Entry(top_list_row, textvariable=self.top_list_size).pack(side=tk.RIGHT, padx=5)
            
            max_vol_row = ttk.Frame(filters_frame)
            max_vol_row.pack(fill=tk.X, pady=2)
            ttk.Label(max_vol_row, text="Max Volatility (%)").pack(side=tk.LEFT, padx=5)
            ttk.Entry(max_vol_row, textvariable=self.max_volatility).pack(side=tk.RIGHT, padx=5)
            
            max_spread_row = ttk.Frame(filters_frame)
            max_spread_row.pack(fill=tk.X, pady=2)
            ttk.Label(max_spread_row, text="Max Spread (%)").pack(side=tk.LEFT, padx=5)
            ttk.Entry(max_spread_row, textvariable=self.max_spread).pack(side=tk.RIGHT, padx=5)
            
            # Add consecutive rises parameter to Market Filters
            consecutive_rises_row = ttk.Frame(filters_frame)
            consecutive_rises_row.pack(fill=tk.X, pady=2)
            ttk.Label(consecutive_rises_row, text="Consecutive Rises").pack(side=tk.LEFT, padx=5)
            ttk.Entry(consecutive_rises_row, textvariable=self.consecutive_rises).pack(side=tk.RIGHT, padx=5)
            
            momentum_row = ttk.Frame(filters_frame)
            momentum_row.pack(fill=tk.X, pady=2)
            ttk.Label(momentum_row, text="Momentum Min (%)").pack(side=tk.LEFT, padx=5)
            ttk.Entry(momentum_row, textvariable=self.momentum_threshold).pack(side=tk.RIGHT, padx=5)

            # Add Trend Filter checkbox
            trend_filter_row = ttk.Frame(filters_frame)
            trend_filter_row.pack(fill=tk.X, pady=2)
            ttk.Label(trend_filter_row, text="Use Trend Filter").pack(side=tk.LEFT, padx=5)
            trend_filter_check = ttk.Checkbutton(trend_filter_row, variable=self.use_trend_filter)
            trend_filter_check.pack(side=tk.RIGHT, padx=5)

            # Add Trend Strength parameter
            trend_strength_row = ttk.Frame(filters_frame)
            trend_strength_row.pack(fill=tk.X, pady=2)
            ttk.Label(trend_strength_row, text="Min Trend Strength").pack(side=tk.LEFT, padx=5)
            ttk.Entry(trend_strength_row, textvariable=self.trend_strength_min).pack(side=tk.RIGHT, padx=5)

            # Add a separator line
            ttk.Separator(filters_frame, orient='horizontal').pack(fill=tk.X, pady=5)

            # Compact advanced filters
            advanced_row1 = ttk.Frame(filters_frame)
            advanced_row1.pack(fill=tk.X, pady=2)
            sr_check = ttk.Checkbutton(advanced_row1, text="S/R", variable=self.use_support_resistance)
            sr_check.pack(side=tk.LEFT, padx=5)
            ttk.Entry(advanced_row1, textvariable=self.sr_lookback, width=4).pack(side=tk.LEFT, padx=2)
            ttk.Label(advanced_row1, text="/").pack(side=tk.LEFT)
            ttk.Entry(advanced_row1, textvariable=self.sr_threshold, width=4).pack(side=tk.LEFT, padx=2)
            ttk.Label(advanced_row1, text="%").pack(side=tk.LEFT)

            advanced_row2 = ttk.Frame(filters_frame)
            advanced_row2.pack(fill=tk.X, pady=2)
            pattern_check = ttk.Checkbutton(advanced_row2, text="Patterns", variable=self.use_candlestick_patterns)
            pattern_check.pack(side=tk.LEFT, padx=5)
            ttk.Entry(advanced_row2, textvariable=self.pattern_confidence_min, width=4).pack(side=tk.LEFT, padx=2)
            ttk.Label(advanced_row2, text="%").pack(side=tk.LEFT)

            volume_check = ttk.Checkbutton(advanced_row2, text="Vol Profile", variable=self.use_volume_profile)
            volume_check.pack(side=tk.LEFT, padx=20)
            ttk.Entry(advanced_row2, textvariable=self.volume_quality_min, width=4).pack(side=tk.LEFT, padx=2)

            # === TECHNICAL INDICATORS (Right Column) ===
            indicators_frame = ttk.LabelFrame(right_params, text="Technical Indicators")
            indicators_frame.pack(fill=tk.X, padx=5, pady=5)

            # Create a frame for the two columns using pack exclusively
            columns_frame = ttk.Frame(indicators_frame)
            columns_frame.pack(fill=tk.BOTH, expand=True)

            # Create two columns within the indicators frame
            left_indicators = ttk.Frame(columns_frame)
            left_indicators.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

            right_indicators = ttk.Frame(columns_frame)
            right_indicators.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

            # Left column indicators - using pack instead of grid
            # RSI Parameters
            rsi_row = ttk.Frame(left_indicators)
            rsi_row.pack(fill=tk.X, pady=2)
            ttk.Label(rsi_row, text="RSI Period").pack(side=tk.LEFT, padx=5)
            ttk.Entry(rsi_row, textvariable=self.rsi_period).pack(side=tk.RIGHT, padx=5)

            rsi_ob_row = ttk.Frame(left_indicators)
            rsi_ob_row.pack(fill=tk.X, pady=2)
            ttk.Label(rsi_ob_row, text="RSI Overbought").pack(side=tk.LEFT, padx=5)
            ttk.Entry(rsi_ob_row, textvariable=self.rsi_overbought).pack(side=tk.RIGHT, padx=5)

            rsi_os_row = ttk.Frame(left_indicators)
            rsi_os_row.pack(fill=tk.X, pady=2)
            ttk.Label(rsi_os_row, text="RSI Oversold").pack(side=tk.LEFT, padx=5)
            ttk.Entry(rsi_os_row, textvariable=self.rsi_oversold).pack(side=tk.RIGHT, padx=5)

            ema_short_row = ttk.Frame(left_indicators)
            ema_short_row.pack(fill=tk.X, pady=2)
            ttk.Label(ema_short_row, text="EMA Short").pack(side=tk.LEFT, padx=5)
            ttk.Entry(ema_short_row, textvariable=self.ema_short).pack(side=tk.RIGHT, padx=5)

            ema_long_row = ttk.Frame(left_indicators)
            ema_long_row.pack(fill=tk.X, pady=2)
            ttk.Label(ema_long_row, text="EMA Long").pack(side=tk.LEFT, padx=5)
            ttk.Entry(ema_long_row, textvariable=self.ema_long).pack(side=tk.RIGHT, padx=5)

            # Right column indicators - using pack instead of grid
            # MACD Parameters
            macd_fast_row = ttk.Frame(right_indicators)
            macd_fast_row.pack(fill=tk.X, pady=2)
            ttk.Label(macd_fast_row, text="MACD Fast").pack(side=tk.LEFT, padx=5)
            ttk.Entry(macd_fast_row, textvariable=self.macd_fast).pack(side=tk.RIGHT, padx=5)

            macd_slow_row = ttk.Frame(right_indicators)
            macd_slow_row.pack(fill=tk.X, pady=2)
            ttk.Label(macd_slow_row, text="MACD Slow").pack(side=tk.LEFT, padx=5)
            ttk.Entry(macd_slow_row, textvariable=self.macd_slow).pack(side=tk.RIGHT, padx=5)

            macd_signal_row = ttk.Frame(right_indicators)
            macd_signal_row.pack(fill=tk.X, pady=2)
            ttk.Label(macd_signal_row, text="MACD Signal").pack(side=tk.LEFT, padx=5)
            ttk.Entry(macd_signal_row, textvariable=self.macd_signal).pack(side=tk.RIGHT, padx=5)

            req_cond_row = ttk.Frame(right_indicators)
            req_cond_row.pack(fill=tk.X, pady=2)
            ttk.Label(req_cond_row, text="Required Conditions").pack(side=tk.LEFT, padx=5)
            ttk.Entry(req_cond_row, textvariable=self.required_conditions).pack(side=tk.RIGHT, padx=5)

            # === FEE STRUCTURE (Right Column) ===
            fee_frame = ttk.LabelFrame(right_params, text="Fee Structure")
            fee_frame.pack(fill=tk.X, padx=5, pady=5)
            
            maker_fee_row = ttk.Frame(fee_frame)
            maker_fee_row.pack(fill=tk.X, pady=2)
            ttk.Label(maker_fee_row, text="Maker Fee (%)").pack(side=tk.LEFT, padx=5)
            ttk.Entry(maker_fee_row, textvariable=self.maker_fee_var).pack(side=tk.RIGHT, padx=5)
            
            taker_fee_row = ttk.Frame(fee_frame)
            taker_fee_row.pack(fill=tk.X, pady=2)
            ttk.Label(taker_fee_row, text="Taker Fee (%)").pack(side=tk.LEFT, padx=5)
            ttk.Entry(taker_fee_row, textvariable=self.taker_fee_var).pack(side=tk.RIGHT, padx=5)
            
            total_fee_row = ttk.Frame(fee_frame)
            total_fee_row.pack(fill=tk.X, pady=2)
            ttk.Label(total_fee_row, text="Total Round-Trip Fee:").pack(side=tk.LEFT, padx=5)
            self.total_fees_label = ttk.Label(total_fee_row, text="0.80%")
            self.total_fees_label.pack(side=tk.RIGHT, padx=5)
            
            # Add checkbox for limit orders
            limit_orders_row = ttk.Frame(fee_frame)
            limit_orders_row.pack(fill=tk.X, pady=2)
            self.use_limit_orders_var = tk.BooleanVar(value=False)
            limit_orders_check = ttk.Checkbutton(limit_orders_row, text="Use Limit Orders", 
                                               variable=self.use_limit_orders_var,
                                               command=self.toggle_order_type)
            limit_orders_check.pack(side=tk.LEFT, padx=5)

            # Create main content area with better proportions
            main_content = ttk.Frame(self.root)
            main_content.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=5)
            
            # Create left and right columns for main content with better proportions
            left_column = ttk.Frame(main_content)
            left_column.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
            
            right_column = ttk.Frame(main_content)
            right_column.pack(side=tk.RIGHT, fill=tk.BOTH, expand=False, padx=5)
            
            # Create a better chart frame with proper initialization
            self.chart_frame = ttk.LabelFrame(left_column, text="Active Trades (% Change from Entry)")
            self.chart_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            
            # Trading Log with increased height
            log_frame = ttk.LabelFrame(left_column, text="Trading Log")
            log_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            self.log_text = scrolledtext.ScrolledText(log_frame)
            self.log_text.pack(fill=tk.BOTH, expand=True)

            # === RIGHT COLUMN CONTENTS ===
            # Performance Metrics Frame with more details
            metrics_frame = ttk.LabelFrame(right_column, text="Performance Metrics")
            metrics_frame.pack(fill=tk.X, padx=5, pady=5)

            # Create a grid layout for metrics
            self.metrics_labels = {}
            metrics_data = [
                ('total_profit', 'Total Profit:'),
                ('total_fees', 'Total Fees:'),
                ('net_profit', 'Net Profit:'),
                ('win_rate', 'Win Rate:'),
                ('total_trades', 'Total Trades:'),
                ('paper_balance', 'Paper Balance:')
            ]
            
            for row_idx, (key, label_text) in enumerate(metrics_data):
                label = ttk.Label(metrics_frame, text=label_text)
                label.grid(row=row_idx, column=0, sticky='w', padx=5, pady=2)
                
                value_label = ttk.Label(metrics_frame, text="0")
                value_label.grid(row=row_idx, column=1, sticky='e', padx=5, pady=2)
                self.metrics_labels[key] = value_label
            
            # Store reference to metrics frame
            self.metrics_frame = metrics_frame
            
            # Active Trades Frame - make it smaller by reducing expand or setting a fixed height
            trades_frame = ttk.LabelFrame(right_column, text="Active Trades")
            trades_frame.pack(fill=tk.BOTH, expand=False, padx=5, pady=5)  # Changed expand=True to expand=False
            self.trades_text = scrolledtext.ScrolledText(trades_frame, height=8)  # Set a specific height
            self.trades_text.pack(fill=tk.BOTH, expand=True)

            # Trade History Frame - make it larger by ensuring it expands
            history_frame = ttk.LabelFrame(right_column, text="Trade History")
            history_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)  # Keep expand=True
            self.history_text = scrolledtext.ScrolledText(history_frame)
            self.history_text.pack(fill=tk.BOTH, expand=True)

            # Night Mode Button Frame
            night_mode_frame = ttk.Frame(right_column)
            night_mode_frame.pack(fill=tk.X, padx=5, pady=5)

            self.night_mode_button = ttk.Button(
                night_mode_frame,
                text="Dark Mode",
                command=self.toggle_night_mode
            )
            self.night_mode_button.pack(fill=tk.X)

            # Set minimum window size
            self.root.minsize(1200, 800)

            # Status bar at the bottom of the window
            status_frame = ttk.Frame(self.root)
            status_frame.pack(fill=tk.X, side=tk.BOTTOM, padx=5, pady=2)

            # Status label
            self.status_label = ttk.Label(status_frame, text="Ready")
            self.status_label.pack(side=tk.LEFT)

            # Initialize chart
            self.setup_chart()
            self.initialize_fees()
            
            # Update metrics display
            self.update_metrics()
            
        except Exception as e:
            self.log_trade(f"Error setting up GUI: {str(e)}")
            import traceback
            self.log_trade(traceback.format_exc())
            raise

    def update_order_type(self):
        """Update fee calculations based on order type selection"""
        try:
            # Get the current state of the checkbox
            use_limit_orders = self.use_limit_orders_var.get()
            
            # Update the fee calculation
            if use_limit_orders:
                # Using limit orders (maker fees)
                self.total_fee_percentage = self.maker_fee * 2  # Entry and exit
                order_type = "Limit"
                fee_type = "maker"
                fee_pct = self.maker_fee * 100
            else:
                # Using market orders (taker fees)
                self.total_fee_percentage = self.taker_fee * 2  # Entry and exit
                order_type = "Market"
                fee_type = "taker"
                fee_pct = self.taker_fee * 100
            
            # Update the total fees label
            total_fee_pct = self.total_fee_percentage * 100
            if hasattr(self, 'total_fees_label'):
                self.total_fees_label.config(text=f"Total Round-Trip Fee: {total_fee_pct:.2f}%")
            
            # Log the change
            self.log_trade(f"Order type set to {order_type} orders ({fee_type} fee: {fee_pct:.2f}%)")
            self.log_trade(f"Total round-trip fee: {total_fee_pct:.2f}%")
            
            # Update tooltips
            if hasattr(self, 'profit_target'):
                self.add_tooltip(self.profit_target, 
                    f"Target profit percentage for trades (min: {total_fee_pct:.2f}% to cover fees)")
            
            # Show/hide limit order settings based on selection
            if hasattr(self, 'limit_frame'):
                if use_limit_orders:
                    self.limit_frame.pack(fill="x", padx=5, pady=5)
                else:
                    self.limit_frame.pack_forget()
            
        except Exception as e:
            self.log_trade(f"Error updating order type: {str(e)}")

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
            
            # Check if we're using limit orders
            use_limit_orders = self.use_limit_orders_var.get() if hasattr(self, 'use_limit_orders_var') else False
            
            # Calculate total round-trip fee based on order type
            if use_limit_orders:
                # Using limit orders (maker fees)
                self.total_fee_percentage = self.maker_fee * 2  # Entry and exit
                fee_type = "maker"
            else:
                # Using market orders (taker fees)
                self.total_fee_percentage = self.taker_fee * 2  # Entry and exit
                fee_type = "taker"
            
            # Debug: Print the updated instance variables
            self.log_trade(f"DEBUG: self.maker_fee = {self.maker_fee}")
            self.log_trade(f"DEBUG: self.taker_fee = {self.taker_fee}")
            self.log_trade(f"DEBUG: self.total_fee_percentage = {self.total_fee_percentage}")
            self.log_trade(f"DEBUG: using limit orders = {use_limit_orders}")
            
            # Update the total fees label
            total_fee_pct = self.total_fee_percentage * 100
            if hasattr(self, 'total_fees_label'):
                self.total_fees_label.config(text=f"Total Round-Trip Fee: {total_fee_pct:.2f}%")
            
            # Log the update
            self.log_trade(f"Fee structure updated: Maker {maker_fee*100:.2f}%, Taker {taker_fee*100:.2f}%, " +
                        f"Total ({fee_type}*2) {total_fee_pct:.2f}%")
            
            # Update tooltips
            if hasattr(self, 'profit_target'):
                self.add_tooltip(self.profit_target, 
                    f"Target profit percentage for trades (min: {total_fee_pct:.2f}% to cover fees)")
            
            # Validate parameters to ensure profit target is above fees
            self.validate_parameters()
            
            return True
        except ValueError as e:
            self.log_trade(f"Error updating fees: {str(e)}")
            return False

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
            self.add_tooltip(self.consecutive_rises,
                "Number of consecutive price increases required before entry")
            self.add_tooltip(self.required_conditions, 
                "Minimum number of conditions that must be met for entry")
            self.add_tooltip(self.max_spread,
                "Maximum bid-ask spread percentage allowed for trading")
                
            # Validation Criteria Tooltips
            self.add_tooltip(self.momentum_beta, 
                "Trend strength indicator (0.1-1.0). Higher values require stronger trends")
            self.add_tooltip(self.price_alpha, 
                "Price momentum indicator (0.01-0.1). Higher values require faster price movement")
            self.add_tooltip(self.momentum_theta, 
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
            
        
            # Technical Indicators Tooltips
            self.add_tooltip(self.ema_short, 
                "Short-term EMA period for crossover signals (typical: 5)")
            self.add_tooltip(self.ema_long, 
                "Long-term EMA period for crossover signals (typical: 15)")
            self.add_tooltip(self.required_conditions, 
                "Minimum number of conditions that must be met for trade entry")
            self.add_tooltip(self.max_spread, 
                "Maximum bid-ask spread percentage allowed for trading")
                
        except Exception as e:
            self.log_trade(f"Error setting up tooltips: {str(e)}")

    def add_tooltip(self, widget, text):
        """Add tooltip to widget with improved styling and positioning"""
        try:
            # Skip if widget is not a proper widget (e.g., StringVar)
            if not hasattr(widget, 'bind'):
                return
            
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

    def enforce_max_trades_limit(self):
        """Enforce the maximum trades limit by closing excess trades if necessary"""
        try:
            max_trades = int(self.max_trades_entry.get()) if hasattr(self, 'max_trades_entry') else 3
            current_trades = len(self.active_trades)
            
            if current_trades <= max_trades:
                return  # No action needed
                
            # We have more trades than allowed, close the least profitable ones
            self.log_trade(f"WARNING: Found {current_trades} active trades but max is {max_trades}. Closing excess trades.")
            
            # Sort trades by profit percentage (ascending)
            sorted_trades = sorted(
                self.active_trades.items(),
                key=lambda x: self.calculate_profit_percentage(x[1]),
            )
            
            # Close the least profitable trades until we're at the limit
            trades_to_close = sorted_trades[:current_trades - max_trades]
            for trade_id, trade in trades_to_close:
                # Use the existing method to get current price (fetch_ticker)
                try:
                    ticker = self.exchange.fetch_ticker(trade['symbol'])
                    current_price = float(ticker['last'])
                    self.log_trade(f"Closing excess trade {trade['symbol']} to enforce max trades limit")
                    self.close_trade(trade_id, trade, current_price, "max trades enforcement")
                except Exception as e:
                    self.log_trade(f"Error closing excess trade {trade['symbol']}: {str(e)}")
                
        except Exception as e:
            self.log_trade(f"Error enforcing max trades limit: {str(e)}")
            
    def calculate_profit_percentage(self, trade):
        """Calculate current profit percentage for a trade"""
        try:
            # Use the existing method to get current price
            ticker = self.exchange.fetch_ticker(trade['symbol'])
            current_price = float(ticker['last'])
            entry_price = float(trade['entry_price'])
            return ((current_price - entry_price) / entry_price) * 100
        except Exception:
            return -999  # Return a very negative number to prioritize closing this trade

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

    def show_symbol_selector(self):
        """Show a dialog to select a trading symbol"""
        try:
            # Get available symbols
            markets = self.exchange.load_markets()
            usd_symbols = [symbol for symbol in markets.keys() if symbol.endswith('/USD') and markets[symbol].get('active', True)]
            
            # Sort by volume if available
            if hasattr(self, 'data_manager') and hasattr(self.data_manager, 'get_sorted_symbols'):
                sorted_symbols = self.data_manager.get_sorted_symbols()
                # Filter to only include USD pairs
                sorted_usd_symbols = [s for s in sorted_symbols if s.endswith('/USD')]
                # Combine with the full list to ensure we don't miss any
                symbol_list = sorted(set(sorted_usd_symbols + usd_symbols))
            else:
                symbol_list = sorted(usd_symbols)
            
            # Create a simple dialog
            dialog = tk.Toplevel(self.root)
            dialog.title("Select Symbol")
            dialog.geometry("300x400")
            dialog.transient(self.root)
            dialog.grab_set()
            
            # Add a label
            label = ttk.Label(dialog, text="Select a symbol to trade:")
            label.pack(pady=10)
            
            # Add a listbox with scrollbar
            frame = ttk.Frame(dialog)
            frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            scrollbar = ttk.Scrollbar(frame)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            
            listbox = tk.Listbox(frame, yscrollcommand=scrollbar.set)
            listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            
            scrollbar.config(command=listbox.yview)
            
            # Add symbols to listbox
            for symbol in symbol_list:
                listbox.insert(tk.END, symbol)
            
            # Add a search entry
            search_var = tk.StringVar()
            search_var.trace("w", lambda name, index, mode, sv=search_var: self._filter_symbol_list(sv, listbox, symbol_list))
            search_entry = ttk.Entry(dialog, textvariable=search_var)
            search_entry.pack(fill=tk.X, padx=10, pady=5)
            ttk.Label(dialog, text="Type to search").pack()
            
            # Add buttons
            button_frame = ttk.Frame(dialog)
            button_frame.pack(fill=tk.X, padx=10, pady=10)
            
            selected_symbol = [None]  # Use a list to store the selected symbol
            
            def on_select():
                selection = listbox.curselection()
                if selection:
                    selected_symbol[0] = listbox.get(selection[0])
                    dialog.destroy()
            
            def on_cancel():
                dialog.destroy()
            
            select_button = ttk.Button(button_frame, text="Select", command=on_select)
            select_button.pack(side=tk.LEFT, padx=5)
            
            cancel_button = ttk.Button(button_frame, text="Cancel", command=on_cancel)
            cancel_button.pack(side=tk.RIGHT, padx=5)
            
            # Wait for the dialog to close
            self.root.wait_window(dialog)
            
            # Return the selected symbol
            return selected_symbol[0]
            
        except Exception as e:
            self.log_trade(f"Error showing symbol selector: {str(e)}")
            return None

    def _filter_symbol_list(self, search_var, listbox, symbol_list):
        """Filter the symbol list based on search text"""
        search_text = search_var.get().upper()
        listbox.delete(0, tk.END)
        for symbol in symbol_list:
            if search_text in symbol.upper():
                listbox.insert(tk.END, symbol)

    def execute_real_trade(self, symbol, quantity, price=None, use_limit_orders=True):
        """Execute a real trade on the exchange"""
        try:
            self.log_trade(f"Executing real {'LIMIT' if use_limit_orders else 'MARKET'} trade for {symbol}")
            
            if use_limit_orders and price is not None:
                # Create limit buy order
                order = self.exchange.create_limit_buy_order(symbol, quantity, price)
                self.log_trade(f"Created LIMIT buy order: {quantity:.6f} {symbol} at ${price:.6f}")
            else:
                # Create market buy order
                order = self.exchange.create_market_buy_order(symbol, quantity)
                self.log_trade(f"Created MARKET buy order: {quantity:.6f} {symbol}")
                
            return order
        except Exception as e:
            self.log_trade(f"Error executing real trade: {str(e)}")
            raise  # Re-raise to be handled by the caller


    def force_trade(self):
        """Force a trade on the selected symbol"""
        try:
            # Get selected symbol
            symbol = self.symbol_var.get() if hasattr(self, 'symbol_var') else None
            if not symbol:
                messagebox.showwarning("Warning", "Please select a symbol first")
                return False
                
            # Get current price
            try:
                ticker = self.exchange.fetch_ticker(symbol)
                price = float(ticker['last'])
            except Exception as e:
                self.log_trade(f"Error fetching price for {symbol}: {str(e)}")
                messagebox.showerror("Error", f"Could not fetch price for {symbol}")
                return False
                
            # Get position size
            position_size = float(self.position_size.get())
            
            # Check if we're using limit orders
            use_limit_orders = self.use_limit_orders_var.get() if hasattr(self, 'use_limit_orders_var') else True
            order_type = "LIMIT" if use_limit_orders else "MARKET"
            
            # Confirm with user
            if not messagebox.askyesno("Confirm Trade", 
                                    f"Execute {order_type} order for {symbol} at ${price:.4f}?\n" +
                                    f"Position size: ${position_size:.2f}"):
                return False
                
            # Execute trade based on mode
            if self.is_paper_trading:
                # Paper trading mode
                self.log_trade(f"[FORCE TRADE] Executing paper trade for {symbol} at ${price:.4f}")
                trade_id = self.execute_trade(symbol, price, use_limit_orders=use_limit_orders)
                if trade_id:
                    messagebox.showinfo("Trade Executed", f"Paper trade executed for {symbol}")
                    return True
                else:
                    messagebox.showerror("Error", "Failed to execute paper trade")
                    return False
                    
            else:
                # Real trading mode
                self.log_trade(f"[FORCE TRADE] Executing real trade for {symbol} at ${price:.4f}")
                
                # Calculate quantity
                quantity = position_size / price
                
                if use_limit_orders:
                    # Create limit buy order
                    order = self.exchange.create_limit_buy_order(symbol, quantity, price)
                    self.log_trade(f"[FORCE TRADE] Real LIMIT trade executed: {quantity:.6f} {symbol} at ${price:.4f}")
                else:
                    # Create market buy order
                    order = self.exchange.create_market_buy_order(symbol, quantity)
                    self.log_trade(f"[FORCE TRADE] Real MARKET trade executed: {quantity:.6f} {symbol}")
                
                # Log success
                self.log_trade(f"Order details: {order}")
                
                # Create trade object and add to active trades
                self.execute_trade(symbol, price, use_limit_orders=use_limit_orders)
                
                self.update_trades_display()
                messagebox.showinfo("Trade Executed", f"Real {order_type} trade executed: {quantity:.6f} {symbol}")
                
                return True
                
        except Exception as e:
            self.log_trade(f"Error forcing trade: {str(e)}")
            messagebox.showerror("Error", f"Failed to execute trade: {str(e)}")
            return False

    def advanced_checks(self, symbol, df):
        """Advanced market checks including order book analysis and RSI"""
        try:
            # Check if market override is active
            market_override = hasattr(self, 'market_override_var') and self.market_override_var.get()
            
            # 1. EMA Cross (5/15)
            if 'ema_5' in df.columns and 'ema_15' in df.columns and len(df) >= 2:
                ema_cross = (df['ema_5'].iloc[-2] < df['ema_15'].iloc[-2]) and \
                            (df['ema_5'].iloc[-1] > df['ema_15'].iloc[-1])
                self.log_trade(f"EMA Cross check for {symbol}: {'Bullish' if ema_cross else 'No cross'}")
            else:
                self.log_trade(f"Insufficient EMA data for {symbol}")
                ema_cross = False
            
            # 2. Custom EMA Cross (using user-defined periods)
            short_period = int(self.ema_short.get())
            long_period = int(self.ema_long.get())
            ema_short_col = f'ema_{short_period}'
            ema_long_col = f'ema_{long_period}'
            
            custom_ema_cross = False
            if ema_short_col in df.columns and ema_long_col in df.columns and len(df) >= 2:
                custom_ema_cross = (df[ema_short_col].iloc[-2] < df[ema_long_col].iloc[-2]) and \
                                  (df[ema_short_col].iloc[-1] > df[ema_long_col].iloc[-1])
                self.log_trade(f"Custom EMA Cross ({short_period}/{long_period}) check for {symbol}: {'Bullish' if custom_ema_cross else 'No cross'}")
                
                # Use custom EMA cross if available
                ema_cross = ema_cross or custom_ema_cross
            
            # 3. MACD Check
            macd_bullish = False
            if 'macd_line' in df.columns and 'macd_signal_line' in df.columns and len(df) >= 2:
                # Check for MACD line crossing above signal line
                macd_cross = (df['macd_line'].iloc[-2] < df['macd_signal_line'].iloc[-2]) and \
                             (df['macd_line'].iloc[-1] > df['macd_signal_line'].iloc[-1])
                             
                # Check for MACD histogram turning positive
                macd_hist_positive = df['macd_histogram'].iloc[-1] > 0
                
                macd_bullish = macd_cross or macd_hist_positive
                self.log_trade(f"MACD check for {symbol}: {'Bullish' if macd_bullish else 'Not bullish'}")
            
            # 4. RSI Check
            rsi_period = int(self.rsi_period.get()) if hasattr(self, 'rsi_period') else 14
            rsi_column = f'rsi_{rsi_period}'
            
            # Check if RSI column exists and has valid data
            if rsi_column in df.columns:
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
                
                # Check if RSI is coming out of oversold (bullish)
                rsi_bullish = rsi_value > rsi_oversold and rsi_value < 50
                if rsi_bullish:
                    self.log_trade(f"RSI bullish for {symbol}: {rsi_value:.2f} (coming out of oversold)")
            else:
                self.log_trade(f"RSI data not available for {symbol}")
                rsi_bullish = False
            
            # Apply different criteria based on market override
            if market_override:
                self.log_trade(f"[OK] Market override ACTIVE - using relaxed criteria")
                return any([
                    ema_cross,          # EMA crossover
                    macd_bullish,       # MACD bullish signal
                    rsi_bullish         # RSI bullish signal
                ])
            else:
                # Standard criteria - require more confirmation
                return all([
                    ema_cross,          # EMA crossover
                    any([macd_bullish, rsi_bullish])  # At least one of MACD or RSI must be bullish
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
        """Update the price chart with active trades using a fixed time window"""
        try:
            # Check if chart objects exist
            if not hasattr(self, 'ax') or not hasattr(self, 'canvas'):
                self.log_trade("Chart objects not initialized, skipping chart update")
                return
                
            # Clear the current plot
            self.ax.clear()
            
            # Set colors based on night mode
            bg_color = '#2d2d2d' if self.night_mode else 'white'
            fg_color = 'white' if self.night_mode else 'black'
            grid_color = '#3d3d3d' if self.night_mode else '#cccccc'
            
            # Set background color
            self.ax.set_facecolor(bg_color)
            self.ax.tick_params(axis='both', colors=fg_color)
            self.ax.grid(True, color=grid_color, linestyle='--', alpha=0.5)
            
            # Set spine colors
            for spine in self.ax.spines.values():
                spine.set_color(grid_color)
            
            # Calculate time window (last 30 minutes)
            now = datetime.now()
            time_window = timedelta(minutes=30)
            window_start = now - time_window
            
            # Draw key levels
            try:
                profit_target = float(self.profit_target.get())
                stop_loss = float(self.stop_loss.get())
                
                # Draw profit target line
                self.ax.axhline(y=profit_target, color='green', linestyle=':', alpha=0.5, 
                            label=f"Profit Target ({profit_target}%)")
                
                # Draw stop loss line
                self.ax.axhline(y=-stop_loss, color='red', linestyle=':', alpha=0.5,
                            label=f"Stop Loss (-{stop_loss}%)")
                
                # Draw zero line
                self.ax.axhline(y=0, color=grid_color, linestyle='-', alpha=0.3)
                
            except Exception as e:
                self.log_trade(f"Error drawing key levels: {str(e)}")
            
            # Plot active trades
            if not self.active_trades:
                self.ax.set_title("No Active Trades", color=fg_color)
                self.canvas.draw()
                return
            
            # Initialize price history if needed
            if not hasattr(self, 'price_history'):
                self.price_history = {}
            
            # Plot each active trade
            for trade_id, trade in self.active_trades.items():
                symbol = trade['symbol']
                entry_price = trade['entry_price']
                entry_time = trade.get('entry_time', now - timedelta(minutes=1))
                
                # Initialize price history for this symbol if it doesn't exist
                if symbol not in self.price_history:
                    self.price_history[symbol] = []
                
                # Add current price point
                try:
                    current_price = float(self.exchange.fetch_ticker(symbol)['last'])
                    self.price_history[symbol].append((now, current_price))
                except Exception as e:
                    self.log_trade(f"Error fetching price for {symbol}: {str(e)}")
                    continue
                
                # Filter price history to time window
                self.price_history[symbol] = [
                    (t, p) for t, p in self.price_history[symbol] 
                    if t >= window_start
                ]
                
                # Plot if we have data points
                if self.price_history[symbol]:
                    times = [point[0] for point in self.price_history[symbol]]
                    prices = [point[1] for point in self.price_history[symbol]]
                    
                    # Calculate percentage change from entry
                    prices_pct = [(p - entry_price) / entry_price * 100 for p in prices]
                    
                    # Plot with proper formatting
                    self.ax.plot(times, prices_pct, 
                            label=f"{symbol} ({prices_pct[-1]:.2f}%)",
                            linewidth=2)
            
            # Set title and labels
            self.ax.set_title("Active Trades (% Change from Entry)", color=fg_color)
            self.ax.set_xlabel('Time', color=fg_color)
            self.ax.set_ylabel('Price Change (%)', color=fg_color)
            
            # Format x-axis to show time properly
            self.ax.xaxis.set_major_formatter(DateFormatter('%H:%M:%S'))
            
            # Set x-axis limits to time window
            self.ax.set_xlim(window_start, now)
            
            # Set reasonable y-axis limits
            try:
                profit_target = float(self.profit_target.get())
                stop_loss = float(self.stop_loss.get())
                
                # Ensure the profit target and stop loss lines are visible
                self.ax.set_ylim(
                    min(-stop_loss * 1.5, -0.5),
                    max(profit_target * 1.5, 0.5)
                )
            except Exception as e:
                self.log_trade(f"Error setting y-axis limits: {str(e)}")
            
            # Add legend
            self.ax.legend(loc='upper left')
            
            # Rotate x-axis labels for better readability
            plt.xticks(rotation=45)
            
            # Adjust layout to prevent label cutoff
            self.fig.tight_layout()
            
            # Draw the chart
            self.canvas.draw()
            
        except Exception as e:
            self.log_trade(f"Error updating chart: {str(e)}")

    def update_chart_theme(self, bg_color, fg_color):
        """Update chart colors to match the current theme"""
        try:
            if not hasattr(self, 'fig') or not hasattr(self, 'ax'):
                return
                
            # Update figure background
            self.fig.set_facecolor(bg_color)
            
            # Update axes colors
            self.ax.set_facecolor(bg_color)
            self.ax.tick_params(axis='both', colors=fg_color)
            
            # Update grid color
            grid_color = '#3d3d3d' if self.night_mode else '#cccccc'
            self.ax.grid(True, color=grid_color, linestyle='--', alpha=0.5)
            
            # Update spine colors
            for spine in self.ax.spines.values():
                spine.set_color(grid_color)
            
            # Update title and labels
            if self.ax.get_title():
                self.ax.set_title(self.ax.get_title(), color=fg_color)
            self.ax.set_xlabel(self.ax.get_xlabel(), color=fg_color)
            self.ax.set_ylabel(self.ax.get_ylabel(), color=fg_color)
            
            # Update legend if it exists
            if self.ax.get_legend():
                for text in self.ax.get_legend().get_texts():
                    text.set_color(fg_color)
            
            # Redraw canvas
            self.canvas.draw()
            
        except Exception as e:
            self.log_trade(f"Error updating chart theme: {str(e)}")

    def _update_chart_internal(self):
        """Internal chart update method that runs on main thread"""
        try:
            # Clear the figure
            self.ax.clear()
            
            # Set colors based on night mode
            bg_color = '#2d2d2d' if self.night_mode else 'white'
            fg_color = 'white' if self.night_mode else 'black'
            grid_color = '#3d3d3d' if self.night_mode else '#cccccc'
            
            # Set background color
            self.ax.set_facecolor(bg_color)
            
            # Always draw the key levels first, regardless of active trades
            try:
                # Draw zero line
                self.ax.axhline(y=0, color='gray', linestyle='--', alpha=0.7, linewidth=1.0)
                
                # Draw profit target line - make it more visible
                profit_target = float(self.profit_target.get())
                self.ax.axhline(y=profit_target, color='green', linestyle=':', alpha=0.9, linewidth=2.0)
                
                # Draw stop loss line - make it more visible
                stop_loss = float(self.stop_loss.get())
                self.ax.axhline(y=-stop_loss, color='red', linestyle=':', alpha=0.9, linewidth=2.0)
                
                # Add text labels for the lines
                self.ax.text(0, profit_target + 0.1, f"Profit Target: {profit_target}%", 
                        color='green', fontweight='bold')
                self.ax.text(0, -stop_loss - 0.1, f"Stop Loss: {stop_loss}%", 
                        color='red', fontweight='bold')
                
                self.log_trade(f"Drew key levels: PT={profit_target}%, SL={stop_loss}%")
            except Exception as e:
                self.log_trade(f"Error drawing key levels: {str(e)}")
            
            if not self.active_trades:
                self.ax.set_title("No Active Trades")
                self.canvas.draw()
                return
            
            # Plot each active trade
            for trade_id, trade in self.active_trades.items():
                symbol = trade['symbol']
                
                if symbol in self.price_history:
                    history = self.price_history[symbol]
                    
                    if history:
                        times = list(range(len(history)))
                        prices = [point[1] for point in history]
                        
                        # Calculate percentage change from entry
                        entry_price = trade['entry_price']
                        prices_pct = [(p - entry_price) / entry_price * 100 for p in prices]
                        
                        self.ax.plot(times, prices_pct, 
                                label=f"{symbol} ({prices_pct[-1]:.2f}%)",
                                linewidth=2)
            
            if self.active_trades:
                self.ax.set_title("Active Trades (% Change from Entry)")
                self.ax.set_xlabel("Time Points")
                self.ax.set_ylabel("Price Change (%)")
                self.ax.legend(loc='upper left')
                self.ax.grid(True, alpha=0.3)
                
                # Set reasonable y-axis limits
                try:
                    profit_target = float(self.profit_target.get())
                    stop_loss = float(self.stop_loss.get())
                    
                    self.ax.set_ylim(
                        min(-stop_loss * 1.5, -0.5),
                        max(profit_target * 1.5, 0.5)
                    )
                except Exception as e:
                    self.log_trade(f"Error setting y-axis limits: {str(e)}")
            
            # Draw without tight_layout
            self.canvas.draw()
            
        except Exception as e:
            self.log_trade(f"Error updating chart: {str(e)}")

    def update_price_history(self, symbol, price):
        """Update price history with proper timestamps"""
        try:
            # Initialize price history for this symbol if it doesn't exist
            if symbol not in self.price_history:
                self.price_history[symbol] = []
            
            # Add current timestamp and price
            current_time = datetime.now()
            self.price_history[symbol].append((current_time, price))
            
            # Limit history length to avoid memory issues
            max_history = 100  # Keep last 100 points
            if len(self.price_history[symbol]) > max_history:
                self.price_history[symbol] = self.price_history[symbol][-max_history:]
            
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
        """Close all open positions"""
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
                    
                    # Get current price with error handling
                    try:
                        current_price = self.exchange.fetch_ticker(symbol)['last']
                    except Exception as e:
                        self.log_trade(f"Error fetching price for {symbol}: {str(e)}")
                        # Use entry price as fallback
                        current_price = trade['entry_price']
                    
                    # Close the trade
                    self.close_trade(trade_id, trade, current_price, "manual close all")
                    
                except Exception as e:
                    self.log_trade(f"Error closing trade {trade_id}: {str(e)}")
            
            self.log_trade("All positions closed")
            
            # Force update of displays
            self.update_active_trades_display()
            self.update_metrics()
            self.update_balance_display()
            
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
        """Update the balance display with paper trading balance"""
        try:
            # Calculate allocated balance
            allocated_balance = 0
            unrealized_pl = 0
            
            for trade in self.active_trades.values():
                # Get position size
                if 'position_size' in trade:
                    position_size = float(trade['position_size'])
                elif 'amount' in trade and 'entry_price' in trade:
                    position_size = float(trade['amount']) * float(trade['entry_price'])
                elif 'quantity' in trade and 'entry_price' in trade:
                    position_size = float(trade['quantity']) * float(trade['entry_price'])
                else:
                    position_size = 0
                
                allocated_balance += position_size
                
                # Calculate unrealized P/L
                try:
                    symbol = trade['symbol']
                    entry_price = float(trade['entry_price'])
                    current_price = float(self.exchange.fetch_ticker(symbol)['last'])
                    
                    if 'quantity' in trade:
                        quantity = float(trade['quantity'])
                    elif 'amount' in trade:
                        quantity = float(trade['amount'])
                    else:
                        quantity = position_size / entry_price if entry_price > 0 else 0
                    
                    trade_pl = (current_price - entry_price) * quantity
                    unrealized_pl += trade_pl
                except Exception as e:
                    self.log_trade(f"Error calculating P/L for {trade.get('symbol', 'unknown')}: {str(e)}")
            
            # Calculate realized P/L
            realized_pl = self.paper_balance - 1000 + allocated_balance  # Assuming starting balance of 1000
            
            # Calculate total balance
            total_balance = self.paper_balance + allocated_balance
            
            # Update the balance label
            if hasattr(self, 'balance_label'):
                self.balance_label.config(text=f"Paper Balance: ${self.paper_balance:.2f} (${total_balance:.2f} total)")
            
            # Update metrics if they exist
            if hasattr(self, 'metrics_labels') and 'paper_balance' in self.metrics_labels:
                self.metrics_labels['paper_balance'].config(
                    text=f"Paper Balance: ${self.paper_balance:.2f} (${total_balance:.2f} total)"
                )
            
            # Format balance info for logging
            balance_info = f"""
                    Balance Breakdown:
                    Available Balance: ${self.paper_balance:.2f}
                    Allocated in Trades: ${allocated_balance:.2f}
                    Unrealized P/L: ${unrealized_pl:.2f}
                    Realized P/L: ${realized_pl:.2f}
                    Total Balance: ${total_balance:.2f}
                    Starting Balance: $1000.00
                    Net P/L: ${total_balance - 1000:.2f}
            """
            
            self.log_trade(balance_info)
            
        except Exception as e:
            self.log_trade(f"Error updating balance display: {str(e)}")

    def setup_balance_display(self, parent_frame):
        """Set up the balance display with allocation information"""
        balance_frame = ttk.Frame(parent_frame)
        balance_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Available balance (free/unused)
        self.balance_label = ttk.Label(
            balance_frame, 
            text=f"Paper Balance: ${self.paper_balance:.2f}",
            font=("Arial", 10, "bold")
        )
        self.balance_label.pack(side=tk.LEFT, padx=5)
        
        # Allocated balance (in active trades)
        allocated = sum(trade.get('position_size', 0) for trade in self.active_trades.values())
        self.allocated_label = ttk.Label(
            balance_frame,
            text=f"Allocated: ${allocated:.2f}",
            font=("Arial", 10)
        )
        self.allocated_label.pack(side=tk.LEFT, padx=5)
        
        # Total balance (available + allocated)
        total = self.paper_balance + allocated
        self.total_balance_label = ttk.Label(
            balance_frame,
            text=f"Total: ${total:.2f}",
            font=("Arial", 10, "bold")
        )
        self.total_balance_label.pack(side=tk.LEFT, padx=5)
        
        # Reset balance button
        self.reset_balance_button = ttk.Button(
            balance_frame,
            text="Reset Balance",
            command=self.reset_paper_balance
        )
        self.reset_balance_button.pack(side=tk.RIGHT, padx=5)

    def live_update_conditions(self):
        """Apply condition changes without restarting the bot"""
        try:
            self.validate_parameters()  # Validate new parameters
            self.force_update_fee_calculations()  # Update fees if needed
            self.update_metrics()  # Refresh metrics
            self.update_balance_display()  # Refresh balance
            self.log_trade("Trading conditions updated and applied.")
            messagebox.showinfo("Success", "Trading conditions updated and applied.")
        except Exception as e:
            self.log_trade(f"Error applying new conditions: {str(e)}")
            messagebox.showerror("Error", f"Failed to apply new conditions: {str(e)}")

    def close_trade(self, trade_id, trade, current_price, reason="closed"):
        """Close a trade and update performance metrics"""
        try:
            if trade_id not in self.active_trades:
                self.log_trade(f"Trade {trade_id} not found in active trades")
                return False

            symbol = trade['symbol']
            entry_price = trade['entry_price']
            position_size = trade.get('position_size', 0)

            # Calculate profit/loss
            price_change = (current_price - entry_price) / entry_price
            gross_pl = position_size * price_change

            # Calculate trade duration
            entry_time = trade.get('entry_time', datetime.now() - timedelta(minutes=1))
            duration = datetime.now() - entry_time
            hours, remainder = divmod(duration.total_seconds(), 3600)
            minutes, seconds = divmod(remainder, 60)
            duration_str = f"{int(hours)}:{int(minutes):02d}:{int(seconds):02d}"

            # Determine fee type (maker or taker)
            if 'is_limit_order' in trade:
                fee_type = "maker" if trade['is_limit_order'] else "taker"
            elif hasattr(self, 'using_limit_orders'):
                fee_type = "maker" if self.using_limit_orders else "taker"
            else:
                fee_type = "taker"  # Default fallback

            fee_pct = self.maker_fee if fee_type == "maker" else self.taker_fee

            # Entry and exit fees
            entry_fee = position_size * fee_pct
            exit_fee = (position_size * (1 + price_change)) * fee_pct
            total_fees = entry_fee + exit_fee

            # Calculate net profit/loss
            net_pl = gross_pl - total_fees

            # Update paper balance
            self.paper_balance += position_size + net_pl
            self.gross_profit += gross_pl      # Track gross profit (before fees)
            self.total_profit += net_pl        # Track net profit (after fees)
            self.total_fees += total_fees      # <-- Only add ONCE

            # Log the trade closure
            order_type = "LIMIT" if fee_type == "maker" else "MARKET"
            self.log_trade(
                f"CLOSED {order_type} {trade['symbol']} @ ${current_price:.6f} | "
                f"P/L: ${net_pl:.2f} ({price_change*100:.2f}%) | Duration: {duration_str}"
            )
            self.log_trade(
                f"Exit Fee: ${exit_fee:.4f} ({fee_type}: {fee_pct*100:.2f}%) | Total Fees: ${total_fees:.4f}"
            )

            # Update performance metrics
            self.total_trades += 1
            if net_pl > 0:
                self.winning_trades += 1
            else:
                self.losing_trades += 1

            # Update trade history - simplified call without status
            self.update_trade_history(
                symbol=symbol,
                percentage=price_change * 100,
                profit=net_pl,
                is_win=(net_pl > 0)
            )

            # Remove from active trades
            if trade_id in self.active_trades:
                del self.active_trades[trade_id]

            # Update displays
            self.update_active_trades_display()
            self.update_balance_display()
            self.update_metrics()

            return True

        except Exception as e:
            self.log_trade(f"Error closing trade: {str(e)}")
            self.log_trade(traceback.format_exc())
            return False

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
                'volume_surge': float(self.volume_surge.get()),  # Use volume_surge instead of volume_increase
                'trailing_activation': float(self.trailing_activation.get()),
                
                # Risk Management
                'max_position_percent': float(self.max_position_percent.get()),
                'daily_loss_limit': float(self.daily_loss_limit.get()),
                
                # Market Filters
                'min_volume': float(self.min_volume_entry.get()),
                'max_trades': float(self.max_trades_entry.get()),
                'max_volatility': float(self.max_volatility.get()),
                'required_conditions': int(self.required_conditions.get()),
                
                # Greek Parameters
                'momentum_beta': float(self.momentum_beta.get()),
                'price_alpha': float(self.price_alpha.get()),
                'momentum_theta': float(self.momentum_theta.get()),
                'vol_vega': float(self.vol_vega.get()),
                'volume_rho': float(self.volume_rho.get())
            }

            # Core Parameter Validation
            assert 0.05 <= conditions['profit_target'] <= 5.0, "Profit target must be between 0.05% and 5%"
            assert 0.05 <= conditions['stop_loss'] <= 2.0, "Stop loss must be between 0.05% and 2%"
            assert 10 <= conditions['position_size'] <= 1000, "Position size must be between $10 and $1000"
            assert 0.05 <= conditions['trailing_stop'] <= 1.0, "Trailing stop must be between 0.05% and 1%"

            # Entry Condition Validation
            assert 0.1 <= conditions['price_rise_min'] <= 2.0, "Minimum price rise must be between 0.1% and 2%"
            assert 1 <= conditions['volume_surge'] <= 500, "Volume surge must be between 1% and 500%"
            assert 0.1 <= conditions['trailing_activation'] <= 2.0, "Trailing activation must be between 0.1% and 2%"

            # Risk Management Validation
            assert 1 <= conditions['max_position_percent'] <= 20, "Maximum position must be between 1% and 20% of balance"
            assert 1 <= conditions['daily_loss_limit'] <= 10, "Daily loss limit must be between 1% and 10%"

            # Market Filter Validation
            assert 50 <= conditions['min_volume'] <= 1000, "Minimum volume must be between $50 and $1000"
            assert 1 <= conditions['max_trades'] <= 10, "Maximum trades must be between 1 and 10"
            assert 0.1 <= conditions['max_volatility'] <= 5.0, "Maximum volatility must be between 0.1% and 5%"
            assert 1 <= conditions['required_conditions'] <= 5, "Required conditions must be between 1 and 5"
            
            # Greek Parameter Validation
            assert 0.0001 <= conditions['momentum_beta'] <= 1.0, "Momentum Beta must be between 0.0001 and 1.0"
            assert 0.0001 <= conditions['price_alpha'] <= 1.0, "Price Alpha must be between 0.0001 and 1.0"
            assert 0.0001 <= conditions['momentum_theta'] <= 1.0, "Momentum Theta must be between 0.0001 and 1.0"
            assert 0.1 <= conditions['vol_vega'] <= 20.0, "Vol Vega must be between 0.1 and 20.0"
            assert 0.0001 <= conditions['volume_rho'] <= 1.0, "Volume Rho must be between 0.0001 and 1.0"

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
            
        except AssertionError as e:
            self.log_trade(f"Validation error: {str(e)}")
            messagebox.showerror("Validation Error", str(e))
            return False
            
        except Exception as e:
            self.log_trade(f"Validation error: {str(e)}")
            messagebox.showerror("Validation Error", f"Invalid parameter: {str(e)}")
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
        """Get market pairs with improved filtering for top volume pairs under $5"""
        try:
            self.log_trade("Fetching market pairs...")
            
            # Get top_pairs parameter from GUI
            max_price = float(self.max_price.get()) if hasattr(self, 'max_price') else 5.0
            top_pairs = int(self.top_list_size.get()) if hasattr(self, 'top_list_size') else 20
            min_volume = float(self.min_volume_entry.get()) if hasattr(self, 'min_volume_entry') else 150
            
            self.log_trade(f"Filtering for top {top_pairs} pairs by volume under ${max_price:.2f}")
            
            # Fetch all tickers
            tickers = self.fetch_tickers_with_retry()
            if not tickers:
                self.log_trade("Failed to fetch tickers")
                return []
                
            self.log_trade(f"Fetched {len(tickers)} tickers, filtering...")
            
            # Process tickers and filter for USD pairs under max_price
            usd_pairs = []
            
            for symbol, ticker in tickers.items():
                try:
                    # Skip non-USD pairs
                    if not symbol.endswith('/USD'):
                        continue
                        
                    # Skip stablecoins
                    base_symbol = symbol.split('/')[0]
                    if base_symbol in ['USDT', 'USDC', 'DAI', 'BUSD']:
                        continue
                        
                    # Get price and volume
                    price = float(ticker['last']) if ticker.get('last') else 0
                    volume = float(ticker['quoteVolume']) if ticker.get('quoteVolume') else 0
                    
                    # Skip if price is too high
                    if price > max_price:
                        continue
                    
                    # Skip if volume is too low
                    if volume < min_volume:
                        continue
                    
                    # Add to list of valid pairs
                    usd_pairs.append({
                        'symbol': symbol,
                        'ticker': ticker,
                        'price': price,
                        'volume': volume
                    })
                    
                except Exception as e:
                    # Skip logging errors for every ticker to reduce log spam
                    continue
                    
            # Sort by volume (highest first)
            usd_pairs.sort(key=lambda x: x['volume'], reverse=True)
            
            # Take only the top N pairs
            top_usd_pairs = usd_pairs[:top_pairs]
            
            self.log_trade(f"Found {len(top_usd_pairs)} top volume pairs under ${max_price:.2f}")
            
            # Process each pair to get price data
            valid_pairs = []
            for pair_data in top_usd_pairs:
                try:
                    symbol = pair_data['symbol']
                    
                    # Check if we have enough data, if not initialize it
                    df = self.data_manager.get_price_data(symbol)
                    if df is None or len(df) < 20:  # Not enough data points
                        self.log_trade(f"Evaluating {symbol}...")
                        if not self.initialize_pair_data(symbol):
                            self.log_trade(f"Insufficient data for {symbol}")
                            continue
                    
                    # Get updated price data after initialization
                    df = self.data_manager.get_price_data(symbol)
                    if df is None:
                        continue
                        
                    # Update pair data with dataframe
                    pair_data['df'] = df
                    valid_pairs.append(pair_data)
                    
                except Exception as e:
                    self.log_trade(f"Error processing {pair_data['symbol']}: {str(e)}")
                    continue
                    
            self.log_trade(f"Found {len(valid_pairs)} valid pairs with sufficient data")
            return valid_pairs
            
        except Exception as e:
            self.log_trade(f"Error getting market pairs: {str(e)}")
            return []

    def fetch_tickers_with_retry(self, max_retries=3, retry_delay=2):
        """Fetch tickers with retry mechanism"""
        for attempt in range(max_retries):
            try:
                self.log_trade(f"Fetching tickers (attempt {attempt+1}/{max_retries})...")
                tickers = self.exchange.fetch_tickers()
                self.log_trade(f"Successfully fetched {len(tickers)} tickers")
                return tickers
            except Exception as e:
                self.log_trade(f"Error fetching tickers: {str(e)}")
                if attempt < max_retries - 1:
                    self.log_trade(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    self.log_trade("Max retries reached, giving up")
                    return {}
        return {}

    def scan_opportunities(self):
        """Scan for trading opportunities with improved error handling"""
        try:
            if not self.running:
                return False
            
            # Check if we're already at max trades
            max_trades = int(self.max_trades_entry.get()) if hasattr(self, 'max_trades_entry') else 3
            available_slots = max_trades - len(self.active_trades)
            
            if available_slots <= 0:
                self.log_trade(f"Already at maximum trades ({len(self.active_trades)}/{max_trades}), skipping scan")
                return False
            
            # Set scanning flag
            self.is_scanning = True
            self.update_status("Scanning for opportunities...")
            
            # Get market pairs
            pairs = self.get_market_pairs()
            if not pairs:
                self.log_trade("No valid pairs found for trading")
                self.is_scanning = False
                return False
            
            self.log_trade(f"Found {len(pairs)} potential pairs to analyze. Looking to fill {available_slots} available slots")
            
            # Track pairs we've analyzed and trades executed
            analyzed_pairs = []
            trades_executed = 0
            
            # Analyze each pair
            for pair_data in pairs:
                try:
                    if not self.running or trades_executed >= available_slots:
                        break
                    
                    symbol = pair_data['symbol']
                
                    # Skip if we're already trading this symbol
                    if self.is_already_trading(symbol):
                        continue
                    
                    # Get current ticker
                    ticker = self.exchange.fetch_ticker(symbol)
                    volume = float(ticker['quoteVolume']) if 'quoteVolume' in ticker else 0
                    
                    # Add to analyzed pairs
                    analyzed_pairs.append(symbol)
                    
                    # Analyze opportunity
                    if self.analyze_opportunity(ticker, volume, pair_data):
                        self.log_trade(f"Found opportunity in {symbol}")
                        
                        # Execute trade with proper error handling
                        try:
                            if self.execute_trade(symbol, ticker):
                                self.log_trade(f"Successfully executed trade for {symbol}")
                                trades_executed += 1
                            else:
                                self.log_trade(f"Failed to execute trade for {symbol}")
                        except Exception as trade_error:
                            self.log_trade(f"Error executing trade for {symbol}: {str(trade_error)}")
                
                except Exception as e:
                    self.log_trade(f"Error analyzing {pair_data['symbol']}: {str(e)}")
                    continue
                
            # Log summary of scan
            self.log_trade(f"Scan complete. Analyzed {len(analyzed_pairs)} pairs, executed {trades_executed} trades")
            
            # Reset scanning flag
            self.is_scanning = False
            self.update_status("Monitoring trades")
            
            return trades_executed > 0
        
        except Exception as e:
            self.log_trade(f"Error in scan_opportunities: {str(e)}")
            self.is_scanning = False
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

    def is_already_trading(self, symbol):
        """Check if we're already trading a specific symbol"""
        try:
            # Check if the symbol exists in any active trade
            for trade_id, trade in self.active_trades.items():
                if trade['symbol'] == symbol:
                    return True
            return False
        except Exception as e:
            self.log_trade(f"Error checking if already trading {symbol}: {str(e)}")
            # Default to False if there's an error
            return False

    def analyze_opportunity(self, ticker, volume, pair_data):
        """Enhanced opportunity analysis with configurable confirmation factors"""
        try:
            symbol = pair_data['symbol']
            
            # Skip if we're already trading this symbol
            if self.is_already_trading(symbol):
                return False
            
            # Skip if we've reached max trades
            max_trades = int(self.max_trades_entry.get()) if hasattr(self, 'max_trades_entry') else 3
            if len(self.active_trades) >= max_trades:
                return False
            
            # Skip if price is above maximum allowed
            price = float(ticker['last'])
            max_price = float(self.max_price.get()) if hasattr(self, 'max_price') else 5.0
            if price > max_price:
                self.log_trade(f"Price too high for {symbol}: ${price:.6f} > ${max_price:.2f}")
                return False
            
            # Get price data
            df = pair_data.get('df')
            
            if df is None or len(df) < 20:
                self.log_trade(f"Insufficient data for {symbol}")
                return False
            
            # Log the pair we're analyzing
            self.log_trade(f"Analyzing {symbol} at ${price:.6f} with volume ${volume:.2f}")
            
            # Get trading parameters from GUI
            required_conditions = int(self.required_conditions.get()) if hasattr(self, 'required_conditions') else 3
            price_rise_threshold = float(self.price_rise_min.get()) if hasattr(self, 'price_rise_min') else 0.5
            volume_increase = float(self.volume_surge.get()) if hasattr(self, 'volume_surge') else 20
            
            # Track conditions met
            conditions_met = 0
            conditions = []
            
            # CONDITION 1: Price above moving averages
            if 'sma_50' in df.columns and 'sma_200' in df.columns:
                sma_50 = df['sma_50'].iloc[-1]
                sma_200 = df['sma_200'].iloc[-1]
                
                if price > sma_50:
                    conditions_met += 1
                    conditions.append(f"Price above SMA50 ({price:.6f} > {sma_50:.6f})")
                    
                if price > sma_200:
                    conditions_met += 1
                    conditions.append(f"Price above SMA200 ({price:.6f} > {sma_200:.6f})")
            
            # CONDITION 2: Recent price rise
            if len(df) >= 10:
                recent_price = df['price'].iloc[-1]
                previous_price = df['price'].iloc[-10]
                recent_change = ((recent_price - previous_price) / previous_price) * 100
                
                if recent_change >= price_rise_threshold:
                    conditions_met += 1
                    conditions.append(f"Price rise ({recent_change:.2f}% >= {price_rise_threshold:.2f}%)")
            
            # CONDITION 3: Volume increase
            if 'volume_ratio' in df.columns and not df['volume_ratio'].isna().all():
                vol_ratio = df['volume_ratio'].iloc[-1] if not pd.isna(df['volume_ratio'].iloc[-1]) else 1
                if vol_ratio >= (volume_increase / 100) + 1:
                    conditions_met += 1
                    conditions.append(f"Volume surge ({(vol_ratio-1)*100:.2f}% >= {volume_increase:.2f}%)")
            
            # CONDITION 4: Trend Strength Check (if enabled)
            use_trend_filter = self.use_trend_filter.get() if hasattr(self, 'use_trend_filter') else False
            
            if use_trend_filter:
                trend_min = float(self.trend_strength_min.get()) if hasattr(self, 'trend_strength_min') else 15
                trend_direction, trend_strength = self.detect_trend_strength(df)
                
                if trend_direction > 0 and trend_strength >= trend_min:
                    conditions_met += 1
                    conditions.append(f"Strong uptrend (strength: {trend_strength:.2f} >= {trend_min})")
                    
                self.log_trade(f"Trend analysis: direction={trend_direction}, strength={trend_strength:.2f}, min={trend_min}")
            
            # CONDITION 5: Support/Resistance Check (if enabled)
            use_sr = self.use_support_resistance.get() if hasattr(self, 'use_support_resistance') else False
            
            if use_sr:
                sr_lookback = int(self.sr_lookback.get()) if hasattr(self, 'sr_lookback') else 50
                sr_threshold = float(self.sr_threshold.get()) if hasattr(self, 'sr_threshold') else 0.2
                
                sr_levels = self.detect_support_resistance_levels(df, lookback=sr_lookback, threshold_pct=sr_threshold)
                supports = sr_levels['supports']
                
                # Check if price is near support
                near_support = False
                support_level = 0
                
                for support in supports:
                    # Check if price is within the threshold percentage of a support level
                    if price >= support * (1 - sr_threshold/100) and price <= support * (1 + sr_threshold/100):
                        near_support = True
                        support_level = support
                        break
                
                if near_support:
                    conditions_met += 1
                    conditions.append(f"Price near support level (${support_level:.6f})")
                
                # Log support/resistance info
                self.log_trade(f"S/R Analysis: Found {len(supports)} support levels")
            
            # CONDITION 6: Candlestick Pattern Recognition (if enabled)
            use_patterns = self.use_candlestick_patterns.get() if hasattr(self, 'use_candlestick_patterns') else False
            
            if use_patterns:
                min_confidence = float(self.pattern_confidence_min.get()) if hasattr(self, 'pattern_confidence_min') else 70
                patterns = self.detect_candlestick_patterns(df)
                
                if patterns:
                    # Get the highest confidence pattern
                    best_pattern = max(patterns.items(), key=lambda x: x[1])
                    pattern_name = best_pattern[0].replace('_', ' ').title()
                    confidence = best_pattern[1]
                    
                    if confidence >= min_confidence:
                        conditions_met += 1
                        conditions.append(f"Bullish pattern: {pattern_name} ({confidence}% confidence)")
                        
                    # Log all detected patterns
                    self.log_trade(f"Patterns detected: {', '.join([f'{k}({v}%)' for k,v in patterns.items()])}")
            
            # CONDITION 7: Volume Profile Analysis (if enabled)
            use_vol_profile = self.use_volume_profile.get() if hasattr(self, 'use_volume_profile') else False
            
            if use_vol_profile:
                min_quality = float(self.volume_quality_min.get()) if hasattr(self, 'volume_quality_min') else 60
                volume_analysis = self.analyze_volume_profile(df)
                
                if volume_analysis['volume_quality'] >= min_quality:
                    conditions_met += 1
                    conditions.append(f"Healthy volume profile (quality: {volume_analysis['volume_quality']:.1f})")
                    
                # Log volume profile details
                self.log_trade(f"Volume Profile: Quality={volume_analysis['volume_quality']:.1f}, Trend={volume_analysis['volume_trend']}")
            
            # Log conditions summary
            self.log_trade(f"Analysis for {symbol}: {conditions_met}/{required_conditions} conditions met")
            for condition in conditions:
                self.log_trade(f"✓ {condition}")
            
            # Log any conditions that were checked but not met
            if conditions_met < required_conditions:
                self.log_trade(f"Insufficient conditions: Need {required_conditions - conditions_met} more")
            
            # Final decision - using actual required conditions from GUI
            if conditions_met >= required_conditions:
                self.log_trade(f"OPPORTUNITY FOUND: {symbol} meets {conditions_met}/{required_conditions} conditions")
                return True
            else:
                return False
            
        except Exception as e:
            self.log_trade(f"Error analyzing opportunity for {pair_data['symbol']}: {str(e)}")
            import traceback
            self.log_trade(traceback.format_exc())
            return False

    def analyze_pairs(self, pairs):
        """Analyze pairs for trading opportunities"""
        try:
            self.log_trade(f"\nAnalyzing {len(pairs)} pairs for trading opportunities...")
            trades_executed = False
            
            # CRITICAL CHECK: Ensure we haven't exceeded max trades
            max_trades = int(self.max_trades_entry.get()) if hasattr(self, 'max_trades_entry') else 3
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
            # Use our tracking flag instead of checking the variable each time
            if hasattr(self, 'using_limit_orders') and self.using_limit_orders:
                total_fees = self.maker_fee * 2  # Entry and exit with maker fees
                fee_type = "LIMIT"
            else:
                total_fees = self.taker_fee * 2  # Entry and exit with taker fees
                fee_type = "MARKET"
            
            # Log the fee calculation
            self.log_trade(f"Using {fee_type} orders: " +
                        f"Total round-trip fee: {total_fees*100:.2f}%")
            
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

    def safe_update_entry(self, entry_widget, value):
        """Safely update an Entry widget or StringVar"""
        try:
            if entry_widget is None:
                return
                
            # Check if it's a StringVar
            if hasattr(entry_widget, 'set'):
                entry_widget.set(str(value))
            else:
                # It's an Entry widget
                entry_widget.delete(0, tk.END)
                entry_widget.insert(0, str(value))
                
        except Exception as e:
            self.log_trade(f"Error updating entry: {str(e)}")

    def auto_adjust_parameters(self):
        """Automatically adjust trading parameters based on market conditions"""
        try:
            # Main loop
            while self.running:
                try:
                    # Update status
                    self.update_status("Running - Monitoring trades")
                    
                    # Monitor active trades (do this first to ensure quick response to exit conditions)
                    self.monitor_trades()
                    
                    # Scan for new opportunities (only if we have room for more trades)
                    max_trades = int(self.max_trades_entry.get()) if hasattr(self, 'max_trades_entry') else 3  # Default to 3 if not set
                    
                    # Update max_active_trades to ensure consistency
                    self.max_active_trades = max_trades
                    
                    # Force a scan every 30 seconds if we're not at max trades
                    if len(self.active_trades) < max_trades:
                        current_time = time.time()
                        if not hasattr(self, 'last_scan_time') or (current_time - self.last_scan_time) > 30:
                            self.log_trade(f"Initiating scheduled scan (active: {len(self.active_trades)}, max: {max_trades})")
                            self.scan_opportunities()
                            self.last_scan_time = current_time
                    
                    # Sleep to avoid high CPU usage
                    time.sleep(1)
                    
                except Exception as e:
                    self.log_trade(f"Error in bot loop: {str(e)}")
                    self.is_scanning = False  # Reset flag in case of error
                    time.sleep(5)
                    
        except Exception as e:
            self.log_trade(f"Bot error: {str(e)}")
            self.stop_bot()

    def force_update_fee_calculations(self):
        """Force update all fee-related calculations to ensure consistency"""
        try:
            # Check if we're using limit orders
            use_limit_orders = False
            if hasattr(self, 'use_limit_orders_var'):
                use_limit_orders = self.use_limit_orders_var.get()
            
                # Update our tracking flag
                self.using_limit_orders = use_limit_orders
            
            # Update the total fee percentage
            if use_limit_orders:
                self.total_fee_percentage = self.maker_fee * 2
                fee_type = "maker"
            else:
                self.total_fee_percentage = self.taker_fee * 2
                fee_type = "taker"
            
            # Update the total fees label
            total_fee_pct = self.total_fee_percentage * 100
            if hasattr(self, 'total_fees_label'):
                self.total_fees_label.config(text=f"Total Round-Trip Fee: {total_fee_pct:.2f}%")
            
            # Log the update
            self.log_trade(f"Fee calculations force-updated: Using {fee_type} fees ({fee_type}*2 = {total_fee_pct:.2f}%)")
            
            # Update tooltips
            if hasattr(self, 'profit_target'):
                self.add_tooltip(self.profit_target, 
                    f"Target profit percentage for trades (min: {total_fee_pct:.2f}% to cover fees)")
            
            return True
        except Exception as e:
            self.log_trade(f"Error updating fee calculations: {str(e)}")
            return False

    def execute_trade(self, symbol, ticker):
        """Execute a trade with proper error handling"""
        try:
            # Check if we're already in a trade for this symbol
            if self.is_already_trading(symbol):
                self.log_trade(f"Already in a trade for {symbol}")
                return False
            
            # Get the current price
            if isinstance(ticker, dict) and 'last' in ticker:
                price = float(ticker['last'])
            elif hasattr(ticker, 'last'):
                price = float(ticker.last)
            else:
                # If ticker is just the price itself
                try:
                    price = float(ticker)
                except (TypeError, ValueError):
                    self.log_trade(f"Invalid ticker format for {symbol}")
                    return False
            
            # CRITICAL CHECK: Ensure we haven't exceeded max trades
            max_trades = int(self.max_trades_entry.get()) if hasattr(self, 'max_trades_entry') else 3
            current_trades = len(self.active_trades)
            
            if current_trades >= max_trades:
                self.log_trade(f"Cannot execute trade for {symbol}: Max trades reached ({current_trades}/{max_trades})")
                return False
                
            # Calculate position size
            position_size = float(self.position_size.get()) if hasattr(self, 'position_size') else 50.0
            
            # Check if we have enough balance
            if self.is_paper_trading:
                available_balance = self.paper_balance
            else:
                available_balance = float(self.exchange.fetch_balance()['USD']['free'])
                
            if position_size > available_balance:
                self.log_trade(f"Insufficient balance for trade: {available_balance:.2f} < {position_size:.2f}")
                return False
                
            # Generate a unique trade ID
            trade_id = f"{'paper' if self.is_paper_trading else 'live'}_{int(time.time())}"
            
            # Create trade object
            trade = {
                'id': trade_id,
                'symbol': symbol,
                'entry_price': price,
                'position_size': position_size,
                'entry_time': datetime.now(),
                'status': 'open',
                'highest_price': price,
                'highest_profit_percentage': 0.0,
                'stop_loss_pct': float(self.stop_loss.get()) / 100 if hasattr(self, 'stop_loss') else 0.008,
                'profit_target_pct': float(self.profit_target.get()) / 100 if hasattr(self, 'profit_target') else 0.015,
                'trailing_stop_pct': float(self.trailing_stop.get()) / 100 if hasattr(self, 'trailing_stop') else 0.003,
                'is_limit_order': self.using_limit_orders
            }
            
            # Double-check max trades one more time before adding
            if len(self.active_trades) >= max_trades:
                self.log_trade(f"Race condition detected: Max trades reached while preparing trade")
                return False
                
            # Add to active trades
            self.active_trades[trade_id] = trade
            
            # Update paper balance if in paper trading mode
            if self.is_paper_trading:
                self.paper_balance -= position_size
                
            # Log the trade - using explicit string formatting to avoid dict.__format__ error
            self.log_trade(
                f"{'PAPER' if self.is_paper_trading else 'LIVE'} TRADE EXECUTED:\n"
                f"Symbol: {symbol}\n"
                f"Entry Price: ${price:.8f}\n"
                f"Position Size: ${position_size:.2f}\n"
                f"Stop Loss: {float(self.stop_loss.get()) if hasattr(self, 'stop_loss') else 0.8}%\n"
                f"Profit Target: {float(self.profit_target.get()) if hasattr(self, 'profit_target') else 1.5}%"
            )
            
            # Update displays
            self.update_active_trades_display()
            self.update_balance_display()
            self.update_chart()
            
            return True
            
        except Exception as e:
            self.log_trade(f"Error executing trade: {str(e)}")
            return False
        
    def verify_paper_balance(self):
        """Verify and fix paper balance if it's incorrect"""
        try:
            if not self.is_paper_trading:
                return  # Only applicable for paper trading
                
            # Calculate what the paper balance should be
            allocated = sum(trade.get('position_size', 0) for trade in self.active_trades.values())
            
            # If we have no active trades but balance is not 1000, reset it
            if not self.active_trades and abs(self.paper_balance - 1000.0) > 0.01:
                self.log_trade(f"Fixing incorrect paper balance: ${self.paper_balance:.2f} -> $1000.00")
                self.paper_balance = 1000.0
                self.update_balance_display()
                return
                
            # Check if we have a reasonable starting balance (1000 - allocated)
            expected_min_balance = 1000.0 - allocated
            
            # If balance is significantly off, fix it
            if self.paper_balance < expected_min_balance - 10.0:  # Allow for some P/L variation
                self.log_trade(f"Paper balance appears incorrect: ${self.paper_balance:.2f} vs expected min ${expected_min_balance:.2f}")
                self.log_trade("Adjusting paper balance to expected minimum")
                self.paper_balance = expected_min_balance
                self.update_balance_display()
                    
        except Exception as e:
            self.log_trade(f"Error verifying paper balance: {str(e)}")
    
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
            # Check if market override is active
            market_override = hasattr(self, 'market_override_var') and self.market_override_var.get()
            
            # 1. EMA Cross (5/15)
            if 'ema_5' in df.columns and 'ema_15' in df.columns and len(df) >= 2:
                ema_cross = (df['ema_5'].iloc[-2] < df['ema_15'].iloc[-2]) and \
                            (df['ema_5'].iloc[-1] > df['ema_15'].iloc[-1])
                
                # If market override is active, also accept if EMAs are close to crossing
                if market_override:
                    ema_ratio = df['ema_5'].iloc[-1] / df['ema_15'].iloc[-1]
                    ema_cross = ema_cross or (ema_ratio > 0.98)  # Accept if within 2% of crossing
                    if ema_ratio > 0.98:
                        self.log_trade(f"[OK] Market override ACTIVE - relaxed EMA crossover check (ratio: {ema_ratio:.4f})")
                
                self.log_trade(f"EMA Cross: {ema_cross}")
            else:
                self.log_trade(f"Insufficient EMA data for {symbol}")
                ema_cross = False
            
            # 2. Check RSI if available
            rsi_period = int(self.rsi_period.get()) if hasattr(self, 'rsi_period') else 14
            rsi_column = f'rsi_{rsi_period}'
            
            if rsi_column in df.columns and not df[rsi_column].isna().all():
                rsi_value = df[rsi_column].iloc[-1]
                rsi_overbought = float(self.rsi_overbought.get()) if hasattr(self, 'rsi_overbought') else 70
                rsi_oversold = float(self.rsi_oversold.get()) if hasattr(self, 'rsi_oversold') else 30
                
                # Check if RSI is in a good range (not overbought)
                if rsi_value > rsi_overbought:
                    self.log_trade(f"Rejected {symbol}: RSI indicates overbought condition ({rsi_value:.2f} > {rsi_overbought})")
                    return False
            else:
                self.log_trade(f"RSI data not available for {symbol}")
                rsi_value = 50  # Default neutral value
                rsi_overbought = 70
                rsi_oversold = 30
            
            # 3. Volume-Weighted Momentum
            try:
                # Calculate volume-weighted momentum
                if len(df) >= 5:
                    # Get recent price and volume data
                    recent_df = df.tail(5)
                    
                    # Calculate price change
                    price_change = (recent_df['price'].iloc[-1] - recent_df['price'].iloc[0]) / recent_df['price'].iloc[0]
                    
                    # Calculate volume-weighted momentum
                    vwm = price_change * 100  # Convert to percentage
                    
                    self.log_trade(f"Volume-Weighted Momentum for {symbol}: {vwm:.2f}%")
                    
                    # Check if momentum is positive
                    if vwm < 0 and not market_override:
                        self.log_trade(f"Negative momentum for {symbol}: {vwm:.2f}%")
                        return False
                else:
                    self.log_trade(f"Insufficient data for volume-weighted momentum calculation")
                    vwm = 0
            except Exception as e:
                self.log_trade(f"Error calculating volume-weighted momentum: {str(e)}")
                vwm = 0
            
            # 4. Order Book Analysis
            try:
                # Get order book if available
                if hasattr(self, 'exchange') and self.exchange is not None:
                    order_book = self.exchange.fetch_order_book(symbol)
                    
                    # Calculate bid/ask ratio
                    total_bids = sum(bid[1] for bid in order_book['bids'][:5])
                    total_asks = sum(ask[1] for ask in order_book['asks'][:5])
                    
                    if total_asks > 0:
                        ob_ratio = total_bids / total_asks
                    else:
                        ob_ratio = 1.0
                    
                    self.log_trade(f"Order Book Imbalance for {symbol}: {ob_ratio:.2f}")
                    
                    # Check if there's significant buy pressure
                    if ob_ratio < 0.5 and not market_override:
                        self.log_trade(f"Insufficient buy pressure for {symbol}: {ob_ratio:.2f}")
                        return False
                else:
                    self.log_trade(f"Order book not available for {symbol}")
                    ob_ratio = 1.0
            except Exception as e:
                self.log_trade(f"Error analyzing order book: {str(e)}")
                ob_ratio = 1.0
            
            # Log all advanced check results
            self.log_trade(f"Advanced checks passed for {symbol}:")
            self.log_trade(f"- EMA Cross: {ema_cross}")
            self.log_trade(f"- VWAP Momentum: {vwm:.2f}%")
            self.log_trade(f"- Order Book Ratio: {ob_ratio:.2f}")
            self.log_trade(f"- RSI: {rsi_value:.2f} (Overbought: {rsi_overbought}, Oversold: {rsi_oversold})")
            
            # Return true if we've passed all checks
            return True
            
        except Exception as e:
            self.log_trade(f"Error in advanced checks for {symbol}: {str(e)}")
            return False
            
    def update_active_trades_display(self):
        """Update the active trades display with current trade information"""
        try:
            # Skip if trades_text widget doesn't exist
            if not hasattr(self, 'trades_text'):
                return
                
            # Clear the text widget
            self.trades_text.config(state=tk.NORMAL)
            self.trades_text.delete(1.0, tk.END)
            
            if not self.active_trades:
                self.trades_text.insert(tk.END, "No active trades\n")
                self.trades_text.config(state=tk.DISABLED)
                return
                
            self.trades_text.insert(tk.END, f"Active Trades: {len(self.active_trades)}\n\n")
            
            for trade_id, trade in list(self.active_trades.items()):
                try:
                    symbol = trade.get('symbol', 'Unknown')
                    entry_price = trade.get('entry_price', 0.0)
                    
                    # Get current price safely
                    current_price = trade.get('current_price', entry_price)
                    
                    # Calculate profit percentage
                    profit_pct = ((current_price - entry_price) / entry_price) * 100
                    
                    # Get entry time and calculate time in trade
                    entry_time_str = "Unknown"
                    time_in_trade_str = "Unknown"
                    
                    if 'entry_time' in trade and isinstance(trade['entry_time'], datetime):
                        entry_time = trade['entry_time']
                        entry_time_str = entry_time.strftime("%H:%M:%S")
                        time_diff = datetime.now() - entry_time
                        minutes, seconds = divmod(time_diff.seconds, 60)
                        time_in_trade_str = f"{minutes}m {seconds}s"
                    
                    # Format position size
                    position_size = trade.get('position_size', 0.0)
                    
                    # Format trade info for display
                    trade_info = (
                        f"{symbol} - ID: {trade_id}\n"
                        f"Entry: ${entry_price:.8f}\n"
                        f"Current: ${current_price:.8f}\n"
                        f"P/L: {profit_pct:.2f}%\n"
                        f"Size: ${position_size:.2f}\n"
                        f"Entry Time: {entry_time_str}\n"
                        f"Time in Trade: {time_in_trade_str}\n"
                        f"Highest P/L: {trade.get('highest_profit_percentage', 0.0):.2f}%\n"
                        f"{'-' * 30}\n"
                    )
                    
                    # Insert trade info into text widget
                    self.trades_text.insert(tk.END, trade_info)
                    
                    # Apply color coding based on profit/loss
                    start_pos = self.trades_text.index(f"end-{len(trade_info)+1}c")
                    end_pos = self.trades_text.index("end-1c")
                    
                    if profit_pct > 0:
                        self.trades_text.tag_add("profit", start_pos, end_pos)
                        self.trades_text.tag_config("profit", foreground="green")
                    else:
                        self.trades_text.tag_add("loss", start_pos, end_pos)
                        self.trades_text.tag_config("loss", foreground="red")
                    
                except Exception as e:
                    # Log error but continue processing other trades
                    self.log_trade(f"Error updating display for trade {trade_id}: {str(e)}")
                    continue
            
            # Disable text widget after updating
            self.trades_text.config(state=tk.DISABLED)
            
        except Exception as e:
            self.log_trade(f"Error updating active trades display: {str(e)}")

    def validate_trade(self, pair_data, entry_price):
        """Validate a trade before execution"""
        try:
            symbol = pair_data['symbol']
            self.log_trade(f"Validating trade for {symbol} at ${entry_price}")
            
            # Check if we're already trading this symbol
            if self.is_already_trading(symbol):
                self.log_trade(f"Already trading {symbol}")
                return False
            
            # Check if we've reached max trades
            max_trades = int(self.max_trades_entry.get()) if hasattr(self, 'max_trades_entry') else 5
            if len(self.active_trades) >= max_trades:
                self.log_trade(f"Maximum trades reached ({len(self.active_trades)}/{max_trades})")
                return False
            
            # Check if we have enough balance
            position_size = float(self.position_size.get()) if hasattr(self, 'position_size') else 150.0
            if position_size > self.paper_balance:
                self.log_trade(f"Insufficient balance: ${self.paper_balance:.2f} < ${position_size:.2f}")
                return False
            
            # Additional validation checks can be added here
            
            # If we reach here, the trade is valid
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
        
    def detect_trend_strength(self, df):
        """
        Detect trend strength using directional movement for lower-priced cryptocurrencies
        Returns a tuple of (trend_direction, trend_strength)
        trend_direction: 1 for uptrend, -1 for downtrend, 0 for neutral
        trend_strength: 0-100 value indicating strength (higher = stronger trend)
        """
        try:
            if df is None or len(df) < 14:
                return 0, 0  # Neutral trend, zero strength
            
            # Use percentage changes instead of absolute values to work well with any price range
            price_series = df['price'].iloc[-14:].copy()
            
            # Calculate true range as percentage of price
            high_pct = price_series.pct_change().fillna(0) + 1
            low_pct = 1 - price_series.pct_change().fillna(0).abs()
            
            # Use percentage changes for calculations to work with any price range
            changes = price_series.pct_change().fillna(0)
            
            # Calculate directional movement
            up_moves = changes.copy()
            up_moves[up_moves < 0] = 0
            down_moves = -changes.copy()
            down_moves[down_moves < 0] = 0
            
            # Calculate smoothed indicators
            pos_di = 100 * up_moves.mean() / changes.abs().mean() if changes.abs().mean() > 0 else 0
            neg_di = 100 * down_moves.mean() / changes.abs().mean() if changes.abs().mean() > 0 else 0
            
            # Calculate trend direction
            if pos_di > neg_di:
                trend_direction = 1  # Uptrend
            elif neg_di > pos_di:
                trend_direction = -1  # Downtrend
            else:
                trend_direction = 0  # Neutral
            
            # Calculate simplified trend strength (0-100)
            di_diff = abs(pos_di - neg_di)
            di_sum = pos_di + neg_di
            trend_strength = 100 * (di_diff / di_sum) if di_sum > 0 else 0
            
            return trend_direction, trend_strength
            
        except Exception as e:
            self.log_trade(f"Error in trend detection: {str(e)}")
            return 0, 0  # Default to neutral trend, zero strength



    def analyze_volume_profile(self, df, lookback=20):
        """
        Analyze volume profile to determine if volume is supporting price movement
        Returns a dict with volume analysis indicators
        """
        try:
            if df is None or len(df) < lookback or 'volume' not in df.columns:
                return {'healthy_profile': False, 'volume_trend': 0, 'volume_quality': 0}
                
            # Get recent data
            recent_data = df.iloc[-lookback:].copy()
            
            # Calculate price changes
            recent_data['price_change'] = recent_data['price'].pct_change()
            
            # Separate up and down days
            up_days = recent_data[recent_data['price_change'] > 0]
            down_days = recent_data[recent_data['price_change'] < 0]
            
            # Calculate average volume on up days vs down days
            avg_up_volume = up_days['volume'].mean() if not up_days.empty else 0
            avg_down_volume = down_days['volume'].mean() if not down_days.empty else 0
            
            # Calculate volume trend (rising or falling)
            volume_trend = 0
            if len(recent_data) >= 5:
                recent_vol = recent_data['volume'].iloc[-5:].mean()
                earlier_vol = recent_data['volume'].iloc[-lookback:-5].mean() if len(recent_data) > 5 else 0
                
                if recent_vol > earlier_vol * 1.1:
                    volume_trend = 1  # Rising volume
                elif recent_vol < earlier_vol * 0.9:
                    volume_trend = -1  # Falling volume
            
            # Calculate volume quality score (0-100)
            volume_quality = 0
            if avg_down_volume > 0:
                # Higher is better - we want more volume on up days
                up_down_ratio = avg_up_volume / avg_down_volume if avg_down_volume > 0 else 1
                volume_quality = min(100, max(0, (up_down_ratio - 0.8) * 50))
            
            # Determine if volume profile is healthy for upward movement
            healthy_profile = volume_quality > 60 and volume_trend >= 0
            
            return {
                'healthy_profile': healthy_profile,
                'volume_trend': volume_trend,
                'volume_quality': volume_quality
            }
            
        except Exception as e:
            self.log_trade(f"Error analyzing volume profile: {str(e)}")
            return {'healthy_profile': False, 'volume_trend': 0, 'volume_quality': 0}

    def detect_candlestick_patterns(self, df):
        """
        Detect bullish candlestick patterns in recent price action
        Returns a dict with pattern names and confidence levels
        """
        try:
            if df is None or len(df) < 5:
                return {}
                
            # Get the last 3 candles
            if 'open' not in df.columns or 'high' not in df.columns or 'low' not in df.columns:
                # Create synthetic OHLC data if only price is available
                df['open'] = df['price'].shift(1)
                df['high'] = df['price']
                df['low'] = df['price']
                df['close'] = df['price']
            
            patterns = {}
            
            # Get last 3 candles
            candles = df.iloc[-3:].copy()
            
            # 1. Bullish Engulfing Pattern
            if len(candles) >= 2:
                prev_candle = candles.iloc[-2]
                curr_candle = candles.iloc[-1]
                
                prev_body_size = abs(prev_candle['open'] - prev_candle['price'])
                curr_body_size = abs(curr_candle['open'] - curr_candle['price'])
                
                if prev_candle['open'] > prev_candle['price'] and \
                curr_candle['open'] < curr_candle['price'] and \
                curr_candle['open'] <= prev_candle['price'] and \
                curr_candle['price'] >= prev_candle['open'] and \
                curr_body_size > prev_body_size:
                    patterns['bullish_engulfing'] = 80
            
            # 2. Morning Star Pattern
            if len(candles) >= 3:
                first = candles.iloc[-3]
                middle = candles.iloc[-2]
                last = candles.iloc[-1]
                
                first_body = abs(first['open'] - first['price'])
                middle_body = abs(middle['open'] - middle['price'])
                last_body = abs(last['open'] - last['price'])
                
                if first['open'] > first['price'] and \
                last['open'] < last['price'] and \
                middle_body < first_body * 0.5 and \
                middle_body < last_body * 0.5:
                    patterns['morning_star'] = 90
            
            # 3. Hammer Pattern
            if len(candles) >= 1:
                curr = candles.iloc[-1]
                
                if 'low' in curr and 'high' in curr:
                    body = abs(curr['open'] - curr['price'])
                    lower_wick = min(curr['open'], curr['price']) - curr['low']
                    upper_wick = curr['high'] - max(curr['open'], curr['price'])
                    
                    if curr['price'] > curr['open'] and \
                    lower_wick > body * 2 and \
                    upper_wick < body * 0.5:
                        patterns['hammer'] = 70
            
            return patterns
            
        except Exception as e:
            self.log_trade(f"Error detecting candlestick patterns: {str(e)}")
            return {}

    def detect_support_resistance_levels(self, df, lookback=50, threshold_pct=0.2):
        """
        Detect key support and resistance levels in the price data
        Returns a dict with 'supports' and 'resistances' lists
        """
        try:
            if df is None or len(df) < lookback:
                return {'supports': [], 'resistances': []}
                
            # Use recent price history for analysis
            prices = df['price'].iloc[-lookback:].values
            
            # Find local maxima and minima
            peaks = []
            troughs = []
            
            for i in range(2, len(prices)-2):
                # Detect peaks (local maxima)
                if prices[i] > prices[i-1] and prices[i] > prices[i-2] and \
                prices[i] > prices[i+1] and prices[i] > prices[i+2]:
                    peaks.append((i, prices[i]))
                    
                # Detect troughs (local minima)
                if prices[i] < prices[i-1] and prices[i] < prices[i-2] and \
                prices[i] < prices[i+1] and prices[i] < prices[i+2]:
                    troughs.append((i, prices[i]))
            
            # Group similar levels together
            threshold = prices[-1] * threshold_pct / 100  # Convert percentage to absolute value
            
            # Cluster resistance levels
            resistances = []
            for peak in peaks:
                peak_price = peak[1]
                
                # Check if this peak is close to an existing resistance
                found_cluster = False
                for i, resistance in enumerate(resistances):
                    if abs(resistance - peak_price) <= threshold:
                        # Average the levels if they're close
                        resistances[i] = (resistance + peak_price) / 2
                        found_cluster = True
                        break
                        
                if not found_cluster:
                    resistances.append(peak_price)
                    
            # Cluster support levels
            supports = []
            for trough in troughs:
                trough_price = trough[1]
                
                # Check if this trough is close to an existing support
                found_cluster = False
                for i, support in enumerate(supports):
                    if abs(support - trough_price) <= threshold:
                        # Average the levels if they're close
                        supports[i] = (support + trough_price) / 2
                        found_cluster = True
                        break
                        
                if not found_cluster:
                    supports.append(trough_price)
                    
            # Sort levels
            supports.sort()
            resistances.sort()
            
            return {'supports': supports, 'resistances': resistances}
            
        except Exception as e:
            self.log_trade(f"Error detecting support/resistance: {str(e)}")
            return {'supports': [], 'resistances': []}

    def smart_trade_filter(self, pair_data):
        """Smart filter for trade opportunities with adaptive thresholds"""
        try:
            symbol = pair_data['symbol']
            self.log_trade(f"Evaluating {symbol} with smart filter...")
            
            # Check if we have enough data
            if 'df' not in pair_data or pair_data['df'] is None or len(pair_data['df']) < 5:
                self.log_trade(f"Insufficient data for {symbol}")
                return False
                
            df = pair_data['df']
            
            # Check if market override is active
            market_override = hasattr(self, 'market_override_var') and self.market_override_var.get()
            if market_override:
                self.log_trade(f"[OK] Market override ACTIVE for {symbol}")
            
            # Log current price
            current_price = pair_data['price']
            self.log_trade(f"Current price for {symbol}: ${current_price}")
            
            # Check EMA crossover if available
            if 'ema_5' in df.columns and 'ema_15' in df.columns and len(df) >= 2:
                # Log EMA values for debugging
                ema5_current = df['ema_5'].iloc[-1]
                ema15_current = df['ema_15'].iloc[-1]
                ema5_prev = df['ema_5'].iloc[-2] if len(df) > 2 else 0
                ema15_prev = df['ema_15'].iloc[-2] if len(df) > 2 else 0
                
                self.log_trade(f"EMA values for {symbol}: EMA_5={ema5_current:.5f}, EMA_15={ema15_current:.5f}, EMA_5_prev={ema5_prev:.5f}, EMA_15_prev={ema15_prev:.5f}")
                
                # Check if EMA5 is crossing above EMA15 or already above
                ema_cross = (df['ema_5'].iloc[-1] > df['ema_15'].iloc[-1])
                
                # If market override is active, also accept if EMAs are close to crossing
                if market_override:
                    ema_ratio = df['ema_5'].iloc[-1] / df['ema_15'].iloc[-1]
                    ema_cross = ema_cross or (ema_ratio > 0.98)  # Accept if within 2% of crossing
                    self.log_trade(f"[OK] Market override ACTIVE - relaxed EMA crossover check (ratio: {ema_ratio:.4f})")
                    
                if ema_cross:
                    self.log_trade(f"EMA check passed for {symbol}: EMA5 > EMA15 or close to crossing")
                else:
                    self.log_trade(f"EMA check failed for {symbol}: No bullish crossover")
                    if not market_override:
                        return False
            else:
                self.log_trade(f"Warning: EMA data not available for {symbol}")
                
            # Check RSI if available
            rsi_period = int(self.rsi_period.get()) if hasattr(self, 'rsi_period') else 14
            rsi_column = f'rsi_{rsi_period}'
            
            if rsi_column in df.columns:
                rsi = df[rsi_column]
                rsi_value = rsi.iloc[-1]
                
                # Get RSI thresholds
                rsi_overbought = float(self.rsi_overbought.get()) if hasattr(self, 'rsi_overbought') else 70
                rsi_oversold = float(self.rsi_oversold.get()) if hasattr(self, 'rsi_oversold') else 30
                
                self.log_trade(f"RSI for {symbol}: {rsi_value:.2f} (Overbought: {rsi_overbought}, Oversold: {rsi_oversold})")
                
                # For testing purposes, temporarily allow higher RSI values
                # In production, you'd want to avoid buying when RSI is overbought
                if rsi_value > rsi_overbought and not market_override:
                    self.log_trade(f"RSI filter rejected {symbol}: Overbought condition ({rsi_value:.2f} > {rsi_overbought})")
                    return False
            else:
                self.log_trade(f"Warning: RSI data not available for {symbol}")
                
            # Check for price momentum
            if 'price' in df.columns and len(df) >= 5:
                # Calculate short-term momentum (last 5 periods)
                recent_prices = df['price'].iloc[-5:]
                price_change = (recent_prices.iloc[-1] - recent_prices.iloc[0]) / recent_prices.iloc[0] * 100
                
                self.log_trade(f"Price momentum for {symbol}: {price_change:.2f}%")
                
                # Require positive momentum unless market override is active
                if price_change <= 0 and not market_override:
                    self.log_trade(f"Momentum filter rejected {symbol}: No positive momentum ({price_change:.2f}%)")
                    return False
            else:
                self.log_trade(f"Warning: Price data not available for momentum calculation for {symbol}")
                
            # Check volume
            if 'volume' in df.columns and len(df) >= 5:
                recent_volume = df['volume'].iloc[-5:]
                avg_volume = recent_volume.mean()
                current_volume = recent_volume.iloc[-1]
                volume_ratio = current_volume / avg_volume if avg_volume > 0 else 0
                
                self.log_trade(f"Volume ratio for {symbol}: {volume_ratio:.2f}x average")
                
                # Require above-average volume unless market override is active
                if volume_ratio < 1.0 and not market_override:
                    self.log_trade(f"Volume filter rejected {symbol}: Below average volume ({volume_ratio:.2f}x)")
                    return False
            else:
                self.log_trade(f"Warning: Volume data not available for {symbol}")
                
            # If we passed all checks, this is a potential opportunity
            self.log_trade(f"Opportunity found for {symbol}")
            return True
            
        except Exception as e:
            self.log_trade(f"Error checking {pair_data['symbol']}: {str(e)}")
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
            win_rate = 0
            if self.total_trades > 0:
                win_rate = (self.winning_trades / self.total_trades) * 100

            # Show gross profit as "Total Profit", net profit as "Net Profit"
            total_profit = self.gross_profit
            net_profit = self.gross_profit - self.total_fees

            if hasattr(self, 'metrics_labels'):
                if 'total_profit' in self.metrics_labels:
                    self.metrics_labels['total_profit'].config(
                        text=f"${total_profit:.2f}",
                        foreground="green" if total_profit > 0 else "red"
                    )
                if 'total_fees' in self.metrics_labels:
                    self.metrics_labels['total_fees'].config(
                        text=f"${self.total_fees:.2f}"
                    )
                if 'net_profit' in self.metrics_labels:
                    self.metrics_labels['net_profit'].config(
                        text=f"${net_profit:.2f}",
                        foreground="green" if net_profit > 0 else "red"
                    )
                if 'win_rate' in self.metrics_labels:
                    self.metrics_labels['win_rate'].config(
                        text=f"{win_rate:.1f}% ({self.winning_trades}/{self.losing_trades})"
                    )
                if 'total_trades' in self.metrics_labels:
                    self.metrics_labels['total_trades'].config(
                        text=f"{self.total_trades}"
                    )
                if 'paper_balance' in self.metrics_labels:
                    self.metrics_labels['paper_balance'].config(
                        text=f"${self.paper_balance:.2f}"
                    )

            self.log_trade(f"Updated metrics: Gross=${total_profit:.2f}, Fees=${self.total_fees:.2f}, "
                        f"Net=${net_profit:.2f}, Trades={self.total_trades}, "
                        f"Wins={self.winning_trades}, Losses={self.losing_trades}, "
                        f"Win Rate={win_rate:.1f}%, Balance=${self.paper_balance:.2f}")

        except Exception as e:
            self.log_trade(f"Error updating metrics: {str(e)}")
            import traceback
            self.log_trade(traceback.format_exc())

    def collect_trade_data(self):
        """Initialize trade data collection system"""
        try:

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

    def update_trade_history(self, symbol, percentage, profit, is_win=True):
        """Update the trade history with a completed trade"""
        try:
            # Ensure percentage is within reasonable bounds
            if abs(percentage) > 50:  # Sanity check
                self.log_trade(f"Warning: Unusual percentage detected for {symbol}: {percentage}%")
                percentage = min(max(percentage, -50), 50)  # Cap at ±50%

            # Format the trade result
            result = f"{symbol}: {percentage:.2f}%, ${profit:.2f}\n"

            if hasattr(self, 'history_text'):
                self.history_text.config(state=tk.NORMAL)
                # Get the index before inserting
                insert_index = self.history_text.index(tk.END)
                line_number = insert_index.split('.')[0]

                self.history_text.insert(tk.END, result)

                # Tag the just-inserted line
                line_start = f"{line_number}.0"
                line_end = f"{line_number}.end"

                color = "green" if profit > 0 else "red"
                self.history_text.tag_add(color, line_start, line_end)
                self.history_text.tag_config("green", foreground="green")
                self.history_text.tag_config("red", foreground="red")

                self.history_text.see(tk.END)
                self.history_text.config(state=tk.DISABLED)
                self.history_text.update_idletasks()

        except Exception as e:
            self.log_trade(f"Error updating trade history: {str(e)}")

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
        """Update the active trades display with current trade information"""
        try:
            # Skip if trades_text widget doesn't exist
            if not hasattr(self, 'trades_text'):
                return
                
            # Clear the text widget
            self.trades_text.config(state=tk.NORMAL)
            self.trades_text.delete(1.0, tk.END)
            
            if not self.active_trades:
                self.trades_text.insert(tk.END, "No active trades\n")
                self.trades_text.config(state=tk.DISABLED)
                return
                
            self.trades_text.insert(tk.END, f"Active Trades: {len(self.active_trades)}\n\n")
            
            for trade_id, trade in list(self.active_trades.items()):
                try:
                    symbol = trade.get('symbol', 'Unknown')
                    entry_price = trade.get('entry_price', 0.0)
                    
                    # Get current price safely
                    current_price = trade.get('current_price', entry_price)
                    
                    # Calculate profit percentage
                    profit_pct = trade.get('current_profit_percentage', 0.0)
                    
                    # Get entry time and calculate time in trade
                    entry_time_str = "Unknown"
                    time_in_trade_str = "Unknown"
                    
                    if 'entry_time' in trade and isinstance(trade['entry_time'], datetime):
                        entry_time = trade['entry_time']
                        entry_time_str = entry_time.strftime("%H:%M:%S")
                        time_diff = datetime.now() - entry_time
                        minutes, seconds = divmod(time_diff.seconds, 60)
                        time_in_trade_str = f"{minutes}m {seconds}s"
                    
                    # Format position size
                    position_size = trade.get('position_size', 0.0)
                    
                    # Format trade info for display
                    trade_info = (
                        f"{symbol} - ID: {trade_id}\n"
                        f"Entry: ${entry_price:.8f}\n"
                        f"Current: ${current_price:.8f}\n"
                        f"P/L: {profit_pct:.2f}%\n"
                        f"Size: ${position_size:.2f}\n"
                        f"Entry Time: {entry_time_str}\n"
                        f"Time in Trade: {time_in_trade_str}\n"
                        f"Highest P/L: {trade.get('highest_profit_percentage', 0.0):.2f}%\n"
                        f"{'-' * 30}\n"
                    )
                    
                    # Insert trade info into text widget
                    self.trades_text.insert(tk.END, trade_info)
                    
                    # Apply color coding based on profit/loss
                    start_pos = self.trades_text.index(f"end-{len(trade_info)+1}c")
                    end_pos = self.trades_text.index("end-1c")
                    
                    if profit_pct > 0:
                        self.trades_text.tag_add("profit", start_pos, end_pos)
                        self.trades_text.tag_config("profit", foreground="green")
                    else:
                        self.trades_text.tag_add("loss", start_pos, end_pos)
                        self.trades_text.tag_config("loss", foreground="red")
                    
                except Exception as e:
                    # Log error but continue processing other trades
                    self.log_trade(f"Error updating display for trade {trade_id}: {str(e)}")
                    continue
            
            # Ensure the display is scrolled to show the latest information
            self.trades_text.see(tk.END)
            self.trades_text.config(state=tk.DISABLED)
            
        except Exception as e:
            self.log_trade(f"Error updating trades display: {str(e)}")

    def monitor_limit_orders(self):
        """Monitor and manage unfilled limit orders"""
        try:
            if self.is_paper_trading or not self.active_trades:
                return
                
            self.log_trade("Checking status of limit orders...")
            
            # Get all open orders from the exchange
            open_orders = self.exchange.fetch_open_orders()
            self.log_trade(f"Found {len(open_orders)} open orders")
            
            # Track orders that need attention
            orders_to_cancel = []
            
            # Current time for timeout calculations
            current_time = datetime.now()
            
            # Check each active trade
            for trade_id, trade in list(self.active_trades.items()):
                # Skip if not a limit order
                if not trade.get('is_limit_order', False):
                    continue
                    
                # Skip if we don't have an order ID
                if 'order_id' not in trade:
                    continue
                    
                # Find matching open order
                matching_order = None
                for order in open_orders:
                    if order['id'] == trade['order_id']:
                        matching_order = order
                        break
                
                # If no matching order found, it might have been filled
                if matching_order is None:
                    # Check if the order was filled
                    try:
                        order_status = self.exchange.fetch_order(trade['order_id'])
                        if order_status['status'] == 'closed':
                            self.log_trade(f"Limit order for {trade['symbol']} was filled")
                            # Update trade with actual fill price if different
                            if 'price' in order_status and order_status['price'] != trade['entry_price']:
                                trade['entry_price'] = float(order_status['price'])
                                self.log_trade(f"Updated entry price to {trade['entry_price']}")
                        continue
                    except Exception as e:
                        self.log_trade(f"Error checking order status: {str(e)}")
                        continue
                
                # Check if the order has been open too long
                order_age = current_time - trade.get('entry_time', current_time)
                max_order_age = int(self.limit_order_timeout.get()) if hasattr(self, 'limit_order_timeout') else 300  # Default 5 minutes
                
                if order_age.total_seconds() > max_order_age:
                    self.log_trade(f"Limit order for {trade['symbol']} has timed out ({order_age.total_seconds():.0f}s > {max_order_age}s)")
                    orders_to_cancel.append((trade_id, trade, matching_order))
                    continue
                
                # Check if price has moved significantly away
                try:
                    current_price = float(self.exchange.fetch_ticker(trade['symbol'])['last'])
                    limit_price = float(trade['entry_price'])
                    price_diff_pct = abs(current_price - limit_price) / limit_price * 100
                    
                    # Cancel if price has moved too far away
                    max_price_diff = float(self.limit_price_diff.get()) if hasattr(self, 'limit_price_diff') else 0.5  # Default 0.5%
                    
                    if price_diff_pct > max_price_diff:
                        self.log_trade(f"Price moved too far for {trade['symbol']} limit order: {price_diff_pct:.2f}% > {max_price_diff:.2f}%")
                        orders_to_cancel.append((trade_id, trade, matching_order))
                except Exception as e:
                    self.log_trade(f"Error checking price movement: {str(e)}")
            
            # Process orders that need to be canceled
            for trade_id, trade, order in orders_to_cancel:
                try:
                    # Cancel the order
                    self.exchange.cancel_order(order['id'])
                    self.log_trade(f"Canceled limit order {order['id']} for {trade['symbol']}")
                    
                    # Decide what to do next
                    action = self.limit_order_action.get() if hasattr(self, 'limit_order_action') else "market"
                    
                    if action == "market":
                        # Replace with market order
                        self.log_trade(f"Replacing with market order for {trade['symbol']}")
                        
                        # Get current price
                        current_price = float(self.exchange.fetch_ticker(trade['symbol'])['last'])
                        
                        # Create market order
                        market_order = self.exchange.create_market_buy_order(
                            trade['symbol'], 
                            trade['amount']
                        )
                        
                        # Update trade details
                        trade['is_limit_order'] = False
                        trade['entry_price'] = current_price
                        trade['order_id'] = market_order['id']
                        trade['fee_type'] = 'taker'
                        trade['entry_fee'] = trade['position_size'] * self.taker_fee
                        
                        self.log_trade(f"Created market order: {market_order['id']} for {trade['symbol']}")
                        
                    elif action == "adjust":
                        # Adjust limit price
                        self.log_trade(f"Adjusting limit price for {trade['symbol']}")
                        
                        # Get current price
                        current_price = float(self.exchange.fetch_ticker(trade['symbol'])['last'])
                        
                        # Create new limit order with adjusted price
                        new_order = self.exchange.create_limit_buy_order(
                            trade['symbol'], 
                            trade['amount'],
                            current_price
                        )
                        
                        # Update trade details
                        trade['entry_price'] = current_price
                        trade['order_id'] = new_order['id']
                        
                        self.log_trade(f"Created adjusted limit order: {new_order['id']} for {trade['symbol']} at {current_price}")
                        
                    else:
                        # Cancel the trade entirely
                        self.log_trade(f"Canceling trade for {trade['symbol']}")
                        del self.active_trades[trade_id]
                        
                        # Update displays
                        self.update_trades_display()
                
                except Exception as e:
                    self.log_trade(f"Error handling unfilled limit order: {str(e)}")
            
            # Update displays
            self.update_trades_display()
            
        except Exception as e:
            self.log_trade(f"Error monitoring limit orders: {str(e)}")

    def setup_advanced_parameters(self, parent):
        """Set up advanced trading parameters"""
        try:
            # Create a frame for Greek parameters
            greek_frame = ttk.LabelFrame(parent, text="Greek Parameters")
            greek_frame.pack(fill=tk.X, padx=5, pady=5)
            
            # Create a grid layout
            row = 0
            
            # Momentum Beta
            ttk.Label(greek_frame, text="Momentum Beta").grid(row=row, column=0, sticky="w", padx=5, pady=2)
            self.momentum_beta = tk.StringVar(self.root, value="0.0001")
            ttk.Entry(greek_frame, textvariable=self.momentum_beta, width=10).grid(row=row, column=1, padx=5, pady=2)
            row += 1
            
            # Price Alpha
            ttk.Label(greek_frame, text="Price Alpha").grid(row=row, column=0, sticky="w", padx=5, pady=2)
            self.price_alpha = tk.StringVar(self.root, value="0.001")
            ttk.Entry(greek_frame, textvariable=self.price_alpha, width=10).grid(row=row, column=1, padx=5, pady=2)
            row += 1
            
            # Momentum Theta
            ttk.Label(greek_frame, text="Momentum Theta").grid(row=row, column=0, sticky="w", padx=5, pady=2)
            self.momentum_theta = tk.StringVar(self.root, value="0.5")
            ttk.Entry(greek_frame, textvariable=self.momentum_theta, width=10).grid(row=row, column=1, padx=5, pady=2)
            row += 1
            
            # Volatility Vega
            ttk.Label(greek_frame, text="Vol Vega").grid(row=row, column=0, sticky="w", padx=5, pady=2)
            self.vol_vega = tk.StringVar(self.root, value="10.0")
            ttk.Entry(greek_frame, textvariable=self.vol_vega, width=10).grid(row=row, column=1, padx=5, pady=2)
            row += 1
            
            # Volume Rho
            ttk.Label(greek_frame, text="Volume Rho").grid(row=row, column=0, sticky="w", padx=5, pady=2)
            self.volume_rho = tk.StringVar(self.root, value="0.001")
            ttk.Entry(greek_frame, textvariable=self.volume_rho, width=10).grid(row=row, column=1, padx=5, pady=2)
            row += 1
            
            # Create a frame for technical indicators
            tech_frame = ttk.LabelFrame(parent, text="Technical Indicators")
            tech_frame.pack(fill=tk.X, padx=5, pady=5)
            
            # Reset row counter for new frame
            row = 0
            
            # RSI Period
            ttk.Label(tech_frame, text="RSI Period").grid(row=row, column=0, sticky="w", padx=5, pady=2)
            ttk.Entry(tech_frame, textvariable=self.rsi_period, width=10).grid(row=row, column=1, padx=5, pady=2)
            row += 1
            
            # RSI Overbought
            ttk.Label(tech_frame, text="RSI Overbought").grid(row=row, column=0, sticky="w", padx=5, pady=2)
            ttk.Entry(tech_frame, textvariable=self.rsi_overbought, width=10).grid(row=row, column=1, padx=5, pady=2)
            row += 1
            
            # RSI Oversold
            ttk.Label(tech_frame, text="RSI Oversold").grid(row=row, column=0, sticky="w", padx=5, pady=2)
            ttk.Entry(tech_frame, textvariable=self.rsi_oversold, width=10).grid(row=row, column=1, padx=5, pady=2)
            row += 1
            
            # EMA Short
            ttk.Label(tech_frame, text="EMA Short").grid(row=row, column=0, sticky="w", padx=5, pady=2)
            ttk.Entry(tech_frame, textvariable=self.ema_short, width=10).grid(row=row, column=1, padx=5, pady=2)
            row += 1
            
            # EMA Long
            ttk.Label(tech_frame, text="EMA Long").grid(row=row, column=0, sticky="w", padx=5, pady=2)
            ttk.Entry(tech_frame, textvariable=self.ema_long, width=10).grid(row=row, column=1, padx=5, pady=2)
            row += 1
            
            # Apply button
            ttk.Button(parent, text="Apply Advanced Settings", command=self.validate_and_apply_advanced).pack(fill=tk.X, padx=5, pady=10)
            
        except Exception as e:
            self.log_trade(f"Error setting up advanced parameters: {str(e)}")

    def validate_and_apply_advanced(self):
        """Validate and apply advanced parameters"""
        try:
            # Validate Greek parameters
            momentum_beta = float(self.momentum_beta.get())
            if momentum_beta < 0 or momentum_beta > 1:
                messagebox.showerror("Validation Error", "Momentum Beta must be between 0 and 1")
                return
                
            price_alpha = float(self.price_alpha.get())
            if price_alpha < 0 or price_alpha > 1:
                messagebox.showerror("Validation Error", "Price Alpha must be between 0 and 1")
                return
                
            momentum_theta = float(self.momentum_theta.get())
            if momentum_theta < 0 or momentum_theta > 1:
                messagebox.showerror("Validation Error", "Momentum Theta must be between 0 and 1")
                return
                
            vol_vega = float(self.vol_vega.get())
            if vol_vega < 0 or vol_vega > 20:
                messagebox.showerror("Validation Error", "Vol Vega must be between 0 and 20")
                return
                
            volume_rho = float(self.volume_rho.get())
            if volume_rho < 0 or volume_rho > 1:
                messagebox.showerror("Validation Error", "Volume Rho must be between 0 and 1")
                return
                
            # Validate technical indicators
            rsi_period = int(float(self.rsi_period.get()))
            if rsi_period < 2 or rsi_period > 30:
                messagebox.showerror("Validation Error", "RSI Period must be between 2 and 30")
                return
                
            rsi_overbought = float(self.rsi_overbought.get())
            if rsi_overbought < 50 or rsi_overbought > 90:
                messagebox.showerror("Validation Error", "RSI Overbought must be between 50 and 90")
                return
                
            rsi_oversold = float(self.rsi_oversold.get())
            if rsi_oversold < 10 or rsi_oversold > 50:
                messagebox.showerror("Validation Error", "RSI Oversold must be between 10 and 50")
                return
                
            if rsi_oversold >= rsi_overbought:
                messagebox.showerror("Validation Error", "RSI Oversold must be less than RSI Overbought")
                return
                
            ema_short = int(float(self.ema_short.get()))
            if ema_short < 1 or ema_short > 50:
                messagebox.showerror("Validation Error", "EMA Short must be between 1 and 50")
                return
                
            ema_long = int(float(self.ema_long.get()))
            if ema_long < 5 or ema_long > 200:
                messagebox.showerror("Validation Error", "EMA Long must be between 5 and 200")
                return
                
            if ema_short >= ema_long:
                messagebox.showerror("Validation Error", "EMA Short must be less than EMA Long")
                return
                
            # All validations passed, apply settings
            self.log_trade("Advanced parameters validated successfully")
            self.log_trade(f"Greek Parameters: Beta={momentum_beta}, Alpha={price_alpha}, Theta={momentum_theta}, Vega={vol_vega}, Rho={volume_rho}")
            self.log_trade(f"Technical Indicators: RSI({rsi_period}, {rsi_oversold}, {rsi_overbought}), EMA({ema_short}, {ema_long})")
            
            # Update status
            self.update_status("Advanced settings applied")
            
            # Show confirmation
            messagebox.showinfo("Success", "Advanced parameters applied successfully")
            
        except ValueError as e:
            messagebox.showerror("Input Error", f"Invalid number format: {str(e)}")
        except Exception as e:
            self.log_trade(f"Error validating advanced parameters: {str(e)}")
            messagebox.showerror("Error", f"Failed to apply advanced parameters: {str(e)}")

    def monitor_trades(self):
        """Monitor active trades and close them when conditions are met"""
        try:
            if not self.active_trades:
                return
                
            # First enforce max trades limit
            self.enforce_max_trades_limit()
                
            self.log_trade(f"Monitoring {len(self.active_trades)} active trades...")
            
            # Get parameters from GUI
            stop_loss = float(self.stop_loss.get())
            profit_target = float(self.profit_target.get())
            trailing_stop = float(self.trailing_stop.get())
            trailing_activation = float(self.trailing_activation.get())
            
            self.log_trade(f"Using stop loss: {stop_loss}%, profit target: {profit_target}%, " +
                          f"trailing stop: {trailing_stop}%, trailing activation: {trailing_activation}%")
            
            # Check each active trade
            for trade_id in list(self.active_trades.keys()):
                try:
                    trade = self.active_trades[trade_id]
                    symbol = trade['symbol']
                    entry_price = float(trade['entry_price'])
                    
                    # Get current price
                    ticker = self.exchange.fetch_ticker(symbol)
                    current_price = float(ticker['last'])
                    
                    # Calculate profit percentage
                    profit_pct = ((current_price - entry_price) / entry_price) * 100
                    
                    # Update highest price and profit if needed
                    if current_price > trade.get('highest_price', 0):
                        trade['highest_price'] = current_price
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
                    
                    # CASE 1: Stop Loss - Exit if price drops below stop loss
                    if profit_pct <= -stop_loss:
                        self.log_trade(f"Stop loss triggered on {trade['symbol']} at {profit_pct:.2f}%")
                        self.close_trade(trade_id, trade, current_price, "stop loss")
                        continue
                    
                    # CASE 2: Take Profit - Exit if price reaches profit target
                    if profit_pct >= profit_target:
                        self.log_trade(f"Profit target reached on {trade['symbol']} at {profit_pct:.2f}%")
                        self.close_trade(trade_id, trade, current_price, "profit target")
                        continue
                    
                    # CASE 3: Trailing Stop - Exit if price drops from highest by trailing stop amount
                    highest_profit = trade.get('highest_profit', 0)
                    if highest_profit >= trailing_activation:
                        drop_from_high = highest_profit - profit_pct
                        if drop_from_high >= trailing_stop:
                            self.log_trade(f"Trailing stop triggered on {trade['symbol']} - Drop from {highest_profit:.2f}% to {profit_pct:.2f}%")
                            self.close_trade(trade_id, trade, current_price, "trailing stop")
                            continue
                    
                    # Update trade data for monitoring
                    trade['current_price'] = current_price
                    trade['current_profit_percentage'] = profit_pct
                    trade['last_update'] = datetime.now()
                    
                    # Update price history for charting
                    if symbol in self.price_history:
                        self.price_history[symbol].append((datetime.now(), current_price))
                        # Limit history size
                        self.price_history[symbol] = self.price_history[symbol][-100:]
                    
                except Exception as e:
                    self.log_trade(f"Error monitoring trade {trade_id}: {str(e)}")
                    continue
            
            # Update displays
            self.update_active_trades_display()
            self.update_chart()
            self.update_balance_display()
            
        except Exception as e:
            self.log_trade(f"Error in monitor_trades: {str(e)}")

    def run_bot(self):
        """Main bot loop"""
        try:
            self.running = True
            self.update_status("Running")
            self.log_trade("=== BOT STARTED ===")
            
            # Start trading thread
            self.trading_thread = threading.Thread(target=self.trading_loop)
            self.trading_thread.daemon = True
            self.trading_thread.start()
            
        except Exception as e:
            self.log_trade(f"Error starting bot: {str(e)}")
            self.update_status("Error")

    def trading_loop(self):
        """Main trading loop running in a separate thread"""
        try:
            # Initialize last scan time
            self.last_scan_time = 0
            
            # Main loop
            while self.running:
                try:
                    # Monitor active trades (do this first to ensure quick response to exit conditions)
                    self.monitor_trades()
                    
                    # Monitor limit orders (check if any need to be canceled/adjusted)
                    if not self.is_paper_trading:
                        self.monitor_limit_orders()
                    
                    # Scan for new opportunities (only if we have room for more trades)
                    max_trades = int(self.max_trades_entry.get()) if hasattr(self, 'max_trades_entry') else 3
                    current_time = time.time()
                    
                    # Force a scan every 30 seconds if we're not at max trades and not currently scanning
                    if len(self.active_trades) < max_trades and not self.is_scanning:
                        if (current_time - self.last_scan_time) > 30:
                            self.log_trade(f"Initiating scheduled scan (active: {len(self.active_trades)}, max: {max_trades})")
                            self.is_scanning = True  # Set flag to prevent multiple scans
                            self.scan_opportunities()
                            self.is_scanning = False  # Reset flag after scan completes
                            self.last_scan_time = current_time
                    
                    # Sleep to avoid high CPU usage
                    time.sleep(1)
                
                except Exception as e:
                    self.log_trade(f"Error in trading loop: {str(e)}")
                    time.sleep(5)
                
        except Exception as e:
            self.log_trade(f"Fatal error in trading loop: {str(e)}")
            self.stop_bot()

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
