import tkinter as tk
from tkinter import messagebox
import json
import hashlib
import os
import sys
import ctypes
from pathlib import Path
import logging

class LicenseWrapper:
    def __init__(self):
        self.license_file = os.path.join(os.getenv('APPDATA'), 'KairosTrading', 'license.dat')
        os.makedirs(os.path.dirname(self.license_file), exist_ok=True)
        
    def get_hardware_id(self):
        system_info = f"{os.getenv('COMPUTERNAME', '')}{os.getenv('PROCESSOR_IDENTIFIER', '')}"
        return hashlib.md5(system_info.encode()).hexdigest()

    def check_license(self):
        try:
            if not Path(self.license_file).exists():
                return False
            
            with open(self.license_file, 'r') as f:
                license_data = json.load(f)
            return license_data.get('hardware_id') == self.get_hardware_id()
        except:
            return False

    def create_license(self, key):
        if self.validate_key(key):
            license_data = {
                'hardware_id': self.get_hardware_id(),
                'key': key
            }
            with open(self.license_file, 'w') as f:
                json.dump(license_data, f)
            return True
        return False

    def validate_key(self, key):
        try:
            # Basic format check
            parts = key.split('-')
            if len(parts) != 4:
                logging.error(f"Invalid key format: {key}")
                return False
                
            prefix, date, random, checksum = parts
            
            # Check prefix
            if prefix != 'KT':  # Changed to KT for Kairos Trading
                logging.error(f"Invalid prefix: {prefix}")
                return False
                
            # Create base string (same as in key generator)
            base = f"{prefix}-{date}-{random}"
            
            # Calculate checksum
            expected_checksum = hashlib.md5(base.encode()).hexdigest()[:4]
            
            # Compare checksums
            if checksum.lower() != expected_checksum.lower():
                logging.error(f"Checksum mismatch: expected {expected_checksum}, got {checksum}")
                return False
                
            # Optional: Load valid_keys.json to check if key exists
            try:
                with open('valid_keys.json', 'r') as f:
                    valid_keys = json.load(f)
                    if key in valid_keys:
                        return True
            except:
                # If we can't read the file, just validate the format
                pass
                
            return True
            
        except Exception as e:
            logging.error(f"Key validation error: {str(e)}")
            return False

    def show_license_window(self):
        window = tk.Tk()
        window.title("Kairos Trading - Activation")
        window.geometry("400x200")
        window.configure(bg='#2c2c2c')
        
        style = {'bg': '#2c2c2c', 'fg': 'white', 'font': ('Arial', 10)}
        
        tk.Label(window, text="Enter License Key:", **style).pack(pady=20)
        
        key_entry = tk.Entry(window, width=40, bg='#3c3c3c', fg='white')
        key_entry.pack(pady=10)
        
        activation_result = {'success': False}  # Use dictionary to store result
        
        def activate():
            key = key_entry.get()
            if self.create_license(key):
                messagebox.showinfo("Success", "License activated successfully!")
                activation_result['success'] = True
                window.destroy()
            else:
                messagebox.showerror("Error", "Invalid license key!")
        
        tk.Button(window, text="Activate", command=activate,
                 bg='#404040', fg='white').pack(pady=20)
        
        window.mainloop()
        return activation_result['success']

if __name__ == "__main__":
    try:
        wrapper = LicenseWrapper()
        
        if not wrapper.check_license():
            if not wrapper.show_license_window():
                sys.exit()

        # Import and start the bot
        from scalper_bot import CryptoScalpingBot
        bot = CryptoScalpingBot()
        bot.run()
        
    except Exception as e:
        messagebox.showerror("Error", f"Failed to start program: {str(e)}")