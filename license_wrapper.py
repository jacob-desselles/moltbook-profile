import tkinter as tk
from tkinter import messagebox
import json
import hashlib
import os
import sys
from pathlib import Path
import logging
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
import base64
import datetime

# Ensure the KairosTrading directory exists before setting up logging
log_dir = os.path.join(os.getenv('APPDATA', os.path.expanduser('~')), 'KairosTrading')
os.makedirs(log_dir, exist_ok=True)

# Configure logging
logging.basicConfig(filename=os.path.join(log_dir, 'license.log'),
                    level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class LicenseWrapper:
    def __init__(self):
        self.base_dir = log_dir  # Use the same directory as logging
        self.license_file = os.path.join(self.base_dir, 'license.dat')
        self.key_file = os.path.join(self.base_dir, 'key.key')
        os.makedirs(self.base_dir, exist_ok=True)  # Redundant but kept for clarity
        
        self.encryption_key = self._load_or_generate_key()
        self.fernet = Fernet(self.encryption_key)
        self.public_key = self._load_public_key()

    def _load_or_generate_key(self):
        """Generate or load a Fernet key for encryption."""
        try:
            if not os.path.exists(self.key_file):
                key = Fernet.generate_key()
                with open(self.key_file, 'wb') as f:
                    f.write(key)
            with open(self.key_file, 'rb') as f:
                return f.read()
        except Exception as e:
            logging.error(f"Failed to load or generate encryption key: {str(e)}")
            raise

    def _load_public_key(self):
        """Load public key, handling bundled files in PyInstaller."""
        try:
            if getattr(sys, 'frozen', False):
                base_path = sys._MEIPASS
            else:
                base_path = self.base_dir
                
            public_key_file = os.path.join(base_path, 'public.pem')
            logging.info(f"Loading public key from {public_key_file}")
            
            with open(public_key_file, 'rb') as f:
                return serialization.load_pem_public_key(f.read())
        except Exception as e:
            logging.error(f"Failed to load public key: {str(e)}")
            raise

    def get_hardware_id(self):
        """Generate a robust hardware ID."""
        identifiers = [
            os.getenv('COMPUTERNAME', ''),
            os.getenv('PROCESSOR_IDENTIFIER', ''),
        ]
        combined = ''.join(identifiers)
        return hashlib.sha256(combined.encode()).hexdigest()

    def check_license(self):
        """Check if a valid license exists."""
        try:
            if not Path(self.license_file).exists():
                logging.warning("License file not found")
                return False
            
            with open(self.license_file, 'rb') as f:
                encrypted_data = f.read()
            decrypted_data = self.fernet.decrypt(encrypted_data)
            license_data = json.loads(decrypted_data.decode())
            
            if license_data.get('hardware_id') != self.get_hardware_id():
                logging.error("Hardware ID mismatch")
                return False
            
            expiry_date = datetime.datetime.fromisoformat(license_data.get('expiry'))
            if datetime.datetime.now() > expiry_date:
                logging.error("License expired")
                return False
            
            return True
        except Exception as e:
            logging.error(f"License check failed: {str(e)}")
            return False

    def create_license(self, key):
        """Create a license file if the key is valid."""
        if self.validate_key(key):
            try:
                expiry_date = (datetime.datetime.now() + datetime.timedelta(days=365)).isoformat()
                license_data = {
                    'hardware_id': self.get_hardware_id(),
                    'key': key,
                    'expiry': expiry_date
                }
                
                encrypted_data = self.fernet.encrypt(json.dumps(license_data).encode())
                with open(self.license_file, 'wb') as f:
                    f.write(encrypted_data)
                logging.info("License created successfully")
                return True
            except Exception as e:
                logging.error(f"License creation failed: {str(e)}")
                return False
        logging.error("Invalid license key during creation")
        return False

    def validate_key(self, key):
        """Validate the license key format and signature."""
        try:
            logging.info(f"Validating key: {key}")
            parts = key.split('-')
            if len(parts) != 5 or parts[0] != 'KT':
                logging.error(f"Invalid key format: {key} (Expected 5 parts with prefix KT)")
                return False
                
            prefix, date, user_id, random, signature = parts
            logging.info(f"Key parts - prefix: {prefix}, date: {date}, user_id: {user_id}, random: {random}, signature: {signature}")
            
            try:
                datetime.datetime.strptime(date, '%Y%m%d')
            except ValueError:
                logging.error(f"Invalid date format: {date}")
                return False
                
            key_data = f"{prefix}-{date}-{user_id}-{random}".encode()
            logging.info(f"Key data for verification: {key_data.decode()}")
            
            try:
                signature_padded = signature + '=' * (4 - len(signature) % 4) if len(signature) % 4 else signature
                signature_bytes = base64.b64decode(signature_padded)
                logging.info(f"Decoded signature length: {len(signature_bytes)} bytes")
                
                self.public_key.verify(
                    signature_bytes,
                    key_data,
                    padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH),
                    hashes.SHA256()
                )
                logging.info(f"Key signature verified successfully: {key}")
                return True
            except Exception as e:
                logging.error(f"Key signature verification failed: {str(e)}")
                return False
        except Exception as e:
            logging.error(f"Key validation error: {str(e)}")
            return False

    def show_license_window(self):
        """Show the license activation window."""
        window = tk.Tk()
        window.title("Kairos Trading - Activation")
        window.geometry("400x200")
        window.configure(bg='#2c2c2c')
        
        style = {'bg': '#2c2c2c', 'fg': 'white', 'font': ('Arial', 10)}
        
        tk.Label(window, text="Enter License Key:", **style).pack(pady=20)
        
        key_entry = tk.Entry(window, width=40, bg='#3c3c3c', fg='white')
        key_entry.pack(pady=10)
        
        activation_result = {'success': False}
        
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

        from scalper_bot import CryptoScalpingBot
        bot = CryptoScalpingBot()
        bot.run()
        
    except Exception as e:
        logging.error(f"Program failed: {str(e)}")
        messagebox.showerror("Error", f"Failed to start program: {str(e)}")