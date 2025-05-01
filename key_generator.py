import hashlib
import os
import random
import string
import datetime
import json
import base64
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
import logging

# Configure logging
logging.basicConfig(filename='key_generator.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

class KeyGenerator:
    def __init__(self):
        # Use APPDATA or fallback to user home directory
        appdata = os.getenv('APPDATA')
        if not appdata:
            appdata = os.path.expanduser('~')
            logging.warning("APPDATA not set, falling back to user home directory")
        
        self.base_dir = os.path.join(appdata, 'Vantrex')
        self.key_file = os.path.join(self.base_dir, 'valid_keys.json')
        self.key_prefix = "VT"
        self.private_key, self.public_key = self._load_or_generate_rsa_keys()

    def _load_or_generate_rsa_keys(self):
        """Load or generate RSA key pair for signing."""
        private_key_file = os.path.join(self.base_dir, 'private.pem')
        public_key_file = os.path.join(self.base_dir, 'public.pem')

        try:
            logging.info(f"Attempting to create directory: {self.base_dir}")
            os.makedirs(self.base_dir, exist_ok=True)

            if not os.path.exists(private_key_file):
                logging.info(f"Generating new RSA key pair at {private_key_file}")
                private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
                public_key = private_key.public_key()
                
                with open(private_key_file, 'wb') as f:
                    f.write(private_key.private_bytes(
                        encoding=serialization.Encoding.PEM,
                        format=serialization.PrivateFormat.PKCS8,
                        encryption_algorithm=serialization.NoEncryption()
                    ))
                with open(public_key_file, 'wb') as f:
                    f.write(public_key.public_bytes(
                        encoding=serialization.Encoding.PEM,
                        format=serialization.PublicFormat.SubjectPublicKeyInfo
                    ))
            else:
                logging.info(f"Loading existing RSA keys from {private_key_file}")
                with open(private_key_file, 'rb') as f:
                    private_key = serialization.load_pem_private_key(f.read(), password=None)
                with open(public_key_file, 'rb') as f:
                    public_key = serialization.load_pem_public_key(f.read())
        
            return private_key, public_key
        except Exception as e:
            logging.error(f"Failed to load or generate RSA keys at {private_key_file}: {str(e)}")
            raise

    def generate_key(self, user_info=None):
        """Generate a license key compatible with LicenseWrapper."""
        try:
            timestamp = datetime.datetime.now().strftime('%Y%m%d')
            user_id = user_info if user_info else f"user_{random.randint(1000, 9999)}"
            random_string = os.urandom(4).hex()
            base = f"{self.key_prefix}-{timestamp}-{user_id}-{random_string}"
            
            signature = self.private_key.sign(
                base.encode(),
                padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH),
                hashes.SHA256()
            )
            signature_b64 = base64.b64encode(signature).decode()  # Full signature
            
            key = f"{base}-{signature_b64}"
            self.save_key(key, user_info, timestamp)
            logging.info(f"Generated key: {key} for user: {user_info}")
            
            return key
        except Exception as e:
            logging.error(f"Key generation failed: {str(e)}")
            raise

    def save_key(self, key, user_info, timestamp):
        """Save the generated key to valid_keys.json."""
        try:
            logging.info(f"Attempting to save key to {self.key_file}")
            os.makedirs(self.base_dir, exist_ok=True)
            
            try:
                with open(self.key_file, 'r') as f:
                    keys = json.load(f)
            except FileNotFoundError:
                keys = {}
            
            keys[key] = {
                'generated_date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'user_info': user_info,
                'timestamp': timestamp,
                'activated': False
            }
            
            with open(self.key_file, 'w') as f:
                json.dump(keys, f, indent=4)
                
            self._add_file_integrity()
        except Exception as e:
            logging.error(f"Error saving key to {self.key_file}: {str(e)}")
            raise

    def _add_file_integrity(self):
        """Add HMAC to valid_keys.json to detect tampering (optional)."""
        try:
            with open(self.key_file, 'rb') as f:
                data = f.read()
            hmac_key = hashlib.sha256(b"secret_key").digest()
            hmac = hashlib.sha256(data + hmac_key).hexdigest()
            
            with open(self.key_file + '.hmac', 'w') as f:
                f.write(hmac)
        except Exception as e:
            logging.error(f"Error adding HMAC: {str(e)}")

    def verify_file_integrity(self):
        """Verify the integrity of valid_keys.json (optional)."""
        try:
            with open(self.key_file, 'rb') as f:
                data = f.read()
            with open(self.key_file + '.hmac', 'r') as f:
                stored_hmac = f.read()
                
            hmac_key = hashlib.sha256(b"secret_key").digest()
            computed_hmac = hashlib.sha256(data + hmac_key).hexdigest()
            
            return computed_hmac == stored_hmac
        except Exception as e:
            logging.error(f"Integrity verification failed: {str(e)}")
            return False

def generate_multiple_keys(num_keys):
    generator = KeyGenerator()
    keys = []
    for i in range(num_keys):
        user_info = f"User_{i+1}"
        key = generator.generate_key(user_info)
        keys.append(key)
    return keys

if __name__ == "__main__":
    print("Vantrex - License Key Generator")
    print("=====================================")
    
    try:
        generator = KeyGenerator()
        while True:
            print("\n1. Generate single key")
            print("2. Generate multiple keys")
            print("3. View all keys")
            print("4. Verify key file integrity")
            print("5. Exit")
            choice = input("\nEnter your choice (1-5): ")
            if choice == '1':
                user_name = input("Enter user name (optional): ")
                try:
                    key = generator.generate_key(user_name)
                    print(f"\nGenerated Key: {key}")
                except Exception as e:
                    print(f"Error: {str(e)}")
            elif choice == '2':
                try:
                    num = int(input("How many keys do you want to generate? "))
                    keys = generate_multiple_keys(num)
                    print("\nGenerated Keys:")
                    for i, key in enumerate(keys, 1):
                        print(f"{i}. {key}")
                except ValueError:
                    print("Please enter a valid number")
            elif choice == '3':
                try:
                    with open(generator.key_file, 'r') as f:
                        keys = json.load(f)
                    print("\nAll Generated Keys:")
                    for key, info in keys.items():
                        print(f"\nKey: {key}")
                        print(f"Generated: {info['generated_date']}")
                        print(f"User: {info['user_info']}")
                        print(f"Timestamp: {info['timestamp']}")
                        print(f"Activated: {info['activated']}")
                except FileNotFoundError:
                    print("No keys have been generated yet.")
            elif choice == '4':
                if generator.verify_file_integrity():
                    print("Key file integrity verified successfully.")
                else:
                    print("Key file integrity check failed. File may have been tampered with.")
            elif choice == '5':
                break
            else:
                print("Invalid choice. Please try again.")
    except Exception as e:
        logging.error(f"KeyGenerator initialization failed: {str(e)}")
        print(f"Failed to initialize KeyGenerator: {str(e)}")