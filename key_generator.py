import hashlib
import random
import string
import datetime
import json

class KeyGenerator:
    def __init__(self):
        self.key_file = "valid_keys.json"
        self.key_prefix = "KT"  # Changed to KT for Kairos Trading
        
    def generate_key(self, user_info=None):
        # Generate timestamp
        timestamp = datetime.datetime.now().strftime('%Y%m%d')
        
        # Generate random string
        random_string = ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))
        
        # Create base string
        base = f"{self.key_prefix}-{timestamp}-{random_string}"
        
        # Add checksum
        checksum = hashlib.md5(base.encode()).hexdigest()[:4]
        
        # Final key format: KT-YYYYMMDD-RANDOM-CHECKSUM
        key = f"{base}-{checksum}"
        
        # Save key with user info
        self.save_key(key, user_info)
        
        return key
    
    def save_key(self, key, user_info):
        try:
            # Load existing keys
            try:
                with open(self.key_file, 'r') as f:
                    keys = json.load(f)
            except FileNotFoundError:
                keys = {}
            
            # Add new key
            keys[key] = {
                'generated_date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'user_info': user_info,
                'activated': False
            }
            
            # Save updated keys
            with open(self.key_file, 'w') as f:
                json.dump(keys, f, indent=4)
                
        except Exception as e:
            print(f"Error saving key: {str(e)}")

def generate_multiple_keys(num_keys):
    generator = KeyGenerator()
    keys = []
    
    for i in range(num_keys):
        user_info = f"User_{i+1}"
        key = generator.generate_key(user_info)
        keys.append(key)
        
    return keys

if __name__ == "__main__":
    print("Kairos Trading - License Key Generator")
    print("=====================================")
    
    while True:
        print("\n1. Generate single key")
        print("2. Generate multiple keys")
        print("3. View all keys")
        print("4. Exit")
        
        choice = input("\nEnter your choice (1-4): ")
        
        if choice == '1':
            user_name = input("Enter user name (optional): ")
            key = KeyGenerator().generate_key(user_name)
            print(f"\nGenerated Key: {key}")
            
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
                with open("valid_keys.json", 'r') as f:
                    keys = json.load(f)
                print("\nAll Generated Keys:")
                for key, info in keys.items():
                    print(f"\nKey: {key}")
                    print(f"Generated: {info['generated_date']}")
                    print(f"User: {info['user_info']}")
                    print(f"Activated: {info['activated']}")
            except FileNotFoundError:
                print("No keys have been generated yet.")
                
        elif choice == '4':
            break
            
        else:
            print("Invalid choice. Please try again.")