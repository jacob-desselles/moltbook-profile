import PyInstaller.__main__
import os
import shutil

def build_executable():
    # Verify public.pem exists
    public_pem_path = os.path.join(os.getenv('APPDATA', os.path.expanduser('~')), 'Vantrex', 'public.pem')
    if not os.path.exists(public_pem_path):
        raise FileNotFoundError(f"public.pem not found at {public_pem_path}. Generate it using KeyGenerator first.")

    # Define additional files
    additional_files = [
        '--add-data', f'{public_pem_path};.',  # Bundle public.pem
        '--add-data', 'scalper_bot.py;.',      # Bot script
        '--add-data', 'license_wrapper.py;.',  # License wrapper script
    ]

    # Define required packages
    hidden_imports = [
        '--hidden-import', 'cryptography',           # For RSA and Fernet
        '--hidden-import', 'tkinter',                # For GUI
        '--hidden-import', 'ccxt',                   # For crypto exchange
        '--hidden-import', 'pandas',                 # Data handling
        '--hidden-import', 'numpy',                  # Numerical operations
        '--hidden-import', 'matplotlib',             # Plotting
        '--hidden-import', 'matplotlib.backends.backend_tkagg',  # Tkinter backend
        '--hidden-import', 'configparser',           # Configuration
        '--hidden-import', 'pathlib',                # Path handling
        '--hidden-import', 'psutil',                 # System resources
    ]

    # Build command
    command = [
        'license_wrapper.py',  # Entry point
        '--onefile',
        '--noconsole',        # Hide console window
        '--name=Vantrex',
        '--clean',
    ]

    # Add icon if available
    if os.path.exists('assets/icon.ico'):
        command.extend(['--icon=assets/icon.ico'])

    # Combine all arguments
    command.extend(additional_files)
    command.extend(hidden_imports)

    # Run PyInstaller
    print("Running PyInstaller...")
    PyInstaller.__main__.run(command)

    # Clean up build artifacts
    if os.path.exists('build'):
        shutil.rmtree('build')
    
    # Create distribution folder
    dist_folder = "distribution"
    if not os.path.exists(dist_folder):
        os.makedirs(dist_folder)
    
    # Move the executable to the distribution folder
    try:
        source = os.path.join('dist', 'Vantrex.exe')
        destination = os.path.join(dist_folder, 'Vantrex.exe')
        
        if os.path.exists(destination):
            os.remove(destination)  # Remove existing file if it exists
            
        shutil.move(source, destination)
        
        # Clean up dist folder
        shutil.rmtree('dist')
        
        print(f"\nBuild completed successfully!")
        print(f"Executable location: {destination}")
        
    except Exception as e:
        print(f"Error moving executable: {str(e)}")

if __name__ == "__main__":
    try:
        print("Starting build process...")
        build_executable()
    except Exception as e:
        print(f"Build failed: {str(e)}")