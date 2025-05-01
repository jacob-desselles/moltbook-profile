import PyInstaller.__main__
import os
import shutil

def build_demo_executable():
    """Build the demo version executable of the scalping bot"""
    print("Starting build process for demo version...")
    
    # Define additional files
    additional_files = [
        '--add-data', 'scalper_bot_paper.py;.',  # Demo bot script
    ]

    # Define required packages
    hidden_imports = [
        '--hidden-import', 'tkinter',                # For GUI
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
        'scalper_bot_paper.py',  # Entry point
        '--onefile',
        '--noconsole',        # Hide console window
        '--name=VantrexDemo',
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
    dist_folder = "demo_distribution"
    if not os.path.exists(dist_folder):
        os.makedirs(dist_folder)
    
    # Move the executable to the distribution folder
    try:
        source = os.path.join('dist', 'VantrexDemo.exe')
        destination = os.path.join(dist_folder, 'VantrexDemo.exe')
        
        if os.path.exists(destination):
            os.remove(destination)  # Remove existing file if it exists
            
        shutil.move(source, destination)
        
        # Clean up dist folder
        shutil.rmtree('dist')
        
        # Create a basic README for the demo
        with open(os.path.join(dist_folder, 'README.txt'), 'w') as f:
            f.write("""Vantrex Demo

This is a demonstration version of the Vantrex Trading Suite, a professional-grade 
automated trading solution for cryptocurrency scalping.

Features in this demo:
- Paper trading with simulated market data
- Full trading interface and charting
- Performance tracking
- Trade history

This demo is provided to showcase the capabilities of the full Vantrex Trading Suite.
For the complete version with real trading capabilities, advanced indicators, and more,
please visit our website.

Thank you for trying the Vantrex Demo!
""")
        
        print(f"\nBuild completed successfully!")
        print(f"Demo executable location: {destination}")
        
    except Exception as e:
        print(f"Error moving executable: {str(e)}")

if __name__ == "__main__":
    try:
        build_demo_executable()
    except Exception as e:
        print(f"Build failed: {str(e)}")