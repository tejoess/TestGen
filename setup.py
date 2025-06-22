#!/usr/bin/env python3
"""
Setup script for TestGen application
"""

import subprocess
import sys
import os

def run_command(command):
    """Run a command and return the result"""
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        return False, e.stderr

def main():
    print("Setting up TestGen application...")
    
    # Check if Python is installed
    print("1. Checking Python installation...")
    success, output = run_command("python --version")
    if not success:
        print("Error: Python is not installed or not in PATH")
        return
    print(f"Python version: {output.strip()}")
    
    # Create virtual environment
    print("\n2. Creating virtual environment...")
    if os.path.exists("venv"):
        print("Virtual environment already exists. Skipping creation.")
    else:
        success, output = run_command("python -m venv venv")
        if not success:
            print(f"Error creating virtual environment: {output}")
            return
        print("Virtual environment created successfully.")
    
    # Activate virtual environment and install packages
    print("\n3. Installing required packages...")
    
    # Determine the activation command based on OS
    if sys.platform == "win32":
        activate_cmd = "venv\\Scripts\\activate"
        pip_cmd = "venv\\Scripts\\pip"
    else:
        activate_cmd = "source venv/bin/activate"
        pip_cmd = "venv/bin/pip"
    
    # Install requirements
    success, output = run_command(f"{pip_cmd} install -r requirements.txt")
    if not success:
        print(f"Error installing packages: {output}")
        return
    print("Packages installed successfully.")
    
    # Create .env file if it doesn't exist
    print("\n4. Setting up environment file...")
    if not os.path.exists(".env"):
        with open(".env", "w") as f:
            f.write("# Add your Google API key here\n")
            f.write("GOOGLE_API_KEY=your_api_key_here\n")
        print("Created .env file. Please add your Google API key.")
    else:
        print(".env file already exists.")
    
    print("\n5. Creating uploads directory...")
    os.makedirs("uploads", exist_ok=True)
    print("Uploads directory ready.")
    
    print("\nSetup completed successfully!")
    print("\nNext steps:")
    print("1. Add your Google API key to the .env file")
    print("2. Activate the virtual environment:")
    if sys.platform == "win32":
        print("   venv\\Scripts\\activate")
    else:
        print("   source venv/bin/activate")
    print("3. Run the application:")
    print("   python main.py")
    print("4. Open http://localhost:5000 in your browser")

if __name__ == "__main__":
    main() 