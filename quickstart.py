#!/usr/bin/env python
"""
quickstart.py
=============
One-click setup and run script for Apple Financial Q&A System
Handles all setup steps automatically
"""

import os
import sys
import subprocess
from pathlib import Path
import time

class QuickStart:
    """Automated setup and execution"""
    
    def __init__(self):
        self.root_dir = Path.cwd()
        self.data_dir = self.root_dir / "financial_data"
        
    def check_python_version(self):
        """Check Python version"""
        print("📌 Checking Python version...")
        if sys.version_info < (3, 8):
            print("❌ Python 3.8+ required")
            return False
        print(f"✅ Python {sys.version.split()[0]} detected")
        return True
    
    def setup_directories(self):
        """Create necessary directories"""
        print("\n📁 Creating directories...")
        dirs = [
            "financial_data",
            "processed_data",
            "models",
            "indices",
            "evaluation_results"
        ]
        for dir_name in dirs:
            Path(dir_name).mkdir(exist_ok=True)
        print("✅ Directories created")
    
    def check_data_files(self):
        """Check if Apple reports are present"""
        print("\n📄 Checking for Apple reports...")
        report_2023 = self.data_dir / "aapl-20230930.pdf"
        report_2022 = self.data_dir / "aapl-20220924.pdf"
        
        if not report_2023.exists() or not report_2022.exists():
            print("⚠️  Apple reports not found!")
            print(f"Please place the following files in '{self.data_dir}':")
            print("  - aapl-20230930.pdf")
            print("  - aapl-20220924.pdf")
            return False
        
        print("✅ Both Apple reports found")
        return True
    
    def install_dependencies(self):
        """Install required packages"""
        print("\n📦 Installing dependencies...")
        print("This may take 5-10 minutes...")
        
        # Upgrade pip
        subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], 
                      capture_output=True, text=True)
        
        # Install requirements
        req_path = self.root_dir / "requirements.txt"
        if not req_path.exists():
            print(f"❌ requirements.txt not found in {self.root_dir}")
            return False
            
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", str(req_path)],
                      check=True)
        
        # Download NLTK data
        print("📚 Downloading NLTK data...")
        subprocess.run([sys.executable, "-c", 
                       "import nltk; nltk.download('punkt', quiet=True); nltk.download('stopwords', quiet=True)"],
                      capture_output=True, text=True)
        
        print("✅ Dependencies installed")
        return True
    
    def run_pipeline(self):
        """Run the main pipeline"""
        print("\n🚀 Running Apple Q&A Pipeline...")
        print("="*60)
        
        # Import and run main
        try:
            from main import AppleQAOrchestrator
            
            orchestrator = AppleQAOrchestrator()
            success = orchestrator.run_complete_pipeline()
            
            if success:
                print("\n✅ Pipeline completed successfully!")
                return True
            else:
                print("\n❌ Pipeline failed")
                return False
                
        except ImportError as e:
            print(f"❌ Import error: {e}")
            print("Please ensure all Python scripts are in the current directory")
            return False
        except Exception as e:
            print(f"❌ An error occurred during the pipeline: {e}")
            return False
    
    def launch_interface(self):
        """Launch Streamlit interface"""
        print("\n🌐 Launching web interface...")
        print("The browser should open automatically")
        print("If not, navigate to: http://localhost:8501")
        print("\nPress Ctrl+C to stop the server")
        
        subprocess.run(["streamlit", "run", "interface.py"])
    
    def run(self):
        """Main execution"""
        print("="*60)
        print("🚀 Apple Financial Q&A System - Quick Start")
        print("="*60)
        
        if not self.check_python_version():
            return False
        
        self.setup_directories()
        
        if not self.check_data_files():
            print("\n⏸️  Please add the Apple reports and run again")
            return False
        
        try:
            import torch
            import transformers
            import streamlit
            print("\n✅ Dependencies appear to be installed.")
        except ImportError:
            print("\n📦 Dependencies not found. Installing...")
            if not self.install_dependencies():
                return False
        
        if not self.run_pipeline():
            return False
        
        print("\n" + "="*60)
        response = input("Would you like to launch the web interface? (y/n): ")
        if response.lower() == 'y':
            self.launch_interface()
        else:
            print("\nTo launch the interface later, run:")
            print("  streamlit run interface.py")
        
        return True

def main():
    """Entry point"""
    quickstart = QuickStart()
    
    try:
        success = quickstart.run()
        if success:
            print("\n🎉 Setup complete! Your Apple Q&A system is ready.")
        else:
            print("\n⚠️  Setup incomplete. Please check the errors above.")
    except KeyboardInterrupt:
        print("\n\n⏹️  Setup interrupted by user")
    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()