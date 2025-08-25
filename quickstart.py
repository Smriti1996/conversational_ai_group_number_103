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
        dirs = ["financial_data", "processed_data", "models", "indices", "evaluation_results"]
        for dir_name in dirs:
            Path(dir_name).mkdir(exist_ok=True)
        print("✅ Directories created")
    
    def check_data_files(self):
        """Check if Apple reports are present"""
        print("\n📄 Checking for Apple reports...")
        report_2023 = self.data_dir / "aapl-20230930.pdf"
        report_2022 = self.data_dir / "aapl-20220924.pdf"
        
        if not report_2023.exists() or not report_2022.exists():
            print(f"⚠️  Apple reports not found! Please place them in '{self.data_dir}'.")
            return False
        
        print("✅ Both Apple reports found")
        return True
    
    def check_huggingface_token(self):
        """Check if Hugging Face token is configured."""
        print("\n🔑 Checking Hugging Face authentication...")
        
        token = os.getenv('HUGGINGFACE_HUB_TOKEN')
        if not token:
            env_file = self.root_dir / ".env"
            if env_file.exists():
                try:
                    with open(env_file, 'r') as f:
                        content = f.read()
                        if "HUGGINGFACE_HUB_TOKEN=" in content:
                            token = content.split("HUGGINGFACE_HUB_TOKEN=")[1].split('\n')[0].strip()
                except Exception as e:
                    print(f"⚠️  Error reading .env file: {e}")

        if token and len(token) > 10:
            print("✅ Hugging Face token found. You can use gated models like Llama-2.")
            return True
        
        print("ℹ️  Hugging Face token not found.")
        print("   The default model (distilgpt2) does not require a token.")
        print("   To use gated models like Llama-2, you will need to set one up by creating a .env file.")
        return True
    
    def install_dependencies(self):
        """Install required packages"""
        print("\n📦 Installing dependencies...")
        req_path = self.root_dir / "requirements.txt"
        if not req_path.exists():
            print(f"❌ requirements.txt not found.")
            return False
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "-r", str(req_path)], check=True)
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install dependencies: {e}")
            return False
        print("✅ Dependencies installed")
        return True
    
    def run_pipeline(self):
        """Run the main pipeline"""
        print("\n🚀 Running Apple Q&A Pipeline...")
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
        except Exception as e:
            print(f"❌ An error occurred during the pipeline: {e}")
            return False
    
    def launch_interface(self):
        """Launch Streamlit interface"""
        print("\n🌐 Launching web interface...")
        try:
            subprocess.run(["streamlit", "run", "interface.py"])
        except Exception as e:
            print(f"❌ Failed to launch interface: {e}")
    
    def run(self):
        """Main execution"""
        print("="*60 + "\n🚀 Apple Financial Q&A System - Quick Start\n" + "="*60)
        if not self.check_python_version(): return False
        self.setup_directories()
        if not self.check_data_files(): return False
        if not self.check_huggingface_token(): return False
        
        try:
            import streamlit
        except ImportError:
            if not self.install_dependencies(): return False
        
        if not self.run_pipeline(): return False
        
        response = input("\nWould you like to launch the web interface? (y/n): ")
        if response.lower() == 'y':
            self.launch_interface()
        
        return True

def main():
    """Entry point"""
    quickstart = QuickStart()
    try:
        if quickstart.run():
            print("\n🎉 Setup complete!")
        else:
            print("\n⚠️  Setup incomplete.")
    except KeyboardInterrupt:
        print("\n\n⏹️  Setup interrupted by user")

if __name__ == "__main__":
    main()