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
    
    def check_huggingface_token(self):
        """Check if Hugging Face token is configured"""
        print("\n🔑 Checking Hugging Face authentication...")
        
        # Check for .env file
        env_file = self.root_dir / ".env"
        if env_file.exists():
            try:
                with open(env_file, 'r') as f:
                    content = f.read()
                    if "HUGGINGFACE_HUB_TOKEN=" in content:
                        token_line = content.split("HUGGINGFACE_HUB_TOKEN=")[1].split('\n')[0].strip()
                        if len(token_line) > 10:  # Valid token should be longer
                            print("✅ Hugging Face token found in .env file")
                            return True
            except Exception as e:
                print(f"⚠️  Error reading .env file: {e}")
        
        # Check environment variable
        token = os.getenv('HUGGINGFACE_HUB_TOKEN')
        if token and len(token) > 10:
            print("✅ Hugging Face token found in environment")
            return True
        
        print("⚠️  Hugging Face token not found!")
        print("\nThe RAG system uses Llama-2 which requires authentication.")
        print("\n📋 To set up authentication:")
        print("1. Get your token from: https://huggingface.co/settings/tokens")
        print("2. Make sure you have access to: https://huggingface.co/meta-llama/Llama-2-7b-chat-hf")
        print("3. Create a .env file in this directory with:")
        print("   HUGGINGFACE_HUB_TOKEN=your_token_here")
        print("4. Or set environment variable:")
        print("   export HUGGINGFACE_HUB_TOKEN=your_token")
        
        print("\n💡 Alternative: The system will work with reduced functionality using the fine-tuned model only.")
        response = input("\nContinue anyway? (y/n): ")
        return response.lower() == 'y'
    
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
            
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "-r", str(req_path)],
                          check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install dependencies: {e}")
            return False
        
        # Download NLTK data
        print("📚 Downloading NLTK data...")
        try:
            subprocess.run([sys.executable, "-c", 
                           "import nltk; nltk.download('punkt', quiet=True); nltk.download('stopwords', quiet=True)"],
                          capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError:
            print("⚠️  NLTK data download failed, but continuing...")
        
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
            import traceback
            traceback.print_exc()
            return False
    
    def launch_interface(self):
        """Launch Streamlit interface"""
        print("\n🌐 Launching web interface...")
        print("The browser should open automatically")
        print("If not, navigate to: http://localhost:8501")
        print("\nPress Ctrl+C to stop the server")
        
        try:
            subprocess.run(["streamlit", "run", "interface.py"])
        except KeyboardInterrupt:
            print("\n⏹️  Interface stopped by user")
        except FileNotFoundError:
            print("❌ Streamlit not found. Please install with: pip install streamlit")
        except Exception as e:
            print(f"❌ Failed to launch interface: {e}")
    
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
        
        if not self.check_huggingface_token():
            print("\n⏸️  Setup stopped due to authentication issues")
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