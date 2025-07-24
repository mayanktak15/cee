#!/usr/bin/env python3
"""
Setup and run script for Loan Approval RAG Chatbot
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    print("📦 Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Installation completed successfully!")
        return True
    except subprocess.CalledProcessError:
        print("❌ Installation failed. Please check your Python environment.")
        return False

def check_dataset_files():
    """Check if dataset files exist"""
    train_file = "Training Dataset.csv"
    test_file = "Test Dataset.csv"
    
    if not os.path.exists(train_file):
        print(f"❌ Missing file: {train_file}")
        return False
    
    if not os.path.exists(test_file):
        print(f"❌ Missing file: {test_file}")
        return False
    
    print("✅ Dataset files found!")
    return True

def main():
    """Main setup function"""
    print("🏦 Loan Approval RAG Chatbot Setup")
    print("=" * 40)
    
    # Check dataset files
    if not check_dataset_files():
        print("\\nPlease ensure the following files are in the current directory:")
        print("- Training Dataset.csv")
        print("- Test Dataset.csv")
        return
    
    # Install requirements
    if not install_requirements():
        return
    
    print("\\n🚀 Choose how to run the chatbot:")
    print("1. Web Interface (Recommended)")
    print("2. Demo Mode")
    print("3. Command Line Interface")
    
    while True:
        try:
            choice = input("\\nEnter your choice (1-3): ").strip()
            
            if choice == "1":
                print("\\n🌐 Starting web interface...")
                subprocess.run([sys.executable, "-m", "streamlit", "run", "streamlit_app.py"])
                break
            elif choice == "2":
                print("\\n🎭 Starting demo mode...")
                subprocess.run([sys.executable, "demo.py"])
                break
            elif choice == "3":
                print("\\n💻 Starting command line interface...")
                subprocess.run([sys.executable, "rag_chatbot.py"])
                break
            else:
                print("Invalid choice. Please enter 1, 2, or 3.")
        except KeyboardInterrupt:
            print("\\n👋 Setup cancelled.")
            break

if __name__ == "__main__":
    main()
