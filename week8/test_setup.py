#!/usr/bin/env python3
"""
Quick test script to verify the RAG chatbot setup
"""

import os
import sys

def check_files():
    """Check if all required files exist"""
    required_files = [
        "Training Dataset.csv",
        "Test Dataset.csv", 
        "document_retriever.py",
        "response_generator.py",
        "rag_chatbot.py",
        "requirements.txt"
    ]
    
    print("📋 Checking required files...")
    missing_files = []
    
    for file in required_files:
        if os.path.exists(file):
            print(f"✅ {file}")
        else:
            print(f"❌ {file} - MISSING")
            missing_files.append(file)
    
    if missing_files:
        print(f"\n⚠️  Missing files: {missing_files}")
        return False
    
    print("\n✅ All required files present!")
    return True

def check_datasets():
    """Quick dataset validation"""
    try:
        import pandas as pd
        
        print("\n📊 Checking datasets...")
        
        train_df = pd.read_csv("Training Dataset.csv")
        test_df = pd.read_csv("Test Dataset.csv")
        
        print(f"✅ Training dataset: {len(train_df)} rows")
        print(f"✅ Test dataset: {len(test_df)} rows")
        print(f"✅ Columns: {list(train_df.columns)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Dataset check failed: {e}")
        return False

def test_imports():
    """Test if core modules can be imported"""
    print("\n🔍 Testing imports...")
    
    modules_to_test = [
        ("pandas", "pd"),
        ("numpy", "np")
    ]
    
    success_count = 0
    
    for module_name, alias in modules_to_test:
        try:
            if alias:
                exec(f"import {module_name} as {alias}")
            else:
                exec(f"import {module_name}")
            print(f"✅ {module_name}")
            success_count += 1
        except ImportError:
            print(f"❌ {module_name} - Need to install")
    
    # Test optional heavy dependencies
    optional_modules = [
        "sentence_transformers",
        "faiss",
        "transformers",
        "streamlit"
    ]
    
    print("\n📦 Checking optional dependencies...")
    for module in optional_modules:
        try:
            exec(f"import {module}")
            print(f"✅ {module}")
        except ImportError:
            print(f"⚠️  {module} - Will be installed during setup")
    
    return success_count == len(modules_to_test)

def main():
    """Main test function"""
    print("🏦 Loan Approval RAG Chatbot - Quick Test")
    print("=" * 50)
    
    # Run checks
    checks = [
        ("File Check", check_files),
        ("Dataset Check", check_datasets), 
        ("Import Check", test_imports)
    ]
    
    all_passed = True
    
    for check_name, check_func in checks:
        try:
            result = check_func()
            if not result:
                all_passed = False
        except Exception as e:
            print(f"❌ {check_name} failed: {e}")
            all_passed = False
    
    print("\n" + "=" * 50)
    
    if all_passed:
        print("🎉 All basic checks passed!")
        print("\n📋 Next steps:")
        print("1. Run: pip install -r requirements.txt")
        print("2. Test: python demo.py")
        print("3. Launch: streamlit run streamlit_app.py")
        print("4. Or use: python rag_chatbot.py")
        
    else:
        print("⚠️  Some checks failed. Please:")
        print("1. Ensure all files are present")
        print("2. Install dependencies: pip install -r requirements.txt")
        print("3. Check dataset files")

if __name__ == "__main__":
    main()
