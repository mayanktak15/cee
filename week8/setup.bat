@echo off
echo 🏦 Loan Approval RAG Chatbot Setup
echo ================================

echo.
echo 📦 Installing required packages...
pip install -r requirements.txt

if %ERRORLEVEL% NEQ 0 (
    echo ❌ Installation failed. Please check your Python environment.
    pause
    exit /b 1
)

echo.
echo ✅ Installation completed successfully!
echo.

echo 🚀 Choose how to run the chatbot:
echo 1. Web Interface (Recommended)
echo 2. Demo Mode
echo 3. Command Line Interface
echo.

set /p choice="Enter your choice (1-3): "

if "%choice%"=="1" (
    echo.
    echo 🌐 Starting web interface...
    streamlit run streamlit_app.py
) else if "%choice%"=="2" (
    echo.
    echo 🎭 Starting demo mode...
    python demo.py
) else if "%choice%"=="3" (
    echo.
    echo 💻 Starting command line interface...
    python rag_chatbot.py
) else (
    echo Invalid choice. Please run the script again.
)

pause
