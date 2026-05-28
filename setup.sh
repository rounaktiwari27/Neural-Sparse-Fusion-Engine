#!/bin/bash

echo "⚡ Initializing Dual Stage Retriever Setup..."

# 1. Ensure data directory exists so C++ doesn't crash on boot
mkdir -p data
echo "✅ Data directory verified."

# 2. Compile the C++ Engine
echo "🔨 Compiling C++ Microservice (-O3 Optimization)..."
# Compiling from src/ directory and outputting the binary to the root directory
clang++ -std=c++17 -O3 src/main.cpp -o hybrid_engine

if [ $? -eq 0 ]; then
    echo "✅ C++ Engine compiled successfully."
else
    echo "❌ C++ Compilation failed! Please check your compiler."
    exit 1
fi

# 3. Setup Python Virtual Environment
if [ ! -d "venv" ]; then
    echo "🐍 Creating Python virtual environment..."
    python3 -m venv venv
fi

# 4. Install Dependencies
echo "📦 Installing Python dependencies (this may take a moment)..."
source venv/bin/activate
pip install -r requirements.txt
echo "✅ Dependencies installed."

echo "===================================================="
echo "🎉 Setup Complete! The engine is ready."
echo "To start the application, run:"
echo "source venv/bin/activate && streamlit run app.py"
echo "===================================================="