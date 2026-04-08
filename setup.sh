#!/bin/bash
set -e

echo "[1/4] Updating system..."
sudo apt-get update -qq || true

echo "[2/4] Installing system packages..."
sudo apt-get install -y python3-pip python3-venv python3-dev gcc

echo "[3/4] Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

echo "[4/4] Installing Python packages..."
pip install --quiet --upgrade pip setuptools wheel
pip install --quiet numpy==2.1.3
pip install --quiet pandas==2.2.3
pip install --quiet scikit-learn==1.5.2
pip install --quiet torch==2.5.1+cpu --index-url https://download.pytorch.org/whl/cpu
pip install --quiet shap==0.46.0
pip install --quiet flask==3.1.0
pip install --quiet flask-socketio==5.3.6
pip install --quiet matplotlib==3.9.2
pip install --quiet scapy==2.6.1

echo ""
echo "Setup complete. Activate with:"
echo "  cd ~/netflowai && source venv/bin/activate"
