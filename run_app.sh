# Set the name of your Python script
PYTHON_SCRIPT="app.py"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Virtual environment not found. Please run the install script first."
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

# Run the FastAPI app with nohup on port 7000
nohup uvicorn $PYTHON_SCRIPT:app --host 0.0.0.0 --port 7000 &

# Save the PID to a file
echo $! > fastapi_app.pid

echo "FastAPI app is running on port 7000. PID: $(cat fastapi_app.pid)"