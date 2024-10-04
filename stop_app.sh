# Check if PID file exists
if [ ! -f fastapi_app.pid ]; then
    echo "PID file not found. Is the app running?"
    exit 1
fi

# Read PID from file
PID=$(cat fastapi_app.pid)

# Check if process is running
if ps -p $PID > /dev/null
then
    echo "Stopping FastAPI app (PID: $PID)"
    kill $PID
    rm fastapi_app.pid
    echo "FastAPI app stopped"
else
    echo "Process not found. It may have already been stopped."
    rm fastapi_app.pid
fi