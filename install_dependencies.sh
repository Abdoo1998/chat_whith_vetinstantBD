python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install requirements
pip install fastapi uvicorn sqlalchemy pymysql python-dotenv langchain langchain_community langchain_openai

echo "Virtual environment created and dependencies installed."
echo "To activate the virtual environment, run: source venv/bin/activate"