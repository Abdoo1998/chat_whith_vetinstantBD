import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, inspect
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI
from langchain_community.agent_toolkits import create_sql_agent
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import traceback
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from urllib.parse import urlparse, parse_qs
from typing import Optional

# Load environment variables
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")

app = FastAPI()

# Global variables to store the agents
sql_agent = None
formatting_chain = None

class JDBCUrl(BaseModel):
    jdbc_url: str

class Query(BaseModel):
    question: str

def parse_jdbc_url(jdbc_url: str) -> str:
    # Parse the JDBC URL
    parsed = urlparse(jdbc_url.replace('jdbc:', ''))
    query_params = parse_qs(parsed.query)
    
    # Extract database name from path
    database = parsed.path.strip('/')
    
    # Construct SQLAlchemy URL
    sqlalchemy_url = f"mysql+pymysql://{parsed.hostname}:{parsed.port}/{database}"
    
    # Add query parameters
    params = []
    if 'useSSL' in query_params:
        params.append(f"ssl={'true' if query_params['useSSL'][0].lower() == 'true' else 'false'}")
    if 'serverTimezone' in query_params:
        params.append(f"time_zone={query_params['serverTimezone'][0]}")
    
    if params:
        sqlalchemy_url += '?' + '&'.join(params)
    
    return sqlalchemy_url

def diagnose_and_connect_database(jdbc_url: str):
    print(f"Diagnosing database: {jdbc_url}")
    sqlalchemy_url = parse_jdbc_url(jdbc_url)
    engine = create_engine(sqlalchemy_url)
    inspector = inspect(engine)
    try:
        tables = inspector.get_table_names()
        print("Tables found in the database:")
        for table in tables:
            print(f"- {table}")
        return engine, tables
    except Exception as e:
        print(f"Error inspecting database: {e}")
        return None, []

def create_agents(jdbc_url: str) -> bool:
    global sql_agent, formatting_chain
    
    engine, available_tables = diagnose_and_connect_database(jdbc_url)
    if not engine:
        print("Failed to connect to the database.")
        return False

    db = SQLDatabase(engine, include_tables=available_tables)
    llm = ChatOpenAI(api_key=openai_key, model="gpt-4", temperature=0)

    # SQL Agent
    sql_agent = create_sql_agent(
        llm=llm,
        db=db,
        agent_type="openai-tools",
        verbose=True
    )

    # Response Formatting Agent
    formatting_prompt = PromptTemplate.from_template(
        """SQL Agent Response: {agent_response}

        Please provide a detailed answer that:
        1. Directly addresses the user's question.
        2. Summarizes the SQL result naturally.
        3. Provides context and explanations where necessary.
        4. Suggests additional queries if more information is needed.

        Formatted Answer:"""
    )
    formatting_chain = formatting_prompt | llm | StrOutputParser()

    return True

def process_query(query: str) -> str:
    try:
        # Get result from SQL agent
        sql_result = sql_agent.invoke({"input": query})
        
        # Format the response
        formatted_response = formatting_chain.invoke({"agent_response": sql_result['output']})
        
        return formatted_response

    except Exception as e:
        error_message = f"An error occurred while processing the query: {e}\n{traceback.format_exc()}"
        return error_message

@app.post("/query")
async def query_endpoint(query: Query):
    if not sql_agent or not formatting_chain:
        raise HTTPException(status_code=500, detail="Agents not initialized. Please use /initialize endpoint first.")

    response = process_query(query.question)
    return {"answer": response}

@app.post("/initialize")
async def initialize_endpoint(jdbc_url: JDBCUrl):
    success = create_agents(jdbc_url.jdbc_url)
    if success:
        return {"message": "Agents initialized successfully"}
    else:
        raise HTTPException(status_code=500, detail="Failed to initialize agents")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7000)