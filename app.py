from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import Optional, Dict
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, inspect
from langchain_community.utilities import SQLDatabase
from langchain_openai import AzureChatOpenAI
from langchain_community.agent_toolkits import create_sql_agent
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

app = FastAPI()

# Load environment variables
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")

# Global dictionary to store initialized components
initialized_components: Dict[str, tuple] = {}

class InitRequest(BaseModel):
    db_url: str = Field(..., description="The database URL to connect to")
    table_descriptions: Optional[str] = Field(None, description="Optional description of database tables")

class QueryRequest(BaseModel):
    query: str = Field(..., description="The natural language query to process")

class InitResponse(BaseModel):
    message: str

class QueryResponse(BaseModel):
    answer: str

def diagnose_and_connect_database(db_url):
    try:
        engine = create_engine(db_url)
        inspector = inspect(engine)
        tables = inspector.get_table_names()
        return engine, tables
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error connecting to database: {str(e)}")

def create_agents(db_url, table_descriptions):
    engine, available_tables = diagnose_and_connect_database(db_url)
    db = SQLDatabase(engine, include_tables=available_tables)
    llm = AzureChatOpenAI(
        azure_deployment="gpt-4o",
        model="gpt-4o",
        api_version="2024-02-15-preview",
        azure_endpoint="https://notice-parser-openai-gpt4.openai.azure.com/",
        temperature=0.5,
        api_key="d6a6fd2468ea4f76993b8323e5a5e736",
        seed=235,
    )
    # SQL Agent
    sql_agent = create_sql_agent(
        llm=llm,
        db=db,
        agent_type="openai-tools",
        verbose=True
    )

    # Response Formatting Agent
    formatting_prompt = PromptTemplate.from_template(
       """
SQL Agent Response: {agent_response}
Table Descriptions: {table_descriptions}
Instructions:

Carefully analyze the user's question to understand the core information they're seeking.
Review the database schema and table descriptions to identify relevant tables and relationships.
(Note: An SQL query has already been constructed and executed to answer the user's question.)
Interpret the query results in a way that directly addresses the user's question.
Provide a clear, concise explanation of the findings in natural language.
If appropriate, offer insights or implications based on the data.
Use analogies or real-world examples to make complex data more relatable.
If the results are numerical, consider providing context or comparisons to make them more meaningful.
Address any potential limitations or caveats in the data or analysis.
Suggest follow-up questions or areas for further investigation if relevant.

Important:

Do not include technical details about the SQL query or database structure.
Focus on explaining the results in a way that a non-technical person can understand.
If the data reveals any trends or patterns, highlight these in your explanation.
Use a conversational tone while maintaining professionalism and accuracy.
    Answer:"""
    )
    formatting_chain = formatting_prompt | llm | StrOutputParser()

    return sql_agent, formatting_chain, table_descriptions

@app.post("/initialize", response_model=InitResponse)
async def initialize(request: InitRequest):
    try:
        sql_agent, formatting_chain, table_descriptions = create_agents(request.db_url, request.table_descriptions)
        initialized_components[request.db_url] = (sql_agent, formatting_chain, table_descriptions)
        return InitResponse(message="Initialization successful")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during initialization: {str(e)}")

def get_initialized_components(db_url: str):
    components = initialized_components.get(db_url)
    if not components:
        raise HTTPException(status_code=400, detail="Database not initialized. Please call /initialize first.")
    return components

@app.post("/process_query", response_model=QueryResponse)
async def process_query(request: QueryRequest, db_url: str):
    try:
        sql_agent, formatting_chain, table_descriptions = get_initialized_components(db_url)
        
        # Get result from SQL agent
        sql_result = sql_agent.invoke({"input": request.query})
        
        # Format the response
        formatted_response = formatting_chain.invoke({
            "agent_response": sql_result['output'],
            "table_descriptions": table_descriptions or "No table descriptions provided."
        })
        
        return QueryResponse(answer=formatted_response)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred while processing the query: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7000)
