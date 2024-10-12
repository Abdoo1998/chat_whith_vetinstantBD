import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, inspect
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI
from langchain_community.agent_toolkits import create_sql_agent
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
import traceback
from urllib.parse import quote_plus
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Load environment variables
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")

app = FastAPI()

def diagnose_and_connect_database():
    print("Connecting to MySQL database...")
    password = quote_plus("Vetinstant@9588#!")
    db_url = f"mysql+mysqlconnector://root:{password}@localhost/vetinstant"
    engine = create_engine(db_url)
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

def create_agents():
    engine, available_tables = diagnose_and_connect_database()
    if not engine:
        print("Failed to connect to the database. Exiting.")
        return None, None, None

    db = SQLDatabase(engine, include_tables=available_tables)
    llm = ChatOpenAI(api_key=openai_key, model="gpt-4")

    # SQL Agent
    sql_agent = create_sql_agent(
        llm=llm,
        db=db,
        agent_type="openai-tools",
        verbose=True
    )

    # Response Formatting Agent
    formatting_prompt = PromptTemplate.from_template(
        """As a specialized Veterinary Database Assistant, please provide a comprehensive and insightful response based on the SQL query results. Your answer should:

        Your primary goals are to:
        1. Understand and accurately interpret veterinary queries.
        2. Utilize the Veterinary SQL Database tool to extract relevant information.
        3. Analyze the data in the context of veterinary practice and animal health.
        4. Provide insights that are valuable for veterinary decision-making and patient care.
        5. Use the Veterinary Response Formatter to present information in a clear, professional manner suited for veterinary staff.

        Remember to consider factors such as:
        - Species-specific health concerns
        - Age-related health issues in animals
        - Seasonal patterns in animal health and diseases
        - Vaccination and preventive care schedules
        - Trends in treatment efficacy
        - Owner compliance patterns
        - Clinic operational insights

        Current conversation:
        Human: {input}
        AI: Certainly, I'd be happy to help with that veterinary query. Let's break this down step by step:

        SQL Agent Response: {agent_response}
        Formatted Answer:"""
    )
    formatting_chain = formatting_prompt | llm | StrOutputParser()

    # Main Agent
    tools = [
        Tool(
            name="SQL Database",
            func=sql_agent.run,
            description="Useful for querying the veterinary database"
        ),
        Tool(
            name="Response Formatter",
            func=formatting_chain.invoke,
            description="Useful for formatting the final response to the user"
        )
    ]

    main_agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )

    return sql_agent, formatting_chain, main_agent

# Create agents at startup
sql_agent, formatting_chain, main_agent = create_agents()

class Query(BaseModel):
    query: str

@app.post("/query")
async def process_query(query: Query):
    if not main_agent:
        raise HTTPException(status_code=500, detail="Agents not initialized properly")

    try:
        # Run SQL agent to get the response from the database
        sql_agent_result = main_agent.run({"input": query.query})

        # Format the response using the response formatting agent
        formatted_response = formatting_chain.invoke({
            "input": query.query,
            "agent_response": sql_agent_result
        })

        return {"response": formatted_response}
    except Exception as e:
        print(f"An error occurred: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7000)
