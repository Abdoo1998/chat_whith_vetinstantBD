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
import json
from urllib.parse import quote_plus  # Add this import

# Load environment variables
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")

def diagnose_and_connect_database():
    print("Connecting to MySQL database...")
    password = quote_plus("Vetinstant@9588#!")
    db_url = f"mysql+mysqlconnector://root:{password}@localhost/vetinstant"
    engine = create_engine(db_url)
    inspector = inspect(engine)
    
    try:
        # Get all tables
        tables = inspector.get_table_names()
        print("\nDatabase Schema Information:")
        
        # Store schema information
        schema_info = []
        
        for table in tables:
            print(f"\nTable: {table}")
            
            # Get columns
            columns = inspector.get_columns(table)
            print("Columns:")
            for column in columns:
                print(f"  - {column['name']} ({column['type']})")
            
            # Get primary keys
            pk = inspector.get_pk_constraint(table)
            if pk['constrained_columns']:
                print(f"Primary Key: {pk['constrained_columns']}")
            
            # Get foreign keys
            fks = inspector.get_foreign_keys(table)
            if fks:
                print("Foreign Keys:")
                for fk in fks:
                    print(f"  - {fk['constrained_columns']} -> {fk['referred_table']}.{fk['referred_columns']}")
            
            # Store schema information for the table
            table_info = {
                'name': table,
                'columns': [{'name': col['name'], 'type': str(col['type'])} for col in columns],
                'primary_key': pk['constrained_columns'] if pk else [],
                'foreign_keys': fks
            }
            schema_info.append(table_info)
            
        return engine, schema_info
    except Exception as e:
        print(f"Error inspecting database: {e}")
        traceback.print_exc()
        return None, []

# Function to create agents
def create_agents():
    engine, schema_info = diagnose_and_connect_database()
    if not engine:
        print("Failed to connect to the database. Exiting.")
        return None, None, None

    # Create schema description for the prompt
    schema_description = "Database Schema:\n"
    for table in schema_info:
        schema_description += f"\nTable: {table['name']}\n"
        schema_description += "Columns:\n"
        for col in table['columns']:
            schema_description += f"  - {col['name']} ({col['type']})\n"
        if table['primary_key']:
            schema_description += f"Primary Key: {', '.join(table['primary_key'])}\n"
        if table['foreign_keys']:
            schema_description += "Foreign Keys:\n"
            for fk in table['foreign_keys']:
                schema_description += f"  - {fk['constrained_columns']} -> {fk['referred_table']}.{fk['referred_columns']}\n"

    db = SQLDatabase(engine, include_tables=[t['name'] for t in schema_info])
    llm = ChatOpenAI(api_key=openai_key, model="gpt-4")

    # Update SQL Agent with schema information
    sql_agent = create_sql_agent(
        llm=llm,
        db=db,
        agent_type="openai-tools",
        verbose=True,
        prefix=f"""You are an expert SQL agent for a veterinary database. 
        You have access to the following database schema:
        
        {schema_description}
        
        When forming SQL queries:
        1. Consider table relationships through foreign keys
        2. Use appropriate JOIN operations when querying related tables
        3. Always consider data types when comparing values
        4. Use appropriate WHERE clauses to filter data
        5. Consider using appropriate indexes for better performance
        """
    )

    # Update formatting prompt with schema awareness
    formatting_prompt = PromptTemplate.from_template(
        """As a specialized Veterinary Database Assistant with full knowledge of the database schema:

        {schema_description}

        Your primary goals are to:
        1. Understand and accurately interpret veterinary queries using the complete schema knowledge
        2. Utilize table relationships and foreign keys effectively
        3. Analyze the data in the context of veterinary practice and animal health
        4. Provide insights that are valuable for veterinary decision-making and patient care
        5. Present information in a clear, professional manner suited for veterinary staff

        Consider:
        - Relationships between different tables in the database
        - Data consistency across related tables
        - Species-specific health concerns
        - Age-related health issues in animals
        - Seasonal patterns in animal health and diseases
        - Vaccination and preventive care schedules
        - Trends in treatment efficacy
        - Owner compliance patterns
        - Clinic operational insights

        Current query:
        Human: {input}
        
        SQL Agent Response: {agent_response}
        
        Formatted Answer:"""
    )
    
    formatting_chain = formatting_prompt | llm | StrOutputParser()

    # Rest of the code remains the same
    tools = [
        Tool(
            name="SQL Database",
            func=sql_agent.run,
            description="Useful for querying the veterinary database with full schema awareness"
        ),
        Tool(
            name="Response Formatter",
            func=lambda x: formatting_chain.invoke({**x, 'schema_description': schema_description}),
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

# Process the user's query through the agent
def process_query(agent, query):
    try:
        result = agent.run({"input": query})
        return result
    except Exception as e:
        error_message = f"""
        An error occurred while processing your query:
        Error type: {type(e).__name__}
        Error details: {str(e)}
        
        This might be due to:
        1. Invalid table or column references
        2. Syntax errors in the generated SQL
        3. Database connection issues
        4. Data type mismatches
        
        Please try rephrasing your question or contact support if the issue persists.
        """
        print(error_message)
        traceback.print_exc()
        return error_message

# Main script to handle user queries
if __name__ == "__main__":
    sql_agent, formatting_chain, main_agent = create_agents()

    if main_agent:
        while True:
            user_query = input("Enter your query (or 'exit' to quit): ")
            if user_query.lower() == 'exit':
                break

            # Run SQL agent to get the response from the database
            sql_agent_result = process_query(sql_agent, user_query)

            # Format the response using the response formatting agent
            formatted_response = formatting_chain.invoke({
                "input": user_query,
                "agent_response": sql_agent_result
            })

            print("Formatted Response:", formatted_response)
    else:
        print("Failed to initialize agents. Please check your database connection and try again.")
