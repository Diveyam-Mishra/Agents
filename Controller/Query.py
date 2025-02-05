import os
import httpx
from phi.agent import Agent
from pathlib import Path
from phi.agent import Agent
from dotenv import load_dotenv
from Query.Dict import SQLTools
from phi.tools.csv_tools import CsvTools
from Schemas.User_input import User_input
from Query.Workflow import SQLQueryWorkflow
from Connection.Database import db_url
from Connection.LLM import azure_model,embedder
from Query.knowledge_base import knowledge_base
from phi.model.azure.openai_chat import AzureOpenAIChat 
from phi.workflow import Workflow, RunResponse, RunEvent
from phi.embedder.azure_openai import AzureOpenAIEmbedder
load_dotenv()


async def process_csv(file_path: Path, user_input: User_input):

    if not file_path.exists():
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    agent = Agent(
        model=azure_model,
        read_chat_history=True,
        tools=[CsvTools(csvs=[file_path])],
        markdown=True,
        show_tool_calls=True,
        instructions=[
            "First read the file",
            "Then check the columns in the file",
            "Then run the query to answer the question",
        ]
    )
    return agent.run(message=user_input,stream=False)

# async def process_DB(Client_code,user_input: User_input):
#     db_url = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    # engine = create_engine(db_url)
    # try:
    #     with engine.connect() as conn:
    #         print("Connected to PostgreSQL via SQLAlchemy!")
    # except Exception as e:
    #     print(f"Error with SQLAlchemy connection: {e}")
    # db_url = "postgresql+psycopg2://supplymint:s3cr3t_passw0rd!@smdbquality.supplymint.com:5432/supplymint"
    # Query_agent = Agent(
    #     model=azure_model,
    #     read_chat_history=True,
    #     description="This is a DB team",
    #     tools=[SQLTools(db_url=db_url, schema="PYDATA")],
    #     markdown=True,
    #     show_tool_calls=True,
    #     instructions=[
    #         "First read the DB this is a PostGresSQL DB", 
    #         "Then check the columns in the PYDATA SCHEMA where PYDATA is case senstive and then run the search command on the table name give",
    #         "Then run the query to answer the question",
    #     ]
    # )
    # Db_knowledge_agent=Agent(
    #     model=azure_model,
    #     read_chat_history=True,
    #     knowledge_base=knowledge_base,
    # )
    # hn_team = Agent(
    # model=azure_model,
    # name="SupplyMint",
    # team=[ Query_agent],
    # instructions=[
    #     "First Search the DB_knowledge_agentabout what we are looking for how the values look like then go to Query agent",
    #     "Then, ask the Query agent to search for the table name and the column name to get more information.",
    #     "Return the results", "Just remember to Search in the right Schema that is PYDATA and  The table name is jk_item"
    #     ],
    # show_tool_calls=True,
    # )
    # return Query_agent.run(message=user_input,stream=False)

async def process_DB(Client_code, user_input: str):
    workflow = SQLQueryWorkflow(db_url=db_url)
    user_question = user_input
    table_name = f'"PYDATA".{Client_code}'
    response = workflow.run(user_question, table_name)
    return response