import os
from phi.knowledge.text import TextKnowledgeBase
from phi.model.azure.openai_chat import AzureOpenAIChat 
from phi.agent import Agent
from phi.vectordb.pgvector import PgVector
from phi.embedder.azure_openai import AzureOpenAIEmbedder
from phi.tools.crawl4ai_tools import Crawl4aiTools
from phi.tools.duckduckgo import DuckDuckGo
from phi.tools.googlesearch import GoogleSearch
from dotenv import load_dotenv
load_dotenv() 
api_key = os.environ.get("AZURE_OPENAI_API_KEY")
endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
deployment = os.environ.get("AZURE_DEPLOYMENT")
model_name = os.environ.get("AZURE_OPENAI_MODEL_NAME")
embedder_endpoint = os.environ.get("AZURE_EMBEDDER_ENDPOINT")
embedder_deployment = os.environ.get("AZURE_EMBEDDER_DEPLOYMENT")

azure_model = AzureOpenAIChat(
    id="gpt-4o",
    api_key=api_key,
    azure_endpoint=endpoint,
    azure_deployment=deployment,
)

db_url = "postgresql+psycopg://ai:ai@localhost:5532/ai"

embedder = AzureOpenAIEmbedder(api_key=api_key,
                               azure_endpoint =embedder_endpoint,
                               azure_deployment= embedder_deployment,
                               dimensions=3072)

agent = Agent(
    model=azure_model,
    instructions=[
        """You are given a Query Look it up on Web and Search at least 3 sources to get the answer""",
    ],
    tools=[DuckDuckGo(), Crawl4aiTools(), GoogleSearch()],
    read_chat_history=True,
    show_tool_calls=True,
    markdown=True
)
agent.cli_app()