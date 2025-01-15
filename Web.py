from phi.model.azure.openai_chat import AzureOpenAIChat 
import os
from phi.agent import Agent
from phi.knowledge.website import WebsiteKnowledgeBase
from phi.vectordb.pgvector import PgVector, SearchType
from dotenv import load_dotenv
from phi.embedder.azure_openai import AzureOpenAIEmbedder

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
                               azure_deployment= embedder_deployment,)
    
knowledge_base = WebsiteKnowledgeBase(
    urls=["https://docs.phidata.com/vectordb/introduction","https://docs.phidata.com/storage/introduction"],
    max_links=20,
    vector_db=PgVector(
        table_name="website_documents",
        db_url="postgresql+psycopg://ai:ai@localhost:5532/ai",
        embedder=embedder,
        search_type=SearchType.hybrid
    ),
)
agent = Agent(
    model=azure_model,
    knowledge=knowledge_base,
    read_chat_history=True,
    show_tool_calls=True,
    markdown=True
)
agent.print_response("Hi", stream=True)