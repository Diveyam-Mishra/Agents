import os
from phi.knowledge.text import TextKnowledgeBase
from phi.model.azure.openai_chat import AzureOpenAIChat 
from phi.agent import Agent
from phi.vectordb.pgvector import PgVector
from phi.embedder.azure_openai import AzureOpenAIEmbedder
from phi.document.chunking.agentic import AgenticChunking
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
                               azure_deployment= embedder_deployment,)
knowledge_base = TextKnowledgeBase(
    #path=r"C:\Users\mishr\OneDrive\Desktop\Python\cp",
    path=r"C:\Users\mishr\OneDrive\Desktop\Python\cloned-backend\backend-1",
    formats=[".txt",'.py','.js','.html','.css','.json','.xml','.yaml','.yml','.md'],
    exclude=["node_modules",".git",'.venv','__pycache__','myenv','.env','div.txt','.pdf','.ini','.sh'],
    vector_db=PgVector(
        table_name="Trabii_plain",
        embedder=embedder,
        db_url="postgresql+psycopg://ai:ai@localhost:5532/ai",
    ),
    chunking_strategy=AgenticChunking(
        model=azure_model,
        max_chunk_size=4000,
    ),
)
# knowledge_base.load(recreate=True)