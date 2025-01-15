import os
from phi.knowledge.text import TextKnowledgeBase
from phi.model.azure.openai_chat import AzureOpenAIChat 
from phi.agent import Agent
from phi.vectordb.pgvector import PgVector
from phi.embedder.azure_openai import AzureOpenAIEmbedder
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
    #path=r"C:\Users\mishr\OneDrive\Desktop\Python\cloned-backend\backend-1",
    path=r"C:\Users\mishr\Downloads\automations_technical_assessment\integrations_technical_assessment",
    formats=[".txt",'.py','.js','.html','.css','.json','.xml','.yaml','.yml','.md'],
    vector_db=PgVector(
        # table_name="text_documents",
        table_name="Rajneesh_vectors",
        embedder=embedder,
        db_url="postgresql+psycopg://ai:ai@localhost:5532/ai",
    ),
)
# knowledge_base.load()

agent = Agent(
    model=azure_model,
    knowledge=knowledge_base,
    instructions=[
        """You are given a Folder which contains data on how to do the current asisgnment and you have to read the instruction.txt file and kkeep updating what code changes u need to make to help me Solve the assignment?""",
    ],
    read_chat_history=True,
    show_tool_calls=True,
    markdown=True
)
while True:
    user_input = input("Enter your question: ")
    if user_input.lower() == 'exit':
        break
    agent.print_response(user_input, stream=True)
# agent.print_response("Read the instruction.txt and tell me what to do?", stream=True)