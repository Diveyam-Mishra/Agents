from phi.agent import Agent
from phi.tools.arxiv_toolkit import ArxivToolkit
import os
from dotenv import load_dotenv
from phi.agent import Agent
from phi.tools.crawl4ai_tools import Crawl4aiTools
from phi.model.azure.openai_chat import AzureOpenAIChat
from phi.knowledge.arxiv import ArxivKnowledgeBase
from phi.vectordb.pgvector import PgVector, SearchType
from phi.document.chunking.agentic import AgenticChunking
from phi.embedder.azure_openai import AzureOpenAIEmbedder
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
knowledge_base = ArxivKnowledgeBase(
    queries=["LArge Concept Model", "Byte Latent Transformer"],
    num_documents=7,
    vector_db=PgVector(
        table_name="arxiv_documents",
        embedder=embedder,
        db_url="postgresql+psycopg://ai:ai@localhost:5532/ai",
    ),
    chunking_strategy=AgenticChunking(
        model=azure_model,
        max_chunk_size=4000,
    ),
)


Research_agent = Agent(
    model=azure_model,
    knowledge=knowledge_base,
    role="Researches a topic on the internet.",
    search_knowledge=True,
    tools=[ArxivToolkit()],
    show_tool_calls=True)

from phi.tools.googlesearch import GoogleSearch

web_searcher = Agent(
    model=azure_model,
    tools=[GoogleSearch()],
    description="You are a Linkedin Content Creator Reading papers For the given topic.",
    instructions=[
        "Given a topic by the user, respond with 4 articles from Medium.",
    ],
    show_tool_calls=True,
    debug_mode=True,
)


article_reader = Agent(
    model=azure_model,
    name="Article Reader",
    role="Reads articles from URLs.",
    tools=[Crawl4aiTools(max_length=None)],
)

hn_team = Agent(
    model=azure_model,
    name="Linkdein Famous",
    team=[Research_agent, web_searcher, article_reader],
    instructions=[
        "First, search Arxiv for what the user is asking about.",
        "Then, ask the article reader to read the links for the stories to get more information.",
        "Important: you must provide the article reader with the links to read.",
        "Then, ask the web searcher to search for each Complex topic to get more information.",
        "Finally, provide a thoughtful and engaging summary.",
    ],
    show_tool_calls=True,
    markdown=True,
)
hn_team.print_response("Write an article Large Conept Model and Why is it better than Large LAnguage Model a paper by Meta", stream=True)
