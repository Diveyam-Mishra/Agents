import json
from typing import Optional, Iterator
from pydantic import BaseModel, Field
from phi.agent import Agent
from phi.workflow import Workflow, RunResponse, RunEvent
from phi.storage.workflow.sqlite import SqlWorkflowStorage
from phi.tools.duckduckgo import DuckDuckGo
from phi.utils.pprint import pprint_run_response
from phi.utils.log import logger
from pydantic import BaseModel, Field
from typing import Optional, Iterator, List
from phi.tools.googlesearch import GoogleSearch
from phi.workflow import Workflow, RunResponse, RunEvent
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
# Define data models
class Article(BaseModel):
    title: str = Field(..., description="Title of the article.")
    url: str = Field(..., description="Link to the article.")
    summary: Optional[str] = Field(..., description="Summary of the article if available.")

class SearchResults(BaseModel):
    articles: List[Article]

# Define the workflow
class KnowledgeBaseWorkflow(Workflow):
    research_agent: Agent = Agent(
        model=azure_model,
        tools=[ArxivToolkit()],
        instructions=[
            "Search Arxiv for papers related to the provided topic.",
            "Return the top 7 documents with metadata.",
        ],
        response_model=SearchResults,
    )

    web_searcher: Agent = Agent(
        model=azure_model,
        tools=[GoogleSearch()],
        instructions=[
            "Search for articles related to the provided topic on Medium and other web sources.",
            "Return 4 articles that are most relevant to the topic.",
        ],
        response_model=SearchResults,
    )

    article_reader: Agent = Agent(
        model=azure_model,
        name="Article Reader",
        role="Reads and summarizes articles from URLs.",
        tools=[Crawl4aiTools(max_length=None)],
        instructions=["Read the provided links and summarize their content."],
    )

    def run(self, topic: str) -> Iterator[RunResponse]:
        logger.info(f"Starting workflow for topic: {topic}")

        # Step 1: Search Arxiv
        logger.info("Searching Arxiv for papers...")
        research_response = self.research_agent.run(topic)
        if not research_response.content or not isinstance(research_response.content, SearchResults):
            yield RunResponse(
                run_id=self.run_id,
                event=RunEvent.workflow_completed,
                content=f"Could not find any papers related to: {topic}",
            )
            

        arxiv_results = research_response.content
        logger.info(f"Found {len(arxiv_results.articles)} papers.")

        logger.info("Searching the web for related articles...")
        search_response = self.web_searcher.run(topic)
        if not search_response.content or not isinstance(search_response.content, SearchResults):
            yield RunResponse(
                run_id=self.run_id,
                event=RunEvent.workflow_completed,
                content=f"Could not find any web articles related to: {topic}",
            )
            return

        web_results = search_response.content
        logger.info(f"Found {len(web_results.articles)} web articles.")

        # Step 3: Read and Summarize Articles
        logger.info("Reading articles for detailed summaries...")
        for article in web_results.articles + arxiv_results.articles:
            reader_response = self.article_reader.run(article.url)
            if reader_response.content:
                article.summary = reader_response.content

        # Combine all results into a comprehensive summary
        summary = self.generate_summary(topic, arxiv_results, web_results)
        yield RunResponse(
            run_id=self.run_id,
            event=RunEvent.workflow_completed,
            content=summary,
        )

    def generate_summary(self, topic: str, arxiv_results: SearchResults, web_results: SearchResults) -> str:
        """Generates a thoughtful and engaging summary."""
        arxiv_summaries = "\n".join([f"- {article.title}: {article.summary or 'No summary available'}" for article in arxiv_results.articles])
        web_summaries = "\n".join([f"- {article.title}: {article.summary or 'No summary available'}" for article in web_results.articles])

        return f"""
# Summary for: {topic}

## Insights from Arxiv Papers:
{arxiv_summaries}

## Insights from Web Articles:
{web_summaries}

Key Takeaways:
- Arxiv provides detailed theoretical insights.
- Web articles provide practical applications and broader context.
"""
topic = "Large Concept Models and their Advantages over Large Language Models"

knowledge_workflow = KnowledgeBaseWorkflow(
    session_id=f"research-on-{topic}",
    vector_db=PgVector(
        table_name="arxiv_documents",
        embedder=embedder,
        db_url="postgresql+psycopg://ai:ai@localhost:5532/ai",
    ),
    chunking_strategy=AgenticChunking(
        model=azure_model,
        max_chunk_size=4000,
    )
)

# Run the workflow
for response in knowledge_workflow.run(topic):
    print(response.content)
