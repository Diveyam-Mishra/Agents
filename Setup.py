from phi.knowledge.text import TextKnowledgeBase
from phi.vectordb.pgvector import PgVector
from phi.document.chunking.agentic import AgenticChunking
from Connection.LLM import azure_model,embedder
knowledge_base = TextKnowledgeBase(
    path="Jk_db_column_meaning.txt",
    vector_db=PgVector(
        table_name="SupplyMint",
        embedder=embedder,
        db_url="postgresql+psycopg://ai:ai@localhost:5532/ai",
    ),
    chunking_strategy=AgenticChunking(
        model=azure_model,
        max_chunk_size=7000,
    ),
)
# knowledge_base.load(recreate=True)