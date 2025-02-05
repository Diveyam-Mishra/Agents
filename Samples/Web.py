import streamlit as st
from PIL import Image
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
    markdown=True)
# )
# agent.print_response("Hi", stream=True)
if "conversation" not in st.session_state:
    st.session_state.conversation = []
def add_message(sender, message):
    st.session_state.conversation.append({"sender": sender, "message": message})
def display_conversation():
    for message in reversed(st.session_state.conversation):
        if message["sender"] == "user":
            st.markdown(f"**You:** {message['message']}")
        else:
            st.markdown(f"**Bot:** {message['message']}")
st.title("Chatbot Interface")
user_input = st.text_input("Enter your message:")

image_input = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
if user_input:
    add_message("user", user_input)
    with st.spinner("Processing..."):
        response_content = ""
        try:
            for resp in agent.run(user_input, stream=True):
                if hasattr(resp, "content") and isinstance(resp.content, str):
                    response_content += resp.content
            add_message("bot", response_content)
        except Exception as e:
            st.error(f"Error occurred: {e}")
    display_conversation()
# input_type = st.selectbox("Select input type:", ["Text", "Image"])

# elif input_type == "Image":
#     uploaded_image = st.file_uploader("Upload an image:", type=["jpg", "jpeg", "png"])
#     if uploaded_image:
#         image = Image.open(uploaded_image)
#         st.image(image, caption="Uploaded Image", use_column_width=True)
#         with st.spinner("Processing..."):
#             st.text("Bot's Response:")
#             agent.print_response("Image uploaded successfully", stream=True)