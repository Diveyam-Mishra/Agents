import os
from phi.model.azure.openai_chat import AzureOpenAIChat
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


embedder = AzureOpenAIEmbedder(api_key=api_key,
                               azure_endpoint =embedder_endpoint,
                               azure_deployment= embedder_deployment,
                               dimensions=3072)