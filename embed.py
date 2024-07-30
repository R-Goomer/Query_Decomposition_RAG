from langchain_openai import OpenAIEmbeddings# type: ignore
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from dotenv import load_dotenv
import os


def __init__():
    # Load environment variables from .env file
    os.environ['OPENAI_API_KEY'] = os.getenv('open_ai_api')

def initialize_embedding(embedding_option):
    """
    Initialize and return the embedding model based on the selected option.
    
    Args:
    - embedding_option (str): The selected embedding option ('BGE', 'OpenAI', or 'ThirdEmbedding').
    
    Returns:
    - object: The initialized embedding model.
    """
    if embedding_option == 'BGE':
        # Initialize and return the BGE embedding model
        model_name = "BAAI/bge-small-en-v1.5"
        model_kwargs = {'device': 'cpu'}
        encode_kwargs = {'normalize_embeddings': True} # set True to compute cosine similarity
        model = HuggingFaceBgeEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
        return model  # Replace with actual initialization code for BGE
    
    elif embedding_option == 'OpenAI':
        # Initialize and return the OpenAI embedding model
        # Ensure to set your OpenAI API key before using this
        return OpenAIEmbeddings()  
    
    elif embedding_option == 'SentenceTransformer':
        # Initialize and return the third embedding model
        return SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    
    else:
        raise ValueError("Invalid embedding option. Choose 'BGE', 'OpenAI', or 'ThirdEmbedding'.")