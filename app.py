import streamlit as st
import embed
import vec_store
import generate
import read_pdfs
from dotenv import load_dotenv
import os

import importlib
importlib.reload(vec_store)
importlib.reload(embed)
importlib.reload(generate)
importlib.reload(read_pdfs)

load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv('open_ai_api')

def main():
    st.title("Query Decomposition for better RAG")
    
    # Initialize store in session state if it doesn't exist
    if 'store' not in st.session_state:
        st.session_state.store = None
    
    with st.sidebar:
        st.title("Menu:")
        uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
        if uploaded_files:
            st.write(f"Uploaded {len(uploaded_files)} PDF(s)")
            # Select embedding type
            embedding_option = st.selectbox("Select Embedding Type", ["BGE", "OpenAI", "SentenceTransformer"])

        if st.button("Submit and Process") and uploaded_files and embedding_option:
            with st.spinner("Processing..."):
                raw_text = ""
                for pdf_file in uploaded_files:
                    raw_text += read_pdfs.get_pdf_text(pdf_file)
                text_chunks = read_pdfs.get_text_chunks(raw_text)

                # Initialize embedding
                embeddings = embed.initialize_embedding(embedding_option)
                # Create VectorStore
                st.session_state.store = vec_store.create_vs(text_chunks, embeddings, embedding_option)
                st.write(f"VectorStore created with {embedding_option}")
                
    user_query = st.text_input("Ask a question:")
    if user_query:
        if st.session_state.store is not None:
            queries, response = generate.llm_out(st.session_state.store, user_query)
            st.write("3 Queries Generated: ", queries)
            st.write("Reply: ", response)
        else:
            st.write("Please process the PDFs and create the VectorStore first.")

if __name__ == "__main__":
    main()