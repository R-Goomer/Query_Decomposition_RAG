# Query Decomposition RAG Application

## Overview

This project is a Retrieval-Augmented Generation (RAG) application that incorporates Query Decomposition to enhance performance. Users can upload multiple PDFs, and the system generates QA pairs to feed as context, improving the accuracy and relevance of the generated responses. The application allows users to select from various embedding models and interact with the content through a chat interface.

Check it out here - https://huggingface.co/spaces/Rgoomer/Query_Decomposition_RAG

<img width="1298" alt="image" src="https://github.com/user-attachments/assets/73afdae4-a637-45d2-b113-e0861ea1816e">


## Features

- **Upload Multiple PDFs:** Users can upload several PDF documents at once.
- **Generate QA Pairs:** The system decomposes queries into QA pairs to provide better context for retrieval.
- **Select Embedding Models:** Choose from BGE (BAAI), OpenAI, or Sentence Transformers for creating embeddings.
- **Create Chroma VectorStore:** The application generates a Chroma VectorStore based on the selected embedding model and QA pairs.
- **Interactive Chat:** Users can chat about the content of the uploaded PDFs.

## Installation

### Using Docker

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/your-username/query-decomposition-rag-app.git
   cd query-decomposition-rag-app
   ```

2. **Build the Docker Image:**
   ```bash
   docker build -t query-decomposition-rag-app .
   ```

3. **Run the Docker Container:**
   ```bash
   docker run -p 8501:8501 query-decomposition-rag-app
   ```

   The application will be accessible at `http://localhost:8501`.

## Usage

1. **Upload PDFs:**
   - Navigate to the Streamlit app in your browser.
   - Use the upload functionality to add multiple PDF files.

2. **Generate QA Pairs:**
   - The application will automatically decompose queries into QA pairs and use them as context for improved retrieval.

3. **Select Embedding Model:**
   - Choose the desired embedding model (BGE, OpenAI, or Sentence Transformers) from the available options.

4. **Interact with PDFs:**
   - Use the chat interface to ask questions and interact with the content of the uploaded PDFs.

## Contributing

If you would like to contribute to this project, please follow these steps:

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/YourFeature`).
3. Commit your changes (`git commit -am 'Add some feature'`).
4. Push to the branch (`git push origin feature/YourFeature`).
5. Create a new Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Streamlit](https://streamlit.io/)
- [LangChain](https://www.langchain.com/)
- [Chroma](https://www.trychroma.com/)
- [BAAI](https://www.baai.ac.cn/)
- [OpenAI](https://www.openai.com/)
- [Sentence Transformers](https://www.sbert.net/)

---

Let me know if you need any more changes or additions!
