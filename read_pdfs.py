from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def get_pdf_text(pdf_file): ## takes 1 pdf file at a time and exract texts from all pages in a single pdf file
  text = ""
  pdf_reader = PdfReader(pdf_file)
  for page in pdf_reader.pages:
    text += page.extract_text()
  return text

def get_text_chunks(text):
  text_splitter = RecursiveCharacterTextSplitter(chunk_size = 10000, chunk_overlap = 1000)
  chunks = text_splitter.split_text(text)
  return chunks
