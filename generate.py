# Answer each sub-question individually 

from langchain import hub
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

def Decomposition(llm):
    template = """You are a helpful assistant that generates multiple sub-questions related to an input question. \n
    The goal is to break down the input into a set of sub-problems / sub-questions that can be answers in isolation. \n
    Generate multiple search queries related to: {question} \n
    Output (3 queries):"""
    prompt_decomposition = ChatPromptTemplate.from_template(template)
    # Chain
    generate_queries_decomposition = ( prompt_decomposition | llm | StrOutputParser() | (lambda x: x.split("\n")))
    return generate_queries_decomposition


def retrieve_and_rag(llm, store, question, prompt_rag, decompostion_chain):
    """RAG on each sub-question"""
    
    # Use our decomposition / 
    sub_questions = decompostion_chain.invoke({"question":question})
    
    # Initialize a list to hold RAG chain results
    rag_results = []
    
    for sub_question in sub_questions:
        
        # Retrieve documents for each sub-question
        retrieved_docs = store.as_retriever().get_relevant_documents(sub_question)
        
        # Use retrieved documents and sub-question in RAG chain
        answer = (prompt_rag | llm | StrOutputParser()).invoke({"context": retrieved_docs, 
                                                                "question": sub_question})
        rag_results.append(answer)
    
    return rag_results,sub_questions



def format_qa_pairs(questions, answers):
    """Format Q and A pairs"""
    
    formatted_string = ""
    for i, (question, answer) in enumerate(zip(questions, answers), start=1):
        formatted_string += f"Question {i}: {question}\nAnswer {i}: {answer}\n\n"
    return formatted_string.strip()


def llm_out(store, user_query):
    llm = ChatOpenAI(temperature=0)

    # Wrap the retrieval and RAG process in a RunnableLambda for integration into a chain
    decomp_answers, decomp_questions = retrieve_and_rag(llm, store, user_query, hub.pull("rlm/rag-prompt"), Decomposition(llm))

    context = format_qa_pairs(decomp_questions, decomp_answers)

    # Prompt
    template = """Here is a set of Q+A pairs:

    {context}

    Use these to synthesize an answer to the question: {question}
    """

    prompt = ChatPromptTemplate.from_template(template)

    final_rag_chain = (
        prompt
        | llm
        | StrOutputParser()
    )

    return [context,final_rag_chain.invoke({"context":context,"question":user_query})]