"""
This script implements a simple Retrieval-Augmented Generation (RAG) pipeline
using LangChain. It loads a PDF document, splits it into chunks, stores embeddings
in a Chroma vector database, retrieves the most relevant chunks for a given question,
and uses an OpenAI chat model to generate an answer based on the retrieved context.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains.question_answering import load_qa_chain
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence

import os
import json

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

# Model initialization
embeddings_model = OpenAIEmbeddings()
llm = ChatOpenAI(model_name="gpt-3.5-turbo", max_tokens=200)


def loadData():
    pdf_link = "DOC-SF238339076816-20230503.pdf"

    loader = PyPDFLoader(pdf_link, extract_images=False)
    pages = loader.load_and_split()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=20,
        length_function=len,
        add_start_index=True
    )

    chunks = text_splitter.split_documents(pages)

    vectordb = Chroma.from_documents(chunks, embedding=embeddings_model)
    retriever = vectordb.as_retriever(search_kwarfs={"k": 3})

    return retriever


def getRelevantDocuments(question):
    retriever = loadData()
    context = retriever.get_relevant_documents(question)
    return context


chain = load_qa_chain(llm, chain_type="stuff")


def ask(question, llm):
    TEMPLATE = """
    You are a legal assistant. Use the following pieces of context to answer the question at the end.

    Context: {context}

    Question: {question}
    """

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=TEMPLATE
    )

    sequence = RunnableSequence(prompt | llm)
    context = getRelevantDocuments(question)

    response = sequence.invoke({
        "context": context,
        "question": question
    })

    return response


def lambda_handler(event, context):
    question = event.get("question")
    answer = ask(question, llm).content

    return {
        "statusCode": 200,
        "headers": {
            "Content-Type": "application/json"
        },
        "body": json.dumps({
            "message": "Ended successfully",
            "details": answer
        })
    }
