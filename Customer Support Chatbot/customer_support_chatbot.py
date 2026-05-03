from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq
#loading data
loader = PyPDFLoader("docs.pdf")  
docs= loader.load()
#split into chunks
splitter= RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)
#convert to embedings + store
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = Chroma.from_documents(chunks, embeddings)


load_dotenv("secret.env") 

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
#Create Retriever
retriever = db.as_retriever(search_kwargs={"k": 4})

llm = ChatGroq(
    model="llama-3.1-8b-instant",  
    api_key= GROQ_API_KEY
)
prompt = ChatPromptTemplate.from_template("""
    You are an AI assistant that provides clear, structured responses **strictly using the provided knowledge base**.

    **User Question:** {query}

    **Relevant Information:**
    {retrieved_text}

    **Instructions:**
    - Answer **only** using the provided knowledge base.
    - Provide a **step-by-step list** if applicable.
    - Use **bullet points or numbered lists** where necessary.
    - **Elaborate** on each step, making sure it's clear and informative.
    - If no relevant information is found, state: 'I couldn't find enough details in my sources.'

    **Answer in this format:**
    1. **Step 1**: Explanation
    2. **Step 2**: Explanation
    3. **Step 3**: Explanation

    **Bullet Points Example:**
    * **Feature 1**: Explanation
    * **Feature 2**: Explanation

    **Final Answer:**
""")

chain = (
    {"retrieved_text": retriever, "query": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
#asking question
response = llm.invoke("How do I keep track of my returned orders?")


print(response.content)
