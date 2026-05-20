from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from google import genai
from dotenv import load_dotenv
import os

load_dotenv()

# Gemini client
client = genai.Client(
    api_key=os.getenv("GOOGLE_API_KEY")
)

# Load knowledge base
with open("sample_kb.txt", "r", encoding="utf-8") as file:
    knowledge = file.read()

# Split text into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)

texts = text_splitter.split_text(knowledge)

# Create embeddings
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Store in vector DB
vector_db = Chroma.from_texts(
    texts,
    embedding_model
)

print("Enterprise RAG Assistant Ready")
print("Type 'exit' to quit")

while True:

    question = input("\nAsk Question:\n")

    if question.lower() == "exit":
        break

    # Semantic search
    docs = vector_db.similarity_search(question, k=2)

    retrieved_context = "\n".join([doc.page_content for doc in docs])

    prompt = f"""
    You are an enterprise IT troubleshooting assistant.

    Use ONLY the provided knowledge base context.

    Context:
    {retrieved_context}

    User Question:
    {question}

    Provide a professional troubleshooting response.
    """

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    print("\nAI Response:\n")
    print(response.text)