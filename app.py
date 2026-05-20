from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from google import genai
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Create Gemini client
client = genai.Client(
    api_key=os.getenv("GOOGLE_API_KEY")
)

# Load knowledge base file
with open("sample_kb.txt", "r", encoding="utf-8") as file:
    knowledge = file.read()

# Split text into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)

texts = text_splitter.split_text(knowledge)

# Create embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Create vector database
vector_db = Chroma.from_texts(
    texts=texts,
    embedding=embedding_model
)

print("====================================")
print(" Enterprise RAG Assistant Ready ")
print(" Type 'exit' to quit ")
print("====================================")

while True:

    # Get user input
    question = input("\nAsk Question:\n")

    # Exit condition
    if question.lower() == "exit":
        print("\nExiting assistant...")
        break

    # Retrieve relevant chunks
    docs = vector_db.similarity_search(
        question,
        k=2
    )

    # Combine retrieved context
    retrieved_context = "\n".join(
        [doc.page_content for doc in docs]
    )

    # Prompt
    prompt = f"""
You are an enterprise IT troubleshooting assistant.

Use ONLY the provided knowledge base context.

If the answer is not present in the context,
say:
"I could not find this information in the knowledge base."

Context:
{retrieved_context}

User Question:
{question}

Provide a professional troubleshooting response.
"""

    try:

        # Generate response using Gemini
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )

        print("\nAI Response:\n")
        print(response.text)

    except Exception as e:

        print("\nError generating response:")
        print(e)