import os
import argparse
import chromadb
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

DB_DIR = os.path.join(os.path.dirname(__file__), "../data/chroma")

def get_chroma_client():
    return chromadb.PersistentClient(path=DB_DIR)

def build_index():
    chroma_client = get_chroma_client()
    collection = chroma_client.get_or_create_collection(name="hackforge_knowledge")
    
    documents = [
        "Lahore air quality data: PM2.5 levels are often elevated in the morning, making strenuous outdoor exercise like morning walks potentially unhealthy depending on current AQI readings.",
        "Mars facts: The capital of Mars is not defined as it has no cities or government, but Olympus Mons is its largest volcano.",
        "Islamabad air quality data: Islamabad generally maintains moderate to good air quality suitable for morning walks."
    ]
    metadatas = [
        {"source": "openaq_lahore.json", "language": "en"},
        {"source": "mars_trivia.txt", "language": "en"},
        {"source": "openaq_islamabad.json", "language": "en"}
    ]
    ids = ["doc_lahore_aqi", "doc_mars_capital", "doc_islamabad_aqi"]
    
    collection.upsert(documents=documents, metadatas=metadatas, ids=ids)
    print("RAG index successfully built and stored in ChromaDB.")

def query_agent(query_text: str):
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("Error: GROQ_API_KEY not found in .env file.")
        return

    chroma_client = get_chroma_client()
    try:
        collection = chroma_client.get_collection(name="hackforge_knowledge")
    except Exception:
        print("Error: RAG index not found. Run 'python -m src.insight_agent --build' first.")
        return

    results = collection.query(query_texts=[query_text], n_results=2)
    retrieved_docs = results['documents'][0] if results['documents'] else []
    retrieved_metas = results['metadatas'][0] if results['metadatas'] else []

    context_chunks = [f"Content: {doc} [cite: {meta.get('source', 'unknown')}]" for doc, meta in zip(retrieved_docs, retrieved_metas)]
    context_str = "\n".join(context_chunks)

    groq_client = Groq(api_key=api_key)

    prompt = f"""
You are an expert AI insight assistant for hackForge. 
Answer the user's query using ONLY the provided context below. Do not assume or fabricate external facts.
Include the exact source citation tag at the end of statements derived from context.
If the query is written in Urdu, respond in natural Urdu. Otherwise, respond in English.

Context:
{context_str}

User Query: {query_text}
"""

    print(f"\nQuery: {query_text}")
    print("-" * 50)
    try:
        chat_completion = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="openai/gpt-oss-20b",
        )
        print("AI Generated Response:")
        print(chat_completion.choices[0].message.content)
    except Exception as e:
        print(f"Groq API Error: {e}")
    print("-" * 50)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", action="store_true")
    parser.add_argument("query", nargs="?")
    args = parser.parse_args()
    
    if args.build:
        build_index()
    elif args.query:
        query_agent(args.query)

if __name__ == "__main__":
    main()