import os
from dotenv import load_dotenv
load_dotenv()
from openai import AzureOpenAI

from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from azure.core.credentials import AzureKeyCredential

endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
subscription_key = os.getenv("AZURE_OPENAI_API_KEY")
api_version = os.getenv("AZURE_OPENAI_API_VERSION")

client = AzureOpenAI(
    api_version=api_version,
    azure_endpoint=endpoint,
    api_key=subscription_key,
)

# Azure AI Search
SEARCH_ENDPOINT      = os.getenv("SEARCH_ENDPOINT")
SEARCH_API_KEY       = os.getenv("SEARCH_API_KEY")
SEARCH_INDEX_NAME    = os.getenv("SEARCH_INDEX_NAME")

SEARCH_TOP_K         = 3       

VECTOR_FIELD_NAME    = "text_vector"
CONTENT_FIELD_NAME   = "chunk"

search_client = SearchClient(
    endpoint=SEARCH_ENDPOINT,
    index_name=SEARCH_INDEX_NAME,
    credential=AzureKeyCredential(SEARCH_API_KEY),
)

EMBEDDING_DEPLOYMENT = os.getenv("EMBEDDING_DEPLOYMENT")

def embed_query(query: str) -> list[float]:
    response = client.embeddings.create(
        input=query,
        model=EMBEDDING_DEPLOYMENT,
    )
    return response.data[0].embedding


def retrieve_chunks(query: str) -> list[dict]:
    """
    Run a hybrid search (vector + keyword) against Azure AI Search.
    Returns the top-k matching chunks from the PDF.
    """
    query_vector = embed_query(query)

    vector_query = VectorizedQuery(
        vector=query_vector,
        k_nearest_neighbors=SEARCH_TOP_K,
        fields=VECTOR_FIELD_NAME,
    )

    results = search_client.search(
        search_text=query,          # keyword part of hybrid search
        vector_queries=[vector_query],
        select=[CONTENT_FIELD_NAME],
        top=SEARCH_TOP_K,
    )

    chunks = []
    for result in results:
        chunks.append({
            "content": result.get(CONTENT_FIELD_NAME, ""),
            "score": result.get("@search.score", 0),
        })

    return chunks


def generate_answer(query: str, chunks: list[dict]) -> str:
    """
    Pass the retrieved chunks as context and ask GPT to answer the question.
    """
    # Build context from retrieved chunks
    context = "\n\n---\n\n".join([c["content"] for c in chunks if c["content"]])

    system_prompt = """You are a helpful assistant. 
Answer the user's question using ONLY the context provided below.
If the answer is not in the context, say "I don't have enough information in the document to answer that."
Be concise and factual."""

    user_message = f"""Context from document:
{context}

Question: {query}"""

    response = client.chat.completions.create(
        model=deployment,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_message},
        ],
        temperature=0,  # 0 = deterministic / factual answers
        max_tokens=500,
    )

    return response.choices[0].message.content

def ask(query: str) -> str:
    print(f"\n{'='*60}")
    print(f"Question: {query}")
    print(f"{'='*60}")

    # Retrieve
    print("\n[1/2] Searching index for relevant chunks...")
    chunks = retrieve_chunks(query)

    if not chunks:
        print("No relevant chunks found in the index.")
        return ""

    print(f"      Found {len(chunks)} chunk(s):")
    for i, chunk in enumerate(chunks, 1):
        preview = chunk["content"][:120].replace("\n", " ")
        print(f"      Chunk {i} (score: {chunk['score']:.3f}): {preview}...")

    # Generate
    print("\n[2/2] Generating answer with GPT...")
    answer = generate_answer(query, chunks)

    print(f"\nAnswer:\n{answer}")
    return answer



    
test_question = "When to use Dalle?"
ask(test_question)

# Uncomment to run an interactive loop:
# while True:
#     q = input("\nAsk a question (or 'quit'): ").strip()
#     if q.lower() == "quit":
#         break
#     ask(q)