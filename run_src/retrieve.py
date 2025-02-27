import threading
from azure.search.documents.models import VectorizedQuery
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from faiss_handler import FaissHandler
import sys
import os

sys.path.append(".")
from models.OpenAI_API import generate_with_OpenAI_model
import time
import cohere

AZURE_STORAGE_CONNECTION_STRING = "YOUR_AZURE_STORAGE_CONNECTION_STRING"
AZURE_SEARCH_SERVICE_NAME = "YOUR_AZURE_SEARCH_SERVICE_NAME"
AZURE_SEARCH_API_KEY = "YOUR_AZURE_SEARCH_API_KEY"

RETRIEVE_PROMPT = (
    "Given a question, generate a search query that would help gather information to answer it. "
    "Your goal is to formulate a query that will retrieve useful evidence or additional details that contribute to answering the question. "
    "The query should aim to obtain new information and be specific and clear enough to ensure that the search results are relevant and helpful. "
    "Please answer in one complete sentence, starting with string \"The query is: <your retrieve query>\". \n\n"
    "Question: {question}"
)

PROMPT_TEMPLATE = [
    '''### Template:

Question: Who lived longer, Muhammad Ali or Alan Turing?
Follow up: How old was Muhammad Ali when he died?
Intermediate answer: Muhammad Ali was 74 years old when he died.
Follow up: How old was Alan Turing when he died?
Intermediate answer: Alan Turing was 41 years old when he died.
So the final answer is: Muhammad Ali 

Question: When was the founder of Craigslist born?
Follow up: Who was the founder of Craigslist?
Intermediate answer: Craigslist was founded by Craig Newmark.
Follow up: When was Craig Newmark born?
Intermediate answer: Craig Newmark was born on December 6, 1952.
So the final answer is: December 6, 1952

### Instruction: Study the template structure and solve the question below. Complete the follow-up questions and end with: Follow up: <query>.

Question: ''',
    '''
Follow up:''',
]


class Retriever:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.co = cohere.Client("cohere_api_key")
        self.search_client = SearchClient(
            endpoint=f"https://{AZURE_SEARCH_SERVICE_NAME}.search.windows.net",
            index_name="index_cohere_wiki",
            credential=AzureKeyCredential(AZURE_SEARCH_API_KEY),
        )

        faiss_index_path = "faiss_index_path"
        evidence_csv_path = "evidence_csv_path"

        self.evidence = None

        if not os.path.exists(faiss_index_path):
            print(f"FAISS index file does not exist: {faiss_index_path}")
            return
        if not os.path.exists(evidence_csv_path):
            print(f"Evidence data file does not exist: {evidence_csv_path}")
            return

        self.faiss_handler = FaissHandler(index_path=faiss_index_path, csv_path=evidence_csv_path)
        try:
            self.faiss_handler.prepare_handler()
            print("FAISS index and evidence data loaded successfully.")
        except FileNotFoundError as e:
            print(f"Initialization error: {e}")
            return
        except Exception as e:
            print(f"Unknown error: {e}")
            return

    def regist_io_system(self, io):
        self.io = io

    def add_evidence(self, evidence):
        self.evidence = evidence

    def embed_with_retry(self, texts, model, input_type, retries=3, delay=2):
        for attempt in range(retries):
            try:
                response = self.co.embed(texts=texts, model=model, input_type=input_type)
                return response
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {e}")
                if attempt < retries - 1:
                    time.sleep(delay)
                else:
                    print("All retry attempts failed.")
                    return None

    def search_with_retry(self, embedding, retries=3, delay=2):
        for attempt in range(retries):
            try:
                vector_query = VectorizedQuery(vector=embedding, k_nearest_neighbors=50, fields="vector")
                results = self.search_client.search(
                    vector_queries=[vector_query], select=["title", "text"], top=4
                )
                return results
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {e}")
                if attempt < retries - 1:
                    time.sleep(delay)
                else:
                    print("All retry attempts failed.")
                    return None

    def _extract_query(self, original_query):
        parts = original_query.split("Follow up:")
        if len(parts) < 2:
            print("No query generated")
            return parts[0].strip()
        return parts[-1].strip()

    def _do_retrieve(self, query: str):
        response = self.embed_with_retry(texts=[query], model="embed-multilingual-v3.0", input_type="search_query")
        embedding = response.embeddings[0]

        results = self.search_with_retry(embedding=embedding)

        output_lines = [f"{idx}. {result['title']}: {result['text']}" for idx, result in enumerate(results)]
        return "\n".join(output_lines)

    def retrieve_search_engine(self, original_question: str):
        if not original_question:
            return None

        rag_prompt = RETRIEVE_PROMPT.format(question=original_question)
        print(f"Processing question: {original_question}")

        io_output_list = generate_with_OpenAI_model(prompt=rag_prompt, model_ckpt="gpt-4o", max_tokens=128, stop=[])

        print("Generated query: " + ", ".join(io_output_list))
        query = self._extract_query(io_output_list[0])

        return self._do_retrieve(query)

    def retrieve_similar_evidence(self, query, top_k=5):
        try:
            return self.faiss_handler.retrieve(query, top_k=top_k)
        except Exception as e:
            print(f"Error during retrieval: {e}")
            return []

    def retrieve(self, original_question: str):
        if not original_question:
            return None

        rag_prompt = RETRIEVE_PROMPT.format(question=original_question)
        print(f"Processing question: {original_question}")

        io_output_list = generate_with_OpenAI_model(prompt=rag_prompt, model_ckpt="gpt-4o", max_tokens=128, stop=[])

        query = self._extract_query(io_output_list[0])

        top_k = 2
        with self.lock:
            similar_evidences = self.retrieve_similar_evidence(query, top_k=top_k)

        if not similar_evidences:
            print("No relevant evidence found.")

        retrieved_context = "\n".join(
            [f"{idx}. {result['evidence']}" for idx, result in enumerate(similar_evidences, start=1)]
        )

        return retrieved_context


if __name__ == "__main__":
    query = None
    retriever = Retriever()
    print(retriever.retrieve(query))
