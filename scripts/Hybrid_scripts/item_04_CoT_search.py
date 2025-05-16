import os
import re
import time
import pandas as pd
from dotenv import load_dotenv, find_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.retrievers import BaseRetriever
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_core.globals import set_verbose, set_debug
from item_01_database_creation_Hybrid import NetworkXGraph
from pydantic import SkipValidation

set_verbose(False)
set_debug(False)
load_dotenv(find_dotenv())

LLM_MODEL = "gpt-4o-mini"

SCRIPT_DIR      = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT    = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir, os.pardir))
VECTOR_STORE_PATH = os.path.join(PROJECT_ROOT, "hybrid_index")
GRAPH_PATH        = os.path.join(PROJECT_ROOT, "hybrid_index", "networkx_graph.pkl")
K_CHUNKS = 4  # Number of chunks per question

# ======= ENTER YOUR QUESTIONS HERE =========
HARDCODED_QUESTIONS = [
    "What are the ecological implications of the roles played by parrotfishes in tropical reef ecosystems?",
"What role do facultative endosymbionts play in the survival of hosts that have lost their essential endosymbionts?"
"What challenges do remote sensing technologies face in studying biological populations in marine environments?",
"How does the presence of parasites like Plagioporus sp. affect the growth and health of coral species such as Porites compressa?",
"In what ways can the physical condition of infected polyps indicate the impact of parasitic infections on coral colonies?",
"What are the key components of a bleaching response plan for coral reef systems?",
"What factors contribute to the differing population trends of Hawaiian monk seals across various locations?"
]
# ===========================================

print("=" * 50)
print("Coral Reef Hybrid Retriever (Final Answer + Chunks as Columns)")
print("=" * 50)

class HybridRetriever(BaseRetriever):
    vector_retriever: SkipValidation[any] = None
    graph: SkipValidation[any] = None
    k: int = 4

    def __init__(self, vector_store, graph, k=5):
        super().__init__()
        self.vector_retriever = vector_store.as_retriever(search_kwargs={"k": k})
        self.graph = graph
        self.k = k

    def _get_relevant_documents(self, query: str):
        try:
            print(f"Performing vector search for query: {query[:50]}...")
            vector_docs = self.vector_retriever.invoke(query)
            print(f"Retrieved {len(vector_docs)} documents from vector search")
            print(f"Performing graph search for query: {query[:50]}...")
            graph_results = self.graph.query_graph(query)
            print(f"Retrieved {len(graph_results)} document references from graph search")
            graph_doc_ids = {r["id"] for r in graph_results}
            hybrid_docs = []
            for doc in vector_docs:
                if doc.metadata.get("doc_id") in graph_doc_ids:
                    doc.metadata["graph_enriched"] = True
                    doc.metadata["relevance_boost"] = graph_results[list(graph_doc_ids).index(doc.metadata["doc_id"])].get("relevance_score", 1)
                hybrid_docs.append(doc)
            hybrid_docs.sort(key=lambda x: x.metadata.get("relevance_boost", 0), reverse=True)
            return hybrid_docs
        except Exception as e:
            print(f"Error in hybrid retrieval: {e}")
            return vector_docs if 'vector_docs' in locals() else []

def generate_cot_prompt():
    return """
    You are a specialized retrieval system analyzing scientific information.

    CONTEXT INFORMATION:
    {context}

    QUESTION: {question}

    REASONING INSTRUCTIONS:
    1. First, evaluate each piece of context for relevance to the question (score 1-5)
    2. For relevant information, extract key facts with direct citations
    3. Identify any knowledge gaps or contradictions between sources
    4. Synthesize the information into a coherent understanding
    5. Apply logical reasoning to reach a conclusion

    RESPONSE FORMAT:
    [Relevance Analysis]
    - Analysis of which context pieces are most useful and why

    [Key Information]
    - Extracted information with source citations

    [Reasoning Process]
    - Step-by-step logical process connecting facts to answer

    [Answer]
    - Clear, concise answer based only on provided context
    - If the answer cannot be determined from context, state: "The provided information is insufficient to answer this question."

    IMPORTANT:
    - Do not introduce facts not present in the context
    - Do not use uncertain language when information is clearly stated
    - Be precise about what information comes from which source
    - Do not prefix your answer with phrases like "based on the context" or "the information suggests"
    """

def clean_final_answer(final_answer):
    try:
        final_answer = final_answer.strip()
        final_answer = re.sub(r'\d+\.\s*\*\*Final Answer\*\*:\s*|\*\*Final Answer\*\*: |Final Answer:\s* |Final answer:\s\s*', '', final_answer, flags=re.IGNORECASE)
        final_answer = re.sub(r'\*\*', '', final_answer)
        return final_answer
    except Exception as e:
        print(f"Error cleaning final answer: {e}")
        return final_answer if 'final_answer' in locals() else "Error processing answer"

def process_llm_response_with_sources(llm_response):
    try:
        full_response = llm_response['result'].strip()
        result_lower = full_response.lower().split(".")[0]
        irrelevant_phrases = [
            "i don't know", "i do not know", "i'm not sure", "i am not sure",
            "not relevant to the context", "sorry i cannot answer this question",
            "the provided information is insufficient"
        ]
        used_chunks = llm_response.get("source_documents", [])
        if is_irrelevant := any(phrase in result_lower for phrase in irrelevant_phrases):
            return full_response, "No reasoning steps provided.", ["No citations available"], used_chunks
        citations = []
        unique_citations = set()
        for source in used_chunks:
            try:
                citation = source.metadata.get('source', 'Unknown source')
                if citation not in unique_citations and citation != 'Unknown source':
                    unique_citations.add(citation)
                    encoded_citation = urllib.parse.quote_plus(citation)
                    google_search_link = f"https://www.google.com/search?q={encoded_citation}"
                    clickable_link = f'<a href="{google_search_link}" target="_blank">{citation}</a>'
                    citations.append(clickable_link)
            except Exception as e:
                print(f"Error processing citation: {e}")
        if "[Answer]" in full_response:
            parts = full_response.split("[Answer]")
            reasoning_steps = parts[0].strip()
            final_answer = parts[1].strip() if len(parts) > 1 else full_response
        elif "Final Answer:" in full_response:
            reasoning_steps = full_response.split("Final Answer:")[0].strip()
            final_answer = full_response.split("Final Answer:")[1].strip()
        else:
            paragraphs = full_response.split("\n\n")
            if len(paragraphs) > 1:
                reasoning_steps = "\n\n".join(paragraphs[:-1]).strip()
                final_answer = paragraphs[-1].strip()
            else:
                reasoning_steps = "No explicit reasoning steps provided."
                final_answer = full_response.strip()
        final_answer = clean_final_answer(final_answer)
        return final_answer, reasoning_steps, citations, used_chunks
    except Exception as e:
        print(f"Error processing LLM response: {e}")
        return "Error processing response", "Error in processing", [], []

def raw_LLM_response(query, retry_attempts=2):
    for attempt in range(retry_attempts + 1):
        try:
            print(f"Attempt {attempt + 1}/{retry_attempts + 1} to get response for query: {query[:50]}...")
            embedding = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
            print(f"Loading vector store from {VECTOR_STORE_PATH}...")
            vectordb = FAISS.load_local(VECTOR_STORE_PATH, embedding, allow_dangerous_deserialization=True)
            print(f"Loading graph from {GRAPH_PATH}...")
            graph = NetworkXGraph.load_graph(GRAPH_PATH)
            print("Creating hybrid retriever...")
            hybrid_retriever = HybridRetriever(vectordb, graph, k=K_CHUNKS)
            print(f"Initializing LLM with model {LLM_MODEL}...")
            llm = ChatOpenAI(model=LLM_MODEL, temperature=0)
            prompt_template = PromptTemplate(
                input_variables=["context", "question"],
                template=generate_cot_prompt()
            )
            print("Creating QA chain...")
            qa_chain = RetrievalQA.from_chain_type(
                llm,
                chain_type="stuff",
                retriever=hybrid_retriever,
                return_source_documents=True,
                chain_type_kwargs={"prompt": prompt_template}
            )
            print("Getting response from LLM...")
            start_time = time.time()
            llm_response = qa_chain.invoke({"query": query})
            end_time = time.time()
            print(f"LLM response received in {end_time - start_time:.2f} seconds")
            return llm_response
        except Exception as e:
            print(f"Error in attempt {attempt + 1}: {e}")
            if attempt < retry_attempts:
                wait_time = 2 ** attempt
                print(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print("All retry attempts failed.")
                return {"result": "Error: Unable to get response from LLM", "source_documents": []}

def answer_to_QA(query):
    print(f"\nProcessing query: {query}")
    llm_response = raw_LLM_response(query)
    final_answer, reasoning_steps, citations, used_chunks = process_llm_response_with_sources(llm_response)
    return final_answer, reasoning_steps, citations, used_chunks

def save_chunks_with_answers_as_columns(results, k=4, excel_file="retrieved_chunks.xlsx"):
    columns = ['Query', 'Final Answer'] + [f"Chunk {i+1}" for i in range(k)]
    rows = []
    for query, final_answer, chunk_list in results:
        row = [query, final_answer]
        for i in range(k):
            if i < len(chunk_list):
                doc = chunk_list[i]
                content = getattr(doc, 'page_content', None) or getattr(doc, 'content', None) or str(doc)
            else:
                content = "Not retrieved"
            row.append(content)
        rows.append(row)
    df = pd.DataFrame(rows, columns=columns)
    df.to_excel(excel_file, index=False)
    print(f"\nSaved all questions (final answers + chunks) to '{excel_file}'")

if __name__ == "__main__":
    all_results = []

    for question in HARDCODED_QUESTIONS:
        final_answer, reasoning_steps, citations, used_chunks = answer_to_QA(question)
        print("\n" + "=" * 50)
        print("RESULTS for:", question)
        print("=" * 50)
        print("\nFinal Answer:")
        print(final_answer)
        print("\nChain of Thoughts:")
        print(reasoning_steps)
        if citations:
            print("\nCitations:")
            for citation in citations:
                print(citation)
        else:
            print("\nNo citations found.")

        # Save (question, final_answer, chunks)
        all_results.append((question, final_answer, used_chunks))

    save_chunks_with_answers_as_columns(all_results, k=K_CHUNKS)
