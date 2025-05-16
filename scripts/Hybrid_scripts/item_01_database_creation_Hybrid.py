# #run it from main folder using : python scripts/Hybrid-scripts/item_01_database_creation_Hybrid.py


###########

import os
import pickle
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import networkx as nx
import spacy
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Constants
PDF_FOLDER = "data/nine_pdfs"  # Folder containing PDFs
EMBEDDING_MODEL = "text-embedding-ada-002"
VECTOR_STORE_PATH = "hybrid_index"
GRAPH_PATH = "hybrid_index/networkx_graph.pkl"

# Load spaCy
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading spaCy model...")
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

# Coral-specific terminology to augment NER
CORAL_TERMS = {
    "CORAL_SPECIES": [
        "acropora", "porites", "pocillopora", "montastraea", "goniastrea", 
        "diploria", "montipora", "favia", "pavona", "fungia", "stylophora"
    ],
    "REEF_FEATURES": [
        "reef", "atoll", "back reef", "fore reef", "reef crest", "lagoon",
        "fringing reef", "barrier reef", "patch reef", "spur and groove"
    ],
    "THREATS": [
        "bleaching", "acidification", "disease", "warming", "overfishing",
        "pollution", "sedimentation", "runoff", "bioerosion", "ocean warming"
    ]
}

def extract_domain_entities(text):
    """
    Extract domain-specific entities from the provided text.

    Parameters:
    - text (str): The input text from which to extract entities.

    Returns:
    - List[Dict]: A list of dictionaries, each containing:
        - 'text' (str): The matched entity text.
        - 'label' (str): The category of the entity (e.g., CORAL_SPECIES).
        - 'start' (int): The starting index of the entity in the text.
        - 'end' (int): The ending index of the entity in the text.

    Side Effects:
    - None.

    Notable Edge Cases:
    - The function is case-insensitive and will match terms regardless of their case.
    """
    entities = []
    text_lower = text.lower()
    
    for category, terms in CORAL_TERMS.items():
        for term in terms:
            start_pos = 0
            while True:
                start_pos = text_lower.find(term, start_pos)
                if start_pos == -1:
                    break
                end_pos = start_pos + len(term)
                entities.append({"text": text[start_pos:end_pos], 
                                "label": category, 
                                "start": start_pos, 
                                "end": end_pos})
                start_pos = end_pos
    return entities

# NetworkX Graph Class
class NetworkXGraph:
    def __init__(self):
        """
        Initialize a directed graph using NetworkX.

        Purpose:
        - To create a graph structure that can store documents and their relationships.

        Attributes:
        - graph (nx.DiGraph): The underlying directed graph structure.
        """
        self.graph = nx.DiGraph()

    def add_document(self, doc_id, content):
        """
        Add a document to the graph, processing its content to extract entities and relationships.

        Parameters:
        - doc_id (str): Unique identifier for the document.
        - content (str): The content of the document to be processed.

        Returns:
        - None

        Side Effects:
        - Modifies the internal graph structure by adding nodes and edges.

        Notable Edge Cases:
        - Handles large documents by processing them in chunks to avoid memory issues.
        """
        try:
            if doc_id not in self.graph.nodes:
                self.graph.add_node(doc_id, type="Document")
                
                # Process in chunks to handle large documents
                MAX_CHUNK_SIZE = 10000
                chunks = [content[i:i+MAX_CHUNK_SIZE] for i in range(0, len(content), MAX_CHUNK_SIZE)]
                
                for chunk in chunks:
                    doc = nlp(chunk)
                    for sent in doc.sents:
                        self._process_sentence(sent, doc_id)
                    
                    # Also extract domain-specific entities
                    domain_entities = extract_domain_entities(chunk)
                    for entity in domain_entities:
                        ent_id = f"{entity['text']}_{entity['label']}"
                        self.graph.add_node(ent_id, type="Entity", name=entity['text'], label=entity['label'])
                        self.graph.add_edge(doc_id, ent_id, relationship="DOMAIN_SPECIFIC")
        except Exception as e:
            print(f"Error processing document {doc_id}: {e}")

    def _process_sentence(self, sent, doc_id):
        """
        Process a single sentence to extract entities and relationships.

        Parameters:
        - sent (spacy.tokens.Span): The sentence to process.
        - doc_id (str): The ID of the document containing the sentence.

        Returns:
        - None

        Side Effects:
        - Modifies the internal graph structure by adding nodes and edges.

        Notable Edge Cases:
        - Handles sentences with multiple entities and relationships.
        """
        try:
            sent_doc = nlp(sent.text)
            # Extract entities
            for ent in sent_doc.ents:
                ent_id = f"{ent.text}_{ent.label_}"
                self.graph.add_node(ent_id, type="Entity", name=ent.text, label=ent.label_)
                self.graph.add_edge(doc_id, ent_id, relationship="MENTIONS")
            
            # Extract relationships between entities in the same sentence
            for token in sent_doc:
                if token.dep_ in ["nsubj", "dobj", "prep"]:  # Focus on key dependency relations
                    subject = [w for w in token.head.lefts if w.dep_ in ["nsubj", "nsubjpass"]]
                    obj = [w for w in token.rights if w.dep_ in ["dobj", "pobj"]]
                    if subject and obj:
                        self.graph.add_edge(
                            subject[0].text,
                            obj[0].text,
                            relationship=token.text.upper(),
                            rel_type="RELATES_TO"
                        )
        except Exception as e:
            print(f"Error processing sentence '{sent.text[:50]}...': {e}")

    def save_graph(self, file_path):
        """
        Save the graph to a file using pickle.

        Parameters:
        - file_path (str): The path where the graph will be saved.

        Returns:
        - None

        Side Effects:
        - Writes the graph structure to a file.

        Notable Edge Cases:
        - Ensure the directory exists before saving.
        """
        with open(file_path, 'wb') as f:
            pickle.dump(self.graph, f)

    @classmethod
    def load_graph(cls, file_path):
        """
        Load the graph from a file using pickle.

        Parameters:
        - file_path (str): The path from which to load the graph.

        Returns:
        - NetworkXGraph: An instance of NetworkXGraph containing the loaded graph.

        Side Effects:
        - None.

        Notable Edge Cases:
        - Handles file not found errors gracefully.
        """
        with open(file_path, 'rb') as f:
            graph = pickle.load(f)
        nx_graph = cls()
        nx_graph.graph = graph
        return nx_graph

    def query_graph(self, query):
        """
        Query the graph to find documents related to entities in the query.

        Parameters:
        - query (str): The input query string to search for related documents.

        Returns:
        - List[Dict]: A list of dictionaries, each containing:
            - 'id' (str): The ID of the related document.
            - 'relevance_score' (int): A score indicating the relevance of the document to the query.

        Side Effects:
        - None.

        Notable Edge Cases:
        - Returns an empty list if no related documents are found.
        """
        try:
            doc = nlp(query)
            # Extract standard NER entities
            standard_entities = {ent.text.lower() for ent in doc.ents}
            
            # Extract domain-specific entities
            domain_entities = extract_domain_entities(query)
            domain_entity_texts = {entity['text'].lower() for entity in domain_entities}
            
            # Combine all entities
            all_entities = standard_entities.union(domain_entity_texts)
            
            # Add keyword search for better recall
            keywords = set([token.lemma_.lower() for token in doc 
                        if not token.is_stop and not token.is_punct and token.is_alpha])
            important_keywords = {word for word in keywords 
                                if len(word) > 3}  # Filter for more significant keywords
            
            if not all_entities and not important_keywords:
                return []
            
            related_docs = []
            visited_nodes = set()
            
            # Helper function for graph traversal
            def traverse_graph(node_id, depth=0, max_depth=2):
                if depth > max_depth or node_id in visited_nodes:
                    return
                
                visited_nodes.add(node_id)
                
                # If this is a document node, add it to results
                if self.graph.nodes[node_id].get("type") == "Document":
                    doc_info = {
                        "id": node_id,
                        "relevance_score": max_depth - depth + 1  # Higher score for closer nodes
                    }
                    related_docs.append(doc_info)
                
                # Explore neighbors in both directions
                for neighbor in list(self.graph.neighbors(node_id)) + list(self.graph.predecessors(node_id)):
                    traverse_graph(neighbor, depth + 1, max_depth)
            
            # Start with entity nodes
            for node in self.graph.nodes:
                node_data = self.graph.nodes[node]
                if node_data.get("type") == "Entity":
                    node_name = node_data.get("name", "").lower()
                    
                    # Match by exact entity
                    if node_name in all_entities:
                        traverse_graph(node, max_depth=2)
                    # Match by important keywords
                    elif any(keyword in node_name for keyword in important_keywords):
                        traverse_graph(node, max_depth=1)
            
            # Sort by relevance score and deduplicate
            related_docs.sort(key=lambda x: x["relevance_score"], reverse=True)
            unique_docs = []
            seen_ids = set()
            for doc in related_docs:
                if doc["id"] not in seen_ids:
                    seen_ids.add(doc["id"])
                    unique_docs.append(doc)
            
            return unique_docs
            
        except Exception as e:
            print(f"Error querying graph: {e}")
            return []

# Function to split documents semantically with improved logic
def split_documents_semantically(documents, max_tokens=300):
    """
    Split documents more intelligently into chunks based on semantic boundaries.

    Parameters:
    - documents (List[Document]): A list of documents to be split.
    - max_tokens (int): The maximum number of tokens allowed in each chunk.

    Returns:
    - List[Document]: A list of semantically split document chunks.

    Side Effects:
    - None.

    Notable Edge Cases:
    - Attempts to maintain context and keep related sentences together.
    """
    chunks = []
    for doc in documents:
        content = doc.page_content
        spacy_doc = nlp(content)
        
        # Group sentences into paragraphs
        paragraphs = []
        current_para = []
        
        for sent in spacy_doc.sents:
            current_para.append(sent.text)
            # If sentence ends with period and next sentence starts with capital, likely paragraph boundary
            if sent.text.strip().endswith('.') and len(current_para) > 0:
                paragraphs.append(" ".join(current_para))
                current_para = []
        
        # Add any remaining sentences
        if current_para:
            paragraphs.append(" ".join(current_para))
        
        # Create chunk from each paragraph, ensuring they're not too long
        for para in paragraphs:
            # Skip empty paragraphs
            if not para.strip():
                continue
                
            para_doc = nlp(para)
            if len(para_doc) <= max_tokens:
                chunk = type(doc)(page_content=para, metadata=doc.metadata.copy())
                chunks.append(chunk)
            else:
                # If paragraph is too long, split by sentences
                current_chunk = []
                current_token_count = 0
                
                for sent in nlp(para).sents:
                    sent_len = len(nlp(sent.text))
                    if current_token_count + sent_len > max_tokens and current_chunk:
                        # Create chunk and reset
                        chunk_text = " ".join(current_chunk)
                        chunk = type(doc)(page_content=chunk_text, metadata=doc.metadata.copy())
                        chunks.append(chunk)
                        current_chunk = [sent.text]
                        current_token_count = sent_len
                    else:
                        current_chunk.append(sent.text)
                        current_token_count += sent_len
                
                # Add any remaining text as a chunk
                if current_chunk:
                    chunk_text = " ".join(current_chunk)
                    chunk = type(doc)(page_content=chunk_text, metadata=doc.metadata.copy())
                    chunks.append(chunk)
    
    return chunks

# Function to load existing database with better error handling
def load_existing_db():
    """
    Load the existing vector store and graph from disk, handling errors gracefully.

    Returns:
    - Tuple[FAISS, NetworkXGraph, Set[str]]: A tuple containing:
        - vector_store (FAISS): The loaded vector store or None if not found.
        - graph (NetworkXGraph): The loaded graph or a new instance if not found.
        - processed_files (Set[str]): A set of document IDs that have been processed.

    Side Effects:
    - None.

    Notable Edge Cases:
    - Handles cases where the vector store or graph files do not exist.
    """
    try:
        embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
        vector_store = None
        
        if os.path.exists(VECTOR_STORE_PATH):
            try:
                print(f"Loading existing vector store from {VECTOR_STORE_PATH}...")
                vector_store = FAISS.load_local(VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True)
                print("Vector store loaded successfully")
            except Exception as e:
                print(f"Error loading FAISS index: {e}. Creating a new one.")
                vector_store = None

        # Ensure directory exists for new database creation
        os.makedirs(VECTOR_STORE_PATH, exist_ok=True)
            
        graph = NetworkXGraph()
        if os.path.exists(GRAPH_PATH):
            try:
                print(f"Loading existing graph from {GRAPH_PATH}...")
                graph = NetworkXGraph.load_graph(GRAPH_PATH)
                print(f"Graph loaded with {len(graph.graph.nodes)} nodes")
            except Exception as e:
                print(f"Error loading graph: {e}. Creating a new one.")
                graph = NetworkXGraph()
        
        # Get processed files from the graph nodes
        processed_files = set()
        for node, attrs in graph.graph.nodes(data=True):
            if attrs.get("type") == "Document":
                processed_files.add(node)
        
        print(f"Found {len(processed_files)} processed documents in the graph")
        return vector_store, graph, processed_files
        
    except Exception as e:
        print(f"Critical error in load_existing_db: {e}")
        return None, NetworkXGraph(), set()

# Function to process new PDFs with improved error handling
def process_new_pdfs(pdf_folder, vector_store, graph, processed_files):
    """
    Process new PDF files from the specified folder, adding them to the vector store and graph.

    Parameters:
    - pdf_folder (str): The folder containing PDF files to process.
    - vector_store (FAISS): The current vector store to update.
    - graph (NetworkXGraph): The current graph to update.
    - processed_files (Set[str]): A set of document IDs that have already been processed.

    Returns:
    - Tuple[FAISS, NetworkXGraph, int]: A tuple containing:
        - vector_store (FAISS): The updated vector store.
        - graph (NetworkXGraph): The updated graph.
        - new_files_processed (int): The count of new files processed.

    Side Effects:
    - Modifies the vector store and graph by adding new documents.

    Notable Edge Cases:
    - Skips empty or corrupted PDFs and logs warnings.
    """
    documents = []
    new_files_processed = 0
    skipped_files = 0
    failed_files = []
    
    try:
        embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
        
        if not os.path.exists(pdf_folder):
            print(f"Warning: PDF folder {pdf_folder} does not exist")
            return vector_store, graph, new_files_processed
            
        for filename in os.listdir(pdf_folder):
            if not filename.endswith(".pdf"):
                continue
                
            if filename in processed_files:
                skipped_files += 1
                continue
                
            pdf_path = os.path.join(pdf_folder, filename)
            try:
                print(f"Processing {filename}...")
                loader = PyPDFLoader(pdf_path)
                pages = loader.load()
                doc_id = filename
                content = " ".join(page.page_content for page in pages)
                
                # Skip empty or corrupted PDFs
                if not content or len(content.strip()) < 50:
                    print(f"Warning: {filename} appears to be empty or corrupted. Skipping.")
                    failed_files.append(filename)
                    continue
                    
                for page in pages:
                    page.metadata["source"] = pdf_path
                    page.metadata["doc_id"] = doc_id
                    
                # Add document to documents list for vectorization    
                documents.extend(pages)
                
                # Add document to graph
                graph.add_document(doc_id, content)
                
                processed_files.add(filename)
                new_files_processed += 1
                print(f"Successfully processed {filename}")
                
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                failed_files.append(filename)
        
        # Add documents to vector store if any new docs were processed
        if documents:
            print(f"Splitting documents semantically and creating vectors...")
            chunks = split_documents_semantically(documents)
            print(f"Created {len(chunks)} chunks from {len(documents)} pages")
            
            if vector_store is None:
                vector_store = FAISS.from_documents(chunks, embeddings)
            else:
                vector_store.add_documents(chunks)
                
            print(f"Saving updated vector store to {VECTOR_STORE_PATH}")
            vector_store.save_local(VECTOR_STORE_PATH)
            
        print(f"Processing complete. New files: {new_files_processed}, Skipped: {skipped_files}, Failed: {len(failed_files)}")
        if failed_files:
            print(f"Failed files: {failed_files}")
            
    except Exception as e:
        print(f"Critical error in process_new_pdfs: {e}")
    
    return vector_store, graph, new_files_processed

# Main Function
def update_database():
    """
    Main function to update the database by loading existing data, processing new PDFs, and saving the updated graph.

    Purpose:
    - Orchestrates the entire database update process in three steps:
        1. Load existing database.
        2. Process new PDFs.
        3. Save the updated graph.

    Returns:
    - Tuple[FAISS, NetworkXGraph]: The updated vector store and graph.

    Side Effects:
    - Modifies the vector store and graph by adding new documents.

    Notable Edge Cases:
    - Handles errors gracefully at each step and logs relevant information.
    """
    print("=" * 50)
    print("Starting database update process")
    print("=" * 50)
    
    # Step 1: Load existing database
    print("\n[Step 1/3] Loading existing database...")
    vector_store, graph, processed_files = load_existing_db()
    
    # Step 2: Process new PDFs
    print("\n[Step 2/3] Processing new PDFs...")
    vector_store, graph, new_files_processed = process_new_pdfs(PDF_FOLDER, vector_store, graph, processed_files)
    
    # Step 3: Save the updated graph
    print("\n[Step 3/3] Saving graph database...")
    try:
        os.makedirs(os.path.dirname(GRAPH_PATH), exist_ok=True)
        graph.save_graph(GRAPH_PATH)
        print(f"Graph saved successfully to {GRAPH_PATH}")
    except Exception as e:
        print(f"Error saving graph: {e}")
    
    # Generate summary
    total_documents = len(processed_files)
    total_entities = sum(1 for _, attrs in graph.graph.nodes(data=True) if attrs.get("type") == "Entity")
    total_relationships = graph.graph.number_of_edges()
    
    print("\n" + "=" * 50)
    print("Database Update Summary")
    print("=" * 50)
    print(f"Total documents: {total_documents}")
    print(f"New documents added: {new_files_processed}")
    print(f"Total entities in graph: {total_entities}")
    print(f"Total relationships: {total_relationships}")
    if vector_store:
        try:
            # Fix for updated FAISS API - use index instead of _index
            print(f"Vector store dimension: {vector_store.index.d}")
            print(f"Vector store size: {vector_store.index.ntotal} vectors")
        except AttributeError:
            print("Vector store statistics unavailable")
    print("=" * 50)
    
    return vector_store, graph

if __name__ == "__main__":
    update_database()