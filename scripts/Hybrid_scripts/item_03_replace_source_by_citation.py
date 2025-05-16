# Run: python scripts/Hybrid-scripts/item_03_replace_source_by_citation.py

import os
import sys
from dotenv import load_dotenv, find_dotenv
import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

# Get the current script's directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Navigate two levels up
working_dir = os.path.abspath(os.path.join(current_dir, '..', '..'))

# Load environment variables
load_dotenv(find_dotenv())

print("=" * 50)
print("Starting source replacement process")
print("=" * 50)

# Initialize OpenAI embeddings
try:
    print("Initializing OpenAI embeddings...")
    embedding = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
    print("Embeddings initialized successfully")
except Exception as e:
    print(f"Error initializing OpenAI embeddings: {e}")
    sys.exit(1)

# Load the FAISS index
faiss_save_path = os.path.join(working_dir, "hybrid_index")
try:
    print(f"Loading vector store from {faiss_save_path}...")
    vectordb = FAISS.load_local(faiss_save_path, embedding, allow_dangerous_deserialization=True)
    print("Vector store loaded successfully")
except Exception as e:
    print(f"Error loading FAISS index: {e}")
    sys.exit(1)

# Load the CSV file with sources and references
additional_files_dir = os.path.join(working_dir, "additional_files")
csv_filename = os.path.join(additional_files_dir, "citations.csv")

try:
    if not os.path.exists(csv_filename):
        print(f"Error: Citations file not found at {csv_filename}")
        print("Did you run item_02_generate_citations_APA_FAISS.py first?")
        sys.exit(1)
        
    print(f"Loading citations from {csv_filename}...")
    citations_df = pd.read_csv(csv_filename)
    print(f"Loaded {len(citations_df)} citations")
except Exception as e:
    print(f"Error loading citations CSV: {e}")
    sys.exit(1)

# Create a dictionary for quick lookup
source_to_reference = dict(zip(citations_df['Source'], citations_df['Reference']))
print(f"Created lookup dictionary with {len(source_to_reference)} sources")

# Get all keys from the vector database
all_keys = list(vectordb.docstore.__dict__['_dict'].keys())
print(f"Found {len(all_keys)} documents in vector store")

# Function to replace source with reference
def replace_source_with_reference(doc):
    try:
        original_source = doc.metadata["source"]
        if original_source in source_to_reference:
            doc.metadata["source"] = source_to_reference[original_source]
        return doc
    except Exception as e:
        print(f"Error replacing source in document: {e}")
        return doc

# Iterate through all documents and replace the source with the reference
print("Replacing sources with citations...")
replaced_count = 0
error_count = 0
for key in all_keys:
    try:
        original_doc = vectordb.docstore.__dict__['_dict'][key]
        updated_doc = replace_source_with_reference(original_doc)
        vectordb.docstore.__dict__['_dict'][key] = updated_doc
        replaced_count += 1
        
        # Progress indicator every 1000 documents
        if replaced_count % 1000 == 0:
            print(f"Processed {replaced_count}/{len(all_keys)} documents...")
    except Exception as e:
        print(f"Error processing document with key {key}: {e}")
        error_count += 1

# Save the modified vectordb to local
try:
    print(f"Saving updated vector store to {faiss_save_path}...")
    vectordb.save_local(faiss_save_path)
    print("Vector store saved successfully")
except Exception as e:
    print(f"Error saving FAISS index: {e}")
    sys.exit(1)

print("\n" + "=" * 50)
print("Source Replacement Summary")
print("=" * 50)
print(f"Total documents processed: {replaced_count}")
print(f"Documents with errors: {error_count}")
print("=" * 50)

# Optionally, verify a few entries
print("\nVerifying a few sample entries:")
sample_size = min(5, len(all_keys))
for i in range(sample_size):
    try:
        source = vectordb.docstore.__dict__['_dict'][all_keys[i]].metadata["source"]
        print(f"Document {i + 1} source: {source[:100]}..." if len(source) > 100 else source)
    except Exception as e:
        print(f"Error accessing document {i + 1}: {e}")
# %%
