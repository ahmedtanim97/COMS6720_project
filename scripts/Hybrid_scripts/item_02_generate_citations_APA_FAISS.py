## Run : python scripts/Hybrid-scripts/item_02_generate_citations_APA_FAISS.py


import os
import pandas as pd
import openai
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
import time

# Get the current script's directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Navigate two levels up
working_dir = os.path.abspath(os.path.join(current_dir, '..', '..'))

# Load environment variables
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

# Initialize OpenAI embeddings
try:
    embedding = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
except Exception as e:
    print(f"Error initializing OpenAI embeddings: {e}")
    exit(1)

# Load the FAISS index
faiss_save_path = os.path.join(working_dir, "hybrid_index")
try:
    print(f"Loading vector store from {faiss_save_path}...")
    vectordb = FAISS.load_local(faiss_save_path, embedding, allow_dangerous_deserialization=True)
    print("Vector store loaded successfully")
except Exception as e:
    print(f"Error loading FAISS index: {e}")
    exit(1)

# Get the document store data
data_to_manipulate = vectordb.docstore.__dict__['_dict']
values_list = list(data_to_manipulate.values())
all_keys = list(data_to_manipulate.keys())

# Create initial DataFrame
df = pd.DataFrame({"Keys": all_keys, "Values": values_list})

# Extract source from metadata
df["Source"] = df["Values"].apply(lambda x: x.metadata["source"])

# Get unique sources
unique_sources = df['Source'].unique()
print(f"Found {len(unique_sources)} unique sources")

# Function to generate APA references using GPT with retry mechanism
def obtain_reference_using_gpt(text_for_obtaining_reference, max_retries=3):
    for attempt in range(max_retries):
        try:
            completion = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a straightforward assistant who provides quick, direct, and answers without unnecessary elaboration. You will be provided a text chunk and you need to generate the APA 7th reference. Please reply 'I do not know' if you cannot generate the reference. Do not use any other information than the text chunk.",
                    },
                    {
                        "role": "user",
                        "content": f"Text chunk: {text_for_obtaining_reference}",
                    }
                ],
            )
            return completion.choices[0].message.content
        except Exception as e:
            print(f"Error calling OpenAI API (attempt {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff
                print(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                return "Citation generation failed"

# Create additional_files directory if it doesn't exist
additional_files_dir = os.path.join(working_dir, "additional_files")
os.makedirs(additional_files_dir, exist_ok=True)

# Generate references for unique sources
print("Generating references for sources...")
reference_dict = {}
for i, source in enumerate(unique_sources):
    print(f"Processing source {i+1}/{len(unique_sources)}: {source}")
    try:
        df_for_each_unique = df[df["Source"] == source].iloc[:3]
        text_for_obtaining_reference = " ".join(df_for_each_unique["Values"].apply(lambda x: x.page_content))
        reference = obtain_reference_using_gpt(text_for_obtaining_reference)
        reference_dict[source] = reference
    except Exception as e:
        print(f"Error processing source {source}: {e}")
        reference_dict[source] = "Reference generation failed"

# Add citations to the metadata of each document
print("Adding citations to document metadata...")
for key, value in data_to_manipulate.items():
    try:
        source = value.metadata["source"]
        if source in reference_dict:
            value.metadata["citation"] = reference_dict[source]
        else:
            value.metadata["citation"] = "No citation available"
    except Exception as e:
        print(f"Error adding citation to document {key}: {e}")

# Save the updated FAISS index with citation metadata
try:
    print(f"Saving updated vector store to {faiss_save_path}...")
    vectordb.save_local(faiss_save_path)
    print("Vector store saved successfully")
except Exception as e:
    print(f"Error saving FAISS index: {e}")

# Create a new DataFrame with Source and Reference columns
citations_df = pd.DataFrame(list(reference_dict.items()), columns=['Source', 'Reference'])

# Save the DataFrame to a CSV file
csv_filename = os.path.join(additional_files_dir, "citations.csv")
try:
    citations_df.to_csv(csv_filename, index=False, encoding='utf-8')
    print(f"CSV file '{csv_filename}' has been created with Source and Reference columns.")
except Exception as e:
    print(f"Error saving CSV file: {e}")

# Optionally, display the first few rows of the DataFrame
print("\nSample of generated citations:")
print(citations_df.head(5))

