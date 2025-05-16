<<<<<<< HEAD
# COMS6720_project
=======
# Coral AI - Hybrid Scripts

This README provides documentation for the Coral AI hybrid retrieval system scripts and their associated test files. The hybrid system combines vector-based (FAISS) and graph-based (NetworkX) retrieval for more comprehensive coral research information retrieval.

## Overview of Hybrid Scripts

The hybrid scripts in this project implement a sophisticated information retrieval system combining semantic search with graph-based knowledge representation. These scripts form the backbone of the Coral AI knowledge system.

## Architecture Diagram

![Coral-ai-architecture]<img width="601" alt="coralai_arch" src="https://github.com/user-attachments/assets/b73064f8-4ef5-440e-8f34-f668561fdb97" />

### Key Scripts

1. **item_01_database_creation_Hybrid.py**
   - Creates a hybrid database with both FAISS vector index and NetworkX graph components
   - Extracts domain-specific entities from coral research documents
   - Builds connections between documents and entities in the graph
   - Splits documents semantically to preserve context using pre-built nlp model from spacy
   - Saves both the FAISS index and graph for later use

2. **item_02_generate_citation_APA_FAISS.py**
   - Generates APA citations for all documents stored in the FAISS index
   - Creates a CSV file mapping document sources to their corresponding APA citations
   - Enhances document retrieval with proper academic referencing

3. **item_03_replace_source_by_citation.py**
   - Updates document metadata in the FAISS index to use APA citations
   - Replaces raw source paths with properly formatted academic citations
   - Ensures all retrieved information can be properly attributed

4. **item_04_CoT_search.py**
   - Implements a single search (single call to LLM) which costs less.
   - Features chain of tought reasoning capability using the hybrid retrieval system
   - Provides document deduplication to ensure diverse results

5. **item_05_deep_research.py**
   - Implements an advanced research capability using the hybrid retrieval system
   - Features context-aware query expansion for more comprehensive results
   - Includes relevance checking to ensure queries are coral-related
   - Uses sub-query generation to explore related topics
   - Provides document deduplication to ensure diverse results

6. **item_06_evaluate_rag_model.py**
   - Creating LLM generated questions using sample questions given by coral researchers and policymakers
   - Features question generator and rating question functions
   

## Running the Hybrid Scripts

To use these scripts, follow the instructions below. Make sure you have all required dependencies installed.

### Prerequisites

1. Python 3.11.5 environment
2. Required packages installed (see `requirements.txt`)
3. OpenAI API key set in your environment variables or .env file
4. PDF documents in the data directory

### Setup

1. Clone the repository and navigate to the project root:
   ```bash
   git clone https://github.com/CoralX-foundation/Coral-AI.git
   cd Coral-AI
   ```

2. Create and activate virtual environment:
   ```bash
   python -m venv <environment_name>
   <environment_name>\Scripts\activate
   ```


3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the project root with your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

### Running the Scripts

**1. Create the Hybrid Database:**
```bash
python scripts/Hybrid_scripts/item_01_database_creation_Hybrid.py
```
This will:
- Process PDF files in the data directory
- Create a FAISS vector index
- Build a NetworkX graph connecting documents and entities
- Save both to the hybrid_index directory

**2. Generate APA Citations:**
```bash
python scripts/Hybrid_scripts/item_02_generate_citation_APA_FAISS.py
```
This will:
- Process documents in the FAISS index
- Generate APA format citations for each document
- Save a mapping CSV file in additional_files/citations.csv

**3. Replace Sources with Citations:**
```bash
python scripts/Hybrid_scripts/item_03_replace_source_by_citation.py
```
This will:
- Update the FAISS index with proper citations
- Replace raw file paths with academic citations

**4. Run CoT search (For Basic Search):**
```bash
python scripts/Hybrid_scripts/item_04_CoT_search.py
```
This will:
- Return 3 variable final_answer, reasoning_steps, citations by calling "answer_to_QA(query)" function
  
**5. Run Deep search (For More detailed indepth search):**
```bash
python scripts/Hybrid_scripts/item_05_deep_search.py
```
This will:
- Return 3 variable Final_answer,Additional_details,Citations by calling "answer_to_QA(query)" function
  
**6. Run evaluation for creating LLM generated questions:**
```bash
python scripts/Hybrid_scripts/item_06_evaluate_rag_model.py
```
This Above process will:
- Launch the deep research capability
- Allow you to enter research queries
- Return comprehensive results using the hybrid RAG system

### Script Execution Order

For proper functionality, execute the scripts in this order:
1. item_01_database_creation_Hybrid.py
2. item_02_generate_citation_APA_FAISS.py
3. item_03_replace_source_by_citation.py
4. item_04_CoT_search.py  or   item_05_deep_search.py


## Test Scripts

The test scripts validate the functionality of the hybrid system components. These tests ensure that the code is working as expected and that the data structures are properly maintained.

### Available Test Scripts

1. **test_hybrid_database_creation.py**
   - Tests the database creation functionality
   - Validates directory structure, document loading, entity extraction, and graph creation
   - Ensures semantic document splitting works correctly

2. **test_citation_generation.py**
   - Tests the citation generation functionality
   - Validates FAISS index loading, document metadata processing, and citation formatting
   - Checks that citations follow APA format

3. **test_source_replacement.py**
   - Tests the source replacement functionality
   - Ensures citations are properly integrated into document metadata
   - Checks that document content integrity is maintained

4. **test_deep_research.py**
   - Tests the deep research code flow
   - Validates NetworkX graph operations, retriever initialization, and content processing
   - Checks that the hybrid retrieval system can be properly initialized

### Running the Tests

To run the tests, use the Python unittest module:

**Run all tests:**
```bash
python -m unittest discover test_scripts
```

**Run a specific test:**
```bash
python -m unittest test_scripts/test_hybrid_database_creation.py
```

**Run tests with verbose output:**
```bash
python -m unittest test_scripts/test_deep_research.py -v
```

## Data Flow

1. **Document Processing**
   ```
   PDFs → database_creation_FAISS.py → FAISS index / Graph  
   FAISS index → generate_citations_APA_FAISS.py → citations.csv
   citations.csv + FAISS index → replace_source_by_citation.py → Updated FAISS index
   ```

2. **Query Processing**
   ```
   User Query → CoT_search.py / deep_search.py→ Answer with citations
   ```
## Project Structure
```
project/
├── scripts/
│   └── FAISS_scripts/
│       ├── item_01_database_creation_FAISS.py            # Initial document processing
│       ├── item_02_generate_citations_APA_FAISS.py       # Citation generation
│       ├── item_03_replace_source_by_citation.py         # Citation integration
│       ├── item_04_retriever_FAISS.py                    # Query processing
│       ├── item_05_streamlit_FAISS.py                    # Web interface
│       ├── item_06_eval_01_save_response_and_context.py  # Evaluation data collection
│       ├── item_07_eval_02_human_evaluation.py           # System evaluation
│       ├── item_08_eval_03_generate_questions_answers_from_chunk.py  
│       ├── item_09_eval_04_save_response_and_context_LLM.py
│       ├── item_10_eval_05_llm_evaluation.py
│   └── Hybrid_scripts/
│       ├── BayesFilter/                              # Filter scripts for query filtering
│       ├── item_01_database_creation_FAISS.py        # Initial document processing
│       ├── item_02_generate_citations_APA_FAISS.py   # Citation generation
│       ├── item_03_replace_source_by_citation.py     # Citation integration
│       ├── item_04_CoT_search.py                     # Query processing with CoT
│       ├── item_05_deep_search.py                    # Query processing with Deep search
│       ├── item_06_evauate_rag_model.py              # Evaluation data collection
│   └── Scraper/
│       ├── PDF downloader script.ipynb       
│       ├── serpapi_Google_scholar.ipynb
│   └── 
│      
├── data/
│   └── nine_pdfs/           # Source PDF documents
│
├── faiss_index/             # faiss vector database storage
│   ├── index.faiss            
│   ├── index.pkl  
│
├── hybrid_index/            # Hybrid vector database storage
│   ├── index.faiss            
│   ├── index.pkl        
│   ├── networkx_graph.pkl 
│
├── evaluation/
│   ├── all_question.csv        
│   ├── high_rated_question.csv
│
├── test_scripts/
│   ├── test_citation_generation.py   
│   ├── test_deep_research.py    
│   ├── test_hybrid_database_creation.py 
│   ├── test_source_replacement.py
│
├── additional_files/
│   ├── citations.csv                    # Generated APA citations
│   ├── background.jpeg                  # UI background image
│   ├── system_architecture.png          # System architecture diagram
│   ├── Q&A-Human_generated.csv          # Human-created test questions
│   ├── Q&A-human_generated_with_context.csv       # System responses
│   ├── Q&A_result-human_generated.csv             # Detailed evaluation results
│   ├── overall_result-human_generated.csv         # Summary evaluation metrics
│   ├── Q&A-LLM_generated.csv                      # LLM-created test questions
│   ├── Q&A-LLM_generated_with_context.csv         # System responses
│   ├── Q&A_result-LLM_generated.csv               # Detailed evaluation results
│   ├── overall_result-LLM_generated.csv           # Summary evaluation metrics
│   └── overall_result-LLM_generated_divided.csv   # Summary evaluation metrics
│
│
└── README.md
```

## Troubleshooting

### Common Issues

1. **Module not found errors:**
   Make sure you're running the scripts from the project root directory. The scripts use relative imports.

2. **FAISS index not found:**
   Ensure you've run the database creation script first (item_01_database_creation_Hybrid.py).

3. **API key errors:**
   Check that your .env file is properly set up with your OpenAI API key.

4. **Memory issues:**
   Processing large PDF collections may require substantial memory. Consider reducing batch sizes or upgrading your hardware.

### Note on DeprecationWarnings

You may see DeprecationWarnings related to SwigPy types when running the FAISS operations. These are internal warnings from the FAISS library and can be safely ignored as they don't affect functionality.

## Further Resources

For more information about the underlying technologies:

- [FAISS Documentation](https://github.com/facebookresearch/faiss)
- [NetworkX Documentation](https://networkx.org/)
- [LangChain Documentation](https://python.langchain.com/docs/get_started/introduction)
- [OpenAI API Documentation](https://platform.openai.com/docs/introduction) 
>>>>>>> cbc7967 (local to remote)
