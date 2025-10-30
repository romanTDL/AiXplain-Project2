# AiXplain-Project2

# 🧠 AiXplain RAG Agent with Document Processing
**Author:** Roman Nemet  
**Date:** August 2025  
**Platform:** [AiXplain](https://www.aixplain.com)  
**Language:** Python 3.10+  
---
## 📘 Overview
This repository provides a **Retrieval-Augmented Generation (RAG)** pipeline built on the **AiXplain platform**.  
It automates the ingestion and processing of PDF documents, creates searchable indexes, and deploys intelligent agents capable of retrieving and reasoning over document content.
### 🧩 Core Capabilities
1. Downloads and uploads PDF documents to AiXplain as **Assets**.  
2. Extracts text via the **Docling** model.  
3. Token-aware **chunking** and embedding-based indexing.  
4. Creates an **AiXplain Index** for document retrieval.  
5. Deploys a **RAG Agent** and optional **Team Agent** for question-answering.  
6. Runs sample queries to validate retrieval and reasoning.  
---
## ⚙️ Dependencies
### **Python Libraries**
| Library | Purpose |
|----------|----------|
| `requests` | Download PDF files |
| `tiktoken` | Tokenize and chunk text |
| `tempfile`, `uuid`, `pathlib`, `datetime` | File and metadata handling |
| `json`, `os`, `time`, `typing` | Utility and control structures |
### **AiXplain SDK Modules**
| Module | Purpose |
|---------|----------|
| `AgentFactory`, `TeamAgentFactory` | Create and deploy LLM agents |
| `IndexFactory` | Create and manage document indexes |
| `ModelFactory` | Access AiXplain models (LLM, Docling) |
| `FileFactory`, `AssetFactory` | Upload or retrieve documents |
| `Record`, `IndexFilter` | Store and filter indexed document data |
---
## 🔐 Environment Setup
### 1. Install Dependencies
```bash
pip install aixplain-sdk requests tiktoken
2. Configure API Key
Set your AiXplain API key in the environment:
export AIXPLAIN_API_KEY=<your_api_key>
or edit directly in the script for testing:
API_KEY = "your_api_key_here"

📄 Data Inputs
Documents (DOCS)
Each document entry defines source and metadata.
DOCS = [
    {
        "id": "doc1",
        "pdf_url": "https://nepis.epa.gov/Exe/ZyPDF.cgi?Dockey=P100NEIJ.txt",
        "author": "EPA",
        "description": "Water Supply Guidance Direct Implementation Programs"
    },
    {
        "id": "doc2",
        "pdf_url": "https://nepis.epa.gov/Exe/ZyPDF.cgi?Dockey=P100NBDM.txt",
        "author": "EPA",
        "description": "Alternative Monitoring Guidelines"
    }
]
Each entry supports:
	• asset_id: (optional) pre-uploaded AiXplain asset
	• pdf_url: (required) source of the PDF
	• author, description: metadata for search context
Queries (QUERIES)
Sample questions for testing the agent:
QUERIES = [
    "Can you list the documents in this index?",
    "What are the sources of these documents?"
]

🧱 Code Architecture
Class: DocumentProcessor
Handles:
	• PDF download
	• Upload to AiXplain as an Asset
	• Text extraction using Docling
Key Methods
Method	Description
download_pdf(url, output_path)	Downloads a PDF to a temporary path
process(asset_id=None, pdf_url=None)	Extracts text from asset or URL
__init__(docling_model_id)	Loads Docling model from AiXplain

Class: RAGAgentPipeline
Coordinates the overall RAG pipeline: indexing, chunking, agent creation, and query testing.
Key Methods
Method	Description
create_index()	Creates a new AiXplain index for embeddings
ingest_documents(documents)	Processes, chunks, and upserts records into the index
chunk_text(text, max_tokens)	Token-aware text chunking
create_agent()	Deploys base retrieval agent
create_team_agent()	Deploys collaborative team agent using the base agent
run_queries(queries, use_team_agent)	Tests responses from agents

🧭 Execution Flow
main()
│
├── Validate AiXplain assets
├── Initialize RAGAgentPipeline
│   ├── Load Docling + LLM
│   └── Create Index
│
├── Ingest Documents
│   ├── Download + Upload PDF
│   ├── Extract text
│   ├── Chunk text
│   └── Upsert to Index
│
├── Create Base Agent
│   └── Deploy RAG Q&A Agent
│
├── Create Team Agent (optional)
│
└── Run Queries
    ├── Base Agent answers
    └── Team Agent answers


🧪 Example Output
Docling model loaded: Model(677bee6c6eb56331f9192a91)
Downloading PDF from URL...
✓ Successfully uploaded, asset ID: 687ff12fb7302f45a9fd2d14
✓ Successfully processed uploaded PDF
✓ Upserted 12 records into index
Agent created and deployed: 68af2ab9b7302f45a9fd1d22
Team agent created and deployed: 68af2ac1b7302f45a9fd1d33
--- Query 1: Can you list the documents in this index? ---
Response: The index contains documents from EPA related to water supply guidance.
--- Query 2: What are the sources of these documents? ---
Response: Documents were sourced from the EPA NEPIS public repository.

⚠️ Error Handling
The script includes:
	• Retry logic for downloads and model processing
	• Validation of asset accessibility
	• Cleanup of temporary files
	• Graceful error messages for AiXplain API or network failures
Common failure causes:
	• Invalid asset_id
	• Expired API key
	• Unreachable or malformed PDF URL
	• Empty extracted text

🧠 Model References
Purpose	Model ID	Description
Docling	677bee6c6eb56331f9192a91	Text extraction from PDFs
Embedding Model	6734c55df127847059324d9e	Vector embedding for index
LLM	669a63646eb56306647e1091	Query answering and reasoning

🚀 How to Run
	1. Save the script as rag_agent_pipeline.py
	2. Ensure dependencies and API key are configured
	3. Run the script:
python rag_agent_pipeline.py
	4. Observe logs for:
		○ Index ID
		○ Agent IDs
		○ Query results

🧩 Integration Ideas
This RAG agent can be integrated into enterprise systems:
Platform	Integration Idea
Microsoft Outlook	Automatically respond to emails related to indexed topics
Slack / Teams	Chat interface for querying government or compliance documents
Power Automate / Power Apps	Build no-code front-end to query AiXplain agents
Web Portal	Provide document Q&A search on internal sites

🧰 Extending the Script
	• Add new document entries to the DOCS list.
	• Replace model IDs with updated AiXplain models.
	• Include custom tools or inspectors in create_team_agent().
	• Integrate with LangChain, Azure Logic Apps, or Zapier for workflow automation.

✅ Sample Completion Log
==================================================
Pipeline completed successfully!
Base Agent ID: 68af2ab9b7302f45a9fd1d22
Team Agent ID: 68af2ac1b7302f45a9fd1d33
Index ID: 6891111bb7302f45a9fd2f77
==================================================
🧑‍💻 Author
Roman Nemet
AI Architect 
<img width="726" height="6063" alt="image" src="https://github.com/user-attachments/assets/35c9a8ec-0853-4370-89dc-5726d62da4f4" />
