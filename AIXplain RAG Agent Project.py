#!/usr/bin/env python3
"""
RAG Agent with Document Processing for AiXplain Platform
========================================================

This script creates a RAG (Retrieval-Augmented Generation) agent that:
1. Downloads and processes PDF documents
2. Creates an index for document storage
3. Sets up an agent with retrieval capabilities
4. Tests the agent with sample queries
5. Deploys a team agent including the base agent

Author: Roman Nemet
Date: 08 2025
"""
import os
import uuid
import json
import time
import requests
import tiktoken
import mimetypes
import tempfile
from datetime import datetime
from typing import List, Dict, Optional
from pathlib import Path
# Set API keys (use environment variable or default)
API_KEY = "8a68b2fe3b33425d8ebacfd3986e8fc79c6d6aa840a1a95ce0e43788ccecf567"
os.environ["TEAM_API_KEY"] = os.getenv("TEAM_API_KEY", API_KEY)
os.environ["AIXPLAIN_API_KEY"] = os.getenv("AIXPLAIN_API_KEY", API_KEY)

if "AIXPLAIN_API_KEY" not in os.environ:
    raise ValueError("AIXPLAIN_API_KEY environment variable is required")
# AiXplain imports
from aixplain.factories import AgentFactory, IndexFactory, FileFactory, ModelFactory, AssetFactory, TeamAgentFactory
from aixplain.modules.model.index_model import IndexFilter, IndexFilterOperator
from aixplain.modules.model.record import Record
from aixplain.modules.asset import Asset

# ------------------
# Configuration
# ------------------

DOCS = [
     
        {
        "id": "doc2",
        "pdf_url": "https://nepis.epa.gov/Exe/ZyPDF.cgi?Dockey=P100NBDM.txt",
        "author": "EPA",
        "description": "ALTERNATIVE MONITORING GUIDELINES, Chemicals Other Than Lead & Copper ",
        },
        {"id": "doc1",
        "pdf_url": "https://nepis.epa.gov/Exe/ZyPDF.cgi?Dockey=P100NEIJ.txt", ##
        "author": "EPA",
        "description": "Water Supply Guidance Direct Implementation Programs"
    }
]
   
QUERIES = [
    #"What available documents do you have to work with?",
    "Can you list the documents in this index?",
    #"Provide details about the indexed documents",
    "What are the sources of these documents?"
]

class DocumentProcessor:
    def __init__(self, docling_model_id: str):
        try:
            self.docling = ModelFactory.get("677bee6c6eb56331f9192a91")  #docling_model_id)
            print(f"Docling model loaded: {self.docling}")
        except Exception as e:
            print(f"Error loading docling model {docling_model_id}: {e}")
            raise
        
        self.temp_dir = Path(tempfile.gettempdir())

    def download_pdf(self, url: str, output_path: str) -> bool:
        """Download PDF from URL to local path"""
        try:
            print(f"Downloading PDF from: {url}")
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            with open(output_path, 'wb') as f:
                f.write(response.content)
            print(f"PDF downloaded successfully to: {output_path}")
            return True
        except Exception as e:
            print(f"Download failed for {url}: {e}")
            return False

    def process(self, asset_id: str = None, pdf_url: str = None, retries: int = 3, delay: int = 5) -> Optional[str]:
        """
        Process document using either:
        1. asset_id: Hex string ID of document already uploaded to AiXplain platform
        2. pdf_url: URL to download PDF and upload to AiXplain first
        """
        
        if asset_id:
            # Case 1: Process existing asset using its hex ID
            print(f"Processing existing AiXplain asset with ID: {asset_id}")
                        
        elif pdf_url:
            # Case 2: Download PDF from URL and upload to AiXplain first
            print(f"Downloading PDF from URL and uploading to AiXplain: {pdf_url}")
            file_path = self.temp_dir / f"temp_{uuid.uuid4()}.pdf"
            
            try:
                # Download the PDF file
                if not self.download_pdf(pdf_url, str(file_path)):
                    return None
                
                # Upload file to AiXplain platform - try different upload methods
                asset = None
                #asset_id = None
                upload_methods = [
                    # Method 1: Most basic upload - just local_path
                    lambda: FileFactory.upload(local_path=str(file_path)),
                    
                ]
                
                for i, method in enumerate(upload_methods):
                    try:
                        print(f"Trying upload method {i+1}...")
                        upload_result = method()
                        
                        # Handle different return types from upload methods
                        asset_id = None
                        if hasattr(upload_result, 'id'):
                            # Standard Asset object with .id attribute
                            asset_id = upload_result.id
                            asset = upload_result
                            print(f"✓ Method {i+1}: Got Asset object with ID: {asset_id}")
                        elif isinstance(upload_result, str):
                            # Sometimes returns just the asset ID as string
                            asset_id = upload_result
                            asset = upload_result  # Use the ID string directly
                            print(f"✓ Method {i+1}: Got asset ID as string: {asset_id}")
                        elif isinstance(upload_result, dict) and 'id' in upload_result:
                            # Dictionary response with ID
                            asset_id = upload_result['id']
                            asset = upload_result['id']
                            print(f"✓ Method {i+1}: Got asset ID from dict: {asset_id}")
                        else:
                            print(f"Method {i+1}: Unexpected return type: {type(upload_result)}")
                            print(f"Upload result: {upload_result}")
                            continue
                        
                        if asset_id:
                            print(f"✓ Successfully uploaded, asset ID: {asset_id}")
                            break
                            
                    except Exception as e:
                        print(f"Upload method {i+1} failed: {e}")
                        if i == len(upload_methods) - 1:  # Last method
                            print("All upload methods failed")
                            return None
                        continue
                
                if not asset_id:
                    print("Failed to get valid asset ID from upload")
                    return None
                
                # Now process the newly uploaded asset
                for attempt in range(retries):
                    try:
                        # Use the asset_id (whether it's a string or object)
                        if hasattr(asset, 'id'):
                            result = self.docling.run(asset.id)
                        else:
                            result = self.docling.run(asset_id)
                        
                        if result and hasattr(result, 'data') and result.data and len(result.data.strip()) > 100:
                            print(f"✓ Successfully processed uploaded PDF (asset: {asset_id})")
                            return result.data
                        else:
                            print(f"Empty or insufficient result from uploaded PDF")
                            
                    except Exception as e:
                        print(f"Attempt {attempt + 1} failed processing uploaded asset: {e}")
                        if attempt < retries - 1:
                            time.sleep(delay)
                            
            except Exception as e:
                print(f"Error in PDF download/upload process: {e}")
            finally:
                # Clean up temporary local file
                if file_path.exists():
                    file_path.unlink()
                    print(f"Cleaned up temporary file: {file_path}")
                    
        else:
            print("Error: Must provide either asset_id or pdf_url")
                    
        return None

class RAGAgentPipeline:
    def __init__(self, docling_model_id: str, llm_model_id: str):
        self.processor = DocumentProcessor("677bee6c6eb56331f9192a91")
        try:
            self.llm = ModelFactory.get(llm_model_id)
            print(f"LLM model loaded: {self.llm}")
        except Exception as e:
            print(f"Error loading LLM model {llm_model_id}: {e}")
            raise

        self.index = None
        self.agent = None
        self.team_agent = None

    @staticmethod
    def chunk_text(text: str, max_tokens: int = 8000) -> List[str]:
        """Split text into token-aware chunks using OpenAI tokenizer."""
        enc = tiktoken.get_encoding("cl100k_base")  # Adjust based on your LLM if needed

        sentences = text.split(". ")
        chunks = []
        current_chunk = ""

        for sentence in sentences:
            candidate = current_chunk + sentence + ". "
            token_count = len(enc.encode(candidate))
            print(f"token count {token_count}")

            if token_count > max_tokens:
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "
            else:
                current_chunk = candidate

        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        print(f"chunks {chunks}")
        return chunks

    def create_index(self):
        """Create a new index for document storage"""
        name = f"RAGIndex-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
        try:
            self.index = IndexFactory.create(
                name=name,
                description="RAG Document Index",
                #embedding_model="6734c55df127847059324d9e"
                embedding_model_id="6734c55df127847059324d9e"
            )
            print(f"Index created: {self.index.id}")
        except Exception as e:
            print(f"Error creating index: {e}")
            raise

    def ingest_documents(self, documents: List[Dict]) -> List[Record]:
        """Process and ingest documents into the index"""
        records = []

        for doc in documents:
            #text = None
            print(f"\n{'='*100}")
            print(f"Processing document: {doc.get('id', 'unknown')}")
            #print(f"Extracted text type: {type(text)}")
            #print(f"Extracted text length (chars): {len(text) if isinstance(text, str) else 'N/A'}")
            text = None
            if doc.get("asset_id"):
                text = self.processor.process(asset_id=doc["asset_id"])

            if not text and doc.get("pdf_url"):
                print(f"Asset processing failed, trying PDF URL for {doc['id']}")
                text = self.processor.process(pdf_url=doc["pdf_url"])

            if not text:
                print(f"Failed to extract text from document {doc.get('id', 'unknown')}")
                continue
            
# Token-aware chunking
            chunks = self.chunk_text(text)
            print(f"✓ Chunked into {len(chunks)} chunks.")

            for i, chunk in enumerate(chunks):
                #print(f"pre upsert {chunks}")
                record_id = f"{doc['id']}_chunk_{i}"
                record = Record(
                    id=record_id,
                    value=chunk,
                    attributes={
                        "document_id": doc["id"],
                        "author": doc.get("author", "Unknown"),
                        "source": doc.get("asset_id", doc.get("pdf_url", "Unknown")),
                        "description": doc.get('description', "No description"),
                        "chunk_index": i,
                        "processed_at": datetime.now().isoformat()
                    }
                )
                records.append(record)

        if records:
            try:
                self.index.upsert(records)
                print(f"✅ Successfully upserted {len(records)} records into index.")
            except Exception as e:
                print(f"Error upserting records: {e}")
                raise
        else:
            print("No records were created — all document processing failed.")

        return records

    def create_agent(self):
        """Create and deploy the RAG agent"""
        name = f"RAGAgent-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
        instructions = """You are a document search and retrieval agent. 

Your primary responsibilities:
1. Use the search tool to find relevant information in indexed documents
2. Always search the index before responding to queries
3. If you don't know, just state that


When responding:
- Include document sources and authors when available
- Mention which document chunks or sections your answer comes from
- Be specific about what information is available in the indexed documents"""

        try:
            self.agent = AgentFactory.create(
                name=name,
                description="Government document retrieval and Q&A agent",
                instructions=instructions,
                llm=self.llm,
                tools=[AgentFactory.create_model_tool(self.index.id)]
            )
            self.agent.deploy()
            print(f"Agent created and deployed: {self.agent.id}")
        except Exception as e:
            print(f"Error creating agent: {e}")
            raise

    def create_team_agent(self):
        """Create and deploy a team agent that uses the base agent"""
        name = f"TeamAgent-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
        
        try:
            self.team_agent = TeamAgentFactory.create(
                name=name,
                agents=[self.agent],  # Pass the agent object, not ID
                tools=[],
                description="Team agent for coordinated document retrieval",
                instructions="Coordinate with the base document agent to provide comprehensive responses about government documents.",
                llm=self.llm,
                use_mentalist=True,
                inspectors=[]
            )
            self.team_agent.deploy()
            print(f"Team agent created and deployed: {self.team_agent.id}")
        except Exception as e:
            print(f"Error creating team agent: {e}")
            # Continue without team agent if creation fails
            self.team_agent = None

    def run_queries(self, queries: List[str], use_team_agent: bool = False):
        """Run test queries against the agent(s)"""
        agent_to_use = self.team_agent if (use_team_agent and self.team_agent) else self.agent
        agent_name = "Team Agent" if (use_team_agent and self.team_agent) else "Base Agent"
        
        print(f"\n{'='*50}")
        print(f"Running queries with {agent_name}")
        print(f"{'='*50}")
        
        for i, query in enumerate(queries, 1):
            print(f"\n--- Query {i}: {query} ---")
            try:
                response = agent_to_use.run(query)
                if hasattr(response, 'data') and hasattr(response.data, 'output'):
                    print("Response:", response.data.output)
                else:
                    print("Response:", str(response))
            except Exception as e:
                print(f"Error during query '{query}': {e}")
                continue

def main():
    """Main execution function with debugging"""
    try:
        # Test asset accessibility first
        print("Testing asset accessibility...")
        for doc in DOCS:
            if doc.get("asset_id"):
                try:
                    # Try multiple methods to access the asset
                    asset = None
                    methods = [
                        lambda: AssetFactory.get(asset_id=doc["asset_id"]),
                        lambda: Asset.get(doc["asset_id"]), 
                        lambda: AssetFactory.get(doc["asset_id"]),
                        lambda: Asset(id=doc["asset_id"])
                    ]
                    
                    for i, method in enumerate(methods):
                        try:
                            asset = method()
                            print(f"✓ Asset {doc['asset_id']} accessible via method {i+1}")
                            break
                        except Exception as e:
                            continue
                    
                    if not asset:
                        print(f" Asset {doc['asset_id']} not accessible with any method")
                        print(f"  You may need to upload this document first or check the asset ID")
                    
                except Exception as e:
                    print(f"✗ Asset {doc['asset_id']} error: {e}")
        
        # Initialize the RAG pipeline
        print("\nInitializing RAG Agent Pipeline...")
        rag = RAGAgentPipeline(
            docling_model_id="671bdab16eb563d378196f51",
            llm_model_id="669a63646eb56306647e1091"
        )
        
        # Create index
        print("\nCreating index...")
        rag.create_index()
        
        # Process and ingest documents
        print(f"\n{'='*10}")
        print("\nIngesting documents...")
        records = rag.ingest_documents(DOCS)
        
        if not records:
            print("No valid records processed. This could mean:")
            print("1. Asset IDs don't exist or aren't accessible")
            print("2. Docling model isn't properly configured for asset processing")
            print("3. Documents are empty or unprocessable")
            return 1
        
        # Create base agent
        print("\nCreating base agent...")
        rag.create_agent()
        
        # Create team agent (optional)
        print("\nCreating team agent...")
        rag.create_team_agent()
        
        # Run queries with base agent
        rag.run_queries(QUERIES, use_team_agent=False)
        
        # Run queries with team agent if available
        if rag.team_agent:
            rag.run_queries(QUERIES, use_team_agent=True)
        
        print(f"\n{'='*50}")
        print("Pipeline completed successfully!")
        print(f"Base Agent ID: {rag.agent.id}")
        if rag.team_agent:
            print(f"Team Agent ID: {rag.team_agent.id}")
        print(f"Index ID: {rag.index.id}")
        print(f"{'='*50}")
        
        return 0
        
    except Exception as e:
        print(f"Pipeline failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())