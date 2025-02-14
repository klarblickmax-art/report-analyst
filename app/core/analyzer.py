from pathlib import Path
from typing import List, Dict, Optional, AsyncGenerator, Any
import os
from dotenv import load_dotenv
import shutil
import yaml
import logging
import sys
import json
import hashlib
import pickle

from langchain_openai import ChatOpenAI
from llama_index.core import Document, Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from llama_index.readers.file import PyMuPDFReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.openai import OpenAI
from llama_index.core.ingestion import IngestionCache

from .prompt_manager import PromptManager
from .storage import LlamaVectorStore

# Setup logging at the top of the file
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Check for required environment variables
if not os.getenv("OPENAI_API_KEY"):
    logger.error("OPENAI_API_KEY environment variable is not set")
    raise ValueError("OPENAI_API_KEY environment variable is required")

if not os.getenv("OPENAI_ORGANIZATION"):
    logger.error("OPENAI_ORGANIZATION environment variable is not set")
    raise ValueError("OPENAI_ORGANIZATION environment variable is required")

def log_analysis_step(message: str, level: str = "info"):
    """Helper function to log analysis steps with consistent formatting"""
    log_func = getattr(logger, level)
    log_func(f"[ANALYSIS] {message}")

def compute_file_hash(file_path: str) -> str:
    """Compute SHA-256 hash of a file"""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def compute_params_hash(params: Dict) -> str:
    """Compute hash of parameters dictionary"""
    param_str = json.dumps(params, sort_keys=True)
    return hashlib.sha256(param_str.encode()).hexdigest()

class DocumentAnalyzer:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DocumentAnalyzer, cls).__new__(cls)
            # Initialize core attributes here
            cls._instance._initialized = False
            cls._instance.question_set = "tcfd"  # Set default question set
            cls._instance.questions = {}
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self.prompt_manager = PromptManager()
        # Use absolute paths for storage, relative to project root
        self.storage_path = Path(__file__).parent.parent.parent / "storage"
        self.cache_path = self.storage_path / "cache"
        self.llm_cache_path = self.storage_path / "llm_cache"
        
        # Create cache directories
        self.cache_path.mkdir(parents=True, exist_ok=True)
        self.llm_cache_path.mkdir(parents=True, exist_ok=True)
        
        log_analysis_step(f"Storage path: {self.storage_path.resolve()}", "debug")
        log_analysis_step(f"Cache path: {self.cache_path.resolve()}", "debug")
        log_analysis_step(f"LLM cache path: {self.llm_cache_path.resolve()}", "debug")
        
        # Load questions
        self.questions = self._load_questions()
        
        model_name = os.getenv("OPENAI_API_MODEL", "gpt-4-turbo-preview")
        log_analysis_step(f"Using model: {model_name}")
        
        try:
            # Initialize LLM with caching
            self.llm = OpenAI(
                model=model_name,
                api_key=os.getenv("OPENAI_API_KEY"),
                api_base=os.getenv("OPENAI_API_BASE"),
                cache_dir=str(self.llm_cache_path),
            )
            
            # Configure embeddings globally for LlamaIndex
            Settings.embed_model = OpenAIEmbedding(
                api_key=os.getenv('OPENAI_API_KEY'),
                api_base=os.getenv('OPENAI_API_BASE'),
                model_name=os.getenv('OPENAI_EMBEDDING_MODEL', 'text-embedding-ada-002'),
                embed_batch_size=100
            )
            
            # Initialize caching
            self.use_cache = True  # Default to True, can be overridden
            Settings.ingestion_cache = IngestionCache(
                cache_dir=str(self.llm_cache_path),
                cache_type="local"
            )
            
            self.text_splitter = SentenceSplitter(
                chunk_size=500,
                chunk_overlap=20
            )
            
            # Cache parameters
            self.chunk_params = {
                "chunk_size": 500,
                "chunk_overlap": 20
            }
            
            self.embedding_params = {
                "model": "text-embedding-ada-002",
                "batch_size": 100
            }
            
        except Exception as e:
            log_analysis_step(f"Error initializing OpenAI clients: {str(e)}", "error")
            raise
        
        self._initialized = True

    def _get_cache_key(self, file_path: str) -> str:
        """Generate a unique cache key for a file based on its path and content hash."""
        try:
            # Use file content hash instead of modification time
            file_hash = compute_file_hash(file_path)
            # Use a shorter hash for parameters
            params_str = f"{self.chunk_params['chunk_size']}_{self.chunk_params['chunk_overlap']}"
            return f"{Path(file_path).stem}_{file_hash[:8]}_{params_str}"
        except Exception as e:
            logger.warning(f"[ANALYSIS] Cache ERROR: Failed to generate cache key: {e}")
            # Fallback to a simple key
            return f"{Path(file_path).stem}_fallback"

    def _get_vector_store_collection_name(self, cache_key: str) -> str:
        """Generate a valid collection name from cache key."""
        # Remove any invalid characters and ensure it's not too long
        valid_name = "".join(c if c.isalnum() or c in "_-" else "_" for c in cache_key)
        return valid_name[:63]  # ChromaDB has a limit on collection name length

    def _load_chunks_cache(self, cache_key: str) -> Optional[List]:
        """Load text chunks from cache if available."""
        try:
            cache_file = self.cache_path / f"{cache_key}_chunks.json"
            if cache_file.exists():
                with open(cache_file, 'r', encoding='utf-8') as f:
                    chunk_data = json.load(f)
                # Convert to LlamaIndex Document objects
                chunks = [Document(
                    text=chunk['page_content'],  # LlamaIndex uses text instead of page_content
                    metadata=chunk['metadata']
                ) for chunk in chunk_data]
                logger.info(f"[ANALYSIS] ✓ Cache HIT: Loaded {len(chunks)} chunks from cache")
                return chunks
            logger.info("[ANALYSIS] Cache MISS: No cached chunks found")
            return None
        except Exception as e:
            logger.warning(f"[ANALYSIS] Cache ERROR: Failed to load chunks cache: {e}")
            return None

    def _save_chunks_cache(self, cache_key: str, chunks: List) -> None:
        """Save text chunks to cache."""
        try:
            cache_file = self.cache_path / f"{cache_key}_chunks.json"
            # Convert Document objects to serializable format
            serializable_chunks = [{
                'page_content': doc.text,  # Store as page_content for backward compatibility
                'metadata': doc.metadata
            } for doc in chunks]
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(serializable_chunks, f)
            logger.info(f"[ANALYSIS] ✓ Cache SAVE: Saved {len(chunks)} chunks to cache")
        except Exception as e:
            logger.warning(f"[ANALYSIS] Cache ERROR: Failed to save chunks cache: {e}")

    def _load_vector_store(self, cache_key: str, chunks: List) -> Optional[LlamaVectorStore]:
        """Load vector store from cache if available."""
        try:
            store_dir = self.cache_path / f"{cache_key}_vectors"
            
            if store_dir.exists():
                logger.info(f"[ANALYSIS] Found vector store directory at {store_dir}")
                try:
                    # Load LlamaVectorStore from local files
                    vector_store = LlamaVectorStore(store_dir)
                    # Try to load the store - this will verify if it's valid
                    if vector_store.load():
                        logger.info(f"[ANALYSIS] ✓ Cache HIT: Loaded vector store from cache")
                        return vector_store
                except Exception as inner_e:
                    logger.error(f"[ANALYSIS] Failed to load existing vector store: {inner_e}", exc_info=True)
                    
            logger.info("[ANALYSIS] Cache MISS: No cached vector store found")
            return None
        except Exception as e:
            logger.warning(f"[ANALYSIS] Cache ERROR: Failed to load vector store cache: {e}")
            logger.debug(f"Full vector store cache error: {str(e)}", exc_info=True)
            return None

    async def score_chunk_relevance(self, question: str, chunk_text: str) -> float:
        """Score the relevance of a chunk to a question using LLM."""
        if not self.use_cache:
            Settings.ingestion_cache = None
            
        log_analysis_step(f"Computing relevance score for chunk: {chunk_text[:100]}...")
        
        try:
            response = await self.llm.acomplete(prompt=f"""As a senior equity analyst with expertise in climate science evaluating a company's sustainability report, you are tasked with evaluating text fragments for their usefulness in answering specific TCFD questions.

Your task is to score the relevance and quality of evidence in each text fragment. Consider:

1. Specificity and Concreteness:
   - Quantitative data and specific metrics (highest value)
   - Concrete policies and procedures
   - Specific commitments with timelines
   - General statements or vague claims (lowest value)

2. Evidence Quality:
   - Verifiable data and third-party verification
   - Clear methodologies and frameworks
   - Specific examples and case studies
   - Unsubstantiated claims (lowest value)

3. Direct Relevance:
   - Direct answers to the question components
   - Related but indirect information
   - Contextual background
   - Unrelated information (lowest value)

4. Disclosure Quality:
   - Comprehensive and transparent disclosure
   - Balanced reporting (both positive and negative)
   - Clear acknowledgment of limitations
   - Potential greenwashing or selective disclosure (lowest value)

Score from 0.0 to 1.0 where:
0.0 = Not useful (generic statements, unrelated content)
0.3 = Contains relevant context but no specific evidence
0.6 = Contains useful specific information but requires additional context
1.0 = Contains critical evidence or specific details that directly answer the question

Question: {question}

Text to evaluate:
{chunk_text}

Output only the numeric score (0.0-1.0):""")
            
            score = float(response.text.strip())
            score = max(0.0, min(1.0, score))
            log_analysis_step(f"Computed relevance score: {score:.4f}")
            return score
            
        except Exception as e:
            log_analysis_step(f"Error scoring chunk relevance: {str(e)}", "error")
            return 0.0

    async def score_chunk_relevance_batch(self, question: str, chunks: List[Dict], single_call: bool = True) -> List[float]:
        """Score a batch of chunks using LLM."""
        try:
            # Format all chunks into a single prompt for batch scoring
            chunks_text = "\n\n".join([
                f"[CHUNK {i+1}]\n{chunk['text']}"
                for i, chunk in enumerate(chunks)
            ])
            
            response = await self.llm.acomplete(prompt=f"""As a senior equity analyst with expertise in climate science evaluating a company's sustainability report, you are tasked with evaluating text fragments for their usefulness in answering specific TCFD questions.

Your task is to score the relevance and quality of evidence in each text fragment. Consider:

1. Specificity and Concreteness:
   - Quantitative data and specific metrics (highest value)
   - Concrete policies and procedures
   - Specific commitments with timelines
   - General statements or vague claims (lowest value)

2. Evidence Quality:
   - Verifiable data and third-party verification
   - Clear methodologies and frameworks
   - Specific examples and case studies
   - Unsubstantiated claims (lowest value)

3. Direct Relevance:
   - Direct answers to the question components
   - Related but indirect information
   - Contextual background
   - Unrelated information (lowest value)

4. Disclosure Quality:
   - Comprehensive and transparent disclosure
   - Balanced reporting (both positive and negative)
   - Clear acknowledgment of limitations
   - Potential greenwashing or selective disclosure (lowest value)

For each chunk marked [CHUNK X], provide a score from 0.0 to 1.0 where:
0.0 = Not useful (generic statements, unrelated content)
0.3 = Contains relevant context but no specific evidence
0.6 = Contains useful specific information but requires additional context
1.0 = Contains critical evidence or specific details that directly answer the question

Question: {question}

Text fragments to evaluate:
{chunks_text}

Output only the scores in a JSON array, like: [0.8, 0.3, 0.0]""")

            scores_text = response.text.strip()
            
            # Extract JSON array from response
            start_idx = scores_text.find('[')
            end_idx = scores_text.rfind(']') + 1
            if start_idx >= 0 and end_idx > start_idx:
                scores_text = scores_text[start_idx:end_idx]
                scores = json.loads(scores_text)
                
                # Validate scores
                if not isinstance(scores, list) or len(scores) != len(chunks):
                    raise ValueError(f"Invalid scores format: expected {len(chunks)} scores, got {len(scores) if isinstance(scores, list) else 'non-list'}")
                
                # Ensure all scores are floats in [0,1]
                scores = [max(0.0, min(1.0, float(score))) for score in scores]
                return scores
            else:
                raise ValueError("No valid JSON array found in response")
                
        except Exception as e:
            log_analysis_step(f"Error in batch scoring: {str(e)}", "error")
            # Return default scores on error
            return [0.0] * len(chunks)

    async def process_document(self, file_path: str, selected_numbers: List[int], use_llm_scoring: bool = False, single_call: bool = True) -> AsyncGenerator[Dict, None]:
        """Process document and analyze questions.
        
        Args:
            file_path (str): Path to the document
            selected_numbers (List[int]): List of question numbers to analyze
            use_llm_scoring (bool): Whether to use LLM for scoring chunk relevance
            single_call (bool): If True, score all chunks in one LLM call
        """
        log_analysis_step(f"Starting document processing: {file_path}")
        log_analysis_step(f"Processing questions: {selected_numbers}")
        log_analysis_step(f"LLM scoring enabled: {use_llm_scoring}")
        
        try:
            # Initial status
            yield {"status": "Starting analysis..."}
            
            # Generate cache key
            cache_key = self._get_cache_key(file_path)
            log_analysis_step(f"Using cache key: {cache_key}")
            
            # Get chunks with caching
            log_analysis_step("Checking cache for document chunks...")
            yield {"status": "Loading and chunking document..."}
            
            chunks = self._load_chunks_cache(cache_key)
            if chunks is None:
                # If not in cache, load and process the document
                log_analysis_step("Building new document chunks...")
                reader = PyMuPDFReader()
                docs = reader.load(file_path=file_path)
                # Convert the documents to text and create new Document objects
                text_chunks = []
                for doc in docs:
                    nodes = self.text_splitter.split_text(doc.text)
                    text_chunks.extend([
                        Document(text=chunk, metadata=doc.metadata)
                        for chunk in nodes
                    ])
                chunks = text_chunks
                self._save_chunks_cache(cache_key, chunks)
            
            log_analysis_step(f"Using {len(chunks)} text chunks")
            yield {"status": f"✓ Using {len(chunks)} text chunks"}
            
            # Get vector store with caching
            log_analysis_step("Checking cache for vector store...")
            yield {"status": "Creating/loading vector store..."}
            
            store_dir = self.cache_path / f"{cache_key}_vectors"
            
            vectorstore = self._load_vector_store(cache_key, chunks)
            if vectorstore is None:
                log_analysis_step("Building new vector store...")
                yield {"status": "Building vector store (this may take a few minutes)..."}
                
                try:
                    # Ensure clean directory
                    if store_dir.exists():
                        shutil.rmtree(store_dir)
                    store_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Create new LlamaVectorStore and add documents
                    vectorstore = LlamaVectorStore(store_dir)
                    vectorstore.add_documents(chunks)
                    
                    # No need to verify by loading again - if add_documents succeeded, it's working
                    log_analysis_step(f"Vector store built and saved successfully")
                    yield {"status": "✓ Vector store built successfully"}
                    
                except Exception as e:
                    error_msg = f"Error building vector store: {str(e)}"
                    log_analysis_step(error_msg, "error")
                    logger.error(f"Full error: {str(e)}", exc_info=True)
                    yield {"error": error_msg}
                    return
            
            log_analysis_step("Vector store ready")
            yield {"status": "✓ Vector store ready"}
            
            # Process each question
            for question_number in selected_numbers:
                try:
                    question_data = self.get_question_by_number(question_number)
                    if not question_data:
                        continue
                    
                    log_analysis_step(f"Processing question {question_number} (key: {question_data['text']})")
                    yield {"status": f"Analyzing question {question_number}"}
                    
                    try:
                        # Get relevant context using TOP_K=20
                        docs_and_scores = vectorstore.similarity_search(question_data['text'], k=20)
                        docs = [doc for doc, _ in docs_and_scores]
                        scores = [score for _, score in docs_and_scores]
                        
                        context = "\n".join(d.text for d in docs)
                        log_analysis_step(f"Retrieved {len(docs)} relevant chunks for question {question_number}", "debug")
                        
                        # Prepare chunks data for passing to frontend
                        chunks_data = [{"text": d.text, "metadata": d.metadata, "relevance_score": float(s)} 
                                     for d, s in docs_and_scores]
                        
                        # Add LLM-based relevance scoring if enabled
                        if use_llm_scoring:
                            log_analysis_step(f"Starting LLM-based relevance scoring for question {question_number}")
                            yield {"status": f"Scoring chunk relevance for question {question_number}..."}
                            
                            computed_scores = await self.score_chunk_relevance_batch(
                                question_data['text'],
                                chunks_data,
                                single_call=single_call
                            )
                            
                            scored_chunks = [
                                {**chunk, "computed_score": float(score)}
                                for chunk, score in zip(chunks_data, computed_scores)
                            ]
                            
                            chunks_data = scored_chunks
                            log_analysis_step(f"Completed scoring {len(chunks_data)} chunks")
                        else:
                            log_analysis_step(f"Skipping LLM-based relevance scoring for question {question_number} (scoring is disabled)")
                        
                        # Sort chunks by computed_score
                        if use_llm_scoring:
                            chunks_data = sorted(
                                chunks_data,
                                key=lambda x: x.get('computed_score', 0.0),
                                reverse=True  # Highest scores first
                            )
                            log_analysis_step(f"Sorted {len(chunks_data)} chunks by computed_score")
                        else:
                            # Sort by vector similarity score
                            chunks_data = sorted(
                                chunks_data,
                                key=lambda x: x.get('relevance_score', 0.0),
                                reverse=True  # Highest scores first
                            )
                            log_analysis_step(f"Sorted {len(chunks_data)} chunks by relevance_score")
                        
                        # Get LLM response with sorted chunks data
                        messages = self.prompt_manager.get_analysis_messages(
                            question=question_data['text'],
                            context=context,
                            guidelines=question_data['guidelines'],
                            chunks_data=chunks_data  # Now sorted by relevance
                        )
                        
                        # Convert messages to a single prompt for LlamaIndex OpenAI
                        prompt = "\n\n".join([
                            f"{msg['role'].upper()}: {msg['content']}"
                            for msg in messages
                        ])
                        
                        result = await self.llm.acomplete(prompt=prompt)
                        log_analysis_step(f"Got LLM response for question {question_number}", "debug")
                        
                        # Extract JSON from response
                        try:
                            result_text = result.text.strip()
                            
                            # Find the first { and last } to extract just the JSON object
                            json_start = result_text.find('{')
                            json_end = result_text.rfind('}') + 1
                            
                            if json_start >= 0 and json_end > json_start:
                                result_text = result_text[json_start:json_end]
                                # Clean up any potential trailing commas
                                result_text = result_text.replace(',}', '}')
                                # Remove any potential markdown code block markers
                                result_text = result_text.replace('```json', '').replace('```', '')
                                
                                log_analysis_step(f"Extracted JSON: {result_text[:100]}...")  # Log first 100 chars
                                
                                result_json = json.loads(result_text)
                                
                                # Ensure we have all required keys
                                required_keys = ["ANSWER", "SCORE", "EVIDENCE", "GAPS", "SOURCES"]
                                missing_keys = [key for key in required_keys if key not in result_json]
                                if missing_keys:
                                    raise ValueError(f"Missing required keys in response: {missing_keys}")
                                
                                # Validate evidence format
                                for evidence in result_json["EVIDENCE"]:
                                    if not isinstance(evidence, dict) or "text" not in evidence or "chunk" not in evidence:
                                        raise ValueError("Evidence items must be dictionaries with 'text' and 'chunk' keys")
                                
                                # Update the result JSON structure
                                # Convert chunks_data to be JSON serializable
                                serializable_chunks = []
                                for chunk in chunks_data:
                                    serializable_chunk = {
                                        "text": chunk["text"],
                                        "metadata": dict(chunk["metadata"]),  # Convert metadata to dict
                                        "relevance_score": float(chunk["relevance_score"]),
                                        "computed_score": float(chunk.get("computed_score", 0.0))
                                    }
                                    serializable_chunks.append(serializable_chunk)

                                result_dict = {
                                    "ANSWER": result_json["ANSWER"],
                                    "SCORE": result_json["SCORE"],
                                    "EVIDENCE": result_json["EVIDENCE"],
                                    "GAPS": result_json["GAPS"],
                                    "SOURCES": result_json["SOURCES"],
                                    "CHUNKS": serializable_chunks
                                }

                                result = {
                                    'result': json.dumps(result_dict),
                                    'question_text': question_data['text'],
                                    'question_number': question_number
                                }

                                yield result
                            else:
                                raise ValueError("No valid JSON object found in response")
                                
                        except json.JSONDecodeError as e:
                            log_analysis_step(f"JSON decode error: {str(e)}\nResponse text: {result_text[:200]}", "error")
                            raise
                        except Exception as e:
                            log_analysis_step(f"Error processing result: {str(e)}", "error")
                            raise
                    except Exception as e:
                        error_msg = f"Error processing question {question_number}: {str(e)}"
                        log_analysis_step(error_msg, "error")
                        logger.error(f"Full error: {str(e)}", exc_info=True)
                        yield {
                            "question_number": question_number,
                            "result": json.dumps({
                                "ANSWER": f"Error: {error_msg}",
                                "SCORE": 0,
                                "EVIDENCE": [],
                                "GAPS": [error_msg],
                                "SOURCES": []
                            }),
                            "error": error_msg
                        }
                    
                except Exception as e:
                    logger.error(f"Error processing question {question_number}: {str(e)}")
                    yield {"error": f"Error processing question {question_number}: {str(e)}"}
                    
        except Exception as e:
            log_analysis_step(f"Error processing document: {str(e)}", "error")
            yield {"error": f"Failed to process document: {str(e)}"}

    def _load_questions(self) -> dict:
        """Load questions from YAML files"""
        # Look for question set file in multiple possible locations
        possible_paths = [
            Path(__file__).parent.parent / "questionsets" / f"{self.question_set}_questions.yaml",  # app/questionsets
            Path(__file__).parent.parent.parent / "questionsets" / f"{self.question_set}_questions.yaml",  # project root
            Path.cwd() / "questionsets" / f"{self.question_set}_questions.yaml"  # current working directory
        ]
        
        log_analysis_step(f"Looking for {self.question_set}_questions.yaml in:")
        for path in possible_paths:
            log_analysis_step(f"- {path.resolve()}")
        
        yaml_file = None
        for path in possible_paths:
            if path.exists():
                yaml_file = path
                log_analysis_step(f"✓ Found questions file at: {path.resolve()}")
                break
                
        if not yaml_file:
            log_analysis_step(f"Could not find questions file for {self.question_set} in any of: {[str(p) for p in possible_paths]}", "error")
            return {}
            
        try:
            with open(yaml_file, 'r') as f:
                config = yaml.safe_load(f)
                log_analysis_step(f"Loaded YAML content: {str(config)[:200]}...")  # Show first 200 chars
                
                questions = {}
                # Convert the questions list into a structured format
                for q in config.get('questions', []):
                    q_id = q.get('id', '')
                    if q_id:
                        questions[q_id] = {
                            'text': q.get('text', ''),
                            'guidelines': q.get('guidelines', '')
                        }
                        log_analysis_step(f"Added question {q_id}: {questions[q_id]['text'][:50]}...")
                
                log_analysis_step(f"✓ Loaded {len(questions)} questions for {self.question_set}")
                log_analysis_step(f"Available question IDs: {list(questions.keys())}")
                return questions
        except Exception as e:
            log_analysis_step(f"Error loading questions: {str(e)}", "error")
            logger.exception("Full error:")  # This will log the full traceback
            return {}

    def get_question_by_number(self, number: int) -> Optional[Dict]:
        """Get question data by its number."""
        try:
            # Convert number to the correct format (e.g., "tcfd_1")
            question_id = f"{self.question_set}_{number}"
            log_analysis_step(f"Looking for question ID: {question_id}")
            log_analysis_step(f"Available questions: {list(self.questions.keys())}")
            
            if question_id in self.questions:
                log_analysis_step(f"✓ Found question {question_id}")
                return self.questions[question_id]
            else:
                log_analysis_step(f"Question {number} (id: {question_id}) not found in question set {self.question_set}", "warning")
                return None
        except Exception as e:
            log_analysis_step(f"Error getting question {number}: {str(e)}", "error")
            logger.exception("Full error:")  # This will log the full traceback
            return None 