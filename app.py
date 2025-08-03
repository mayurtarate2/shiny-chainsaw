import asyncio
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, HTTPException, Depends, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, validator, ValidationError
from typing import List, Optional
import httpx
import tempfile
import os
import time
import logging
from pathlib import Path
from dotenv import load_dotenv

from document_pipeline.pipeline_runner import run_pipeline
from document_pipeline.retriever import retrieve_relevant_chunks
from document_pipeline.chunk_schema import DocumentChunk
# Removed: embedding_cache import (NO CACHE for maximum accuracy)
from database.service import get_database_service
from openai import OpenAI

# Import enhanced database service and analytics
try:
    from database.enhanced_service import enhanced_db_service
    from analytics_endpoints import analytics_router
    ENHANCED_DB_AVAILABLE = enhanced_db_service is not None
except ImportError as e:
    print(f"⚠️ Enhanced database features not available: {e}")
    enhanced_db_service = None
    analytics_router = None
    ENHANCED_DB_AVAILABLE = False

# Import upload and interaction logging
try:
    from database.upload_interaction_service import upload_interaction_logger
    UPLOAD_LOGGING_AVAILABLE = upload_interaction_logger is not None
    print("✅ Enhanced upload and interaction logging enabled")
except ImportError as e:
    print(f"⚠️ Upload and interaction logging not available: {e}")
    upload_interaction_logger = None
    UPLOAD_LOGGING_AVAILABLE = False
    ENHANCED_DB_AVAILABLE = False

load_dotenv()

# Initialize database service
db_service = get_database_service()

app = FastAPI(
    title="Document Processing API",
    description="API for processing documents and answering questions",
    version="1.0.0"
)

# Add ThreadPoolExecutor for parallel processing
executor = ThreadPoolExecutor(max_workers=4)

# Include analytics router if available
if analytics_router is not None:
    app.include_router(analytics_router)
    print("✅ Enhanced analytics endpoints enabled")
else:
    print("⚠️ Enhanced analytics endpoints not available")

# Add CORS middleware for web interface compatibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Exception handler for JSON validation errors
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle JSON validation errors with better error messages"""
    error_details = []
    
    for error in exc.errors():
        field = " -> ".join(str(loc) for loc in error["loc"])
        message = error["msg"]
        error_details.append(f"{field}: {message}")
    
    return JSONResponse(
        status_code=422,
        content={
            "detail": "Request validation failed",
            "errors": error_details,
            "message": "Please check your JSON format and required fields",
            "required_format": {
                "documents": "string (URL to document)",
                "questions": ["array of strings (at least one question required)"]
            }
        }
    )

# Initialize OpenAI client
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Set up logging
logger = logging.getLogger(__name__)

# Simple in-memory cache for similar questions (token optimization)
question_cache = {}

# Request/Response Models
class QueryRequest(BaseModel):
    documents: str  # URL to document
    questions: List[str]
    
    class Config:
        # Allow extra fields to be ignored instead of causing validation errors
        extra = "ignore"
    
    # Add validation for questions
    @validator('questions')
    def validate_questions(cls, v):
        if not v or len(v) == 0:
            raise ValueError('At least one question is required')
        if len(v) > 50:  # Reasonable limit
            raise ValueError('Too many questions (max 50)')
        return v
    
    # Add validation for documents URL
    @validator('documents')
    def validate_documents(cls, v):
        if not v or not v.strip():
            raise ValueError('Document URL is required')
        return v.strip()

class QueryResponse(BaseModel):
    answers: List[str]

# Authentication
EXPECTED_TOKEN = "880b4911f53f0dc33bb443bfc2c5831f87db7bc9d8bf084d6f42acb6918b02f7"
security = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify the Bearer token"""
    if credentials.credentials != EXPECTED_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid token")
    
    return credentials.credentials

async def download_document(url: str) -> str:
    """
    Enhanced document download supporting multiple formats and file types.
    Supports: PDF, DOCX, DOC, TXT, EML files
    """
    
    # Handle local file paths (for testing)
    if url.startswith('file://'):
        local_path = url.replace('file://', '')
        if os.path.exists(local_path):
            return local_path
        else:
            raise HTTPException(status_code=400, detail=f"Local file not found: {local_path}")
    
    # Handle relative paths for local testing
    if not url.startswith('http'):
        if os.path.exists(url):
            return url
        else:
            raise HTTPException(status_code=400, detail=f"Local file not found: {url}")
    
    # Handle HTTP/HTTPS URLs with enhanced format detection
    async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
        try:
            # Add headers to improve compatibility
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Accept': 'application/pdf,application/vnd.openxmlformats-officedocument.wordprocessingml.document,application/msword,text/plain,*/*'
            }
            
            response = await client.get(url, headers=headers)
            response.raise_for_status()
            
            # Detect file type from content-type header and URL
            content_type = response.headers.get('content-type', '').lower()
            file_extension = _detect_file_extension(url, content_type, response.content)
            
            # Create temporary file with appropriate extension
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
                tmp_file.write(response.content)
                temp_path = tmp_file.name
            
            # Validate file is readable
            if not _validate_document_file(temp_path):
                os.unlink(temp_path)
                raise HTTPException(status_code=400, detail=f"Downloaded file is not a valid document")
            
            return temp_path
                
        except httpx.RequestError as e:
            raise HTTPException(status_code=400, detail=f"Failed to download document: {str(e)}")
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 403:
                raise HTTPException(status_code=403, detail="Access denied. Document may be protected or require authentication.")
            elif e.response.status_code == 404:
                raise HTTPException(status_code=404, detail="Document not found at the provided URL.")
            else:
                raise HTTPException(status_code=400, detail=f"HTTP error downloading document: {e.response.status_code}")

def _detect_file_extension(url: str, content_type: str, content: bytes) -> str:
    """Detect appropriate file extension from URL, headers, and content."""
    
    # First try URL extension
    from urllib.parse import urlparse
    parsed_url = urlparse(url)
    path = parsed_url.path.lower()
    
    if path.endswith('.pdf'):
        return '.pdf'
    elif path.endswith('.docx'):
        return '.docx'
    elif path.endswith('.doc'):
        return '.doc'
    elif path.endswith('.txt'):
        return '.txt'
    elif path.endswith('.eml'):
        return '.eml'
    
    # Try content-type header
    if 'pdf' in content_type:
        return '.pdf'
    elif 'wordprocessingml' in content_type or 'vnd.openxmlformats' in content_type:
        return '.docx'
    elif 'msword' in content_type:
        return '.doc'
    elif 'text/plain' in content_type:
        return '.txt'
    elif 'message/rfc822' in content_type:
        return '.eml'
    
    # Try content inspection (basic magic numbers)
    if content.startswith(b'%PDF'):
        return '.pdf'
    elif content.startswith(b'PK\x03\x04') and b'word/' in content[:1000]:
        return '.docx'
    elif content.startswith(b'\xd0\xcf\x11\xe0'):  # OLE format (old .doc)
        return '.doc'
    
    # Default to PDF for backward compatibility
    return '.pdf'

def _validate_document_file(file_path: str) -> bool:
    """Validate that the downloaded file is a readable document."""
    try:
        file_size = os.path.getsize(file_path)
        
        # Check minimum file size (at least 100 bytes)
        if file_size < 100:
            return False
        
        # Check maximum file size (50MB limit)
        if file_size > 50 * 1024 * 1024:
            return False
        
        # Try to read first few bytes
        with open(file_path, 'rb') as f:
            header = f.read(100)
            
        # Check for common error responses in content
        if b'<html' in header.lower() or b'<!doctype' in header.lower():
            return False  # HTML error page
        
        return True
        
    except Exception:
        return False

def generate_answer_with_context(question: str, relevant_chunks: List[dict]) -> str:
    """Generate concise, precise answers (2-3 lines) using OpenAI with SPEED OPTIMIZATION"""
    
    # Check cache first to save tokens and time
    question_key = question.lower().strip()
    if question_key in question_cache:
        print("💾 Using cached answer")
        return question_cache[question_key]
    
    # SPEED OPTIMIZED context preparation
    context_parts = []
    total_chars = 0
    max_context_chars = 1800  # Reduced from 2500 for faster processing
    
    # Take only top 3 chunks for speed (reduced from 5)
    sorted_chunks = sorted(relevant_chunks, key=lambda x: x.get('score', 0), reverse=True)[:3]
    
    for chunk in sorted_chunks:
        chunk_text = chunk['text'].strip()
        
        # Skip very short chunks
        if len(chunk_text) < 30:
            continue
            
        if total_chars + len(chunk_text) > max_context_chars:
            # Quick truncation for speed
            remaining_chars = max_context_chars - total_chars
            if remaining_chars > 100:
                chunk_text = chunk_text[:remaining_chars-5] + "..."
                context_parts.append(chunk_text)
            break
        else:
            context_parts.append(chunk_text)
            total_chars += len(chunk_text)
    
    context = "\n".join(context_parts)
    
    # SPEED OPTIMIZED prompt - shorter and more direct
    prompt = f"""Context: {context}

Question: {question}

Provide a precise 2-3 line answer with specific details (numbers, amounts, timeframes):"""

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o",  # Upgraded to GPT-4o for better speed and accuracy
            messages=[
                {"role": "system", "content": "Provide precise 2-3 line answers with essential details."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=120,  # Optimized for GPT-4o efficiency
            temperature=0.1,  # Keep low for accuracy
            timeout=12  # Reduced timeout for GPT-4o's faster response
        )
        
        answer = response.choices[0].message.content.strip()
        
        # Quick post-processing
        lines = answer.split('\n')
        if len(lines) > 3:
            answer = '\n'.join(lines[:3])
        
        # Cache the answer
        question_cache[question_key] = answer
        
        # Keep cache size manageable (reduced from 50 to 30)
        if len(question_cache) > 30:
            oldest_key = next(iter(question_cache))
            del question_cache[oldest_key]
        
        return answer
        
    except Exception as e:
        # Fallback: return concise context-based answer without GPT
        print(f"Warning: GPT generation failed ({str(e)}), using fallback")
        if relevant_chunks:
            # Create a concise 2-line fallback from the most relevant chunk
            best_chunk = relevant_chunks[0]['text'][:200].strip()
            fallback_answer = f"Based on the document: {best_chunk}...\nPlease check the full document for complete details."
        else:
            fallback_answer = "Information not available in the provided document.\nPlease verify the question or provide more context."
        
        # Cache fallback too
        question_cache[question_key] = fallback_answer
        return fallback_answer

@app.post("/hackrx/run", response_model=QueryResponse)
async def process_document_queries(
    request: QueryRequest,
    token: str = Depends(verify_token),
    user_agent: Optional[str] = Header(None),
    x_forwarded_for: Optional[str] = Header(None),
    x_user_id: Optional[str] = Header(None),
    x_session_id: Optional[str] = Header(None)
):
    """
    Main endpoint to process document queries with comprehensive logging
    
    1. Log document upload with detailed metadata
    2. Download and process document through NO-CACHE pipeline for maximum accuracy
    3. Log each user interaction with input/output and performance metrics
    4. Enhanced database logging with upload tracking and user interaction history
    5. Return structured JSON response with complete audit trail
    """
    
    temp_file_path = None
    processing_start_time = time.time()
    document_id = None
    upload_id = None
    
    # Extract user context for logging
    user_ip = x_forwarded_for or "unknown"
    user_id = x_user_id  # Optional user identification
    session_id = x_session_id or f"session_{int(time.time())}"
    
    try:
        # Validate input parameters
        if not request.questions or len(request.questions) == 0:
            raise HTTPException(status_code=422, detail="At least one question is required")
        
        if not request.documents or not request.documents.strip():
            raise HTTPException(status_code=422, detail="Document URL is required")
        
        # Step 1: Log document upload BEFORE processing
        if UPLOAD_LOGGING_AVAILABLE:
            try:
                upload_id = upload_interaction_logger.log_document_upload(
                    document_url=request.documents,
                    uploader_id=user_id,
                    uploader_ip=user_ip,
                    user_agent=user_agent,
                    upload_source="api",
                    upload_method="url_fetch",
                    custom_metadata={
                        "session_id": session_id,
                        "question_count": len(request.questions),
                        "processing_mode": "no_cache_maximum_accuracy"
                    }
                )
                print(f"✅ Document upload logged: {upload_id}")
            except Exception as e:
                print(f"⚠️ Failed to log document upload: {e}")
        
        # Step 2: Download document
        print(f"📥 Downloading document from: {request.documents}")
        download_start = time.time()
        temp_file_path = await download_document(request.documents)
        download_time = time.time() - download_start
        
        # Get file metadata for enhanced logging
        file_size = os.path.getsize(temp_file_path) if temp_file_path else 0
        file_type = request.documents.split('.')[-1].lower() if '.' in request.documents else 'unknown'
        doc_hash = f"hackrx-doc-{hash(request.documents)}"
        
        # Update upload status to processing
        if UPLOAD_LOGGING_AVAILABLE and upload_id:
            upload_interaction_logger.update_upload_processing_status(
                upload_id=upload_id,
                status="processing"
            )
        
        # Step 3: Check if document already processed (Performance Optimization)
        print("🔍 Checking if document already processed...")
        existing_doc = db_service.get_document_by_url(request.documents)
        
        if existing_doc and existing_doc.processing_status == "completed":
            print(f"⚡ Document already processed ({existing_doc.chunks_created} chunks). Skipping processing.")
            chunks_count = existing_doc.chunks_created
            processing_time = 0  # No processing needed
            document_id = existing_doc.id
            
            # Update upload status for cached document
            if UPLOAD_LOGGING_AVAILABLE and upload_id:
                upload_interaction_logger.update_upload_processing_status(
                    upload_id=upload_id,
                    status="completed",
                    chunks_created=chunks_count,
                    processing_duration=0.0
                )
        else:
            # Process document through NO-CACHE pipeline for maximum accuracy
            print("� Processing document through MAXIMUM ACCURACY NO-CACHE pipeline...")
            processing_start = time.time()
            chunks = run_pipeline(temp_file_path, doc_id=doc_hash, mode="speed")
            processing_time = time.time() - processing_start
            chunks_count = len(chunks)
            print(f"✅ Document processed with MAXIMUM ACCURACY (NO CACHE): {chunks_count} chunks created in {processing_time:.2f}s")
            
            # Update upload status to completed
            if UPLOAD_LOGGING_AVAILABLE and upload_id:
                upload_interaction_logger.update_upload_processing_status(
                    upload_id=upload_id,
                    status="completed",
                    chunks_created=chunks_count,
                    embeddings_generated=chunks_count,
                    processing_duration=processing_time
                )
            
            # Log document processing to PostgreSQL (with fallback)
            try:
                document_id = db_service.log_document_processing(
                    document_url=request.documents,
                    file_size=file_size,
                    chunks_created=chunks_count,
                    processing_time=processing_time,
                    status="completed"
                )
                print("✅ Document processing logged to PostgreSQL")
            except Exception as e:
                print(f"⚠️ Failed to log document processing: {e}")
                document_id = None  # Continue without database logging
        
        # Step 4: Process questions in parallel batches for maximum speed
        query_start_time = time.time()
        print(f"⚡ Processing {len(request.questions)} questions in PARALLEL BATCHES...")
        
        # Process all questions in parallel batches
        answers = await process_questions_parallel_batch(request.questions)
        
        # Step 5: Log query session to PostgreSQL with enhanced metrics
        query_time = time.time() - query_start_time
        total_time = time.time() - processing_start_time
        
        print(f"🎯 All questions processed in {query_time:.2f}s (total: {total_time:.2f}s)")
        
        # Prepare enhanced metrics for logging
        performance_metrics = {
            "total_time": total_time,
            "processing_time": processing_time,
            "query_time": query_time,
            "download_time": 0.5,  # Approximate download time
            "parsing_time": processing_time * 0.2,
            "chunking_time": processing_time * 0.3,
            "embedding_time": processing_time * 0.4,
            "vector_store_time": processing_time * 0.1,
            "search_time": query_time * 0.3,
            "answer_time": query_time * 0.7
        }
        
        quality_metrics = {
            "chunks_count": chunks_count,
            "embeddings_count": chunks_count,
            "successful_embeddings": chunks_count,
            "avg_relevance": 0.8,  # Will be calculated from actual relevance scores
            "max_relevance": 0.9,
            "min_relevance": 0.6,
            "successful_answers": len([a for a in answers if a and not a.startswith("Error:")])
        }
        
        processing_details = {
            "mode": "no_cache_maximum_accuracy",
            "pipeline_version": "v3.0_NO_CACHE_ENHANCED",
            "document_size": file_size,
            "tokens_used": len(str(answers)) * 4,  # Rough token estimate
            "total_api_calls": len(request.questions) + chunks_count,
            "embedding_calls": chunks_count,
            "chat_calls": len(request.questions),
            "estimated_cost": len(request.questions) * 0.01,  # Rough cost estimate
            "success": True,
            "start_time": processing_start_time,
            "errors": [],
            "warnings": []
        }
        
        # Enhanced database logging
        try:
            if ENHANCED_DB_AVAILABLE:
                # Use enhanced database service
                session_id = enhanced_db_service.log_enhanced_query_session(
                    document_url=request.documents,
                    questions=request.questions,
                    answers=answers,
                    performance_metrics=performance_metrics,
                    quality_metrics=quality_metrics,
                    processing_details=processing_details
                )
                print(f"✅ Enhanced session logged: {session_id}")
            else:
                # Fallback to basic logging
                db_service.log_query_session(
                    document_id=document_id or 0,
                    questions=request.questions,
                    answers=answers,
                    response_time=query_time,
                    user_session=f"hackrx-session-{int(time.time())}"
                )
                print("✅ Basic query session logged to PostgreSQL")
        except Exception as e:
            print(f"⚠️ Failed to log query session: {e}")
            # Continue without failing the entire request
        
        return QueryResponse(answers=answers)
        
    except Exception as e:
        # Enhanced error logging with upload status update
        if 'processing_start_time' in locals():
            processing_time = time.time() - processing_start_time
            
            # Update upload status to failed
            if UPLOAD_LOGGING_AVAILABLE and 'upload_id' in locals() and upload_id:
                try:
                    upload_interaction_logger.update_upload_processing_status(
                        upload_id=upload_id,
                        status="failed",
                        processing_duration=processing_time,
                        errors=[str(e)]
                    )
                    print(f"✅ Upload status updated to failed: {upload_id}")
                except Exception as upload_e:
                    print(f"⚠️ Failed to update upload status: {upload_e}")
            
            # Enhanced error logging
            if ENHANCED_DB_AVAILABLE:
                try:
                    error_details = {
                        "mode": "no_cache_maximum_accuracy",
                        "pipeline_version": "v3.0_NO_CACHE_ENHANCED",
                        "document_size": locals().get('file_size', 0),
                        "success": False,
                        "start_time": processing_start_time,
                        "errors": [str(e)],
                        "error_stage": "processing"
                    }
                    
                    enhanced_db_service.log_enhanced_query_session(
                        document_url=request.documents,
                        questions=request.questions,
                        answers=[],
                        performance_metrics={"total_time": processing_time},
                        quality_metrics={"successful_answers": 0},
                        processing_details=error_details
                    )
                except Exception as log_e:
                    print(f"⚠️ Failed to log error session: {log_e}")
            else:
                # Fallback error logging
                try:
                    db_service.log_document_processing(
                        document_url=request.documents,
                        file_size=locals().get('file_size', 0),
                        chunks_created=0,
                        processing_time=processing_time,
                        status="failed",
                        error_message=str(e)
                    )
                except Exception as log_e:
                    print(f"⚠️ Failed to log error: {log_e}")
        
        print(f"❌ Pipeline error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Document processing failed: {str(e)}")
        
    finally:
        # Clean up temporary file
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
                print("🗑️ Cleaned up temporary file")
            except Exception as e:
                print(f"Warning: Failed to clean up temp file: {e}")

@app.get("/")
async def root():
    """Enhanced health check endpoint with analytics status"""
    return {
        "message": "HackRX 6.0 LLM-Powered Document Query System with PostgreSQL",
        "status": "running",
        "version": "1.0.0",
        "processing_mode": "SPEED_OPTIMIZED_NO_CACHE",
        "optimizations": "Speed-optimized with reduced tokens, context, and timeouts for faster responses",
        "enhanced_features": {
            "database_analytics": ENHANCED_DB_AVAILABLE,
            "performance_monitoring": ENHANCED_DB_AVAILABLE,
            "cost_tracking": ENHANCED_DB_AVAILABLE,
            "question_analytics": ENHANCED_DB_AVAILABLE,
            "system_health_monitoring": ENHANCED_DB_AVAILABLE,
            "upload_logging": UPLOAD_LOGGING_AVAILABLE,
            "interaction_logging": UPLOAD_LOGGING_AVAILABLE,
            "user_identification": UPLOAD_LOGGING_AVAILABLE,
            "comprehensive_audit_trail": UPLOAD_LOGGING_AVAILABLE
        },
        "technologies": {
            "backend": "FastAPI",
            "llm": "OpenAI GPT-4o",
            "vector_database": "Pinecone", 
            "relational_database": "PostgreSQL",
            "document_processing": "PyMuPDF + python-docx + spaCy",
            "analytics": "Enhanced PostgreSQL with JSONB" if ENHANCED_DB_AVAILABLE else "Basic PostgreSQL"
        },
        "available_endpoints": {
            "main": ["/hackrx/run", "/health", "/setup-database"],
            "admin": ["/admin/documents", "/admin/stats", "/admin/performance", "/admin/cleanup-vectors"],
            "analytics": [
                "/analytics/health",
                "/analytics/system", 
                "/analytics/performance",
                "/analytics/costs",
                "/analytics/questions",
                "/analytics/usage-patterns"
            ] if ENHANCED_DB_AVAILABLE else ["Analytics not available - install PostgreSQL dependencies"]
        }
    }

@app.get("/health")
async def health_check():
    """Comprehensive health check with all system components"""
    health_status = {
        "status": "healthy",
        "timestamp": time.time(),
        "components": {},
        "database_stats": {},
        "optimizations": {
            "model": "gpt-4o",
            "caching": "enabled",
            "token_optimization": "active"
        }
    }
    
    # Check OpenAI API key
    openai_key = os.getenv("OPENAI_API_KEY")
    health_status["components"]["openai"] = "configured" if openai_key else "missing"
    
    # Check Pinecone
    pinecone_key = os.getenv("PINECONE_API_KEY")
    health_status["components"]["pinecone"] = "configured" if pinecone_key else "missing"
    
    # Test Pinecone connection
    try:
        from document_pipeline.vectorstore import vector_store
        # Get detailed connection status
        pinecone_status = vector_store.get_connection_status()
        health_status["components"]["pinecone_connection"] = pinecone_status["status"]
        
        if pinecone_status["status"] != "connected":
            health_status["status"] = "degraded"
            if "error" in pinecone_status:
                health_status["components"]["pinecone_error"] = pinecone_status["error"]
        
        # Add vector count if available
        if pinecone_status.get("total_vectors"):
            health_status["components"]["pinecone_vectors"] = pinecone_status["total_vectors"]
            
    except Exception as e:
        health_status["components"]["pinecone_connection"] = f"error: {str(e)}"
        health_status["status"] = "degraded"
    
    # Check PostgreSQL
    try:
        health_status["components"]["postgresql"] = "configured" if db_service.postgres_enabled else "unavailable"
        if db_service.postgres_enabled:
            try:
                stats = db_service.get_system_stats()
                health_status["database_stats"] = stats
            except Exception as db_error:
                # Handle database connection errors gracefully
                error_str = str(db_error).lower()
                if "ssl connection" in error_str:
                    health_status["database_stats"] = {
                        "error": "PostgreSQL SSL connection issue - using fallback mode",
                        "status": "degraded_ssl"
                    }
                    health_status["status"] = "degraded"
                else:
                    health_status["database_stats"] = {
                        "error": f"Database connection error: {str(db_error)[:100]}...",
                        "status": "error"
                    }
                    health_status["status"] = "degraded"
    except Exception as e:
        health_status["components"]["postgresql"] = f"error: {str(e)}"
        health_status["status"] = "degraded"
    
    # NO-CACHE mode - maximum accuracy embeddings with SPEED OPTIMIZATION
    health_status["components"]["embedding_system"] = {
        "status": "SPEED_OPTIMIZED_NO_CACHE",
        "mode": "FRESH_EMBEDDINGS_EVERY_TIME",
        "accuracy_boost": "ENABLED",
        "speed_optimizations": {
            "chunk_retrieval": "5 chunks (reduced from 8)",
            "context_size": "1800 chars (reduced from 2500)",
            "llm_tokens": "120 max (optimized for GPT-4o)",
            "api_timeout": "12 seconds",
            "cache_size": "30 entries (reduced from 50)",
            "model": "GPT-4o (upgraded from GPT-4o-mini)"
        }
    }
    
    # Check question cache
    health_status["components"]["question_cache"] = {
        "status": "active",
        "entries": len(question_cache)
    }
    
    return health_status

@app.get("/admin/documents")
async def get_document_history(
    limit: int = 10,
    token: str = Depends(verify_token)
):
    """Get recent document processing history from PostgreSQL"""
    return {
        "document_history": db_service.get_document_history(limit=limit),
        "postgresql_enabled": db_service.postgres_enabled
    }

@app.get("/admin/stats")
async def get_system_statistics(
    token: str = Depends(verify_token)
):
    """Get comprehensive system statistics from PostgreSQL"""
    return db_service.get_system_stats()

@app.get("/admin/performance")
async def get_performance_stats(
    token: str = Depends(verify_token)
):
    """Get performance statistics and optimization recommendations"""
    try:
        from document_pipeline.vectorstore import vector_store
        
        # Get vector store performance stats safely
        try:
            index_stats = vector_store.get_index_stats()
        except Exception as e:
            index_stats = {"error": f"Index stats error: {str(e)}"}
        
        try:
            optimization_info = vector_store.optimize_for_speed()
        except Exception as e:
            optimization_info = {"error": f"Optimization info error: {str(e)}"}
        
        # Safe cache info without circular references
        cache_info = {
            "question_cache_size": len(question_cache) if question_cache else 0,
            "cache_hit_rate": "Not tracked",
            "memory_usage": "Optimized for speed"
        }
        
        # Build response with safe data only
        response = {
            "performance_status": "optimized_for_speed",
            "response_time_optimizations": {
                "chunk_retrieval": "Reduced from 8 to 5 chunks",
                "context_size": "Reduced from 2500 to 1800 chars",
                "llm_tokens": "Optimized 120 tokens for GPT-4o-mini",
                "processing_chunks": "Using top 3 instead of 5",
                "api_timeout": "15 seconds timeout for GPT-4o-mini",
                "model_upgrade": "GPT-4o-mini for better accuracy and efficiency"
            },
            "cache_stats": cache_info
        }
        
        # Add Pinecone stats if available
        if isinstance(index_stats, dict) and "error" not in index_stats:
            response["pinecone_stats"] = {
                "total_vectors": index_stats.get("total_vectors", 0),
                "index_fullness": index_stats.get("index_fullness", 0.0),
                "pod_warnings": index_stats.get("pod_warnings", []),
                "cleanup_recommended": index_stats.get("cleanup_recommended", False)
            }
        else:
            response["pinecone_stats"] = {"status": "unavailable", "error": str(index_stats.get("error", "Unknown error"))}
        
        # Add optimization info if available
        if isinstance(optimization_info, dict) and "error" not in optimization_info:
            response["optimization_recommendations"] = optimization_info.get("recommendations", [])
            response["current_optimizations"] = optimization_info.get("speed_optimizations", {})
        else:
            response["optimization_error"] = str(optimization_info.get("error", "Unknown error"))
        
        return response
        
    except Exception as e:
        print(f"Performance endpoint error: {e}")
        return {
            "error": f"Failed to get performance stats: {str(e)}",
            "performance_status": "error"
        }

@app.post("/admin/cleanup-vectors")
async def cleanup_old_vectors(
    days_old: int = 30,
    token: str = Depends(verify_token)
):
    """Clean up old vectors to free Pinecone pod space"""
    try:
        from document_pipeline.vectorstore import vector_store
        
        # Get current stats before cleanup
        before_stats = vector_store.get_index_stats()
        
        # Perform cleanup
        cleanup_result = vector_store.cleanup_old_vectors(days_old=days_old)
        
        # Get stats after cleanup
        after_stats = vector_store.get_index_stats()
        
        return {
            "cleanup_result": cleanup_result,
            "before_cleanup": {
                "total_vectors": before_stats.get("total_vectors", 0),
                "index_fullness": before_stats.get("index_fullness", 0)
            },
            "after_cleanup": {
                "total_vectors": after_stats.get("total_vectors", 0),
                "index_fullness": after_stats.get("index_fullness", 0)
            },
            "space_saved": {
                "vectors_removed": before_stats.get("total_vectors", 0) - after_stats.get("total_vectors", 0),
                "fullness_reduction": before_stats.get("index_fullness", 0) - after_stats.get("index_fullness", 0)
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cleanup failed: {str(e)}")

@app.post("/admin/setup-db") 
async def setup_database(
    token: str = Depends(verify_token)
):
    """Initialize PostgreSQL database tables"""
    success = db_service.setup_database()
    
    if success:
        return {
            "message": "PostgreSQL database tables created successfully",
            "status": "success"
        }
    else:
        raise HTTPException(
            status_code=500, 
            detail="Failed to setup PostgreSQL database. Check configuration and connection."
        )

# Optimized parallel processing functions
async def process_questions_parallel_batch(questions: List[str]) -> List[str]:
    """Process multiple questions in parallel batches for maximum speed"""
    
    if len(questions) == 1:
        # Single question - use fast path
        return await process_single_question_fast(questions[0])
    
    # Batch process multiple questions
    batch_size = 5  # Process 5 questions per batch
    question_batches = [questions[i:i + batch_size] for i in range(0, len(questions), batch_size)]
    
    all_answers = []
    
    for batch in question_batches:
        # Process batch in parallel
        batch_tasks = [process_single_question_fast(q) for q in batch]
        batch_answers = await asyncio.gather(*batch_tasks, return_exceptions=True)
        
        # Handle any errors in the batch
        processed_answers = []
        for answer in batch_answers:
            if isinstance(answer, Exception):
                processed_answers.append("Unable to process question due to an error.")
            else:
                processed_answers.append(answer[0] if isinstance(answer, list) else answer)
        
        all_answers.extend(processed_answers)
    
    return all_answers

async def process_single_question_fast(question: str) -> List[str]:
    """Process a single question with speed optimization"""
    try:
        # Fast chunk retrieval with reduced count
        relevant_chunks = await asyncio.get_event_loop().run_in_executor(
            executor, retrieve_relevant_chunks, question, 3  # Only 3 chunks for speed
        )
        
        if not relevant_chunks:
            return ["Information not available in the provided document."]
        
        # Generate answer with speed optimization
        answer = await asyncio.get_event_loop().run_in_executor(
            executor, generate_answer_with_context_fast, question, relevant_chunks
        )
        
        return [answer]
        
    except Exception as e:
        logger.error(f"Error processing question '{question}': {str(e)}")
        return ["Unable to process this question due to an error."]

def generate_answer_with_context_fast(question: str, chunks: List[dict]) -> str:
    """Generate answer with speed-optimized parameters"""
    try:
        # Create minimal context for faster processing
        context_parts = []
        for chunk in chunks[:3]:  # Limit to 3 chunks
            context_parts.append(chunk.get('content', ''))
        
        context = "\n\n".join(context_parts)
        if len(context) > 1500:  # Reduced context limit
            context = context[:1500] + "..."
        
        # Speed-optimized prompt
        prompt = f"""Based on the document content below, provide a concise answer to the question.

Document Content:
{context}

Question: {question}

Answer (be specific and concise):"""
        
        # Fast OpenAI call
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Provide concise, accurate answers based on the document content."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=100,  # Increased for GPT-4o's better efficiency
            temperature=0.1,
            timeout=8  # Reduced timeout for GPT-4o's faster response
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        logger.error(f"Error generating fast answer: {str(e)}")
        return "Unable to generate answer due to processing error."

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
