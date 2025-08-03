"""
PostgreSQL Database Models for HackRX 6.0 Project
Handles document metadata, processing history, and user sessions
"""

from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text, Float, Boolean, JSON, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()

# Database configuration
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://username:password@localhost:5432/hackrx_db")

Base = declarative_base()

class DocumentMetadata(Base):
    """Store document processing metadata"""
    __tablename__ = "document_metadata"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    document_url = Column(String(2048), nullable=False)
    document_name = Column(String(500))
    file_size = Column(Integer)  # in bytes
    mime_type = Column(String(100))
    upload_timestamp = Column(DateTime, default=datetime.utcnow)
    processing_status = Column(String(50), default="pending")  # pending, processing, completed, failed
    chunks_created = Column(Integer, default=0)
    processing_time = Column(Float)  # in seconds
    error_message = Column(Text, nullable=True)
    
    def to_dict(self):
        return {
            "id": self.id,
            "document_url": self.document_url,
            "document_name": self.document_name,
            "file_size": self.file_size,
            "mime_type": self.mime_type,
            "upload_timestamp": self.upload_timestamp.isoformat() if self.upload_timestamp else None,
            "processing_status": self.processing_status,
            "chunks_created": self.chunks_created,
            "processing_time": self.processing_time,
            "error_message": self.error_message
        }

class QueryLog(Base):
    """Store question-answer sessions"""
    __tablename__ = "query_logs"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    document_id = Column(Integer, nullable=False)  # Reference to DocumentMetadata.id
    questions = Column(JSON)  # Array of questions
    answers = Column(JSON)  # Array of corresponding answers
    query_timestamp = Column(DateTime, default=datetime.utcnow)
    response_time = Column(Float)  # in seconds
    user_session = Column(String(255))  # Optional session tracking
    confidence_scores = Column(JSON, nullable=True)  # Array of confidence scores
    
    def to_dict(self):
        return {
            "id": self.id,
            "document_id": self.document_id,
            "questions": self.questions,
            "answers": self.answers,
            "query_timestamp": self.query_timestamp.isoformat() if self.query_timestamp else None,
            "response_time": self.response_time,
            "user_session": self.user_session,
            "confidence_scores": self.confidence_scores
        }

class ProcessingStats(Base):
    """Store overall system processing statistics"""
    __tablename__ = "processing_stats"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    total_documents_processed = Column(Integer, default=0)
    total_queries_answered = Column(Integer, default=0)
    average_processing_time = Column(Float, default=0.0)
    average_response_time = Column(Float, default=0.0)
    last_updated = Column(DateTime, default=datetime.utcnow)
    system_status = Column(String(50), default="active")
    
    def to_dict(self):
        return {
            "id": self.id,
            "total_documents_processed": self.total_documents_processed,
            "total_queries_answered": self.total_queries_answered,
            "average_processing_time": self.average_processing_time,
            "average_response_time": self.average_response_time,
            "last_updated": self.last_updated.isoformat() if self.last_updated else None,
            "system_status": self.system_status
        }

# Database engine and session with SSL configuration
engine_args = {
    "echo": False,
    "pool_pre_ping": True,  # Enable connection health checks
    "pool_recycle": 300,    # Recycle connections every 5 minutes
}

# Add SSL configuration for PostgreSQL if using remote database
if "localhost" not in DATABASE_URL and "127.0.0.1" not in DATABASE_URL:
    engine_args["connect_args"] = {
        "sslmode": "prefer",  # Prefer SSL but allow non-SSL
        "sslcert": None,
        "sslkey": None,
        "sslrootcert": None,
        "application_name": "hackrx_app"
    }

engine = create_engine(DATABASE_URL, **engine_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def create_tables():
    """Create all database tables"""
    try:
        Base.metadata.create_all(bind=engine)
        print("✅ PostgreSQL tables created successfully")
        return True
    except Exception as e:
        print(f"❌ Error creating PostgreSQL tables: {e}")
        return False

def get_db_session():
    """Get database session"""
    db = SessionLocal()
    try:
        return db
    except Exception as e:
        print(f"❌ Database connection error: {e}")
        return None

def test_connection():
    """Test PostgreSQL connection with enhanced error handling"""
    try:
        db = get_db_session()
        if db:
            # Test query with proper SQLAlchemy 2.0 syntax
            result = db.execute(text("SELECT 1 as test")).fetchone()
            db.close()
            print("✅ PostgreSQL connection successful")
            return True
        else:
            print("❌ Failed to get database session")
            return False
    except Exception as e:
        error_str = str(e).lower()
        if "ssl connection" in error_str:
            print(f"⚠️ PostgreSQL SSL connection issue: {e}")
            print("💡 Tip: Check SSL configuration or use local PostgreSQL")
        elif "textual sql expression" in error_str:
            print(f"⚠️ PostgreSQL SQLAlchemy version issue: {e}")
            print("💡 Database connection works but SQL syntax needs update")
        else:
            print(f"❌ PostgreSQL connection failed: {e}")
        return False

if __name__ == "__main__":
    print("🗄️ Testing PostgreSQL Configuration...")
    
    # Test connection
    if test_connection():
        print("🔧 Creating database tables...")
        create_tables()
        print("🎉 PostgreSQL setup complete!")
    else:
        print("❌ PostgreSQL setup failed. Please check your configuration.")
        print("📋 Required environment variables:")
        print("  - DATABASE_URL or individual POSTGRES_* variables")
        print("  - Make sure PostgreSQL is running")
