"""
TEST SUITE FOR FIN-SIGHT AI
Covers: Database connection, Embeddings, Vector Search, RAG Pipeline
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
from config import DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD
from embedding_generator import generate_single_embedding, generate_batch_embeddings
from vector_store import cosine_similarity
from rag_pipeline import ask_question


class TestDatabaseConnection:
    """Test PostgreSQL connectivity."""
    
    def test_connection(self):
        import psycopg2
        conn = psycopg2.connect(
            host=DB_HOST, port=DB_PORT,
            dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD
        )
        assert conn.closed == 0  # 0 = open
        conn.close()


class TestEmbeddingGeneration:
    """Test OpenAI embedding API."""
    
    def test_single_embedding(self):
        result = generate_single_embedding("Hello world")
        assert result is not None
        assert len(result) == 1536
        assert isinstance(result[0], float)
    
    def test_batch_embeddings(self):
        texts = ["Hello", "World", "Test"]
        results = generate_batch_embeddings(texts)
        assert len(results) == 3
        assert all(len(r) == 1536 for r in results if r is not None)


class TestVectorSearch:
    """Test cosine similarity and vector operations."""
    
    def test_identical_vectors(self):
        v1 = [1.0, 0.0, 0.0]
        v2 = [1.0, 0.0, 0.0]
        score = cosine_similarity(v1, v2)
        assert abs(score - 1.0) < 0.001  # Should be ~1.0
    
    def test_orthogonal_vectors(self):
        v1 = [1.0, 0.0, 0.0]
        v2 = [0.0, 1.0, 0.0]
        score = cosine_similarity(v1, v2)
        assert abs(score) < 0.001  # Should be ~0.0
    
    def test_opposite_vectors(self):
        v1 = [1.0, 0.0, 0.0]
        v2 = [-1.0, 0.0, 0.0]
        score = cosine_similarity(v1, v2)
        assert abs(score - (-1.0)) < 0.001  # Should be ~-1.0


class TestRAGPipeline:
    """Test end-to-end RAG pipeline."""
    
    def test_loan_documents_question(self):
        result = ask_question("What documents do I need for a loan?")
        assert result['success'] == True
        assert 'document' in result['answer'].lower() or 'id proof' in result['answer'].lower()
        assert len(result['sources']) > 0
        assert result['timing']['total_pipeline'] < 30  # Should complete in <30s
    
    def test_interest_rate_question(self):
        result = ask_question("What is the interest rate?")
        assert result['success'] == True
        assert '%' in result['answer'] or 'percent' in result['answer'].lower()
    
    def test_unknown_question(self):
        result = ask_question("What is the meaning of life?")
        assert result['success'] == True
        # Should admit it doesn't know or give generic answer
        assert len(result['answer']) > 0


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("RUNNING TEST SUITE")
    print("=" * 60 + "\n")
    
    pytest.main([__file__, "-v", "--tb=short"])