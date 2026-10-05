"""
VECTOR STORE MODULE
Stores embeddings in PostgreSQL (pgvector) and performs similarity search

CORE CONCEPT:
- Regular SQL: SELECT * FROM faqs WHERE question LIKE '%password%'
- Vector Search: Find FAQs with SIMILAR MEANING (even if words don't match)

HOW IT WORKS:
1. User asks: "I forgot my login credentials"
2. Convert question to embedding (1536 numbers) - Day 3 code
3. Compare with all stored FAQ embeddings using COSINE SIMILARITY
4. Return top-3 most similar FAQs (highest similarity score)
"""

import psycopg2
from psycopg2.extras import execute_values
import pandas as pd
import numpy as np
import ast
from config import DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD


def get_connection():
    """Connect to PostgreSQL."""
    return psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD
    )


def enable_pgvector_extension(conn):
    """
    Step 1: Enable pgvector extension.
    Must run once per database.
    """
    cursor = conn.cursor()
    
    print("📦 Enabling pgvector extension...")
    cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
    conn.commit()
    
    # Verify
    cursor.execute("SELECT extname FROM pg_extension WHERE extname = 'vector'")
    result = cursor.fetchone()
    
    if result:
        print(f"   ✅ pgvector enabled successfully")
    else:
        print("   ❌ Failed to enable pgvector")
    
    cursor.close()
    return result is not None


def create_faq_table_with_embeddings(conn):
    """
    Step 2: Create table to store FAQs WITH their vector embeddings.
    
    Table Structure:
    - faq_id: Primary key (matches bank_faqs.csv)
    - question: Original question text
    - answer: Original answer text  
    - category: FAQ category (Loan Application, Eligibility, etc.)
    - embedding: THE VECTOR (1536 dimensions) - this is the magic column!
    - created_at: Timestamp
    """
    cursor = conn.cursor()
    
    print("\n📊 Creating FAQ table with vector column...")
    
    cursor.execute("""
        DROP TABLE IF EXISTS faqs_with_embeddings;
        
        CREATE TABLE faqs_with_embeddings (
            faq_id INTEGER PRIMARY KEY,
            question TEXT NOT NULL,
            answer TEXT,
            category VARCHAR(50),
            embedding VECTOR(1536),  -- pgvector type! Stores 1536 numbers
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    conn.commit()
    cursor.close()
    print("   ✅ Table 'faqs_with_embeddings' created with VECTOR(1536) column")


def load_embeddings_from_csv(csv_path):
    """
    Step 3: Read embeddings from CSV file (created on Day 3).
    
    The CSV has:
    - faq_id: Integer
    - embedding: String representation of list (e.g., "[0.01, -0.02, ...]")
    
    We need to convert string → actual list of floats
    """
    print(f"\n📂 Loading embeddings from {csv_path}...")
    
    df = pd.read_csv(csv_path)
    
    # Convert embedding strings to actual lists
    # CSV stores: "[0.001, -0.002, ...]" → Python list: [0.001, -0.002, ...]
    df['embedding'] = df['embedding'].apply(lambda x: ast.literal_eval(x))
    
    print(f"   ✅ Loaded {len(df)} embeddings")
    print(f"   Sample: FAQ ID {df.iloc[0]['faq_id']}, Embedding length: {len(df.iloc[0]['embedding'])}")
    
    return df


def load_faqs_and_embeddings_to_db(conn, embeddings_df, faqs_df):
    """
    Step 4: Insert FAQs + embeddings into PostgreSQL.
    
    Joins:
    - embeddings_df (from faq_embeddings.csv): has faq_id + embedding
    - faqs_df (from bank_faqs.csv): has faq_id + question + answer + category
    
    Merges on faq_id, inserts into faqs_with_embeddings table
    """
    cursor = conn.cursor()
    
    print("\n💾 Inserting FAQs with embeddings into database...")
    
    # Merge embeddings with original FAQ data
    merged = embeddings_df.merge(faqs_df, left_on='faq_id', right_on='id', how='inner')
    
    # Prepare data for insert
    data_to_insert = []
    for _, row in merged.iterrows():
        data_to_insert.append((
            row['faq_id'],
            row['question'],
            row['answer'],
            row['category'],
            row['embedding']  # This is a list of 1536 floats
        ))
    
    # Bulk insert using execute_values (fast!)
    execute_values(cursor, """
        INSERT INTO faqs_with_embeddings (faq_id, question, answer, category, embedding)
        VALUES %s
    """, data_to_insert)
    
    conn.commit()
    
    print(f"   ✅ Inserted {len(data_to_insert)} FAQs with embeddings")
    cursor.close()


def cosine_similarity(vec_a, vec_b):
    """
    Calculate cosine similarity between two vectors.
    
    FORMULA:
    cosine_sim(a, b) = (a · b) / (||a|| * ||b||)
    
    WHERE:
    - a · b = dot product (sum of element-wise multiplication)
    - ||a|| = magnitude (square root of sum of squares)
    
    RETURNS:
    - 1.0 = identical meaning
    - 0.0 = completely unrelated
    - -1.0 = opposite meaning
    
    WHY COSINE (not Euclidean distance)?
    - Measures ANGLE between vectors, not distance
    - Better for text (length of document doesn't matter, only direction/meaning)
    """
    # Convert to numpy arrays for fast math
    a = np.array(vec_a, dtype=np.float32)

    if isinstance(vec_b, str):
        vec_b = ast.literal_eval(vec_b)

    b = np.array(vec_b, dtype=np.float32)
    
    # Dot product
    dot_product = np.dot(a, b)
    
    # Magnitudes
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    
    # Avoid division by zero
    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    # Cosine similarity
    similarity = dot_product / (norm_a * norm_b)
    
    return float(similarity)


def search_similar_faqs(conn, query_embedding, top_k=3):
    """
    Step 5: SEARCH - Find most similar FAQs to a query.
    
    This is the CORE function of vector databases!
    
    PROCESS:
    1. Take query embedding (user's question converted to 1536 numbers)
    2. Compare with ALL stored FAQ embeddings using cosine similarity
    3. Sort by similarity score (descending)
    4. Return top-k results
    
    Args:
        conn: Database connection
        query_embedding: List of 1536 floats (user's question embedding)
        top_k: How many results to return (default: 3)
    
    Returns:
        List of dicts: [{'faq_id', 'question', 'answer', 'score', 'category'}, ...]
    """
    cursor = conn.cursor()
    
    # Fetch ALL FAQs with embeddings from database
    cursor.execute("""
        SELECT faq_id, question, answer, category, embedding 
        FROM faqs_with_embeddings
    """)
    
    all_faqs = cursor.fetchall()
    cursor.close()
    
    # Calculate similarity for each FAQ
    results = []
    for faq in all_faqs:
        faq_id, question, answer, category, stored_embedding = faq
        
        # Calculate cosine similarity
        score = cosine_similarity(query_embedding, stored_embedding)
        
        results.append({
            'faq_id': faq_id,
            'question': question,
            'answer': answer,
            'category': category,
            'score': round(score, 4)  # Round to 4 decimal places
        })
    
    # Sort by score (descending - highest similarity first)
    results.sort(key=lambda x: x['score'], reverse=True)
    
    # Return only top-k
    return results[:top_k]


def test_vector_search(conn):
    """
    Test the vector search with a sample query.
    Uses OpenAI to embed the query, then searches.
    """
    import openai
    from config import OPENAI_API_KEY, EMBEDDING_MODEL
    
    openai.api_key = OPENAI_API_KEY
    
    print("\n" + "=" * 60)
    print("🔍 TESTING VECTOR SEARCH")
    print("=" * 60)
    
    # Test queries (realistic banking questions)
    test_queries = [
        "I forgot my password, how do I reset it?",
        "What documents do I need for loan?",
        "How much interest will I pay?"
    ]
    
    for query in test_queries:
        print(f"\n❓ Query: '{query}'")
        print("-" * 50)
        
        # Step 1: Generate embedding for query (same as Day 3)
        response = openai.embeddings.create(
            model=EMBEDDING_MODEL,
            input=query
        )
        query_embedding = response.data[0].embedding
        
        # Step 2: Search
        results = search_similar_faqs(conn, query_embedding, top_k=3)
        
        # Step 3: Display results
        print(f"\n📋 Top {len(results)} Results:")
        for i, result in enumerate(results, 1):
            print(f"\n   #{i} [Score: {result['score']:.4f}] ({result['category']})")
            print(f"   Q: {result['question'][:80]}...")
            print(f"   A: {result['answer'][:100]}...")


# ============================================
# MAIN EXECUTION
# ============================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("FIN-SIGHT AI: VECTOR STORE SETUP")
    print("=" * 60)
    
    # Connect to database
    conn = get_connection()
    print("✅ Connected to PostgreSQL")
    
    # Step 1: Enable pgvector
    if not enable_pgvector_extension(conn):
        print("❌ Cannot continue without pgvector. Please install it.")
        exit(1)
    
    # Step 2: Create table
    create_faq_table_with_embeddings(conn)
    
    # Step 3: Load embeddings from CSV (Day 3 output)
    embeddings_df = load_embeddings_from_csv('data/faq_embeddings.csv')
    
    # Step 4: Load original FAQs
    faqs_df = pd.read_csv('data/bank_faqs.csv')
    print(f"\n📂 Loaded {len(faqs_df)} original FAQs")
    
    # Step 5: Insert into database
    load_faqs_and_embeddings_to_db(conn, embeddings_df, faqs_df)
    
    # Step 6: Test vector search!
    test_vector_search(conn)
    
    # Close connection
    conn.close()
    
    print("\n" + "=" * 60)
    print("✅ DAY 4 COMPLETE: Vector database ready!")
    print("=" * 60)
    print("\n📌 NEXT STEP (Day 5): Build RAG pipeline (search + LLM generation)")