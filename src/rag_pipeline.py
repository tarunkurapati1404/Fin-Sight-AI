"""
RAG (RETRIEVAL AUGMENTED GENERATION) PIPELINE

THIS IS THE CORE OF YOUR AI DATA ENGINEERING PROJECT!

ARCHITECTURE:
User Question
    ↓
[1] EMBEDDING GENERATOR → Convert question to 1536 numbers (vector)
    ↓  
[2] VECTOR SEARCH → Find top-3 most similar FAQs from database
    ↓
[3] CONTEXT ASSEMBLY → Format retrieved FAQs as context for LLM
    ↓
[4] LLM CALL (GPT-3.5) → Generate answer USING the context (not guessing!)
    ↓
Grounded Answer (accurate, sourced from real FAQs)

WHY RAG MATTERS:
- Without RAG: ChatGPT can "hallucinate" (make up fake policies)
- With RAG: Answer is GROUNDED in your actual FAQ documents
- Result: 80%+ reduction in inaccurate answers
"""

import openai
import psycopg2
import time
from config import (
    OPENAI_API_KEY, 
    EMBEDDING_MODEL, 
    CHAT_MODEL, 
    TOP_K_RESULTS,
    DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD
)
from mlflow_tracker import setup_mlflow, log_rag_query

openai.api_key = OPENAI_API_KEY


def get_db_connection():
    """Connect to PostgreSQL."""
    return psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD
    )


# ============================================
# STEP 1: EMBEDDING GENERATION (from Day 3)
# ============================================

def generate_query_embedding(user_question):
    """
    Convert user's question into embedding (1536 numbers).
    
    Same as Day 3's generate_single_embedding(), 
    but specifically named for clarity in RAG context.
    """
    try:
        response = openai.embeddings.create(
            model=EMBEDDING_MODEL,
            input=user_question
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"❌ Error generating query embedding: {e}")
        return None


# ============================================
# STEP 2: VECTOR SEARCH (from Day 4)
# ============================================

import numpy as np

def cosine_similarity(vec_a, vec_b):
    """Calculate cosine similarity between two vectors."""
    a = np.array(vec_a, dtype=np.float32)

    if isinstance(vec_b, str):
        import ast
        vec_b = ast.literal_eval(vec_b)

    b = np.array(vec_b, dtype=np.float32)
    
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    return float(dot_product / (norm_a * norm_b))


def retrieve_relevant_context(db_conn, query_embedding, top_k=TOP_K_RESULTS):
    """
    Search vector database for similar FAQs.
    
    Returns list of dicts with FAQ content + similarity scores.
    """
    cursor = db_conn.cursor()
    
    # Fetch all FAQs with embeddings
    cursor.execute("""
        SELECT faq_id, question, answer, category, embedding 
        FROM faqs_with_embeddings
    """)
    
    all_faqs = cursor.fetchall()
    cursor.close()
    
    # Calculate similarities
    results = []
    for faq in all_faqs:
        faq_id, question, answer, category, stored_embedding = faq
        score = cosine_similarity(query_embedding, stored_embedding)
        
        results.append({
            'faq_id': faq_id,
            'question': question,
            'answer': answer,
            'category': category,
            'score': round(score, 4)
        })
    
    # Sort by score (highest first) and return top-k
    results.sort(key=lambda x: x['score'], reverse=True)
    return results[:top_k]


# ============================================
# STEP 3: CONTEXT ASSEMBLY
# ============================================

def assemble_context_for_llm(retrieved_faqs):
    """
    Format retrieved FAQs into a clean context string for the LLM.
    
    The LLM needs CLEAR instructions on what information to use.
    Bad context: Just dump raw text
    Good context: Numbered, structured, with source attribution
    """
    context_parts = []
    
    for i, faq in enumerate(retrieved_faqs, 1):
        context_part = f"""
SOURCE {i} (Relevance Score: {faq['score']}, Category: {faq['category']}):
Question: {faq['question']}
Answer: {faq['answer']}
"""
        context_parts.append(context_part)
    
    # Join all parts with separator
    full_context = "\n" + "=" * 70 + "\n".join(context_parts)
    
    return full_context


# ============================================
# STEP 4: LLM GENERATION (THE AI PART)
# ============================================

SYSTEM_PROMPT = """You are FinSight Bank's AI Customer Support Assistant.

YOUR ROLE:
Answer customer questions about banking products, loans, and services using ONLY the provided context below.

RULES YOU MUST FOLLOW:
1. Use ONLY the information in the provided context sources
2. If the context doesn't contain the answer, say: "I'm sorry, I don't have specific information about that. Please contact our support team at 1800-123-4567 or visit your nearest branch."
3. Do NOT make up or guess information (no hallucination!)
4. Be concise but helpful (2-4 sentences typically)
5. If multiple sources are relevant, synthesize information from all of them
6. Always mention which source(s) you used (e.g., "According to our policy...")
7. For numerical questions (interest rates, amounts), quote exact numbers from context

TONE:
- Professional and friendly
- Clear and easy to understand
- Empathetic to customer concerns

CONTEXT FROM KNOWLEDGE BASE:
{context}

IMPORTANT: Base your answer EXCLUSIVELY on the above context. Do not use external knowledge."""


def generate_answer_with_llm(user_question, context):
    """
    Send question + context to GPT and get grounded answer.
    
    This is where the "Augmented Generation" happens.
    The LLM sees both the user's question AND the relevant FAQ context,
    so it can answer accurately without making things up.
    """
    try:
        # Fill in the context placeholder
        system_message = SYSTEM_PROMPT.format(context=context)
        
        # Call OpenAI Chat API
        response = openai.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_question}
            ],
            temperature=0.3,      # Low temperature = more focused, less creative
            max_tokens=300,       # Limit response length
            presence_penalty=0,   # Don't encourage new topics
            frequency_penalty=0   # Don't repeat same phrases
        )
        
        return {
            'answer': response.choices[0].message.content,
            'model': response.model,
            'usage': {
                'prompt_tokens': response.usage.prompt_tokens,
                'completion_tokens': response.usage.completion_tokens,
                'total_tokens': response.usage.total_tokens
            }
        }
        
    except Exception as e:
        return {
            'answer': f"Error generating answer: {str(e)}",
            'model': None,
            'usage': None
        }


# ============================================
# MAIN RAG FUNCTION (COMBINES ALL STEPS)
# ============================================

def ask_question(user_question, db_conn=None):
    """
    COMPLETE RAG PIPELINE - One function call does everything!
    
    This is what your application will call.
    
    Args:
        user_question (str): The customer's question
        db_conn: Database connection (optional, creates new one if not provided)
    
    Returns:
        dict: Complete result with answer, sources, timing, etc.
    """
    
    # Timing the entire pipeline
    start_time = time.time()
    
    # Create connection if not provided
    close_connection = False
    if db_conn is None:
        db_conn = get_db_connection()
        close_connection = True
    
    try:
        # ===== STEP 1: Generate Query Embedding =====
        step1_time = time.time()
        query_embedding = generate_query_embedding(user_question)
        step1_duration = time.time() - step1_time
        
        if query_embedding is None:
            return {
                'success': False,
                'error': 'Failed to generate embedding',
                'answer': 'Sorry, I encountered an error processing your question.'
            }
        
        # ===== STEP 2: Retrieve Relevant Context =====
        step2_time = time.time()
        retrieved_faqs = retrieve_relevant_context(db_conn, query_embedding, TOP_K_RESULTS)
        step2_duration = time.time() - step2_time
        
        # ===== STEP 3: Assemble Context =====
        step3_time = time.time()
        context = assemble_context_for_llm(retrieved_faqs)
        step3_duration = time.time() - step3_time
        
        # ===== STEP 4: Generate Answer with LLM =====
        step4_time = time.time()
        llm_result = generate_answer_with_llm(user_question, context)
        step4_duration = time.time() - step4_time
        
        # Total time
        total_duration = time.time() - start_time

        # ===== NEW: Log to MLflow =====
        log_rag_query({
            'success': True,
            'question': user_question,
            'answer': llm_result['answer'],
            'sources': retrieved_faqs,
            'model_used': llm_result['model'],
            'token_usage': llm_result['usage'],
            'timing': {
                'embedding_generation': step1_duration,
                'vector_search': step2_duration,
                'context_assembly': step3_duration,
                'llm_generation': step4_duration,
                'total_pipeline': total_duration
            }
        })
        
        # Return complete result
        return {
            'success': True,
            'question': user_question,
            'answer': llm_result['answer'],
            'sources': retrieved_faqs,
            'model_used': llm_result['model'],
            'token_usage': llm_result['usage'],
            'timing': {
                'embedding_generation': round(step1_duration, 3),
                'vector_search': round(step2_duration, 3),
                'context_assembly': round(step3_duration, 3),
                'llm_generation': round(step4_duration, 3),
                'total_pipeline': round(total_duration, 3)
            }
        }
        
    finally:
        # Close connection if we created it
        if close_connection:
            db_conn.close()


# ============================================
# PRETTY PRINTING FOR CLI
# ============================================

def print_rag_result(result):
    """Format and display RAG result nicely in terminal."""
    
    print("\n" + "=" * 70)
    print("🤖 FIN-SIGHT AI RESPONSE")
    print("=" * 70)
    
    if not result['success']:
        print(f"\n❌ ERROR: {result.get('error', 'Unknown error')}")
        print(f"\n{result.get('answer', '')}")
        return
    
    # Question
    print(f"\n❓ YOUR QUESTION:")
    print(f"   {result['question']}")
    
    # Answer
    print(f"\n💡 AI ANSWER:")
    print(f"   {result['answer']}")
    
    # Sources used
    print(f"\n📚 SOURCES USED ({len(result['sources'])} FAQs retrieved):")
    print("-" * 70)
    for i, source in enumerate(result['sources'], 1):
        print(f"\n   Source #{i} [Score: {source['score']:.4f}] ({source['category']})")
        print(f"   Q: {source['question'][:80]}...")
    
    # Timing
    timing = result['timing']
    print(f"\n⏱️  PERFORMANCE:")
    print(f"   Embedding generation: {timing['embedding_generation']}s")
    print(f"   Vector search:       {timing['vector_search']}s")
    print(f"   Context assembly:    {timing['context_assembly']}s")
    print(f"   LLM generation:      {timing['llm_generation']}s")
    print(f"   ────────────────────────────────")
    print(f"   TOTAL TIME:          {timing['total_pipeline']}s")
    
    # Token usage (cost tracking)
    if result['token_usage']:
        usage = result['token_usage']
        print(f"\n💰 TOKEN USAGE:")
        print(f"   Input tokens:  {usage['prompt_tokens']}")
        print(f"   Output tokens: {usage['completion_tokens']}")
        print(f"   Total tokens:  {usage['total_tokens']}")
        estimated_cost = (usage['prompt_tokens'] * 0.0000015 + 
                         usage['completion_tokens'] * 0.000002)
        print(f"   Est. cost:     ${estimated_cost:.6f}")
    
    print("\n" + "=" * 70)


# ============================================
# INTERACTIVE CLI MODE
# ============================================

def run_interactive_mode():
    """
    Interactive chat mode - keeps running until user quits.
    This is what you'll demo to interviewers!
    """
    print("\n" + "🎯" * 35)
    print("FIN-SIGHT AI: BANKING ASSISTANT (RAG-Powered)")
    print("🎯" * 35)
    print("\nType your banking questions below.")
    print("Type 'quit' or 'exit' to close.\n")
    
    # Connect to database once (reuse connection)
    conn = get_db_connection()
    setup_mlflow()
    print("✅ Connected to database\n")
    
    try:
        while True:
            # Get user input
            user_input = input("❓ You: ").strip()
            
            # Handle commands
            if user_input.lower() in ['quit', 'exit', 'q', 'bye']:
                print("\n👋 Thank you for using Fin-Sight AI. Goodbye!")
                break
            
            if not user_input:
                continue
            
            if user_input.lower() == 'help':
                print("\nAvailable commands:")
                print("  • Any banking question → Ask the AI")
                print("  • 'help' → Show this message")
                print("  • 'quit' → Exit the program\n")
                continue
            
            # Process question through RAG pipeline
            print("\n🤔 Thinking...\n")
            result = ask_question(user_input, conn)
            
            # Display result
            print_rag_result(result)
            
            print()  # Spacer
            
    finally:
        conn.close()


# ============================================
# MAIN EXECUTION / TESTING
# ============================================

if __name__ == "__main__":
    import sys
    
    print("\n" + "=" * 70)
    print("FIN-SIGHT AI: RAG PIPELINE TEST")
    print("=" * 70)
    
    # Check if running in test mode or interactive mode
    if len(sys.argv) > 1 and sys.argv[1] == '--test':
        setup_mlflow()
        # Test mode: Run predefined questions
        print("\n🧪 Running in TEST MODE (predefined questions)...\n")
        
        test_questions = [
            "What documents do I need for a personal loan?",
            "How do I check my loan application status?",
            "What happens if I miss an EMI payment?",
            "What is the minimum credit score required?",
            "I want to foreclose my loan early"
        ]
        
        conn = get_db_connection()
        
        for i, question in enumerate(test_questions, 1):
            print(f"\n{'🔵' * 35}")
            print(f"TEST QUESTION {i}/{len(test_questions)}")
            print(f"{'🔵' * 35}")
            
            result = ask_question(question, conn)
            print_rag_result(result)
            
            if i < len(test_questions):
                input("\nPress Enter to continue to next question...")
        
        conn.close()
        
        print("\n" + "=" * 70)
        print("✅ ALL TESTS COMPLETED")
        print("=" * 70)
        print("\nTo run in INTERACTIVE mode, type: python rag_pipeline.py")
        print("To run in TEST mode again, type: python rag_pipeline.py --test")
        
    else:
        # Default: Interactive mode
        run_interactive_mode()