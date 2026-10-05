"""
EMBEDDING GENERATOR MODULE
Converts text (FAQs) into vector embeddings using OpenAI API

OUTPUT: List of 1536 numbers per text (captures semantic meaning)
"""

import openai
import time
import pandas as pd
from config import OPENAI_API_KEY, EMBEDDING_MODEL

openai.api_key = OPENAI_API_KEY


def generate_single_embedding(text):
    """
    Convert ONE piece of text to embedding.
    
    Args:
        text (str): Any text (e.g., "How do I apply for loan?")
    
    Returns:
        list: 1536 floating-point numbers, OR None if error
    """
    try:
        response = openai.embeddings.create(
            model=EMBEDDING_MODEL,
            input=text
        )
        
        return response.data[0].embedding
        
    except Exception as e:
        print(f"   ❌ Error generating embedding: {e}")
        return None


def generate_batch_embeddings(texts, batch_size=10):
    """
    Convert MULTIPLE texts to embeddings (efficient batch processing).
    
    Why batches? 
    - Faster than calling API one-by-one
    - Reduces network overhead
    - OpenAI allows up to 2048 texts per batch
    
    Args:
        texts (list): List of strings (e.g., 20 FAQ answers)
        batch_size (int): How many to process at once (default: 10)
    
    Returns:
        list: List of embeddings (same length as input), failed items are None
    """
    all_embeddings = []
    total_texts = len(texts)
    
    print(f"\n🔄 Generating embeddings for {total_texts} texts...")
    print(f"   Model: {EMBEDDING_MODEL}")
    print(f"   Batch size: {batch_size}")
    print(f"   Estimated cost: ~${total_texts * 0.00002:.4f} (less than 1 cent!)\n")
    
    # Process in batches
    for i in range(0, total_texts, batch_size):
        batch = texts[i:i + batch_size]
        batch_num = (i // batch_size) + 1
        total_batches = (total_texts + batch_size - 1) // batch_size
        
        try:
            # Call OpenAI API for this batch
            response = openai.embeddings.create(
                model=EMBEDDING_MODEL,
                input=batch
            )
            
            # Extract embeddings from response
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)
            
            print(f"   ✅ Batch {batch_num}/{total_batches}: Processed {len(batch)} texts")
            
            # Small delay to avoid rate limits (good practice)
            time.sleep(0.5)
            
        except Exception as e:
            print(f"   ❌ Batch {batch_num} failed: {e}")
            # Add None for failed items so we maintain alignment
            all_embeddings.extend([None] * len(batch))
    
    # Summary
    successful = sum(1 for e in all_embeddings if e is not None)
    failed = total_texts - successful
    
    print(f"\n📊 EMBEDDING GENERATION COMPLETE:")
    print(f"   ✅ Successful: {successful}/{total_texts}")
    print(f"   ❌ Failed: {failed}/{total_texts}")
    print(f"   💰 Total cost: ~${successful * 0.00002:.4f}")
    
    return all_embeddings


def load_faqs_from_csv(filepath):
    """
    Load FAQs from CSV file.
    
    Returns:
        pandas.DataFrame: Columns [id, question, answer, category]
    """
    print(f"\n📂 Loading FAQs from {filepath}...")
    df = pd.read_csv(filepath)
    print(f"   ✅ Loaded {len(df)} FAQs")
    return df


def prepare_text_for_embedding(df):
    """
    Combine question + answer into single text for better embeddings.
    
    Why combine? 
    - Embedding captures meaning of BOTH question AND answer context
    - When user asks "documents needed", we match to FAQ that mentions documents
    
    Args:
        df (DataFrame): FAQs dataframe
    
    Returns:
        list: List of combined text strings
    """
    texts = []
    
    for _, row in df.iterrows():
        # Combine Q&A with clear separator
        combined = f"Question: {row['question']}\nAnswer: {row['answer']}"
        texts.append(combined)
    
    print(f"\n📝 Prepared {len(texts)} texts for embedding")
    print(f"   Example text:\n")
    print(f"   {texts[0][:200]}...")
    
    return texts


def save_embeddings_to_file(embeddings, faq_ids, output_path):
    """
    Save embeddings to file (temporary storage before DB).
    
    Args:
        embeddings (list): List of embedding vectors
        faq_ids (list): Corresponding FAQ IDs
        output_path (str): Where to save (CSV format)
    """
    import numpy as np
    
    print(f"\n💾 Saving embeddings to {output_path}...")
    
    # Create DataFrame
    df = pd.DataFrame({
        'faq_id': faq_ids,
        'embedding': embeddings
    })
    
    # Save to CSV (we'll load into PostgreSQL tomorrow)
    df.to_csv(output_path, index=False)
    
    print(f"   ✅ Saved {len(df)} embeddings to file")
    print(f"   File size: ~{len(df) * 1536 * 8 / 1024 / 1024:.2f} MB")


# ============================================
# MAIN EXECUTION
# ============================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("FIN-SIGHT AI: EMBEDDING GENERATION PIPELINE")
    print("=" * 60)
    
    # Step 1: Load FAQs
    faqs_df = load_faqs_from_csv('data/bank_faqs.csv')
    
    # Show what we loaded
    print("\n📋 Sample FAQs:")
    print(faqs_df[['id', 'question', 'category']].head(3).to_string())
    
    # Step 2: Prepare text for embedding
    texts = prepare_text_for_embedding(faqs_df)
    
    # Step 3: Generate embeddings (THE AI PART!)
    embeddings = generate_batch_embeddings(texts, batch_size=5)
    
    # Step 4: Save to file (for tomorrow's step)
    save_embeddings_to_file(
        embeddings=embeddings,
        faq_ids=faqs_df['id'].tolist(),
        output_path='data/faq_embeddings.csv'
    )
    
    # Step 5: Quick verification
    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)
    
    if embeddings[0] is not None:
        print(f"\n✅ First FAQ embedding sample:")
        print(f"   FAQ ID: {faqs_df.iloc[0]['id']}")
        print(f"   Question: {faqs_df.iloc[0]['question']}")
        print(f"   Embedding length: {len(embeddings[0])} dimensions")
        print(f"   First 10 values: {embeddings[0][:10]}")
        print(f"   Last 10 values: {embeddings[0][-10:]}")
        print(f"\n   Min value: {min(embeddings[0]):.6f}")
        print(f"   Max value: {max(embeddings[0]):.6f}")
        print(f"   Mean value: {sum(embeddings[0])/len(embeddings[0]):.6f}")
    
    print("\n" + "=" * 60)
    print("✅ DAY 3 COMPLETE: Embeddings generated and saved!")
    print("=" * 60)
    print("\n📌 NEXT STEP (Day 4): Load these embeddings into pgvector (vector database)")