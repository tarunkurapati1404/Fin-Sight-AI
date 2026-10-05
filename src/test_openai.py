# Test OpenAI API connection
import openai
from config import OPENAI_API_KEY

openai.api_key = OPENAI_API_KEY

try:
    # Simple test: Generate embedding for one sentence
    response = openai.embeddings.create(
        model="text-embedding-3-small",
        input="Hello, this is a test"
    )
    
    embedding = response.data[0].embedding
    
    print("✅ OpenAI API connection SUCCESS!")
    print(f"   Model used: {response.model}")
    print(f"   Embedding length: {len(embedding)} numbers")
    print(f"   First 5 numbers: {embedding[:5]}")
    print(f"   Cost: ~$0.00002 (basically free)")
    
except Exception as e:
    print(f"❌ ERROR: {e}")
    print("\nPossible issues:")
    print("   1. No credits added (go to platform.openai.com → Add credits)")
    print("   2. Wrong API key (check config.py)")
    print("   3. Internet connection issue")