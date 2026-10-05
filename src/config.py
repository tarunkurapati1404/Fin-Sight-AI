# ============================================
# CONFIGURATION - FIN-SIGHT AI
# ============================================

# OPENAI API KEY (paste your key here between quotes)
from dotenv import load_dotenv
import os

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# MODELS
EMBEDDING_MODEL = "text-embedding-3-small"  # Cheap, good for embeddings
CHAT_MODEL = "gpt-3.5-turbo"                 # Cheap chat model

# POSTGRESQL DATABASE SETTINGS
DB_HOST = os.getenv("DB_HOST", "127.0.0.1")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "finsight_db")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "your_password_here")

# RAG SETTINGS
TOP_K_RESULTS = 3             # How many FAQs to retrieve

# MLFLOW SETTINGS
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "./mlruns")