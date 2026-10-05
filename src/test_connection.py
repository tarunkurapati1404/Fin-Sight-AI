# Test PostgreSQL connection
import psycopg2
from config import DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD

try:
    conn = psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD
    )
    print("✅ SUCCESS! Connected to PostgreSQL")
    print(f"   Database: {DB_NAME}")
    print(f"   User: {DB_USER}")
    
    conn.close()
    print("✅ Connection closed.")
    
except Exception as e:
    print(f"❌ ERROR: {e}")
    print("\nCheck these:")
    print("   1. Is PostgreSQL running? (Start pgAdmin or check Services)")
    print("   2. Is password correct in config.py?")
    print("   3. Does database 'finsight_db' exist?")