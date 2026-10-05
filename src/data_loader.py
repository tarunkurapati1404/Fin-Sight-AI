"""
DATA LOADER MODULE
Reads CSV → Cleans data → Loads to PostgreSQL
"""

import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
from config import DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD


def get_connection():
    """Connect to PostgreSQL database."""
    return psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD
    )


def load_raw_data(filepath):
    """
    Step 1: Read CSV into Pandas DataFrame.
    Returns raw dataframe.
    """
    print(f"📂 Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    print(f"   ✅ Loaded {len(df)} rows, {len(df.columns)} columns")
    return df


def clean_data(df):
    """
    Step 2: Clean the data.
    - Handle missing values
    - Fix data types
    - Remove outliers
    """
    print("🧹 Cleaning data...")
    
    # Make a copy to avoid warnings
    df_clean = df.copy()
    
    # 1. Remove rows with missing critical values
    before_count = len(df_clean)
    df_clean = df_clean.dropna(subset=['loan_int_rate', 'person_emp_length'])
    after_count = len(df_clean)
    print(f"   🗑️  Removed {before_count - after_count} rows with nulls in critical columns")
    
    # 2. Fix unrealistic ages (keep 18-100 only)
    df_clean = df_clean[(df_clean['person_age'] >= 18) & (df_clean['person_age'] <= 100)]
    
    # 3. Fix employment length (keep 0-50 years)
    df_clean = df_clean[(df_clean['person_emp_length'] >= 0) & (df_clean['person_emp_length'] <= 50)]
    
    # 4. Ensure loan amount is positive
    df_clean = df_clean[df_clean['loan_amnt'] > 0]
    
    # 5. Ensure income is positive
    df_clean = df_clean[df_clean['person_income'] > 0]
    
    print(f"   ✅ Final clean dataset: {len(df_clean)} rows")
    return df_clean


def create_tables(conn):
    """
    Step 3: Create Star Schema tables.
    
    STAR SCHEMA DESIGN:
    
    FACT TABLE:
    - fact_loans (loan_id, person_id, intent_id, amount, rate, status, 
                  percent_income, cb_default)
    
    DIMENSION TABLES:
    - dim_person (person_id, age, income, emp_length, home_ownership, 
                  cred_hist_length)
    - dim_loan_intent (intent_id, intent_type, grade)
    - dim_loan_status (status_id, status_name, default_flag)
    """
    
    cursor = conn.cursor()
    
    print("📊 Creating Star Schema tables...")
    
    # ============================================
    # DIMENSION TABLES FIRST (no foreign keys yet)
    # ============================================
    
    # 1. dim_person - Customer demographics
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS dim_person (
            person_id SERIAL PRIMARY KEY,
            age INTEGER NOT NULL,
            income DECIMAL(12,2) NOT NULL,
            employment_length DECIMAL(5,2),
            home_ownership VARCHAR(20),
            credit_history_length INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    print("   ✅ Table created: dim_person")
    
    # 2. dim_loan_intent - Loan purpose and grade
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS dim_loan_intent (
            intent_id SERIAL PRIMARY KEY,
            intent_type VARCHAR(50) NOT NULL,
            loan_grade VARCHAR(5),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    print("   ✅ Table created: dim_loan_intent")
    
    # 3. dim_loan_status - Loan outcome
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS dim_loan_status (
            status_id SERIAL PRIMARY KEY,
            status_name VARCHAR(20) NOT NULL,
            is_default BOOLEAN NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    print("   ✅ Table created: dim_loan_status")
    
    # ============================================
    # FACT TABLE (references dimensions)
    # ============================================
    
    # 4. fact_loans - Main transactional table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS fact_loans (
            loan_id SERIAL PRIMARY KEY,
            person_id INTEGER REFERENCES dim_person(person_id),
            intent_id INTEGER REFERENCES dim_loan_intent(intent_id),
            status_id INTEGER REFERENCES dim_loan_status(status_id),
            loan_amount DECIMAL(10,2) NOT NULL,
            interest_rate DECIMAL(5,2),
            percent_income DECIMAL(5,2),
            cb_person_default_on_file BOOLEAN,
            loan_date DATE DEFAULT CURRENT_TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    print("   ✅ Table created: fact_loans")
    
    conn.commit()
    cursor.close()
    print("\n✅ All Star Schema tables created successfully!")


def transform_to_star_schema(df_clean):
    """
    Step 4: Transform flat CSV into Star Schema format.
    Returns dictionaries ready for INSERT.
    """
    print("🔄 Transforming to Star Schema...")
    
    # Extract unique persons (deduplicate)
    persons = df_clean[['person_age', 'person_income', 'person_emp_length', 
                    'person_home_ownership', 'cb_person_cred_hist_length']].copy()

    persons = persons.drop_duplicates().reset_index(drop=True)

    persons.rename(columns={
    'person_age': 'age',
    'person_income': 'income',
    'person_emp_length': 'employment_length',
    'person_home_ownership': 'home_ownership',
    'cb_person_cred_hist_length': 'credit_history_length'
    }, inplace=True)

    persons.insert(0, 'person_id', range(1, len(persons) + 1))
    
    # Extract unique loan intents
    intents = df_clean[['loan_intent', 'loan_grade']].copy()
    intents = intents.drop_duplicates().reset_index(drop=True)
    intents.insert(0, 'intent_id', range(1, len(intents) + 1))
    intents.rename(columns={'loan_intent': 'intent_type', 'loan_grade': 'loan_grade'}, inplace=True)
    
    # Extract unique statuses
    statuses = pd.DataFrame({
        'status_name': ['Non-Default', 'Default'],
        'is_default': [False, True]
    })
    statuses.insert(0, 'status_id', [1, 2])
    
    # Build fact table by looking up IDs
    # Create lookup dictionaries
    person_lookup = {
        (row['age'], row['income'], row['employment_length'],
        row['home_ownership'], row['credit_history_length']): row['person_id']
        for _, row in persons.iterrows()
    }
    
    intent_lookup = {
        (row['intent_type'], row['loan_grade']): row['intent_id']
        for _, row in intents.iterrows()
    }
    
    status_lookup = {0: 1, 1: 2}  # loan_status 0=Non-Default, 1=Default
    
    # Build fact loans
    fact_loans = []
    for _, row in df_clean.iterrows():
        person_key = (row['person_age'], row['person_income'], row['person_emp_length'],
              row['person_home_ownership'], row['cb_person_cred_hist_length'])
        intent_key = (row['loan_intent'], row['loan_grade'])
        
        fact_loans.append({
            'person_id': person_lookup.get(person_key),
            'intent_id': intent_lookup.get(intent_key),
            'status_id': status_lookup.get(row['loan_status']),
            'loan_amount': row['loan_amnt'],
            'interest_rate': row['loan_int_rate'],
            'percent_income': row['loan_percent_income'],
            'cb_person_default_on_file': row['cb_person_default_on_file'] == 'Y'
        })
    
    fact_df = pd.DataFrame(fact_loans)
    fact_df.insert(0, 'loan_id', range(1, len(fact_df) + 1))
    
    print(f"   ✅ Transformed:")
    print(f"      - dim_person: {len(persons)} rows")
    print(f"      - dim_loan_intent: {len(intents)} rows")
    print(f"      - dim_loan_status: {len(statuses)} rows")
    print(f"      - fact_loans: {len(fact_df)} rows")
    
    return persons, intents, statuses, fact_df


def load_to_database(conn, persons, intents, statuses, fact_df):
    """
    Step 5: Insert all data into PostgreSQL.
    Uses execute_values for fast bulk inserts.
    """
    cursor = conn.cursor()
    
    print("💾 Loading data into database...")
    
    # Insert dim_person
    person_tuples = [tuple(x) for x in persons[['age', 'income', 'employment_length', 
                                                  'home_ownership', 'credit_history_length']].values]
    execute_values(cursor, """
        INSERT INTO dim_person (age, income, employment_length, home_ownership, credit_history_length)
        VALUES %s
    """, person_tuples)
    print(f"   ✅ Loaded {len(person_tuples)} persons")
    
    # Insert dim_loan_intent
    intent_tuples = [tuple(x) for x in intents[['intent_type', 'loan_grade']].values]
    execute_values(cursor, """
        INSERT INTO dim_loan_intent (intent_type, loan_grade)
        VALUES %s
    """, intent_tuples)
    print(f"   ✅ Loaded {len(intent_tuples)} loan intents")
    
    # Insert dim_loan_status
    status_tuples = [tuple(x) for x in statuses[['status_name', 'is_default']].values]
    execute_values(cursor, """
        INSERT INTO dim_loan_status (status_name, is_default)
        VALUES %s
    """, status_tuples)
    print(f"   ✅ Loaded {len(status_tuples)} statuses")
    
    # Insert fact_loans
    fact_tuples = [tuple(x) for x in fact_df[['person_id', 'intent_id', 'status_id',
                                               'loan_amount', 'interest_rate', 'percent_income',
                                               'cb_person_default_on_file']].values]
    execute_values(cursor, """
        INSERT INTO fact_loans (person_id, intent_id, status_id, loan_amount, 
                                interest_rate, percent_income, cb_person_default_on_file)
        VALUES %s
    """, fact_tuples)
    print(f"   ✅ Loaded {len(fact_tuples)} loan records")
    
    conn.commit()
    cursor.close()
    print("\n✅ All data loaded successfully!")


def verify_data(conn):
    """
    Step 6: Run verification queries to confirm data integrity.
    """
    cursor = conn.cursor()
    
    print("\n" + "=" * 50)
    print("📊 DATA VERIFICATION")
    print("=" * 50)
    
    # Row counts
    tables = ['dim_person', 'dim_loan_intent', 'dim_loan_status', 'fact_loans']
    for table in tables:
        cursor.execute(f"SELECT COUNT(*) FROM {table}")
        count = cursor.fetchone()[0]
        print(f"   {table}: {count:,} rows")
    
    # Sample queries
    print("\n📈 Sample Analytics Queries:")
    
    # Average loan amount by status
    cursor.execute("""
        SELECT ds.status_name, 
               COUNT(fl.loan_id) as num_loans,
               AVG(fl.loan_amount) as avg_loan_amt,
               ROUND(AVG(fl.interest_rate)::numeric, 2) as avg_interest_rate
        FROM fact_loans fl
        JOIN dim_loan_status ds ON fl.status_id = ds.status_id
        GROUP BY ds.status_name
    """)
    print("\n   Loan Statistics by Status:")
    for row in cursor.fetchall():
        print(f"      {row[0]:12} | Loans: {row[1]:6,} | Avg Amount: ₹{row[2]:10,.2f} | Avg Rate: {row[3]}%")
    
    # Top 5 loan intents by volume
    cursor.execute("""
        SELECT di.intent_type, 
               COUNT(fl.loan_id) as num_loans,
               SUM(fl.loan_amount) as total_volume
        FROM fact_loans fl
        JOIN dim_loan_intent di ON fl.intent_id = di.intent_id
        GROUP BY di.intent_type
        ORDER BY num_loans DESC
        LIMIT 5
    """)
    print("\n   Top 5 Loan Purposes:")
    for row in cursor.fetchall():
        print(f"      {row[0]:15} | Count: {row[1]:5,} | Total Volume: ₹{row[2]:15,.2f}")
    
    # Default rate by income bracket
    cursor.execute("""
        SELECT 
            CASE 
                WHEN income < 30000 THEN '< ₹30K'
                WHEN income < 60000 THEN '₹30K - ₹60K'
                WHEN income < 90000 THEN '₹60K - ₹90K'
                ELSE '> ₹90K'
            END as income_bracket,
            COUNT(*) as total_loans,
            SUM(CASE WHEN ds.is_default THEN 1 ELSE 0 END) as defaults,
            ROUND(100.0 * SUM(CASE WHEN ds.is_default THEN 1 ELSE 0 END) / COUNT(*), 2) as default_pct
        FROM fact_loans fl
        JOIN dim_person p ON fl.person_id = p.person_id
        JOIN dim_loan_status ds ON fl.status_id = ds.status_id
        GROUP BY income_bracket
        ORDER BY MIN(p.income)
    """)
    print("\n   Default Rate by Income Bracket:")
    print(f"   {'Bracket':15} | {'Total':>7} | {'Defaults':>9} | {'Default %':>10}")
    print(f"   {'-'*55}")
    for row in cursor.fetchall():
        print(f"      {row[0]:13} | {row[1]:7,} | {row[2]:9,} | {row[3]:9}%")
    
    cursor.close()


# ============================================
# MAIN EXECUTION
# ============================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("FIN-SIGHT AI: DATA LOADING PIPELINE")
    print("=" * 60 + "\n")
    
    # Step 1: Load raw data
    df_raw = load_raw_data('data/credit_risk_dataset.csv')    
    
    # Step 2: Clean data
    df_clean = clean_data(df_raw)
    
    # Step 3: Connect to database
    conn = get_connection()
    
    # Step 4: Create tables
    create_tables(conn)
    
    # Step 5: Transform to star schema
    persons, intents, statuses, fact_df = transform_to_star_schema(df_clean)
    
    # Step 6: Load to database
    load_to_database(conn, persons, intents, statuses, fact_df)
    
    # Step 7: Verify
    verify_data(conn)
    
    # Close connection
    conn.close()
    
    print("\n" + "=" * 60)
    print("✅ DAY 2 COMPLETE: Data loaded and verified!")
    print("=" * 60)