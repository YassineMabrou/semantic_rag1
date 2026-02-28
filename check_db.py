"""Quick script to check PostgreSQL connection and database contents."""

import psycopg2

# Docker container credentials
DB_CONFIGS = [
    {"host": "localhost", "port": 5432, "dbname": "database", "user": "user", "password": "password"},
]

print("=" * 50)
print("  PostgreSQL Connection Check")
print("=" * 50)

for config in DB_CONFIGS:
    try:
        print(f"\nTrying: {config['dbname']}...")
        conn = psycopg2.connect(**config)
        cur = conn.cursor()
        
        # List all tables
        cur.execute("""
            SELECT table_name FROM information_schema.tables 
            WHERE table_schema = 'public'
        """)
        tables = cur.fetchall()
        
        print(f"  [OK] Connected to '{config['dbname']}'")
        print(f"  Tables: {[t[0] for t in tables]}")
        
        # Check if embeddings table exists
        if any('embeddings' in t[0] for t in tables):
            cur.execute("SELECT COUNT(*) FROM embeddings")
            count = cur.fetchone()[0]
            print(f"  Embeddings count: {count}")
            
            # Show sample
            cur.execute("SELECT id, texte_fragment FROM embeddings LIMIT 2")
            samples = cur.fetchall()
            for s in samples:
                print(f"    ID {s[0]}: {s[1][:50]}...")
        
        cur.close()
        conn.close()
        
    except psycopg2.OperationalError as e:
        print(f"  [FAIL] {e}")
    except Exception as e:
        print(f"  [ERROR] {e}")

print("\n" + "=" * 50)
