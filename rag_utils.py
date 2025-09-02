import psycopg2
import numpy as np

def get_connection():
    return psycopg2.connect(
        dbname="aidb",
        user="aiuser",
        password="aipassword",
        host="localhost",
        port="5432"
    )

def get_embedding(text):
    # TODO: use sentence-transformers or OpenAI embedding API
    return np.random.rand(768).tolist()  # placeholder

def get_context(query, top_k=3):
    conn = get_connection()
    cur = conn.cursor()

    query_vec = get_embedding(query)

    cur.execute("""
        SELECT content
        FROM documents
        ORDER BY embedding <-> %s::vector
        LIMIT %s;
    """, (list(query_vec), top_k))   # cast numpy array to list

    rows = cur.fetchall()
    conn.close()
    return "\n".join(r[0] for r in rows) if rows else "No context found."

