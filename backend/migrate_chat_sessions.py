import sqlite3

conn = sqlite3.connect('healthcare_ai.db')
cursor = conn.cursor()

# Check if document_id column exists
cursor.execute("SELECT name FROM pragma_table_info('chat_sessions') WHERE name='document_id'")
exists = cursor.fetchone()
print('document_id column exists:', bool(exists))

if not exists:
    cursor.execute("ALTER TABLE chat_sessions ADD COLUMN document_id TEXT REFERENCES documents(id) ON DELETE CASCADE")
    conn.commit()
    print('Added document_id column to chat_sessions')
else:
    print('Column already exists, no action needed')

# Verify
cursor.execute("SELECT name FROM pragma_table_info('chat_sessions')")
cols = [r[0] for r in cursor.fetchall()]
print('Columns:', cols)

conn.close()
