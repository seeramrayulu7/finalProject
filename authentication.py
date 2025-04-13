import sqlite3
import hashlib

DB_FILE = "user_data.db"

# Initialize SQLite database
def init_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password TEXT,
            name TEXT
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_data (
            username TEXT,
            chat_history TEXT,
            uploaded_images TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

# Function to hash passwords
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# Function to authenticate users
def authenticate_user(username, password):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT password FROM users WHERE username = ?", (username,))
    result = cursor.fetchone()
    conn.close()
    if result and result[0] == hash_password(password):
        return True
    return False

# Function to check if a username already exists
def username_exists(username):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT username FROM users WHERE username = ?", (username,))
    result = cursor.fetchone()
    conn.close()
    return result is not None

# Function to register a new user
def register_user(username, password, name):
    if username_exists(username):
        return False  # Username already exists
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    hashed_password = hash_password(password)
    try:
        cursor.execute("INSERT INTO users (username, password, name) VALUES (?, ?, ?)", (username, hashed_password, name))
        conn.commit()
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()
    return True

# Initialize the database
init_db()