import sqlite3
import hashlib
import json
from datetime import date

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
            chat_history TEXT,  -- JSON string to store chat history
            uploaded_images TEXT,  -- JSON string to store image paths
            date DATE,  -- Date for which the data is stored
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

# Function to save user data
def save_user_data(username, chat_history, uploaded_images):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    today = date.today().isoformat()  # Get today's date as a string (YYYY-MM-DD)

    # Serialize chat history and images as JSON
    chat_history_json = json.dumps(chat_history)  # Chat history as a JSON string
    uploaded_images_json = json.dumps(uploaded_images)  # Uploaded images as a JSON string

    # Check if a row for the user and today's date already exists
    cursor.execute("SELECT * FROM user_data WHERE username = ? AND date = ?", (username, today))
    existing_row = cursor.fetchone()

    if existing_row:
        # Update the existing row
        cursor.execute(
            "UPDATE user_data SET chat_history = ?, uploaded_images = ?, timestamp = CURRENT_TIMESTAMP WHERE username = ? AND date = ?",
            (chat_history_json, uploaded_images_json, username, today)
        )
    else:
        # Insert a new row
        cursor.execute(
            "INSERT INTO user_data (username, chat_history, uploaded_images, date) VALUES (?, ?, ?, ?)",
            (username, chat_history_json, uploaded_images_json, today)
        )

    conn.commit()
    conn.close()

# Function to get user data
def get_user_data(username, specific_date=None):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    if specific_date:
        # Retrieve data for a specific date
        cursor.execute("SELECT chat_history, uploaded_images, date FROM user_data WHERE username = ? AND date = ?", (username, specific_date))
    else:
        # Retrieve all data for the user
        cursor.execute("SELECT chat_history, uploaded_images, date FROM user_data WHERE username = ?", (username,))

    rows = cursor.fetchall()
    conn.close()

    # Deserialize JSON fields
    data = []
    for row in rows:
        chat_history = json.loads(row[0]) if row[0] else []
        uploaded_images = json.loads(row[1]) if row[1] else []
        data.append({
            "chat_history": chat_history,
            "uploaded_images": uploaded_images,
            "date": row[2]
        })

    return data

# Initialize the database
init_db()