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
            chat_history TEXT,  
            uploaded_images TEXT,  
            date DATE  
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
    today = date.today().isoformat()

    # Serialize new data as JSON
    new_chat_history_json = json.dumps(chat_history)
    new_uploaded_images_json = json.dumps(uploaded_images)

    # Check if a row for the user and today's date already exists
    cursor.execute("SELECT chat_history, uploaded_images FROM user_data WHERE username = ? AND date = ?", (username, today))
    existing_row = cursor.fetchone()

    if existing_row:
        # Deserialize existing data
        existing_chat_history = json.loads(existing_row[0]) if existing_row[0] else []
        existing_uploaded_images = json.loads(existing_row[1]) if existing_row[1] else []

        # Append new data to existing data
        updated_chat_history = existing_chat_history + chat_history
        updated_uploaded_images = existing_uploaded_images + uploaded_images

        # Serialize updated data
        updated_chat_history_json = json.dumps(updated_chat_history)
        updated_uploaded_images_json = json.dumps(updated_uploaded_images)

        # Update the row with appended data
        cursor.execute(
            "UPDATE user_data SET chat_history = ?, uploaded_images = ? WHERE username = ? AND date = ?",
            (updated_chat_history_json, updated_uploaded_images_json, username, today)
        )
    else:
        # Insert a new row if no existing row is found
        cursor.execute(
            "INSERT INTO user_data (username, chat_history, uploaded_images, date) VALUES (?, ?, ?, ?)",
            (username, new_chat_history_json, new_uploaded_images_json, today)
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