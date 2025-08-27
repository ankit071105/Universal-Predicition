import streamlit as st
import google.generativeai as genai
from PIL import Image
import os
from dotenv import load_dotenv
import io
import requests
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import hashlib
import time
from datetime import datetime

# Load environment variables
load_dotenv()

# Configure page
st.set_page_config(
    page_title="Universal Scanner & Identifier",
    page_icon="🔍",
    layout="wide"
)

# Initialize database
def init_db():
    conn = sqlite3.connect('scanner_app.db')
    c = conn.cursor()
    
    # Create users table
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  username TEXT UNIQUE NOT NULL,
                  password TEXT NOT NULL,
                  email TEXT,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    
    # Create search history table
    c.execute('''CREATE TABLE IF NOT EXISTS search_history
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  user_id INTEGER NOT NULL,
                  search_type TEXT NOT NULL,
                  search_query TEXT,
                  result TEXT,
                  timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                  FOREIGN KEY (user_id) REFERENCES users (id))''')
    
    conn.commit()
    conn.close()

# Initialize database on app start
init_db()

# Initialize Gemini API
def configure_genai():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        st.error("Please set GEMINI_API_KEY in your .env file")
        st.stop()
    genai.configure(api_key=api_key)

configure_genai()

# Password hashing
def make_hashes(password):
    return hashlib.sha256(str.encode(password)).hexdigest()

def check_hashes(password, hashed_text):
    return make_hashes(password) == hashed_text

# User authentication
def create_user(username, password, email):
    conn = sqlite3.connect('scanner_app.db')
    c = conn.cursor()
    hashed_password = make_hashes(password)
    
    try:
        c.execute('INSERT INTO users (username, password, email) VALUES (?, ?, ?)', 
                 (username, hashed_password, email))
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        conn.close()
        return False

def login_user(username, password):
    conn = sqlite3.connect('scanner_app.db')
    c = conn.cursor()
    
    c.execute('SELECT * FROM users WHERE username = ?', (username,))
    data = c.fetchone()
    conn.close()
    
    if data and check_hashes(password, data[2]):
        return data
    return None

# Search history management
def add_to_search_history(user_id, search_type, search_query, result):
    conn = sqlite3.connect('scanner_app.db')
    c = conn.cursor()
    
    c.execute('INSERT INTO search_history (user_id, search_type, search_query, result) VALUES (?, ?, ?, ?)',
             (user_id, search_type, search_query, result))
    conn.commit()
    conn.close()

def get_search_history(user_id, limit=20):
    conn = sqlite3.connect('scanner_app.db')
    c = conn.cursor()
    
    c.execute('SELECT * FROM search_history WHERE user_id = ? ORDER BY timestamp DESC LIMIT ?', (user_id, limit))
    data = c.fetchall()
    conn.close()
    return data

def delete_search_history_item(user_id, item_id):
    conn = sqlite3.connect('scanner_app.db')
    c = conn.cursor()
    
    c.execute('DELETE FROM search_history WHERE id = ? AND user_id = ?', (item_id, user_id))
    conn.commit()
    conn.close()

def clear_search_history(user_id):
    conn = sqlite3.connect('scanner_app.db')
    c = conn.cursor()
    
    c.execute('DELETE FROM search_history WHERE user_id = ?', (user_id,))
    conn.commit()
    conn.close()

# Analytics functions
def get_search_analytics(user_id):
    conn = sqlite3.connect('scanner_app.db')
    
    # Get search type distribution
    type_df = pd.read_sql_query(
        'SELECT search_type, COUNT(*) as count FROM search_history WHERE user_id = ? GROUP BY search_type', 
        conn, params=(user_id,)
    )
    
    # Get daily search count
    daily_df = pd.read_sql_query(
        '''SELECT DATE(timestamp) as date, COUNT(*) as count 
           FROM search_history WHERE user_id = ? 
           GROUP BY DATE(timestamp) ORDER BY date''', 
        conn, params=(user_id,)
    )
    
    # Get most common searches
    common_df = pd.read_sql_query(
        '''SELECT search_query, COUNT(*) as count 
           FROM search_history WHERE user_id = ? AND search_query IS NOT NULL
           GROUP BY search_query ORDER BY count DESC LIMIT 10''', 
        conn, params=(user_id,)
    )
    
    conn.close()
    
    return type_df, daily_df, common_df

# Function to get Gemini response
def get_gemini_response(image, prompt):
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content([image, prompt])
    return response.text

# Function to identify object
def identify_object(image):
    prompt = """
    Analyze this image and identify what it contains. 
    Provide a detailed response in the following format:
    
    **Object/Subject:** [What is in the image]
    **Type/Category:** [Category it belongs to - animal, plant, product, etc.]
    
    **Key Characteristics:**
    - [Characteristic 1]
    - [Characteristic 2]
    - [Characteristic 3]
    
    **Additional Information:**
    [Any relevant information about what's in the image]
    
    **Interesting Facts:**
    - [Fact 1]
    - [Fact 2]
    
    **Practical Applications/Uses:**
    - [Use 1]
    - [Use 2]
    
    If the image doesn't contain anything recognizable, please state that clearly.
    """
    return get_gemini_response(image, prompt)

# Function to get detailed facts
def get_detailed_facts(subject):
    prompt = f"""
    Provide comprehensive information about {subject} in a structured format.
    Include information about:
    - Origin/history
    - Key features
    - Varieties/types (if applicable)
    - Significance to humans
    - Interesting trivia
    """
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content(prompt)
    return response.text

# Authentication UI
def authentication_section():
    st.sidebar.title("User Account")
    
    menu = ["Login", "Register"]
    choice = st.sidebar.selectbox("Menu", menu)
    
    if choice == "Login":
        st.sidebar.subheader("Login")
        username = st.sidebar.text_input("Username")
        password = st.sidebar.text_input("Password", type="password")
        
        if st.sidebar.button("Login"):
            user = login_user(username, password)
            if user:
                st.session_state.user = user
                st.session_state.logged_in = True
                st.sidebar.success(f"Logged in as {username}")
                st.rerun()
            else:
                st.sidebar.error("Invalid username or password")
    
    elif choice == "Register":
        st.sidebar.subheader("Create Account")
        new_username = st.sidebar.text_input("Username")
        new_email = st.sidebar.text_input("Email")
        new_password = st.sidebar.text_input("Password", type="password")
        confirm_password = st.sidebar.text_input("Confirm Password", type="password")
        
        if st.sidebar.button("Register"):
            if new_password == confirm_password:
                if create_user(new_username, new_password, new_email):
                    st.sidebar.success("Account created successfully. Please login.")
                else:
                    st.sidebar.error("Username already exists")
            else:
                st.sidebar.error("Passwords do not match")

# User profile section
def user_profile_section():
    if st.session_state.logged_in:
        st.sidebar.subheader(f"Welcome, {st.session_state.user[1]}")
        
        if st.sidebar.button("Logout"):
            st.session_state.logged_in = False
            st.session_state.user = None
            st.sidebar.success("Logged out successfully")
            st.rerun()
        
        # Show search history
        st.sidebar.subheader("Search History")
        history = get_search_history(st.session_state.user[0], 5)
        
        if history:
            for item in history:
                st.sidebar.text(f"{item[3]} - {item[5]}")
            
            if st.sidebar.button("View Full History & Analytics"):
                st.session_state.show_analytics = True
        else:
            st.sidebar.info("No search history yet")

# Analytics section
def analytics_section():
    if st.session_state.get('show_analytics', False):
        st.header("Search Analytics")
        
        # Get analytics data
        type_df, daily_df, common_df = get_search_analytics(st.session_state.user[0])
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Search type distribution
            if not type_df.empty:
                st.subheader("Search Type Distribution")
                fig, ax = plt.subplots()
                ax.pie(type_df['count'], labels=type_df['search_type'], autopct='%1.1f%%')
                st.pyplot(fig)
        
        with col2:
            # Daily search activity
            if not daily_df.empty:
                st.subheader("Daily Search Activity")
                fig, ax = plt.subplots()
                ax.plot(pd.to_datetime(daily_df['date']), daily_df['count'])
                plt.xticks(rotation=45)
                st.pyplot(fig)
        
        # Most common searches
        if not common_df.empty:
            st.subheader("Most Common Searches")
            fig, ax = plt.subplots()
            ax.barh(common_df['search_query'], common_df['count'])
            st.pyplot(fig)
        
        # Full search history with delete options
        st.subheader("Full Search History")
        full_history = get_search_history(st.session_state.user[0], 50)
        
        if full_history:
            for item in full_history:
                col1, col2, col3 = st.columns([3, 1, 1])
                with col1:
                    st.write(f"**{item[3]}** - {item[5]}")
                with col2:
                    if st.button("View", key=f"view_{item[0]}"):
                        st.session_state.view_result = item[4]
                with col3:
                    if st.button("Delete", key=f"delete_{item[0]}"):
                        delete_search_history_item(st.session_state.user[0], item[0])
                        st.rerun()
            
            if st.button("Clear All History"):
                clear_search_history(st.session_state.user[0])
                st.rerun()
        
        # View result if requested
        if st.session_state.get('view_result'):
            st.subheader("Search Result")
            st.markdown(st.session_state.view_result)
        
        if st.button("Back to Scanner"):
            st.session_state.show_analytics = False
            st.rerun()

# Main scanner app
def scanner_app():
    st.title("🔍 Universal Scanner & Identifier")
    st.markdown("Use your camera to scan any object, animal, or plant to identify it and learn more about it.")
    
    # Create tabs for different input methods
    tab1, tab2, tab3 = st.tabs(["Camera Scanner", "Upload Image", "Sample Images"])
    
    with tab1:
        st.subheader("Real-time Camera Scanner")
        st.write("Allow camera access and point at an object to identify it.")
        
        # Use Streamlit's built-in camera input
        camera_img = st.camera_input("Take a picture", key="camera_scanner")
        
        if camera_img is not None:
            # Display the captured image
            img = Image.open(camera_img)
            st.image(img, caption="Captured Image", use_column_width=True)
            
            # Process the image
            if st.button("Identify This Image", key="identify_camera_btn"):
                with st.spinner("Analyzing image..."):
                    try:
                        # Identify object
                        response = identify_object(img)
                        
                        # Save to search history
                        if st.session_state.logged_in:
                            add_to_search_history(
                                st.session_state.user[0], 
                                "Camera Scan", 
                                "Camera Image", 
                                response
                            )
                        
                        # Display results
                        st.success("Analysis Complete!")
                        st.markdown(response)
                        
                        # Extract subject for more facts
                        if "**Object/Subject:**" in response:
                            subject_line = response.split("**Object/Subject:**")[1].split("\n")[0].strip()
                            subject = subject_line.split("[")[1].split("]")[0] if "[" in subject_line else subject_line
                            st.subheader(f"More About {subject}")
                            facts = get_detailed_facts(subject)
                            st.markdown(facts)
                    
                    except Exception as e:
                        st.error(f"An error occurred: {str(e)}")
                        st.info("Please try again.")
    
    with tab2:
        st.subheader("Upload an Image")
        # File upload
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], key="uploader")
        
        if uploaded_file is not None:
            # Display the uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Process the image
            if st.button("Identify This Image", key="identify_btn"):
                with st.spinner("Analyzing image..."):
                    try:
                        # Identify object
                        response = identify_object(image)
                        
                        # Save to search history
                        if st.session_state.logged_in:
                            add_to_search_history(
                                st.session_state.user[0], 
                                "Image Upload", 
                                uploaded_file.name, 
                                response
                            )
                        
                        # Display results
                        st.success("Analysis Complete!")
                        st.markdown(response)
                        
                        # Extract subject for more facts
                        if "**Object/Subject:**" in response:
                            subject_line = response.split("**Object/Subject:**")[1].split("\n")[0].strip()
                            subject = subject_line.split("[")[1].split("]")[0] if "[" in subject_line else subject_line
                            st.subheader(f"More About {subject}")
                            facts = get_detailed_facts(subject)
                            st.markdown(facts)
                    
                    except Exception as e:
                        st.error(f"An error occurred: {str(e)}")
                        st.info("Please try again with a different image.")

    with tab3:
        st.subheader("Try with these sample images:")
        col1, col2, col3 = st.columns(3)
        
        sample_images = {
            "Cat": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4d/Cat_November_2010-1a.jpg/800px-Cat_November_2010-1a.jpg",
            "Flower": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/40/Sunflower_sky_backdrop.jpg/800px-Sunflower_sky_backdrop.jpg",
            "Landmark": "https://upload.wikimedia.org/wikipedia/commons/thumb/6/6a/Eiffel_Tower_from_the_Champ_de_Mars%2C_Paris_May_2008.jpg/800px-Eiffel_Tower_from_the_Champ_de_Mars%2C_Paris_May_2008.jpg"
        }
        
        def analyze_sample(image_url, sample_name):
            with st.spinner("Analyzing..."):
                try:
                    response = requests.get(image_url)
                    img = Image.open(io.BytesIO(response.content))
                    result = identify_object(img)
                    
                    # Save to search history
                    if st.session_state.logged_in:
                        add_to_search_history(
                            st.session_state.user[0], 
                            "Sample Image", 
                            sample_name, 
                            result
                        )
                    
                    st.markdown(result)
                except Exception as e:
                    st.error(f"Error loading sample image: {str(e)}")
        
        with col1:
            st.image(sample_images["Cat"], caption="Sample Cat", use_column_width=True)
            if st.button("Identify Cat", key="cat_btn"):
                analyze_sample(sample_images["Cat"], "Cat")
        
        with col2:
            st.image(sample_images["Flower"], caption="Sample Flower", use_column_width=True)
            if st.button("Identify Flower", key="flower_btn"):
                analyze_sample(sample_images["Flower"], "Flower")
        
        with col3:
            st.image(sample_images["Landmark"], caption="Sample Landmark", use_column_width=True)
            if st.button("Identify Landmark", key="landmark_btn"):
                analyze_sample(sample_images["Landmark"], "Landmark")

# Main app
def main():
    # Initialize session state
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if 'user' not in st.session_state:
        st.session_state.user = None
    if 'show_analytics' not in st.session_state:
        st.session_state.show_analytics = False
    
    # Show authentication or app based on login status
    if not st.session_state.logged_in:
        authentication_section()
        st.info("Please login or register to use the scanner and save your search history")
    else:
        user_profile_section()
        
        if st.session_state.get('show_analytics', False):
            analytics_section()
        else:
            scanner_app()

if __name__ == "__main__":
    main()
