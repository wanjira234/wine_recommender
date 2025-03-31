# Import python lib
import streamlit as st
import time
import pandas as pd
import numpy as np
from surprise import Dataset, Reader
from surprise import KNNBaseline
from PIL import Image
import os
import hashlib
import json
import plotly.express as px
import plotly.figure_factory as ff
from datetime import datetime

# User data functions
def load_user_data():
    try:
        with open('user_data.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {"users": {}}

def save_user_data(data):
    with open('user_data.json', 'w') as f:
        json.dump(data, f, indent=4)

def create_user_account(username, password, email, name):
    user_data = load_user_data()
    if username in user_data["users"]:
        return False, "Username already exists"
    
    user_data["users"][username] = {
        "password": hash_password(password),
        "email": email,
        "name": name,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "preferences": {
            "wine_types": [],
            "traits": []
        }
    }
    save_user_data(user_data)
    return True, "Account created successfully"

def check_user_login(username, password):
    user_data = load_user_data()
    if username not in user_data["users"]:
        return False, "User not found"
    
    if user_data["users"][username]["password"] == hash_password(password):
        return True, "Login successful"
    return False, "Invalid password"

def update_user_preferences(username, wine_types, traits):
    user_data = load_user_data()
    if username not in user_data["users"]:
        return False, "User not found"
    
    user_data["users"][username]["preferences"]["wine_types"] = wine_types
    user_data["users"][username]["preferences"]["traits"] = traits
    save_user_data(user_data)
    return True, "Preferences updated successfully"

def load_image(image_path):
    try:
        return Image.open(image_path)
    except Exception as e:
        st.error(f"Error loading image: {e}")
        return None

def get_wine_icon(variety):
    variety = variety.lower()
    if 'sparkling' in variety:
        return '<i class="fa-solid fa-champagne-glasses"></i>'
    elif 'rosé' in variety or 'rose' in variety:
        return '<i class="fa-solid fa-wine-glass-empty"></i>'
    elif 'red' in variety or 'blend' in variety:
        return '<i class="fa-solid fa-wine-glass"></i>'
    else:
        return '<i class="fa-solid fa-wine-bottle"></i>'

# Import wine dataframes
df_wine_model = pd.read_pickle('data/df_wine_us_rate.pkl')
df_wine_combi = pd.read_pickle('data/df_wine_combi.pkl')

# Load wine images
wine_images = {
    'red': 'images/red-wine.jpg',
    'white': 'images/white-wine.jpg',
    'rose': 'images/rose-wine.jpg',
    'sparkling': 'images/sparkling-wine.jpg'
}

# Admin credentials file
ADMIN_FILE = 'admin_credentials.json'

def load_admin_credentials():
    if os.path.exists(ADMIN_FILE):
        with open(ADMIN_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_admin_credentials(credentials):
    with open(ADMIN_FILE, 'w') as f:
        json.dump(credentials, f)

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def check_login(username, password):
    credentials = load_admin_credentials()
    hashed_password = hash_password(password)
    return username in credentials and credentials[username] == hashed_password

def create_admin_account(username, password):
    credentials = load_admin_credentials()
    if username in credentials:
        return False, "Username already exists"
    
    credentials[username] = hash_password(password)
    save_admin_credentials(credentials)
    return True, "Admin account created successfully"

# Initialize session state
if 'is_admin' not in st.session_state:
    st.session_state.is_admin = False
if 'wines_displayed' not in st.session_state:
    st.session_state.wines_displayed = 10
if 'cart' not in st.session_state:
    st.session_state.cart = []
if 'cart_total' not in st.session_state:
    st.session_state.cart_total = 0.0

# Initialize session state for orders
if 'orders' not in st.session_state:
    st.session_state.orders = []

# Instantiate the list of wine traits
all_traits = ['almond', 'anise', 'apple', 'apricot', 'baked', 'baking_spices', 'berry', 'black_cherry', 'black_currant', 'black_pepper', 'black_tea', 'blackberry', 'blueberry', 
              'boysenberry', 'bramble', 'bright', 'butter', 'candy', 'caramel', 'cardamom', 'cassis', 'cedar', 'chalk', 'cherry', 'chocolate', 'cinnamon', 'citrus', 'clean', 'closed',
              'clove', 'cocoa', 'coffee', 'cola', 'complex', 'concentrated', 'cranberry', 'cream', 'crisp', 'dark', 'dark_chocolate', 'dense', 'depth', 'dried_herb', 'dry', 'dust',
              'earth', 'edgy', 'elderberry', 'elegant', 'fennel', 'firm', 'flower', 'forest_floor', 'french_oak', 'fresh', 'fruit', 'full_bodied', 'game', 'grapefruit', 'graphite',
              'green', 'gripping', 'grippy', 'hearty', 'herb', 'honey', 'honeysuckle', 'jam', 'juicy', 'lavender', 'leafy', 'lean', 'leather', 'lemon', 'lemon_peel', 'length', 'licorice',
              'light_bodied', 'lime', 'lush', 'meaty', 'medium_bodied', 'melon', 'milk_chocolate', 'minerality', 'mint', 'nutmeg', 'oak', 'olive', 'orange', 'orange_peel', 'peach',
              'pear', 'pencil_lead', 'pepper', 'pine', 'pineapple', 'plum', 'plush', 'polished', 'pomegranate', 'powerful', 'purple', 'purple_flower', 'raspberry', 'refreshing',
              'restrained', 'rich', 'ripe', 'robust', 'rose', 'round', 'sage', 'salt', 'savory', 'sharp', 'silky', 'smoke', 'smoked_meat', 'smooth', 'soft', 'sparkling', 'spice',
              'steel', 'stone', 'strawberry', 'succulent', 'supple', 'sweet', 'tangy', 'tannin', 'tar', 'tart', 'tea', 'thick', 'thyme', 'tight', 'toast', 'tobacco', 'tropical_fruit',
              'vanilla', 'velvety', 'vibrant', 'violet', 'warm', 'weight', 'wet_rocks', 'white', 'white_pepper', 'wood']

# Add custom theme and styling at the beginning of the app
def set_custom_theme():
    st.markdown("""
        <style>
        /* Theme Variables */
        :root {
            --primary-bg: #ffffff;
            --secondary-bg: #f8f9fa;
            --accent-burgundy: #722F37;
            --accent-gold: #B4A169;
            --text-dark: #2C1810;
            --text-gray: #495057;
            --border-light: #dee2e6;
        }

        /* Form Labels and Text */
        .stTextInput label, 
        .stNumberInput label, 
        .stTextArea label, 
        .stSelectbox label,
        .stDateInput label,
        div[data-baseweb="select"] label {
            color: var(--text-dark) !important;
            font-family: 'Playfair Display', serif;
            font-size: 1rem;
            font-weight: 500;
        }

        /* Input field text */
        .stTextInput input,
        .stNumberInput input,
        .stTextArea textarea,
        .stSelectbox select,
        .stDateInput input {
            color: var(--text-dark) !important;
            background-color: white !important;
            border: 1px solid var(--border-light);
            border-radius: 4px;
            padding: 0.75rem;
            font-family: 'Playfair Display', serif;
        }

        /* Expander styling */
        .streamlit-expanderHeader {
            color: var(--text-dark) !important;
            background-color: var(--secondary-bg);
            border: 1px solid var(--border-light);
        }

        /* Data editor styling */
        .stDataFrame {
            color: var(--text-dark);
        }

        .stDataFrame td {
            color: var(--text-dark) !important;
        }

        .stDataFrame th {
            color: white !important;
            background-color: var(--accent-burgundy) !important;
        }

        /* Radio buttons */
        .stRadio label {
            color: var(--text-dark) !important;
        }

        /* Markdown text */
        .stMarkdown {
            color: var(--text-dark) !important;
        }

        /* Headers */
        h1, h2, h3, h4, h5, h6 {
            color: var(--text-dark) !important;
        }

        .stMarkdown p {
            color: var(--text-dark) !important;
        }

        /* Selectbox text */
        div[data-baseweb="select"] span {
            color: var(--text-dark) !important;
        }

        /* Dropdown options */
        div[role="listbox"] div[role="option"] {
            color: var(--text-dark) !important;
        }

        /* Global Styles */
        .stApp {
            background-color: var(--primary-bg);
            color: var(--text-dark);
            font-family: 'Playfair Display', serif;
        }

        /* Tabs styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            background-color: var(--secondary-bg);
            padding: 0.5rem;
            border-radius: 4px;
            border-bottom: 2px solid var(--accent-burgundy);
        }

        .stTabs [data-baseweb="tab"] {
            background-color: transparent;
            color: var(--text-gray);
            border: none;
            padding: 0.5rem 1rem;
            font-family: 'Playfair Display', serif;
            font-size: 1rem;
        }

        .stTabs [data-baseweb="tab"][aria-selected="true"] {
            background-color: var(--accent-burgundy);
            color: white;
            border-radius: 4px;
        }

        /* Links */
        a {
            color: var(--accent-burgundy);
            text-decoration: none;
            transition: color 0.2s ease;
        }

        a:hover {
            color: var(--accent-gold);
        }

        /* Buttons */
        .stButton button {
            background-color: var(--accent-burgundy);
            color: white;
            border: none;
            border-radius: 4px;
            padding: 0.75rem 1.5rem;
            font-family: 'Playfair Display', serif;
            font-size: 1rem;
            font-weight: 500;
            transition: all 0.2s ease;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        .stButton button:hover {
            background-color: var(--accent-gold);
            transform: translateY(-2px);
        }

        /* Input fields */
        .stTextInput input:focus,
        .stNumberInput input:focus,
        .stTextArea textarea:focus {
            border-color: var(--accent-burgundy);
            box-shadow: 0 0 0 2px rgba(114, 47, 55, 0.1);
        }

        /* Multiselect */
        div[data-baseweb="select"] {
            background-color: white;
            border-radius: 4px;
        }

        div[data-baseweb="select"] > div {
            background-color: white;
            border: 1px solid var(--border-light);
            color: var(--text-dark);
        }

        div[role="listbox"] {
            background-color: white;
            border: 1px solid var(--border-light);
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }

        div[role="option"] {
            color: var(--text-dark);
            padding: 0.5rem 1rem;
        }

        div[role="option"]:hover {
            background-color: var(--secondary-bg);
            color: var(--accent-burgundy);
        }

        /* Tables */
        .dataframe {
            background-color: white;
            border: 1px solid var(--border-light);
            border-radius: 4px;
        }

        .dataframe th {
            background-color: var(--accent-burgundy);
            color: white;
            padding: 0.75rem 1rem;
            font-family: 'Playfair Display', serif;
        }

        .dataframe td {
            color: var(--text-dark);
            border-top: 1px solid var(--border-light);
            padding: 0.75rem 1rem;
        }

        /* Wine Cards */
        .wine-card {
            background-color: white;
            border-radius: 12px;
            overflow: hidden;
            margin: 0.5rem;
            transition: all 0.3s ease;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            height: 100%;
            display: flex;
            flex-direction: column;
            position: relative;
            cursor: pointer;
            width: calc(100% - 1rem);  /* Reduced margin */
            min-width: 250px;  /* Smaller minimum width */
            max-width: 400px;  /* Smaller maximum width */
            margin: 0.5rem auto;  /* Reduced margin */
        }

        /* Optimize image loading */
        .wine-image-container {
            position: relative;
            width: 100%;
            padding-bottom: 0;
            height: 200px;
            overflow: hidden;
            background-color: var(--secondary-bg);
        }

        .wine-card img {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            object-fit: cover;
            transform: scale(1);
            transition: transform 0.5s cubic-bezier(0.4, 0, 0.2, 1);
            will-change: transform;  /* Optimize performance */
        }

        .wine-card-content {
            padding: 1.5rem;
            background: white;
            flex-grow: 1;
            display: flex;
            flex-direction: column;
            gap: 0.75rem;
            width: 100%;
        }

        .wine-title {
            color: #2C1810;
            font-size: 1rem;  /* Smaller font */
            font-weight: 600;
            font-family: 'Playfair Display', serif;
            line-height: 1.4;
            margin: 0;
            overflow: hidden;
            text-overflow: ellipsis;
            display: -webkit-box;
            -webkit-line-clamp: 2;
            -webkit-box-orient: vertical;
            min-height: 2.4em;  /* Reduced minimum height */
        }

        .wine-price {
            color: #B4A169;
            font-size: 1.1rem;  /* Smaller font */
            font-weight: 600;
            margin: 0;
            margin-bottom: 0.25rem;  /* Reduced margin */
        }

        .wine-details {
            color: #495057;
            font-size: 0.9rem;  /* Smaller font */
            line-height: 1.4;
            margin: 0;
            margin-bottom: 0.25rem;  /* Reduced margin */
        }

        /* Adjust the grid layout for better spacing */
        .stMarkdown {
            width: 100%;
        }

        [data-testid="stHorizontalBlock"] {
            gap: 2rem;
            padding: 0 1rem;
        }

        /* Wine Details Page */
        .wine-traits-detail {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            margin-top: 1rem;
        }

        .wine-traits-detail .wine-trait {
            background-color: #F8F9FA;
            color: #2C1810;
            padding: 8px 16px;
            border-radius: 20px;
            font-size: 0.9rem;
            border: 1px solid #E9ECEF;
        }

        .add-to-cart-btn {
            background-color: #B4A169;
            color: white;
            border: none;
            padding: 16px 32px;
            border-radius: 30px;
            font-family: 'Playfair Display', serif;
            font-size: 1.1rem;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.3s ease;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-top: 2rem;
            width: 100%;
        }

        .add-to-cart-btn:hover {
            background-color: #2C1810;
            transform: translateY(-2px);
        }

        /* Load More Button */
        [data-testid="stButton"] button {
            background-color: transparent;
            color: #2C1810;
            border: 2px solid #2C1810;
            padding: 12px 24px;
            border-radius: 25px;
            font-family: 'Playfair Display', serif;
            font-size: 1rem;
            margin-top: 2rem;
            transition: all 0.3s ease;
        }

        [data-testid="stButton"] button:hover {
            background-color: #2C1810;
            color: white;
            transform: translateY(-2px);
        }

        /* Grid Layout */
        [data-testid="column"] {
            padding: 1rem;
            min-width: 350px;
        }

        /* Masonry-like grid */
        .wine-grid {
            column-count: 3;
            column-gap: 1rem;
            padding: 1rem;
        }

        .wine-card-wrapper {
            break-inside: avoid;
            margin-bottom: 1rem;
        }

        /* Metrics */
        [data-testid="stMetricValue"] {
            color: var(--accent-burgundy);
            font-size: 2rem;
            font-weight: 600;
            font-family: 'Playfair Display', serif;
        }

        [data-testid="stMetricLabel"] {
            color: var(--text-gray);
            font-size: 0.875rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        /* Sidebar */
        [data-testid="stSidebar"] {
            background-color: var(--secondary-bg);
            border-right: 1px solid var(--border-light);
        }

        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
            color: var(--text-dark);
        }

        /* Success/Error messages */
        .stSuccess {
            background-color: var(--accent-burgundy) !important;
            color: white !important;
            border: 1px solid #5a252c !important;
            padding: 0.75rem 1rem !important;
            border-radius: 4px !important;
            font-family: 'Playfair Display', serif !important;
            font-size: 1rem !important;
            font-weight: 500 !important;
        }

        /* Target all elements inside success message with high specificity */
        .stSuccess > div {
            color: white !important;
        }
        
        .stSuccess > div > div {
            color: white !important;
        }
        
        .stSuccess > div > div > div {
            color: white !important;
        }
        
        .stSuccess > div > div > div > div {
            color: white !important;
        }
        
        .stSuccess [data-testid="stMarkdownContainer"] {
            color: white !important;
        }
        
        .stSuccess [data-testid="stMarkdownContainer"] p {
            color: white !important;
        }

        .stWarning {
            background-color: #FFF3E0 !important;
            color: #884A00 !important;
            border: 1px solid #FFB74D !important;
            padding: 0.75rem 1rem !important;
            border-radius: 4px !important;
            font-family: 'Playfair Display', serif !important;
            font-size: 1rem !important;
        }

        .stError {
            background-color: #FCE8E8 !important;
            color: #C62828 !important;
            border: 1px solid #F5C2C2 !important;
            padding: 0.75rem 1rem !important;
            border-radius: 4px !important;
            font-family: 'Playfair Display', serif !important;
            font-size: 1rem !important;
        }

        /* Make sure message text is visible */
        .stWarning div, .stError div {
            color: inherit !important;
            font-weight: 500 !important;
        }

        /* Remove background image */
        [data-testid="stAppViewContainer"] {
            background-image: none !important;
        }

        [data-testid="stVerticalBlock"] {
            background-color: transparent !important;
        }

        /* Wine Catalog Container */
        .wine-catalog-container {
            width: 100%;
            max-width: 1200px;  /* Reduced max width */
            margin: 0 auto;
            padding: 1rem;  /* Reduced padding */
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));  /* Smaller minimum column width */
            gap: 1rem;  /* Reduced gap */
        }

        .wine-item {
            background: white;
            border-radius: 8px;  /* Smaller border radius */
            overflow: hidden;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);  /* Lighter shadow */
            transition: transform 0.3s ease;
            height: 100%;
            display: flex;
            flex-direction: column;
        }

        .wine-image {
            position: relative;
            width: 100%;
            height: 200px;
            background-color: var(--secondary-bg);
            overflow: hidden;
        }

        .wine-image img {
            width: 100%;
            height: 100%;
            object-fit: cover;
            transition: transform 0.3s ease;
        }

        .wine-image img.hidden {
            display: none;
        }

        .wine-icon-fallback {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            display: none;
            align-items: center;
            justify-content: center;
            background-color: var(--secondary-bg);
            color: var(--accent-burgundy);
            font-size: 4rem;
            transition: all 0.3s ease;
        }

        .wine-icon-fallback.show {
            display: flex;
        }

        .wine-icon-fallback:hover {
            color: var(--accent-gold);
        }

        .wine-item:hover {
            transform: translateY(-4px);
        }

        .wine-item:hover .wine-image img {
            transform: scale(1.05);
        }

        .wine-content {
            padding: 1rem;  /* Reduced padding */
            flex-grow: 1;
            display: flex;
            flex-direction: column;
            gap: 0.5rem;  /* Reduced gap */
        }

        .wine-rating {
            padding: 0.25rem 0.75rem;  /* Reduced padding */
            font-size: 0.8rem;  /* Smaller font */
        }

        @media (max-width: 1200px) {
            .wine-catalog-container {
                grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));  /* Even smaller on medium screens */
                padding: 1rem;
                gap: 1rem;
            }
        }

        @media (max-width: 768px) {
            .wine-catalog-container {
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));  /* Smallest on mobile */
                padding: 0.75rem;
                gap: 0.75rem;
            }
            
            .wine-content {
                padding: 0.75rem;
            }
        }

        /* Performance optimizations */
        * {
            -webkit-font-smoothing: antialiased;
            -moz-osx-font-smoothing: grayscale;
        }

        /* Reduce layout shifts */
        .wine-title {
            min-height: 3em;
            display: -webkit-box;
            -webkit-line-clamp: 2;
            -webkit-box-orient: vertical;
            overflow: hidden;
        }

        /* Optimize button rendering */
        .stButton button {
            backface-visibility: hidden;
            transform: translateZ(0);
            -webkit-font-smoothing: subpixel-antialiased;
        }

        /* Shopping Cart Styles */
        .stMarkdown {
            color: var(--text-dark) !important;
        }

        .stMarkdown p, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
            color: var(--text-dark) !important;
        }

        /* Form text color */
        .stTextInput label, 
        .stNumberInput label, 
        .stTextArea label, 
        .stSelectbox label,
        .stDateInput label {
            color: var(--text-dark) !important;
        }

        /* Input field text */
        .stTextInput input,
        .stNumberInput input,
        .stTextArea textarea,
        .stSelectbox select,
        .stDateInput input {
            color: var(--text-dark) !important;
        }

        /* Cart item text */
        [data-testid="stVerticalBlock"] {
            color: var(--text-dark) !important;
        }

        [data-testid="stVerticalBlock"] p {
            color: var(--text-dark) !important;
        }

        /* Success message text */
        .stSuccess {
            background-color: var(--accent-burgundy) !important;
            color: white !important;
        }

        .stSuccess > div {
            color: white !important;
        }

        /* Info message text */
        .stInfo {
            color: var(--text-dark) !important;
            background-color: var(--secondary-bg) !important;
        }

        .stInfo > div {
            color: var(--text-dark) !important;
        }

        /* Error message text */
        .stError {
            color: #C62828 !important;
        }

        .stError > div {
            color: #C62828 !important;
        }

        /* Form field text */
        .stForm label {
            color: var(--text-dark) !important;
        }

        .stForm [data-baseweb="input"] {
            color: var(--text-dark) !important;
        }

        /* Total price text */
        [data-testid="stHeader"] {
            color: var(--text-dark) !important;
        }

        /* Cart item details */
        .wine-details-cart {
            color: var(--text-dark) !important;
            font-family: 'Playfair Display', serif !important;
        }

        .wine-price-cart {
            color: var(--accent-gold) !important;
            font-weight: 600 !important;
        }

        /* Button text */
        .stButton button {
            color: white !important;
        }

        /* Wine Icon Placeholders */
        .wine-icon-placeholder {
            width: 100%;
            height: 200px;
            display: flex;
            align-items: center;
            justify-content: center;
            background-color: var(--secondary-bg);
            color: var(--accent-burgundy);
            font-size: 4rem;
            transition: all 0.3s ease;
        }

        .wine-icon-placeholder:hover {
            color: var(--accent-gold);
            transform: scale(1.05);
        }
        </style>

        <script>
        function handleImageError(img) {
            img.classList.add('hidden');
            img.nextElementSibling.classList.add('show');
        }
        </script>

        <link href="https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;500;600&display=swap" rel="stylesheet">
        <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    """, unsafe_allow_html=True)

# Call the theme function at the start
set_custom_theme()

#---------------------------------------------------------------------------------------------------------

# Function to instantiate the model & return the est recsys scores
def recommend_scores():
    
    # Instantiate reader & data for surprise
    reader = Reader(rating_scale=(88, 100))
    data = Dataset.load_from_df(df_wine_model, reader)
    
    # Instantiate recsys model
    sim_options={'name':'cosine'}
    model = KNNBaseline(k=35, min_k=1, sim_options=sim_options, verbose=False)

    # Train & fit the data into model
    train=data.build_full_trainset()
    model.fit(train)

    # Start the model to compute the best estimate match score on wine list
    recommend_list = []
    user_wines = df_wine_model[df_wine_model.taster_name == 'mockuser']['title'].unique()
    not_user_wines = []
    
    for wine in df_wine_model['title'].unique():
        if wine not in user_wines:
            not_user_wines.append(wine)

    for wine in not_user_wines:
        wine_compatibility = []
        prediction = model.predict(uid='mockuser', iid=wine)
        wine_compatibility.append(prediction.iid)
        wine_compatibility.append(prediction.est)
        recommend_list.append(wine_compatibility)
        
    result_df = pd.DataFrame(recommend_list, columns = ['title', 'est_match_pts'])
    
    return result_df

# Function for background image
def add_bg_from_url():
    st.markdown(
        f"""
        <style>
        
        [data-testid="stAppViewContainer"] {{
        background-image: url("images/learn-more.jpg");
        background-attachment: fixed;
        background-size: cover      
        }}
        
        [data-testid="stVerticalBlock"] {{
        background-color: rgba(255,255,255,0.5)
        }}
        
        </style>
        """,
        unsafe_allow_html=True
    )

#----------------------------------------------------------------------------------------------------------

# Create plotly charts with theme
def create_plotly_chart(df, x, title, xlabel, ylabel="Number of Wines", nbins=50):
    fig = px.histogram(df, x=x, nbins=nbins,
                      title=title,
                      labels={x: xlabel, 'count': ylabel})
    
    fig.update_layout(
        plot_bgcolor='#2d2d2d',
        paper_bgcolor='#2d2d2d',
        font_color='#E5E5E5',
        title_font_color='#FFD700',
        showlegend=False,
        title_font_family="Playfair Display",
        title_font_size=24,
    )
    
    fig.update_xaxes(gridcolor='#404040', tickfont_color='#E5E5E5')
    fig.update_yaxes(gridcolor='#404040', tickfont_color='#E5E5E5')
    
    return fig

# Create tabs for different sections
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(["Home", "About Us", "Learn", "Contact", "Wine Recommender", "Wine Catalog", "Shopping Cart", "Admin Dashboard"])

# Home Section
with tab1:
    # Hero Section with learn-more.jpg background
    hero_image = load_image('images/learn-more.jpg')
    if hero_image:
        st.image(hero_image, use_container_width=True)
    
    st.markdown("""
        <h1 style="text-align: center; font-size: 4rem; margin-bottom: 1rem; font-family: 'Playfair Display', serif; color: #2C1810;">Welcome to WineWise</h1>
    """, unsafe_allow_html=True)

    featured_cols = st.columns(2)
    with featured_cols[0]:
        red_wine_image = load_image('images/red-wine.jpg')
        if red_wine_image:
            st.image(red_wine_image, caption="Premium Red Wines", use_container_width=True)
        
        rose_wine_image = load_image('images/rose-wine.jpg')
        if rose_wine_image:
            st.image(rose_wine_image, caption="Rosé Collection", use_container_width=True)
    
    with featured_cols[1]:
        white_wine_image = load_image('images/white-wine.jpg')
        if white_wine_image:
            st.image(white_wine_image, caption="White Wine Selection", use_container_width=True)
        
        sparkling_wine_image = load_image('images/sparkling-wine.jpg')
        if sparkling_wine_image:
            st.image(sparkling_wine_image, caption="Sparkling Wines", use_container_width=True)

# About Us Section
with tab2:
    st.markdown("""
        <div style='padding: 2rem; background-color: rgba(255, 255, 255, 0.9); border-radius: 10px;'>
            <h1 style='text-align: center; color: #722F37; margin-bottom: 2rem;'>About WineWise</h1>
            
            <h2 style='color: #2C1810; margin-bottom: 1rem;'>Our Story</h2>
            <p style='color: #495057; font-size: 1.1rem; line-height: 1.6; margin-bottom: 2rem;'>
                WineWise was founded with a passion for connecting wine enthusiasts with their perfect bottle. 
                Our sophisticated recommendation system combines expert knowledge with advanced technology to 
                provide personalized wine suggestions tailored to your unique taste preferences.
            </p>

            <h2 style='color: #2C1810; margin-bottom: 1rem;'>Our Mission</h2>
            <p style='color: #495057; font-size: 1.1rem; line-height: 1.6; margin-bottom: 2rem;'>
                We strive to make the world of wine accessible to everyone, from novices to connoisseurs. 
                Through our carefully curated selection and intelligent recommendations, we help you discover 
                wines that perfectly match your palate and preferences.
            </p>

            <h2 style='color: #2C1810; margin-bottom: 1rem;'>Why Choose Us</h2>
            <ul style='color: #495057; font-size: 1.1rem; line-height: 1.6; margin-bottom: 2rem;'>
                <li>Expert-curated wine selection</li>
                <li>Personalized recommendations</li>
                <li>Educational resources and wine knowledge</li>
                <li>Secure shopping experience</li>
                <li>Exceptional customer service</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

# Learn Section
with tab3:
    st.markdown("""
        <div style='padding: 2rem; background-color: rgba(255, 255, 255, 0.9); border-radius: 10px;'>
            <h1 style='text-align: center; color: #722F37; margin-bottom: 2rem;'>Learn About Wine</h1>
        </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
            <div style='padding: 1rem; background-color: rgba(255, 255, 255, 0.9); border-radius: 10px;'>
                <h2 style='color: #2C1810; margin-bottom: 1rem;'>Wine Basics</h2>
                <ul style='color: #495057; font-size: 1.1rem; line-height: 1.6;'>
                    <li>Understanding Wine Types</li>
                    <li>Wine Tasting Techniques</li>
                    <li>Food and Wine Pairing</li>
                    <li>Wine Storage Tips</li>
                    <li>Reading Wine Labels</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
            <div style='padding: 1rem; background-color: rgba(255, 255, 255, 0.9); border-radius: 10px;'>
                <h2 style='color: #2C1810; margin-bottom: 1rem;'>Advanced Topics</h2>
                <ul style='color: #495057; font-size: 1.1rem; line-height: 1.6;'>
                    <li>Wine Regions</li>
                    <li>Winemaking Process</li>
                    <li>Wine Investment</li>
                    <li>Vintage Guides</li>
                    <li>Wine Certification</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)

# Contact Section
with tab4:
    st.markdown("""
        <div style='padding: 2rem; background-color: rgba(255, 255, 255, 0.9); border-radius: 10px;'>
            <h1 style='text-align: center; color: #722F37; margin-bottom: 2rem;'>Contact Us</h1>
        </div>
    """, unsafe_allow_html=True)

    contact_col1, contact_col2 = st.columns(2)

    with contact_col1:
        st.markdown("""
            <div style='padding: 1rem; background-color: rgba(255, 255, 255, 0.9); border-radius: 10px;'>
                <h2 style='color: #2C1810; margin-bottom: 1rem;'>Get in Touch</h2>
                <form>
                    <div style='margin-bottom: 1rem;'>
                        <label style='display: block; margin-bottom: 0.5rem; color: #2C1810;'>Name</label>
                        <input type='text' style='width: 100%; padding: 0.5rem; border-radius: 4px; border: 1px solid #dee2e6;'>
                    </div>
                    <div style='margin-bottom: 1rem;'>
                        <label style='display: block; margin-bottom: 0.5rem; color: #2C1810;'>Email</label>
                        <input type='email' style='width: 100%; padding: 0.5rem; border-radius: 4px; border: 1px solid #dee2e6;'>
                    </div>
                    <div style='margin-bottom: 1rem;'>
                        <label style='display: block; margin-bottom: 0.5rem; color: #2C1810;'>Message</label>
                        <textarea style='width: 100%; padding: 0.5rem; border-radius: 4px; border: 1px solid #dee2e6; height: 150px;'></textarea>
                    </div>
                </form>
            </div>
        """, unsafe_allow_html=True)
        st.button("Send Message")

    with contact_col2:
        st.markdown("""
            <div style='padding: 1rem; background-color: rgba(255, 255, 255, 0.9); border-radius: 10px;'>
                <h2 style='color: #2C1810; margin-bottom: 1rem;'>Support Hours</h2>
                <p style='color: #495057; font-size: 1.1rem; line-height: 1.6;'>
                    Monday - Friday: 9:00 AM - 6:00 PM<br>
                    Saturday: 10:00 AM - 4:00 PM<br>
                    Sunday: Closed
                </p>
                
                <h2 style='color: #2C1810; margin: 1.5rem 0 1rem;'>Contact Information</h2>
                <p style='color: #495057; font-size: 1.1rem; line-height: 1.6;'>
                    Email: support@winewise.com<br>
                    Phone: (555) 123-4567<br>
                    Address: 123 Wine Street, Vintage Valley, CA 90210
                </p>
            </div>
        """, unsafe_allow_html=True)

# Wine Recommender Section
with tab5:
    st.markdown("""
        <div style='padding: 2rem; background-color: rgba(255, 255, 255, 0.9); border-radius: 10px;'>
            <h1 style='text-align: center; color: #722F37; margin-bottom: 2rem;'>Wine Recommender</h1>
            <p style='text-align: center; color: #495057; font-size: 1.1rem; line-height: 1.6; margin-bottom: 2rem;'>
                Get personalized wine recommendations based on your preferences
            </p>
        </div>
    """, unsafe_allow_html=True)

    # Add your existing wine recommendation logic here
    st.write("Select your wine preferences:")
    
    col1, col2 = st.columns(2)
    with col1:
        price_range = st.slider("Price Range ($)", 0, 500, (20, 100), key="recommender_price")
        wine_type = st.multiselect("Wine Type", ["Red", "White", "Rosé", "Sparkling"], key="recommender_wine_type")
    
    with col2:
        rating_min = st.slider("Minimum Rating", 80, 100, 85, key="recommender_rating")
        country = st.multiselect("Country", df_wine_combi['country'].unique().tolist(), key="recommender_country")

    if st.button("Get Recommendations"):
        st.write("Based on your preferences, here are our recommendations:")
        # Add your recommendation display logic here

# Wine Catalog Section
with tab6:
    st.markdown("""
        <div style='padding: 2rem; background-color: rgba(255, 255, 255, 0.9); border-radius: 10px;'>
            <h1 style='text-align: center; color: #722F37; margin-bottom: 2rem;'>Wine Catalog</h1>
        </div>
    """, unsafe_allow_html=True)

    # Add search and filter options
    search = st.text_input("Search wines by name, variety, or winery")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        price_filter = st.slider("Price Range ($)", 0, 500, (0, 500), key="catalog_price")
    with col2:
        variety_filter = st.multiselect("Variety", df_wine_combi['variety'].unique().tolist(), key="catalog_variety")
    with col3:
        country_filter = st.multiselect("Country", df_wine_combi['country'].unique().tolist(), key="catalog_country")

    # Display wine catalog
    filtered_wines = df_wine_combi  # Add your filtering logic here
    st.write(f"Showing {len(filtered_wines)} wines")
    
    # Display wines in a grid
    for i in range(0, len(filtered_wines), 3):
        cols = st.columns(3)
        for j in range(3):
            if i + j < len(filtered_wines):
                with cols[j]:
                    wine = filtered_wines.iloc[i + j]
                    st.markdown(f"""
                        <div style='padding: 1rem; background-color: white; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
                            <h3 style='color: #2C1810;'>{wine['title']}</h3>
                            <p style='color: #B4A169; font-weight: 600;'>${wine['price']}</p>
                            <p style='color: #495057;'>{wine['variety']} • {wine['country']}</p>
                            <p style='color: #495057;'>{wine['points']} points</p>
                        </div>
                    """, unsafe_allow_html=True)

# Shopping Cart Section
with tab7:
    st.markdown("""
        <div style='padding: 2rem; background-color: rgba(255, 255, 255, 0.9); border-radius: 10px;'>
            <h1 style='text-align: center; color: #722F37; margin-bottom: 2rem;'>Shopping Cart</h1>
        </div>
    """, unsafe_allow_html=True)

    if not st.session_state.user_logged_in:
        st.warning("Please create an account or log in to access the shopping cart")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Create Account")
            new_username = st.text_input("Username", key="new_username")
            new_password = st.text_input("Password", type="password", key="new_password")
            new_email = st.text_input("Email", key="new_email")
            new_name = st.text_input("Full Name", key="new_name")
            
            if st.button("Create Account"):
                success, message = create_user_account(new_username, new_password, new_email, new_name)
                if success:
                    st.session_state.user_logged_in = True
                    st.session_state.current_user = new_username
                    st.success(message)
                else:
                    st.error(message)
        
        with col2:
            st.markdown("### Login")
            login_username = st.text_input("Username", key="login_username")
            login_password = st.text_input("Password", type="password", key="login_password")
            
            if st.button("Login"):
                success, message = check_user_login(login_username, login_password)
                if success:
                    st.session_state.user_logged_in = True
                    st.session_state.current_user = login_username
                    st.success(message)
                else:
                    st.error(message)
    
    elif st.session_state.account_creation_step == 1:
        st.markdown("### Step 1: Select Your Preferred Wine Types")
        wine_types = st.multiselect(
            "Choose your favorite wine types",
            ["Red", "White", "Rosé", "Sparkling"],
            key="setup_wine_types"
        )
        
        if st.button("Next", key="next_step1"):
            if wine_types:
                st.session_state.account_creation_step = 2
            else:
                st.warning("Please select at least one wine type")
    
    elif st.session_state.account_creation_step == 2:
        st.markdown("### Step 2: Select Your Preferred Wine Traits")
        traits = st.multiselect(
            "Choose your preferred wine traits",
            all_traits,
            key="setup_traits"
        )
        
        if st.button("Complete Setup", key="complete_setup"):
            if traits:
                success, message = update_user_preferences(
                    st.session_state.current_user,
                    st.session_state.pref_wine_types,
                    traits
                )
                if success:
                    st.session_state.account_creation_step = 3
                    st.success("Preferences saved successfully!")
                else:
                    st.error(message)
            else:
                st.warning("Please select at least one trait")
    
    elif st.session_state.account_creation_step == 3:
        st.success("Account setup complete! You can now start shopping.")
        if st.button("Start Shopping", key="start_shopping"):
            st.session_state.account_creation_step = 4
    
    else:
        if not st.session_state.cart:
            st.info("Your cart is empty")
        else:
            for item in st.session_state.cart:
                st.markdown(f"""
                    <div style='padding: 1rem; background-color: white; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 1rem;'>
                        <h3 style='color: #2C1810;'>{item['title']}</h3>
                        <p style='color: #B4A169; font-weight: 600;'>${item['price']}</p>
                        <p style='color: #495057;'>{item['variety']} • {item['country']}</p>
                    </div>
                """, unsafe_allow_html=True)
            
            st.markdown(f"""
                <div style='padding: 1rem; background-color: white; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
                    <h2 style='color: #2C1810;'>Total: ${st.session_state.cart_total:.2f}</h2>
                </div>
            """, unsafe_allow_html=True)
            
            if st.button("Proceed to Checkout"):
                st.success("Thank you for your order!")
                st.session_state.cart = []
                st.session_state.cart_total = 0.0

# Admin Dashboard Section
with tab8:
    st.markdown("""
        <div style='padding: 2rem; background-color: rgba(255, 255, 255, 0.9); border-radius: 10px;'>
            <h1 style='text-align: center; color: #722F37; margin-bottom: 2rem;'>Admin Dashboard</h1>
        </div>
    """, unsafe_allow_html=True)

    if not st.session_state.is_admin:
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Login"):
                if check_login(username, password):
                    st.session_state.is_admin = True
                    st.success("Successfully logged in!")
                else:
                    st.error("Invalid credentials")
        
        with col2:
            if st.button("Create Admin Account"):
                if username and password:
                    success, message = create_admin_account(username, password)
                    if success:
                        st.success(message)
                    else:
                        st.error(message)
                else:
                    st.error("Please enter both username and password")
    
    else:
        # Display admin dashboard content
        st.markdown("""
            <div style='padding: 1rem; background-color: white; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
                <h2 style='color: #2C1810;'>Recent Orders</h2>
            </div>
        """, unsafe_allow_html=True)

        # Display orders if any
        if st.session_state.orders:
            for order in st.session_state.orders:
                st.markdown(f"""
                    <div style='padding: 1rem; background-color: white; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 1rem;'>
                        <h3 style='color: #2C1810;'>Order #{order['order_id']}</h3>
                        <p style='color: #495057;'>Customer: {order['customer']}</p>
                        <p style='color: #495057;'>Amount: ${order['amount']:.2f}</p>
                        <p style='color: #495057;'>Status: {order['status']}</p>
                        <p style='color: #495057;'>Date: {order['date']}</p>
                    </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No orders yet")

        if st.button("Logout"):
            st.session_state.is_admin = False
            st.success("Successfully logged out!")