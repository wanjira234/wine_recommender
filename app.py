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
        st.image(hero_image, use_column_width=True)
    
    st.markdown("""
        <h1 style="text-align: center; font-size: 4rem; margin-bottom: 1rem; font-family: 'Playfair Display', serif; color: #2C1810;">Welcome to WineWise</h1>
        <p style="text-align: center; font-size: 1.8rem; margin-bottom: 2rem; font-family: 'Playfair Display', serif; color: #2C1810;">Your Personal Wine Companion</p>
        <div style="text-align: center;">
            <a href="#wine-recommender" style="background-color: #722F37; color: white; padding: 1.2rem 2.5rem; border-radius: 30px; text-decoration: none; font-size: 1.3rem; transition: all 0.3s ease; font-family: 'Playfair Display', serif;">Discover Your Perfect Wine</a>
        </div>
    """, unsafe_allow_html=True)
    
    # Featured Collections with wine images
    st.markdown("""
        <h2 style="text-align: center; color: #722F37; font-size: 2.5rem; margin: 3rem 0; font-family: 'Playfair Display', serif;">Our Collections</h2>
    """, unsafe_allow_html=True)
    
    featured_cols = st.columns(2)
    with featured_cols[0]:
        red_wine_image = load_image('images/red-wine.jpg')
        if red_wine_image:
            st.image(red_wine_image, caption="Premium Red Wines", use_column_width=True)
        
        rose_wine_image = load_image('images/rose-wine.jpg')
        if rose_wine_image:
            st.image(rose_wine_image, caption="Rosé Collection", use_column_width=True)
    
    with featured_cols[1]:
        white_wine_image = load_image('images/white-wine.jpg')
        if white_wine_image:
            st.image(white_wine_image, caption="White Wine Selection", use_column_width=True)
        
        sparkling_wine_image = load_image('images/sparkling-wine.jpg')
        if sparkling_wine_image:
            st.image(sparkling_wine_image, caption="Sparkling Wines", use_column_width=True)
    
    # Featured Experience Section
    st.markdown("""
        <div style="text-align: center; padding: 4rem 2rem; margin-top: 3rem; background: linear-gradient(rgba(0,0,0,0.7), rgba(0,0,0,0.7)), url('images/learn-more.jpg'); background-size: cover; background-position: center; color: white; border-radius: 10px;">
            <h2 style="font-size: 2.5rem; margin-bottom: 2rem; font-family: 'Playfair Display', serif;">Experience the Art of Wine</h2>
            <div style="display: flex; justify-content: space-around; flex-wrap: wrap; gap: 2rem; max-width: 1200px; margin: 0 auto;">
                <div style="flex: 1; min-width: 250px; padding: 2rem; background: rgba(255,255,255,0.1); border-radius: 10px;">
                    <i class="fas fa-wine-glass" style="font-size: 2.5rem; margin-bottom: 1rem;"></i>
                    <h3 style="font-size: 1.5rem; margin-bottom: 1rem; font-family: 'Playfair Display', serif;">Expert Selection</h3>
                    <p style="font-size: 1.1rem; font-family: 'Playfair Display', serif;">Curated by wine connoisseurs</p>
                </div>
                <div style="flex: 1; min-width: 250px; padding: 2rem; background: rgba(255,255,255,0.1); border-radius: 10px;">
                    <i class="fas fa-star" style="font-size: 2.5rem; margin-bottom: 1rem;"></i>
                    <h3 style="font-size: 1.5rem; margin-bottom: 1rem; font-family: 'Playfair Display', serif;">Premium Quality</h3>
                    <p style="font-size: 1.1rem; font-family: 'Playfair Display', serif;">Only the finest wines</p>
                </div>
                <div style="flex: 1; min-width: 250px; padding: 2rem; background: rgba(255,255,255,0.1); border-radius: 10px;">
                    <i class="fas fa-truck" style="font-size: 2.5rem; margin-bottom: 1rem;"></i>
                    <h3 style="font-size: 1.5rem; margin-bottom: 1rem; font-family: 'Playfair Display', serif;">Fast Delivery</h3>
                    <p style="font-size: 1.1rem; font-family: 'Playfair Display', serif;">Right to your doorstep</p>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

# About Us Section
with tab2:
    st.markdown("""
        <div style="text-align: center; padding: 2rem;">
            <h1 style="color: #722F37; margin-bottom: 2rem;">About WineWise</h1>
            <div style="max-width: 800px; margin: 0 auto;">
                <p style="font-size: 1.2rem; line-height: 1.6; margin-bottom: 2rem;">
                    WineWise is your trusted companion in the world of wine. We combine cutting-edge technology with expert knowledge to help you discover the perfect wine for your palate.
                </p>
                <p style="font-size: 1.2rem; line-height: 1.6; margin-bottom: 2rem;">
                    Our mission is to make wine selection accessible and enjoyable for everyone, from beginners to connoisseurs. We believe that every wine lover deserves to find their perfect match.
                </p>
                <p style="font-size: 1.2rem; line-height: 1.6;">
                    Join us on this journey of discovery, where technology meets tradition in the world of wine.
                </p>
            </div>
        </div>
    """, unsafe_allow_html=True)

# Learn Section
with tab3:
    st.markdown("""
        <div style="text-align: center; padding: 2rem;">
            <h1 style="color: #722F37; margin-bottom: 2rem;">Learn About Wine</h1>
        </div>
    """, unsafe_allow_html=True)
    
    learn_cols = st.columns(2)
    with learn_cols[0]:
        st.markdown("""
            <div style="padding: 2rem; background: white; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                <h2 style="color: #722F37; margin-bottom: 1rem;">Wine Basics</h2>
                <ul style="list-style-type: none; padding: 0;">
                    <li style="margin-bottom: 1rem;">• Understanding Wine Varieties</li>
                    <li style="margin-bottom: 1rem;">• Reading Wine Labels</li>
                    <li style="margin-bottom: 1rem;">• Wine Tasting Techniques</li>
                    <li style="margin-bottom: 1rem;">• Food and Wine Pairing</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
    with learn_cols[1]:
        st.markdown("""
            <div style="padding: 2rem; background: white; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                <h2 style="color: #722F37; margin-bottom: 1rem;">Advanced Topics</h2>
                <ul style="list-style-type: none; padding: 0;">
                    <li style="margin-bottom: 1rem;">• Wine Regions and Terroir</li>
                    <li style="margin-bottom: 1rem;">• Wine Production Process</li>
                    <li style="margin-bottom: 1rem;">• Wine Storage and Aging</li>
                    <li style="margin-bottom: 1rem;">• Wine Investment</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)

# Contact Section
with tab4:
    st.markdown("""
        <div style="text-align: center; padding: 2rem;">
            <h1 style="color: #722F37; margin-bottom: 2rem;">Contact Us</h1>
        </div>
    """, unsafe_allow_html=True)
    
    contact_cols = st.columns(2)
    with contact_cols[0]:
        st.markdown("""
            <div style="padding: 2rem; background: white; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                <h2 style="color: #722F37; margin-bottom: 1rem;">Get in Touch</h2>
                <p style="margin-bottom: 1rem;">Have questions about our wine recommendations?</p>
                <p style="margin-bottom: 1rem;">Need help with your order?</p>
                <p style="margin-bottom: 1rem;">Want to learn more about our services?</p>
                <p>We're here to help!</p>
            </div>
        """, unsafe_allow_html=True)
    with contact_cols[1]:
        with st.form("contact_form"):
            st.text_input("Name")
            st.text_input("Email")
            st.selectbox("Subject", ["General Inquiry", "Order Support", "Wine Recommendations", "Other"])
            st.text_area("Message")
            st.form_submit_button("Send Message")

# Admin Login Section
with tab7:
    st.title("Shopping Cart")
    
    if not st.session_state.cart:
        st.info("Your cart is empty. Browse our Wine Catalog to add some wines!")
    else:
        # Display cart items
        st.subheader("Cart Items")
        for idx, item in enumerate(st.session_state.cart):
            col1, col2, col3 = st.columns([3, 1, 1])
            with col1:
                st.markdown(f'<div class="wine-details-cart"><strong>{item["title"]}</strong><br/><em>{item["variety"]}</em> - {item["winery"]}</div>', unsafe_allow_html=True)
            with col2:
                st.markdown(f'<div class="wine-price-cart">${item["price"]:.2f}</div>', unsafe_allow_html=True)
            with col3:
                if st.button("Remove", key=f"remove_{idx}"):
                    st.session_state.cart_total -= item['price']
                    st.session_state.cart.pop(idx)
                    st.rerun()
        
        # Display total and checkout form
        st.markdown("---")
        st.markdown(f"### Total: ${st.session_state.cart_total:.2f}")
        
        # Checkout Form
        with st.form("checkout_form"):
            st.subheader("Checkout Information")
            col1, col2 = st.columns(2)
            with col1:
                first_name = st.text_input("First Name")
                email = st.text_input("Email")
                address = st.text_area("Shipping Address")
            with col2:
                last_name = st.text_input("Last Name")
                phone = st.text_input("Phone Number")
                
            # Payment Information
            st.subheader("Payment Information")
            col3, col4 = st.columns(2)
            with col3:
                card_number = st.text_input("Card Number")
                card_name = st.text_input("Name on Card")
            with col4:
                expiry = st.text_input("Expiry Date (MM/YY)")
                cvv = st.text_input("CVV", type="password")
                
            submit_order = st.form_submit_button("Place Order")
            
            if submit_order:
                if all([first_name, last_name, email, phone, address, card_number, card_name, expiry, cvv]):
                    # Create order record
                    order = {
                        'order_id': f"#{len(st.session_state.orders) + 1:03d}",
                        'customer': f"{first_name} {last_name}",
                        'amount': st.session_state.cart_total,
                        'status': 'Processing',
                        'date': datetime.now().strftime('%Y-%m-%d'),
                        'items': st.session_state.cart.copy()
                    }
                    st.session_state.orders.append(order)
                    
                    st.success("Order placed successfully! You will receive a confirmation email shortly.")
                    st.session_state.cart = []
                    st.session_state.cart_total = 0.0
                    st.rerun()
                else:
                    st.error("Please fill in all required fields.")

with tab5:
    st.title("Which wine should I get?")
    st.write("By Lee Wan Xian")
    st.write("[GitHub](https://github.com/leewanxian) | [LinkedIn](https://www.linkedin.com/in/wanxianlee)")
    st.write("You can type the wine traits that you want in the dropdown list below")
    add_bg_from_url()

    select_temptrait = st.multiselect(label = " ", options = all_traits, label_visibility = "collapsed")

    if st.button('Show me the wines!'):
        with st.spinner('Should you have some wine now?'):
            
            time.sleep(2)
            # Instantiate selected wine traits
            if len(select_temptrait) == 0:
                selected_traits = all_traits
            else:
                selected_traits = select_temptrait

            # Run recommender model
            recommend_df = recommend_scores()
        
            # Instantiate traits filter
            trait_filter = ['title']

            # Add on any traits selected by user
            trait_filter.extend(selected_traits)

            # Create dataframe for wine name and traits
            df_temp_traits = df_wine_combi.drop(columns=['taster_name', 'points', 'variety', 'designation', 'winery', 'country', 'province', 'region_1', 'region_2', 'price', 'description',
                                                         'desc_wd_count', 'traits'])

            # Code to start filtering out wines with either one of the selected traits
            df_temp_traits = df_temp_traits[trait_filter]
            df_temp_traits['sum'] = df_temp_traits.sum(axis=1, numeric_only=True)
            df_temp_traits = df_temp_traits[df_temp_traits['sum'] != 0]

            # Merge the selected wines traits with recommend scores
            df_selectrec_temp = df_temp_traits.merge(recommend_df, on='title', how='left')

            # Merge the selected wines with recommendations with df on details
            df_selectrec_detail = df_selectrec_temp.merge(df_wine_combi, on='title', how='left')
            df_selectrec_detail.drop_duplicates(inplace=True)

            # Pull out the top 10 recommendations (raw)
            df_rec_raw = df_selectrec_detail.sort_values('est_match_pts', ascending=False).head(10)
            
            # Prepare the display for the top 10 recommendations
            df_rec_final = df_rec_raw[['title', 'points', 'price', 'variety', 'country', 'province', 'winery', 'description', 'traits']].reset_index(drop=True)
            df_rec_final.index = df_rec_final.index + 1
            df_rec_final['traits']=df_rec_final['traits'].str.replace(" ", " | ")
            df_rec_final.rename(columns={'title':'Name',
                                         'country':'Country',
                                         'province':'State/Province',
                                         'variety':'Type',
                                         'winery':'Winery',
                                         'points':'Rating (Out of 100)',
                                         'price':'Price',
                                         'description':'Review',
                                         'traits':'Key Traits'}, inplace=True)
            st.balloons()
            st.dataframe(df_rec_final.style.format({"Price": "${:,.2f}"}))

with tab6:
    st.title("Wine Catalog")
    st.write("Browse through our collection of wines")
    
    # Add search functionality
    search_term = st.text_input("Search wines by name, variety, or winery:", "")
    
    # Get unique wines from the dataframe
    catalog_df = df_wine_combi.drop_duplicates(subset=['title'])
    
    # Apply search filter if search term is provided
    if search_term:
        search_term = search_term.lower()
        catalog_df = catalog_df[
            catalog_df['title'].str.lower().str.contains(search_term, na=False) |
            catalog_df['variety'].str.lower().str.contains(search_term, na=False) |
            catalog_df['winery'].str.lower().str.contains(search_term, na=False)
        ]
        
        # Show search results feedback
        result_count = len(catalog_df)
        if result_count == 0:
            st.warning("No wines found matching your search. Please try different keywords.")
        else:
            st.success(f"Found {result_count} wine{'s' if result_count != 1 else ''} matching your search.")
    
    # Get the wines to display
    catalog_display = catalog_df[['title', 'variety', 'winery', 'country', 'province', 'points', 'price', 'description', 'traits']].head(st.session_state.wines_displayed)
    
    # Initialize session states
    if 'view_wine_details' not in st.session_state:
        st.session_state.view_wine_details = False
    if 'selected_wine' not in st.session_state:
        st.session_state.selected_wine = None
    
    # Show either the catalog or the wine details
    if st.session_state.view_wine_details and st.session_state.selected_wine is not None:
        # Back button
        if st.button("← Back to Catalog"):
            st.session_state.view_wine_details = False
            st.session_state.selected_wine = None
            st.rerun()
        
        # Wine Details Page
        wine = st.session_state.selected_wine
        
        # Determine which image to use
        if 'sparkling' in wine['variety'].lower():
            image_type = 'sparkling'
        elif 'rosé' in wine['variety'].lower() or 'rose' in wine['variety'].lower():
            image_type = 'rose'
        elif 'red' in wine['variety'].lower() or 'blend' in wine['variety'].lower():
            image_type = 'red'
        else:
            image_type = 'white'
        
        # Create two columns for the details page
        col1, col2 = st.columns([3, 2])
        
        with col1:
            wine_image = load_image(wine_images[image_type])
            if wine_image:
                st.image(wine_image, caption=wine['title'], use_column_width=True)
            else:
                st.markdown(f'<div class="wine-icon-fallback show">{get_wine_icon(wine["variety"])}</div>', unsafe_allow_html=True)
            
            st.markdown(f"<h1 style='font-family: Playfair Display; color: #2C1810;'>{wine['title']}</h1>", unsafe_allow_html=True)
            st.markdown(f"<h2 style='font-family: Playfair Display; color: #B4A169; font-size: 24px;'>${wine['price']:,.2f}</h2>", unsafe_allow_html=True)
            st.markdown("### Description")
            st.write(wine['description'])
        
        with col2:
            st.markdown("### Wine Details")
            st.markdown(f"**Variety:** {wine['variety']}")
            st.markdown(f"**Winery:** {wine['winery']}")
            st.markdown(f"**Region:** {wine['province']}, {wine['country']}")
            st.markdown(f"**Rating:** {wine['points']}/100")
            
            st.markdown("### Characteristics")
            traits_list = wine['traits'].split() if isinstance(wine['traits'], str) else []
            traits_html = ''.join([f'<span class="wine-trait">{trait.replace("_", " ").title()}</span>' for trait in traits_list])
            st.markdown(f'<div class="wine-traits-detail">{traits_html}</div>', unsafe_allow_html=True)
            
            if st.button("Add to Cart", key=f"add_to_cart_{wine['title']}", type="primary", help="Add this wine to your shopping cart"):
                st.session_state.cart.append(wine)
                st.session_state.cart_total += wine['price']
                st.success(f"{wine['title']} has been added to your cart!")
                time.sleep(1)
                st.rerun()
    
    else:
        # Create a grid layout for the catalog
        cols = st.columns(3)
        for idx, row in catalog_display.iterrows():
            col = cols[idx % 3]
            with col:
                # Determine image type based on variety
                if 'sparkling' in row['variety'].lower():
                    image_type = 'sparkling'
                elif 'rosé' in row['variety'].lower() or 'rose' in row['variety'].lower():
                    image_type = 'rose'
                elif 'red' in row['variety'].lower() or 'blend' in row['variety'].lower():
                    image_type = 'red'
                else:
                    image_type = 'white'

                wine_image = load_image(wine_images[image_type])
                if wine_image:
                    st.image(wine_image, use_column_width=True)
                else:
                    st.markdown(f'<div class="wine-icon-fallback show">{get_wine_icon(row["variety"])}</div>', unsafe_allow_html=True)
                
                st.markdown(f"""
                    <div class="wine-content">
                        <div class="wine-price">${row['price']:,.2f}</div>
                        <div class="wine-title">{row['title']}</div>
                        <div class="wine-details">{row['variety']}</div>
                        <div class="wine-details">{row['winery']}</div>
                        <div class="wine-rating">Rating: {row['points']}/100</div>
                    </div>
                """, unsafe_allow_html=True)
                
                if st.button("View Details", key=f"view_details_{idx}"):
                    st.session_state.selected_wine = row
                    st.session_state.view_wine_details = True
                    st.rerun()
        
        # Load More button
        if st.session_state.wines_displayed < len(catalog_df):
            st.markdown('<div style="text-align: center; padding: 2rem;">', unsafe_allow_html=True)
            if st.button("Load More Wines"):
                st.session_state.wines_displayed += 10
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

# Add Admin Dashboard tab content
with tab8:
    st.title("Admin Dashboard")
    
    if not st.session_state.is_admin:
        st.subheader("Admin Login")
        login_col1, login_col2 = st.columns(2)
        
        with login_col1:
            st.subheader("Login")
            login_username = st.text_input("Username", key="login_username")
            login_password = st.text_input("Password", type="password", key="login_password")
            if st.button("Login", key="login_button"):
                if check_login(login_username, login_password):
                    st.session_state.is_admin = True
                    st.session_state.admin_username = login_username
                    st.success("Login successful!")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error("Invalid username or password")
        
        with login_col2:
            st.subheader("Create Admin Account")
            new_username = st.text_input("New Username", key="new_username")
            new_password = st.text_input("New Password", type="password", key="new_password")
            confirm_password = st.text_input("Confirm Password", type="password", key="confirm_password")
            if st.button("Create Account", key="create_account"):
                if new_password != confirm_password:
                    st.error("Passwords do not match")
                else:
                    success, message = create_admin_account(new_username, new_password)
                    if success:
                        st.success(message)
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error(message)
    else:
        # Admin Navigation
        st.sidebar.title(f"Welcome, {st.session_state.admin_username}!")
        
        # Navigation Menu
        admin_menu = st.sidebar.radio(
            "Dashboard Navigation",
            ["Overview", "Wine Management", "Discount Management", "Sales & Reports", "User Management", "Orders & Payments"]
        )
        
        if st.sidebar.button("Logout", key="logout"):
            st.session_state.is_admin = False
            st.session_state.admin_username = None
            st.rerun()
        
        # Overview Section
        if admin_menu == "Overview":
            st.header("Dashboard Overview")
            
            # Key Metrics
            metrics_cols = st.columns(4)
            with metrics_cols[0]:
                st.metric("Total Wines", len(df_wine_combi))
            with metrics_cols[1]:
                st.metric("Avg. Price", f"${df_wine_combi['price'].mean():.2f}")
            with metrics_cols[2]:
                st.metric("Avg. Rating", f"{df_wine_combi['points'].mean():.1f}")
            with metrics_cols[3]:
                st.metric("Total Revenue", "$127,890")  # Example static value
            
            # Charts Row
            chart_cols = st.columns(2)
            with chart_cols[0]:
                st.subheader("Sales Trend")
                # Create and style sales trend chart
                sales_data = pd.DataFrame({
                    'Month': ['Jan', 'Feb', 'Mar', 'Apr', 'May'],
                    'Sales': [12000, 15000, 18000, 16000, 21000]
                })
                fig_sales = px.line(sales_data, x='Month', y='Sales', 
                                  title='Monthly Sales Trend',
                                  markers=True)
                fig_sales.update_layout(
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    font_color='#1B2B4B',
                    title_font_color='#1B2B4B',
                    title_font_family="Cormorant Garamond",
                    title_font_size=24,
                    margin=dict(t=50, l=50, r=30, b=50),
                    yaxis=dict(
                        title=dict(
                            text='Sales ($)',
                            font=dict(size=16, color='#1B2B4B')
                        ),
                        tickfont=dict(size=14, color='#1B2B4B'),
                        tickprefix='$',
                        gridcolor='#E8C7C3'
                    ),
                    xaxis=dict(
                        title=dict(
                            text='Month',
                            font=dict(size=16, color='#1B2B4B')
                        ),
                        tickfont=dict(size=14, color='#1B2B4B'),
                        gridcolor='#E8C7C3'
                    ),
                    height=400
                )
                fig_sales.update_traces(
                    line=dict(color='#1B2B4B', width=3),
                    marker=dict(
                        color='#E8C7C3',
                        size=10,
                        line=dict(color='#1B2B4B', width=2)
                    )
                )
                st.plotly_chart(fig_sales, use_container_width=True)
            
            with chart_cols[1]:
                st.subheader("Top Wine Categories")
                # Process the data to show only top varieties
                variety_counts = df_wine_combi['variety'].value_counts()
                top_10_varieties = variety_counts.head(10)
                others_count = variety_counts[10:].sum()
                
                # Create a new series with top 10 and Others
                plot_data = pd.concat([top_10_varieties, pd.Series({'Others': others_count})])
                
                # Create and style the pie chart
                fig_category = px.pie(
                    values=plot_data.values,
                    names=plot_data.index,
                    title='Top 10 Wine Categories',
                    color_discrete_sequence=['#1B2B4B', '#0F1C33', '#E8C7C3', '#F5E6E4', '#B4A169', 
                                          '#1B2B4B', '#0F1C33', '#E8C7C3', '#F5E6E4', '#B4A169', '#ADB5BD']
                )
                fig_category.update_layout(
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    font_color='#1B2B4B',
                    title_font_color='#1B2B4B',
                    title_font_family="Cormorant Garamond",
                    title_font_size=24,
                    legend_font_family="Cormorant Garamond",
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=-0.5,
                        xanchor="center",
                        x=0.5
                    ),
                    height=500
                )
                fig_category.update_traces(
                    textposition='inside',
                    textinfo='percent+label',
                    textfont_color='white',
                    hole=0.3
                )
                st.plotly_chart(fig_category, use_container_width=True)

            # Update metrics styling
            st.markdown("""
                <style>
                [data-testid="stMetricValue"] {
                    color: #722F37 !important;
                    font-size: 2rem !important;
                    font-weight: 600 !important;
                }
                [data-testid="stMetricLabel"] {
                    color: #2C1810 !important;
                    font-size: 1rem !important;
                }
                </style>
            """, unsafe_allow_html=True)
        
        # Wine Management Section
        elif admin_menu == "Wine Management":
            st.header("Wine Management")
            
            # Add New Wine Form
            with st.expander("Add New Wine"):
                col1, col2 = st.columns(2)
                with col1:
                    st.text_input("Wine Name", key="new_wine_name")
                    st.text_input("Variety", key="new_wine_variety")
                    st.text_input("Winery", key="new_wine_winery")
                    st.number_input("Price ($)", min_value=0.0, key="new_wine_price")
                with col2:
                    st.text_input("Country", key="new_wine_country")
                    st.text_input("Province", key="new_wine_province")
                    st.number_input("Rating (0-100)", min_value=0, max_value=100, key="new_wine_rating")
                    st.number_input("Stock", min_value=0, key="new_wine_stock")
                st.text_area("Description", key="new_wine_description")
                if st.button("Add Wine", key="add_wine"):
                    st.success("Wine added successfully!")
            
            # Wine List with Edit/Delete Options
            st.subheader("Manage Existing Wines")
            wine_list = df_wine_combi[['title', 'variety', 'winery', 'price', 'points']].head(10)
            edited_df = st.data_editor(
                wine_list,
                num_rows="dynamic",
                use_container_width=True
            )
        
        # Discount Management Section
        elif admin_menu == "Discount Management":
            st.header("Discount Management")
            
            # Add New Discount
            with st.expander("Add New Discount"):
                disc_col1, disc_col2 = st.columns(2)
                with disc_col1:
                    st.selectbox("Select Wine", df_wine_combi['title'].unique(), key="discount_wine")
                    st.number_input("Discount Percentage", min_value=0, max_value=100, key="discount_percent")
                with disc_col2:
                    st.date_input("Start Date", key="discount_start")
                    st.date_input("End Date", key="discount_end")
                if st.button("Apply Discount", key="apply_discount"):
                    st.success("Discount applied successfully!")
            
            # Active Discounts Table
            st.subheader("Active Discounts")
            example_discounts = pd.DataFrame({
                'Wine': ['Wine A', 'Wine B', 'Wine C'],
                'Discount': ['20%', '15%', '25%'],
                'Start Date': ['2024-01-01', '2024-01-15', '2024-02-01'],
                'End Date': ['2024-02-01', '2024-02-15', '2024-03-01'],
                'Status': ['Active', 'Active', 'Upcoming']
            })
            st.dataframe(example_discounts, use_container_width=True)
        
        # Sales & Reports Section
        elif admin_menu == "Sales & Reports":
            st.header("Sales & Reports")
            
            # Date Range Filter
            col1, col2 = st.columns(2)
            with col1:
                st.date_input("Start Date", key="sales_start_date")
            with col2:
                st.date_input("End Date", key="sales_end_date")
            
            # Sales Metrics
            metrics_cols = st.columns(3)
            with metrics_cols[0]:
                st.metric("Total Sales", "$45,678")
            with metrics_cols[1]:
                st.metric("Orders", "234")
            with metrics_cols[2]:
                st.metric("Avg. Order Value", "$195.20")
            
            # Sales Charts
            st.subheader("Sales Analytics")
            chart_tabs = st.tabs(["Daily Sales", "Weekly Trend", "Monthly Revenue"])
            
            with chart_tabs[0]:
                st.line_chart(np.random.randn(30, 1))
            
            with chart_tabs[1]:
                st.bar_chart(np.random.randn(20, 1))
            
            with chart_tabs[2]:
                st.area_chart(np.random.randn(12, 1))
        
        # User Management Section
        elif admin_menu == "User Management":
            st.header("User Management")
            
            # User Stats
            user_cols = st.columns(3)
            with user_cols[0]:
                st.metric("Total Users", "1,234")
            with user_cols[1]:
                st.metric("Active Users", "892")
            with user_cols[2]:
                st.metric("New Users (This Month)", "+45")
            
            # User List
            st.subheader("User List")
            example_users = pd.DataFrame({
                'Username': ['user1', 'user2', 'user3'],
                'Email': ['user1@example.com', 'user2@example.com', 'user3@example.com'],
                'Role': ['Admin', 'Staff', 'Customer'],
                'Join Date': ['2024-01-01', '2024-01-15', '2024-02-01'],
                'Status': ['Active', 'Active', 'Inactive']
            })
            st.data_editor(example_users, use_container_width=True)
        
        # Orders & Payments Section
        elif admin_menu == "Orders & Payments":
            st.header("Orders & Payments")
            
            # Order Stats
            order_cols = st.columns(4)
            with order_cols[0]:
                st.metric("New Orders", "12")
            with order_cols[1]:
                st.metric("Processing", "8")
            with order_cols[2]:
                st.metric("Shipped", "15")
            with order_cols[3]:
                st.metric("Delivered", "45")
            
            # Recent Orders
            st.subheader("Recent Orders")
            
            # Define payment columns first
            payment_cols = st.columns(2)
            
            if st.session_state.orders:
                # Create DataFrame with only serializable columns
                orders_df = pd.DataFrame([{
                    'order_id': order['order_id'],
                    'customer': order['customer'],
                    'amount': order['amount'],
                    'status': order['status'],
                    'date': order['date']
                } for order in st.session_state.orders])
                st.dataframe(orders_df, use_container_width=True)
                
                # Update Order Stats
                total_revenue = sum(order['amount'] for order in st.session_state.orders)
                processing_orders = len([order for order in st.session_state.orders if order['status'] == 'Processing'])
                shipped_orders = len([order for order in st.session_state.orders if order['status'] == 'Shipped'])
                delivered_orders = len([order for order in st.session_state.orders if order['status'] == 'Delivered'])
                
                with order_cols[0]:
                    st.metric("New Orders", len(st.session_state.orders))
                with order_cols[1]:
                    st.metric("Processing", processing_orders)
                with order_cols[2]:
                    st.metric("Shipped", shipped_orders)
                with order_cols[3]:
                    st.metric("Delivered", delivered_orders)
                
                # Update Payment Analytics
                with payment_cols[0]:
                    st.metric("Total Revenue", f"${total_revenue:,.2f}")
                    st.metric("Pending Payments", f"${total_revenue:,.2f}")
                with payment_cols[1]:
                    st.metric("Refunds", "$0.00")
                    st.metric("Average Order Value", f"${total_revenue/len(st.session_state.orders):,.2f}")
            else:
                st.info("No orders have been placed yet.")
            
            # Payment Analytics
            st.subheader("Payment Analytics")
            with payment_cols[0]:
                st.metric("Total Revenue", "$12,456")
                st.metric("Pending Payments", "$890")
            with payment_cols[1]:
                st.metric("Refunds", "$234")
                st.metric("Average Order Value", "$245")

def load_image(image_path):
    try:
        return Image.open(image_path)
    except Exception as e:
        st.error(f"Error loading image: {e}")
        return None
