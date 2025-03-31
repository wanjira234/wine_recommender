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
        /* Main theme colors */
        :root {
            --background-color: #FFFFFF;
            --navy: #1B2B4B;
            --navy-dark: #0F1C33;
            --blush: #E8C7C3;
            --blush-light: #F5E6E4;
            --gold: #B4A169;
            --text-color: #1B2B4B;
            --gray-100: #F8F9FA;
            --gray-200: #E9ECEF;
            --gray-300: #DEE2E6;
            --gray-400: #CED4DA;
            --gray-500: #ADB5BD;
            --success: #28A745;
            --warning: #FFC107;
            --danger: #DC3545;
        }
        
        /* Global styles */
        .stApp {
            background-color: var(--background-color);
            color: var(--text-color);
            font-family: 'Cormorant Garamond', serif;
        }
        
        /* Headers */
        h1, h2, h3 {
            color: var(--navy);
            font-family: 'Cormorant Garamond', serif;
            font-weight: 600;
            letter-spacing: 0.5px;
            margin-bottom: 1.5rem;
        }
        
        /* Metrics styling */
        [data-testid="stMetricValue"] {
            color: var(--navy);
            font-size: 2rem;
            font-family: 'Cormorant Garamond', serif;
            font-weight: 600;
        }
        
        [data-testid="stMetricDelta"] {
            color: var(--success);
            font-size: 1rem;
        }
        
        [data-testid="stMetricLabel"] {
            color: var(--text-color);
            font-family: 'Cormorant Garamond', serif;
            font-size: 1rem;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        /* Cards styling */
        .wine-card {
            background-color: white;
            border-radius: 8px;
            padding: 1.5rem;
            margin: 1rem 0;
            box-shadow: 0 2px 4px rgba(27, 43, 75, 0.05);
            transition: all 0.3s ease;
            border: 1px solid var(--blush-light);
        }
        
        .wine-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(27, 43, 75, 0.1);
            border-color: var(--blush);
        }
        
        /* Dashboard cards */
        .dashboard-card {
            background: white;
            padding: 1.5rem;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
            margin-bottom: 1rem;
        }
        
        /* Sidebar styling */
        [data-testid="stSidebar"] {
            background-color: white;
            border-right: 1px solid var(--gray-200);
            padding-top: 2rem;
        }

        /* Navigation menu styling */
        [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
            background-color: transparent !important;
        }

        [data-testid="stSidebar"] .st-bk {
            background-color: transparent !important;
        }

        /* Radio buttons in sidebar */
        [data-testid="stSidebar"] .st-cc {
            color: var(--navy) !important;
            font-family: 'Cormorant Garamond', serif;
            font-size: 1.1rem;
            font-weight: 500;
        }

        /* Style all text in sidebar */
        [data-testid="stSidebar"] div {
            color: var(--navy) !important;
        }

        /* Radio button labels */
        [data-testid="stSidebar"] [role="radiogroup"] {
            color: var(--navy) !important;
        }

        [data-testid="stSidebar"] [role="radiogroup"] label {
            color: var(--navy) !important;
            font-family: 'Cormorant Garamond', serif;
            font-size: 1.1rem;
            padding: 0.5rem 0;
        }

        /* Radio button text */
        [data-testid="stSidebar"] [role="radio"] {
            color: var(--navy) !important;
        }

        [data-testid="stSidebar"] [role="radio"] div {
            color: var(--navy) !important;
        }

        /* Sidebar title */
        [data-testid="stSidebar"] h1 {
            color: var(--navy) !important;
            font-size: 1.5rem;
            margin-bottom: 2rem;
            padding: 0 1rem;
        }

        /* Sidebar buttons */
        [data-testid="stSidebar"] .stButton button {
            background-color: var(--navy);
            color: white !important;
            margin-top: 1rem;
        }

        /* Override any white text in sidebar */
        [data-testid="stSidebar"] * {
            color: var(--navy) !important;
        }

        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
            color: var(--navy) !important;
        }
        
        /* Button styling */
        .stButton button {
            background-color: var(--navy);
            color: white;
            border: none;
            padding: 0.5rem 1.5rem;
            border-radius: 4px;
            font-family: 'Cormorant Garamond', serif;
            font-size: 1rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 1px;
            transition: all 0.3s ease;
            width: 100%;
        }
        
        .stButton button:hover {
            background-color: var(--navy-dark);
            transform: translateY(-1px);
        }
        
        /* Input fields */
        .stTextInput input, .stNumberInput input, .stDateInput input {
            background-color: white;
            border: 1px solid var(--gray-300);
            border-radius: 4px;
            padding: 0.5rem;
            font-family: 'Cormorant Garamond', serif;
            transition: all 0.3s ease;
            color: var(--text-color) !important;
        }
        
        .stTextInput input:focus, .stNumberInput input:focus, .stDateInput input:focus {
            border-color: var(--navy);
            box-shadow: 0 0 0 2px rgba(27, 43, 75, 0.1);
        }

        /* Select box and multiselect styling */
        .stSelectbox select, .stMultiSelect select {
            background-color: white;
            color: var(--text-color) !important;
            border: 1px solid var(--gray-300);
            border-radius: 4px;
        }

        div[data-baseweb="select"] > div {
            background-color: white;
            color: var(--text-color);
        }

        div[data-baseweb="select"] input {
            color: var(--text-color) !important;
        }

        div[data-baseweb="select"] > div:hover {
            border-color: var(--navy);
        }

        div[data-baseweb="select"] > div[data-focused="true"] {
            border-color: var(--navy);
            box-shadow: 0 0 0 2px rgba(27, 43, 75, 0.1);
        }

        /* Dropdown menu items */
        div[role="listbox"] {
            background-color: white;
        }

        div[role="listbox"] div[role="option"] {
            color: var(--text-color);
        }

        div[role="listbox"] div[role="option"]:hover {
            background-color: var(--blush-light);
        }

        /* Selected items in multiselect */
        div[data-baseweb="tag"] {
            background-color: var(--navy) !important;
            color: white !important;
        }

        div[data-baseweb="tag"] span {
            color: white !important;
        }
        
        /* Tables */
        .dataframe {
            background-color: white;
            border: 1px solid var(--gray-200);
            border-radius: 4px;
            font-family: 'Cormorant Garamond', serif;
        }
        
        .dataframe th {
            background-color: var(--blush-light);
            color: var(--navy);
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            padding: 0.75rem 1rem;
        }
        
        .dataframe td {
            padding: 0.75rem 1rem;
            border-top: 1px solid var(--gray-200);
        }
        
        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 0;
            background-color: transparent;
            border-bottom: 2px solid var(--gray-200);
            padding: 0 1rem;
        }
        
        .stTabs [data-baseweb="tab"] {
            background-color: transparent;
            color: var(--text-color);
            border: none;
            padding: 1rem 1.5rem;
            font-family: 'Cormorant Garamond', serif;
            font-size: 1rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
            color: var(--navy);
            border-bottom: 2px solid var(--navy);
            background-color: transparent;
        }
        
        /* Charts */
        [data-testid="stPlotlyChart"] > div {
            background-color: white;
            border-radius: 8px;
            padding: 1rem;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        }
        
        /* Expander */
        .streamlit-expanderHeader {
            background-color: white;
            border: 1px solid var(--gray-200);
            border-radius: 4px;
            color: var(--navy);
            font-family: 'Cormorant Garamond', serif;
            font-weight: 600;
        }
        
        /* Success/Error messages */
        .stSuccess, .stError {
            padding: 0.75rem 1rem;
            border-radius: 4px;
            margin: 1rem 0;
        }
        
        .stSuccess {
            background-color: var(--blush-light);
            color: var(--navy);
            border: 1px solid var(--blush);
        }
        
        .stError {
            background-color: #F8D7DA;
            color: #721C24;
            border: 1px solid #F5C6CB;
        }
        
        /* Form styling */
        [data-testid="stForm"] {
            background-color: white;
            padding: 1.5rem;
            border-radius: 8px;
            border: 1px solid var(--gray-200);
        }

        /* Form labels */
        .stTextInput label, .stNumberInput label, .stTextArea label, .stSelectbox label {
            color: var(--navy) !important;
            font-family: 'Cormorant Garamond', serif;
            font-size: 1rem;
            font-weight: 500;
            margin-bottom: 0.5rem;
        }

        /* Form inputs */
        .stTextInput input, .stNumberInput input, .stTextArea textarea {
            background-color: white !important;
            border: 1px solid var(--gray-300) !important;
            border-radius: 4px !important;
            padding: 0.5rem !important;
            font-family: 'Cormorant Garamond', serif !important;
            color: var(--navy) !important;
            font-size: 1rem !important;
        }

        /* Input focus state */
        .stTextInput input:focus, .stNumberInput input:focus, .stTextArea textarea:focus {
            border-color: var(--navy) !important;
            box-shadow: 0 0 0 2px rgba(27, 43, 75, 0.1) !important;
        }

        /* Expander styling */
        .streamlit-expanderHeader {
            background-color: var(--blush-light) !important;
            color: var(--navy) !important;
            font-family: 'Cormorant Garamond', serif !important;
            font-weight: 600 !important;
            padding: 1rem !important;
            border: none !important;
            border-radius: 4px !important;
        }

        .streamlit-expanderContent {
            background-color: white !important;
            border: 1px solid var(--gray-200) !important;
            border-radius: 0 0 4px 4px !important;
            padding: 1.5rem !important;
        }

        /* Data editor styling */
        .stDataFrame {
            background-color: white !important;
        }

        .stDataFrame td {
            color: var(--navy) !important;
            font-family: 'Cormorant Garamond', serif !important;
        }

        .stDataFrame th {
            background-color: var(--blush-light) !important;
            color: var(--navy) !important;
            font-family: 'Cormorant Garamond', serif !important;
            font-weight: 600 !important;
        }

        /* Headers in sections */
        [data-testid="stHeader"] {
            color: var(--navy) !important;
        }

        .main h1, .main h2, .main h3, .main .stMarkdown {
            color: var(--navy) !important;
        }

        /* Subheader styling */
        .stSubheader {
            color: var(--navy) !important;
            font-family: 'Cormorant Garamond', serif !important;
            font-size: 1.25rem !important;
            font-weight: 600 !important;
            margin-bottom: 1rem !important;
        }

        /* Success/Error messages */
        .stSuccess, .stError {
            padding: 0.75rem 1rem;
            border-radius: 4px;
            margin: 1rem 0;
            color: var(--navy) !important;
        }
        
        </style>
        
        <link href="https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@400;600&display=swap" rel="stylesheet">
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
tab1, tab2, tab3 = st.tabs(["Wine Recommender", "Wine Catalog", "Admin Dashboard"])

# Admin Login Section
with tab3:
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
            example_orders = pd.DataFrame({
                'Order ID': ['#001', '#002', '#003'],
                'Customer': ['John D.', 'Jane S.', 'Mike R.'],
                'Amount': ['$245.00', '$189.00', '$567.00'],
                'Status': ['Processing', 'Shipped', 'Delivered'],
                'Date': ['2024-02-01', '2024-02-02', '2024-02-03']
            })
            st.dataframe(example_orders, use_container_width=True)
            
            # Payment Analytics
            st.subheader("Payment Analytics")
            payment_cols = st.columns(2)
            with payment_cols[0]:
                st.metric("Total Revenue", "$12,456")
                st.metric("Pending Payments", "$890")
            with payment_cols[1]:
                st.metric("Refunds", "$234")
                st.metric("Average Order Value", "$245")

with tab1:
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

with tab2:
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
    
    # Get the wines to display
    catalog_display = catalog_df[['title', 'variety', 'winery', 'country', 'province', 'points', 'price', 'description']].head(st.session_state.wines_displayed)
    
    # Create columns for the grid layout
    cols = st.columns(3)
    
    # Display wines in a grid
    for idx, row in catalog_display.iterrows():
        # Determine which image to use based on index
        image_type = ['red', 'white', 'rose', 'sparkling'][idx % 4]
        
        # Create a card for each wine
        with cols[idx % 3]:
            st.markdown('<div class="wine-card">', unsafe_allow_html=True)
            st.image(wine_images[image_type], use_container_width=True)
            st.markdown(f"""
                <div class="wine-title">{row['title']}</div>
                <div class="wine-details">Variety: {row['variety']}</div>
                <div class="wine-details">Winery: {row['winery']}</div>
                <div class="wine-details">Country: {row['country']}</div>
                <div class="wine-details">Province: {row['province']}</div>
                <div class="wine-details">Rating: {row['points']}/100</div>
                <div class="wine-price">${row['price']:,.2f}</div>
                <div class="wine-details">{row['description'][:100]}...</div>
                </div>
            """, unsafe_allow_html=True)
    
    # Load More button
    if st.session_state.wines_displayed < len(catalog_df):
        if st.button("Load More Wines"):
            st.session_state.wines_displayed += 10
            st.rerun()
