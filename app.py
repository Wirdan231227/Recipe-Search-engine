import streamlit as st
import pandas as pd
import ast
import base64
from main import load_data, build_tfidf_model, search_recipes

# --- Set Background Image ---
def set_background(image_file):
    with open(image_file, "rb") as image:
        encoded = base64.b64encode(image.read()).decode()
    css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{encoded}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

set_background("image.jpg") 

# --- Custom CSS ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Orbitron', sans-serif;
        background-color: #0f1117;
        color: #ffffff;
    }
    .title {
        font-size: 48px;
        text-align: center;
        margin-top: 20px;
        margin-bottom: 0;
        color: #00f2ff;
        text-shadow: 0 0 20px #00f2ff;
    }
    .subtitle {
        font-size: 20px;
        text-align: center;
        margin-bottom: 30px;
        color: #aaaaaa;
    }
    .box {
        border: 1px solid #00f2ff;
        border-radius: 15px;
        padding: 20px;
        margin-bottom: 20px;
        background: rgba(0, 242, 255, 0.05);
        box-shadow: 0 0 15px rgba(0, 242, 255, 0.3);
    }
    .sidebar .css-1d391kg {
        background-color: #1f2633;
    }
    .stSlider, .stMultiSelect, .stTextInput {
        background-color: #1f2633;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# --- Title ---
st.markdown('<div class="title">RECIPE SEARCH ENGINE</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Enter ingredients you want to search for</div>', unsafe_allow_html=True)

# --- Load Data ---
df = load_data()
vectorizer, tfidf_matrix = build_tfidf_model(df)

# --- Sidebar Filters ---
with st.sidebar:
    st.markdown("### 🔧 Search Filters")

    vegetarian_only = st.checkbox("🥗 Vegetarian only")
    max_cook_time = st.slider("⏱️ Max cooking time", 5, 300, 60)

    common_ingredients = [
        'salt', 'butter', 'sugar', 'onion', 'water', 'eggs', 'olive oil', 'flour',
        'milk', 'garlic cloves', 'baking powder', 'pepper', 'vanilla extract',
        'ground cinnamon', 'all-purpose flour', 'baking soda', 'ground black pepper',
        'vegetable oil', 'brown sugar'
    ]
    selected_ingredients = st.multiselect("🧂 Common Ingredients", common_ingredients)
    top_n = st.slider("📋 Number of recipes to show", 1, 20, 5)

# --- Main Input ---
query = st.text_input("🔍 Enter ingredients:")

# --- Recipe Results ---
if query or selected_ingredients:
    final_query = query + ' ' + ' '.join(selected_ingredients)
    results = search_recipes(final_query, vectorizer, tfidf_matrix, df, top_n, vegetarian_only, max_cook_time)

    if not results.empty:
        for _, row in results.iterrows():
            st.markdown('<div class="box">', unsafe_allow_html=True)
            st.subheader(row['name'])
            st.markdown(f"**🏷️ Tags:** {row['tags']}")
            st.markdown(f"**🕒 Cooking Time:** {row['minutes']} mins")

            # Show full step
            if row['steps_list']:
                steps_formatted = "<br>".join([f"{i+1}. {step}" for i, step in enumerate(row['steps_list'])])
                st.markdown(f"**👨‍🍳 Steps:**<br>{steps_formatted}", unsafe_allow_html=True)
            else:
                st.markdown("**👨‍🍳 Steps:** No steps available.")

            st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.warning("❌ No recipes found. Try different ingredients.")
else:
    st.info("👆 Start by typing ingredients or selecting from the sidebar.")
