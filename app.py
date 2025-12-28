import streamlit as st
import pandas as pd
import ast
import base64
from main import load_data, build_tfidf_model, search_recipes

# -----------------------------
# Background Image
# -----------------------------

def set_background(image_file):
    with open(image_file, "rb") as img:
        encoded = base64.b64encode(img.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-position: center;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_background("image.jpg")

# -----------------------------
# Custom CSS
# -----------------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500&display=swap');

html, body, [class*="css"] {
    font-family: 'Orbitron', sans-serif;
    color: white;
}

.title {
    font-size: 48px;
    text-align: center;
    color: #00f2ff;
}

.subtitle {
    text-align: center;
    color: #aaaaaa;
    margin-bottom: 30px;
}

.box {
    border: 1px solid #00f2ff;
    border-radius: 15px;
    padding: 20px;
    margin-bottom: 25px;
    background: rgba(0, 242, 255, 0.05);
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Title
# -----------------------------

st.markdown('<div class="title">RECIPE SEARCH ENGINE</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Enter ingredients you want to search for</div>', unsafe_allow_html=True)

# -----------------------------
# Load Data
# -----------------------------

df = load_data()
vectorizer, tfidf_matrix = build_tfidf_model(df)

# -----------------------------
# Sidebar
# -----------------------------

with st.sidebar:
    vegetarian_only = st.checkbox("🥗 Vegetarian only")
    max_cook_time = st.slider("⏱️ Max cooking time", 5, 300, 60)
    top_n = st.slider("📋 Recipes to show", 1, 20, 5)

# -----------------------------
# Search
# -----------------------------

query = st.text_input("🔍 Enter ingredients:")

if query:
    results = search_recipes(
        query,
        vectorizer,
        tfidf_matrix,
        df,
        top_n,
        vegetarian_only,
        max_cook_time
    )

    if not results.empty:
        for _, row in results.iterrows():
            st.markdown('<div class="box">', unsafe_allow_html=True)

            # Name
            st.subheader(row['name'])

            # Description
            if isinstance(row['description'], str) and row['description'].strip():
                st.markdown(f"**📝 Description:** {row['description']}")

            # Ingredients
            try:
                ingredients = ast.literal_eval(row['ingredients'])
                st.markdown(f"**🧂 Ingredients:** {', '.join(ingredients)}")
            except:
                st.markdown("**🧂 Ingredients:** Not available")

            # Cooking Time
            st.markdown(f"**🕒 Cooking Time:** {row['minutes']} mins")

            # Nutrition
            try:
                nutrition = ast.literal_eval(row['nutrition'])
                st.markdown("""
**🥗 Nutrition (per serving):**
- 🔥 Calories: {} kcal
- 🧈 Total Fat: {} g
- 🍬 Sugar: {} g
- 🧂 Sodium: {} mg
- 💪 Protein: {} g
- 🧀 Saturated Fat: {} g
- 🍞 Carbohydrates: {} g
                """.format(*nutrition))
            except:
                st.markdown("**🥗 Nutrition:** Not available")

            # Steps (✅ FIXED)
            try:
                steps_list = ast.literal_eval(row['steps'])
                if isinstance(steps_list, list) and steps_list:
                    steps = "<br>".join(
                        [f"{i+1}. {step}" for i, step in enumerate(steps_list)]
                    )
                    st.markdown(f"**👨‍🍳 Steps:**<br>{steps}", unsafe_allow_html=True)
                else:
                    st.markdown("**👨‍🍳 Steps:** No steps available.")
            except:
                st.markdown("**👨‍🍳 Steps:** No steps available.")

            # ✅ Correct indentation
            st.markdown('</div>', unsafe_allow_html=True)

    else:
        st.warning("❌ No recipes found. Try different ingredients.")
else:
    st.info("👆 Start by typing ingredients.")
