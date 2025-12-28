import streamlit as st
import ast
import base64
from main import load_data, build_tfidf_model, search_recipes

# -----------------------------
# Background
# -----------------------------

def set_background(image):
    with open(image, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
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
# Styling
# -----------------------------

st.markdown("""
<style>
.box {
    border: 1px solid #00f2ff;
    border-radius: 15px;
    padding: 20px;
    margin-bottom: 25px;
    background: rgba(0, 242, 255, 0.06);
}
.highlight {
    color: #00f2ff;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Title
# -----------------------------

st.title("🍽️ Recipe Search Engine")
st.caption("Smart ingredient-based recipe finder")

# -----------------------------
# Load Data
# -----------------------------

df = load_data()
vectorizer, tfidf_matrix = build_tfidf_model(df)

# -----------------------------
# Sidebar Filters
# -----------------------------

with st.sidebar:
    vegetarian_only = st.checkbox("🥗 Vegetarian only")
    max_cook_time = st.slider("⏱️ Max cooking time", 5, 300, 60)
    max_calories = st.slider("🔥 Max calories", 50, 1500, 600)
    min_protein = st.slider("💪 Min protein (g)", 0, 50, 5)
    top_n = st.slider("📋 Results", 1, 20, 5)

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
        max_cook_time,
        max_calories,
        min_protein
    )

    if not results.empty:
        for _, row in results.iterrows():
            st.markdown('<div class="box">', unsafe_allow_html=True)

            st.subheader(row["name"])
            st.progress(min(row["final_score"], 1.0))
            st.caption(f"Similarity Score: {round(row['final_score'] * 100, 1)}%")

            if row["description"]:
                st.markdown(f"**📝 Description:** {row['description']}")

            # Ingredients (highlight matches)
            try:
                ingredients = ast.literal_eval(row["ingredients"])
                highlighted = []
                for ing in ingredients:
                    if ing.lower() in query.lower():
                        highlighted.append(f"<span class='highlight'>{ing}</span>")
                    else:
                        highlighted.append(ing)
                st.markdown(
                    "**🧂 Ingredients:** " + ", ".join(highlighted),
                    unsafe_allow_html=True
                )
            except:
                pass

            st.markdown(f"**🕒 Cooking Time:** {row['minutes']} mins")

            # Nutrition bars
            try:
                n = ast.literal_eval(row["nutrition"])
                st.markdown("**🥗 Nutrition:**")
                st.progress(min(n[0] / 1000, 1.0))
                st.caption(f"Calories: {n[0]} kcal | Protein: {n[4]} g")
            except:
                pass

            # Steps
            try:
                steps = ast.literal_eval(row["steps"])
                steps_html = "<br>".join(
                    [f"{i+1}. {s}" for i, s in enumerate(steps)]
                )
                st.markdown(f"**👨‍🍳 Steps:**<br>{steps_html}", unsafe_allow_html=True)
            except:
                st.markdown("Steps unavailable.")

            st.markdown("</div>", unsafe_allow_html=True)

    else:
        st.warning("No recipes matched your filters.")
else:
    st.info("Start typing ingredients to search.")
