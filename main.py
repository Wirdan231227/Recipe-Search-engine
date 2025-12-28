import pandas as pd
import ast
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# Utility Functions
# -----------------------------

def parse_ingredients(ingredients_string):
    try:
        ingredients_list = ast.literal_eval(ingredients_string)
        return " ".join(
            ing.lower() for ing in ingredients_list if isinstance(ing, str)
        )
    except:
        return ""

def clean_text(text):
    return "".join(c for c in text if c.isalpha() or c == " ")

def is_vegetarian(tags_text):
    try:
        return "vegetarian" in tags_text.lower()
    except:
        return False


# -----------------------------
# Load Data (SAFE)
# -----------------------------

@st.cache_data
def load_data():
    df = pd.read_csv("RAW_recipes.csv")

    required = [
        'name', 'ingredients', 'steps',
        'tags', 'minutes', 'nutrition', 'description'
    ]
    df = df.dropna(subset=required)

    df['ingredients_clean'] = df['ingredients'].apply(
        lambda x: clean_text(parse_ingredients(x))
    )

    df['minutes'] = df['minutes'].astype(int)

    return df


# -----------------------------
# TF-IDF
# -----------------------------

@st.cache_resource
def build_tfidf_model(df):
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(df['ingredients_clean'])
    return vectorizer, tfidf_matrix


# -----------------------------
# Search (AND-BOOST + SCORE)
# -----------------------------

def search_recipes(
    query,
    vectorizer,
    tfidf_matrix,
    df,
    top_n,
    vegetarian_only,
    max_cook_time,
    max_calories,
    min_protein
):
    if not query.strip():
        return pd.DataFrame()

    query_terms = clean_text(query.lower()).split()
    query_vector = vectorizer.transform([" ".join(query_terms)])
    scores = cosine_similarity(query_vector, tfidf_matrix).flatten()

    results = df.copy()
    results["similarity"] = scores

    # AND-based ingredient boost
    def and_match_boost(text):
        return sum(1 for term in query_terms if term in text)

    results["and_score"] = results["ingredients_clean"].apply(and_match_boost)
    results["final_score"] = results["similarity"] + (results["and_score"] * 0.15)

    results = results.sort_values("final_score", ascending=False)

    if vegetarian_only:
        results = results[results["tags"].apply(is_vegetarian)]

    results = results[results["minutes"] <= max_cook_time]

    # Nutrition filtering
    def nutrition_ok(nut):
        try:
            n = ast.literal_eval(nut)
            return n[0] <= max_calories and n[4] >= min_protein
        except:
            return False

    results = results[results["nutrition"].apply(nutrition_ok)]

    return results.head(top_n)
