import pandas as pd
import ast
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# Utility Functions
# -----------------------------

def parse_ingredients(ingredients_string):
    """Safely convert stringified list of ingredients into lowercase text"""
    try:
        ingredients_list = ast.literal_eval(ingredients_string)
        if not isinstance(ingredients_list, list):
            return ""
        return " ".join(
            ingredient.lower()
            for ingredient in ingredients_list
            if isinstance(ingredient, str)
        )
    except Exception:
        return ""


def clean_text(text):
    """Remove numbers and special characters"""
    return "".join(
        char for char in text if char.isalpha() or char == " "
    )


def parse_steps(steps_string):
    """Safely parse cooking steps"""
    try:
        steps_list = ast.literal_eval(steps_string)
        return steps_list if isinstance(steps_list, list) else []
    except Exception:
        return []


def is_vegetarian(tags_text):
    """Check if recipe is vegetarian"""
    try:
        return "vegetarian" in tags_text.lower()
    except Exception:
        return False


# -----------------------------
# Data Loader
# -----------------------------

@st.cache_data
def load_data():
    try:
        df = pd.read_csv("RAW_recipes.csv")
    except Exception as e:
        st.error(f"Failed to read CSV file: {e}")
        return pd.DataFrame()

    required_columns = ['name', 'ingredients', 'tags', 'steps', 'minutes']
    df = df.dropna(subset=required_columns)

    if df.empty:
        st.error("Dataset is empty after cleaning.")
        return pd.DataFrame()

    # Clean ingredients
    df['ingredients_clean'] = df['ingredients'].apply(
        lambda x: clean_text(parse_ingredients(x))
    )

    # Parse steps
    df['steps_list'] = df['steps'].apply(parse_steps)

    # Ensure minutes is integer
    df['minutes'] = df['minutes'].astype(int)

    return df


# -----------------------------
# TF-IDF Model Builder
# -----------------------------

@st.cache_resource
def build_tfidf_model(dataframe):
    if dataframe.empty:
        raise ValueError("DataFrame is empty. Cannot build TF-IDF model.")

    if 'ingredients_clean' not in dataframe.columns:
        raise ValueError(
            "Column 'ingredients_clean' not found. "
            f"Available columns: {list(dataframe.columns)}"
        )

    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(dataframe['ingredients_clean'])

    return vectorizer, tfidf_matrix


# -----------------------------
# Search Function
# -----------------------------

def search_recipes(
    query,
    vectorizer,
    tfidf_matrix,
    dataframe,
    top_n=10,
    vegetarian_only=False,
    max_cook_time=1000
):
    if not query.strip():
        return pd.DataFrame()

    cleaned_query = clean_text(query.lower())
    query_vector = vectorizer.transform([cleaned_query])

    similarity_scores = cosine_similarity(query_vector, tfidf_matrix).flatten()
    sorted_indices = similarity_scores.argsort()[::-1]

    results = dataframe.iloc[sorted_indices]

    # Vegetarian filter
    if vegetarian_only:
        results = results[results['tags'].apply(is_vegetarian)]

    # Cooking time filter
    results = results[results['minutes'] <= max_cook_time]

    return results.head(top_n)
