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
            ingredient.lower()
            for ingredient in ingredients_list
            if isinstance(ingredient, str)
        )
    except:
        return ""

def clean_text(text):
    return "".join(char for char in text if char.isalpha() or char == " ")

def is_vegetarian(tags_text):
    try:
        return "vegetarian" in tags_text.lower()
    except:
        return False


# -----------------------------
# Data Loader (NO LIST COLUMNS)
# -----------------------------

@st.cache_data
def load_data():
    df = pd.read_csv("RAW_recipes.csv")

    required_columns = [
        'name', 'ingredients', 'steps',
        'tags', 'minutes', 'nutrition', 'description'
    ]
    df = df.dropna(subset=required_columns)

    df['ingredients_clean'] = df['ingredients'].apply(
        lambda x: clean_text(parse_ingredients(x))
    )

    df['minutes'] = df['minutes'].astype(int)

    return df


# -----------------------------
# TF-IDF Model
# -----------------------------

@st.cache_resource
def build_tfidf_model(dataframe):
    vectorizer = TfidfVectorizer(stop_words="english")
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

    query_vector = vectorizer.transform([clean_text(query.lower())])
    similarity_scores = cosine_similarity(query_vector, tfidf_matrix).flatten()

    results = dataframe.iloc[similarity_scores.argsort()[::-1]]

    if vegetarian_only:
        results = results[results['tags'].apply(is_vegetarian)]

    results = results[results['minutes'] <= max_cook_time]

    return results.head(top_n)
