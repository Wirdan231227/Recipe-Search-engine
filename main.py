import pandas as pd
import ast
import streamlit as st  # Added for caching
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Function to safely convert a string of list-like ingredients into lowercase text
def parse_ingredients(ingredients_string):
    try:
        ingredients_list = ast.literal_eval(ingredients_string)  # Convert string to Python list
        final_list = []
        for ingredient in ingredients_list:
            if isinstance(ingredient, str):  # Only add if it's a string
                ingredient_lower = ingredient.lower()
                final_list.append(ingredient_lower)
        combined = " ".join(final_list)  # Join the list into a single string
        return combined
    except Exception as e:
        print("Error parsing ingredients:", e)
        return ""

# Function to remove any special characters (numbers, punctuation)
def clean_text(text):
    result = ""
    for character in text:
        if character.isalpha() or character == " ":
            result = result + character
    return result

# Function to safely parse the cooking steps
def parse_steps(steps_string):
    try:
        steps_list = ast.literal_eval(steps_string)
        if isinstance(steps_list, list):
            return steps_list
        else:
            return []
    except Exception as e:
        print("Error parsing steps:", e)
        return []

# Function to check if recipe is vegetarian
def is_vegetarian(tags_text):
    try:
        return "vegetarian" in tags_text.lower()
    except:
        return False

# Cached MAIN DATA LOADER FUNCTION
@st.cache_data
def load_data():
    try:
        dataframe = pd.read_csv("RAW_recipes.csv")
    except Exception as e:
        print("Error reading CSV file:", e)
        return pd.DataFrame()

    # Drop rows with missing important values
    required_columns = ['name', 'ingredients', 'tags', 'steps', 'minutes']
    dataframe = dataframe.dropna(subset=required_columns)

    # Parse ingredients for every row
    all_clean_ingredients = []
    for index in range(len(dataframe)):
        ingredients_raw = dataframe.iloc[index]['ingredients']
        ingredients_parsed = parse_ingredients(ingredients_raw)
        ingredients_clean = clean_text(ingredients_parsed)
        all_clean_ingredients.append(ingredients_clean)

    dataframe['ingredients_clean'] = all_clean_ingredients

    # Parse steps for every row
    all_steps = []
    for index in range(len(dataframe)):
        steps_raw = dataframe.iloc[index]['steps']
        steps_list = parse_steps(steps_raw)
        all_steps.append(steps_list)

    dataframe['steps_list'] = all_steps

    # Ensure minutes is integer
    dataframe['minutes'] = dataframe['minutes'].astype(int)

    return dataframe

# Cached TF-IDF model builder
@st.cache_resource
def build_tfidf_model(dataframe):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(dataframe['ingredients_clean'])
    return vectorizer, tfidf_matrix

# Function to perform search based on query
def search_recipes(query, vectorizer, tfidf_matrix, dataframe, top_n, vegetarian_only, max_cook_time):
    # Make sure query is in lowercase
    query_lower = query.lower()

    # Remove special characters
    cleaned_query = clean_text(query_lower)

    # Convert query into TF-IDF vector
    query_vector = vectorizer.transform([cleaned_query])

    # Compute cosine similarity between query and all recipes
    similarity_scores = cosine_similarity(query_vector, tfidf_matrix)
    similarity_scores_flattened = similarity_scores.flatten()

    # Sort indices by score (highest first)
    sorted_indices = similarity_scores_flattened.argsort()[::-1]

    # Get top results from dataframe
    top_recipes = dataframe.iloc[sorted_indices]

    # Apply vegetarian filter if enabled
    if vegetarian_only:
        filtered = []
        for index in range(len(top_recipes)):
            tags = top_recipes.iloc[index]['tags']
            if is_vegetarian(tags):
                filtered.append(top_recipes.iloc[index])
        top_recipes = pd.DataFrame(filtered)

    # Apply cooking time filter
    final_results = top_recipes[top_recipes['minutes'] <= max_cook_time]

    # Return only the top N rows
    return final_results.head(top_n)
