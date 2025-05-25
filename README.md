🍽️ Recipe Search Engine

A Streamlit-based web app that lets users search for recipes by entering ingredients. The app uses TF-IDF vectorization and cosine similarity to recommend the most relevant recipes from a dataset of thousands.

🚀 Features

* 🔍 Search recipes based on custom and common ingredients
* 🥗 Vegetarian-only filter
* ⏱️ Max cooking time slider
* 📋 Choose how many top recipes to display
* 📖 View full step-by-step instructions
* 🎨 Custom background and modern dark-themed UI
* ⚡ Fast performance with Streamlit caching

---

🧠 How It Works

* The dataset (`RAW_recipes.csv`) contains thousands of recipes with ingredients, tags, cooking time, and steps.
* Input ingredients are cleaned and transformed using **TF-IDF**.
* Cosine similarity is computed to match user queries to the most relevant recipes.
* Recipes are filtered based on user preferences (vegetarian, cook time) and displayed in a styled layout.

---

🛠️ Tech Stack

* **Frontend:** Streamlit + HTML/CSS styling
* **Backend:** Python, pandas, scikit-learn
* **Model:** TF-IDF + Cosine Similarity
* **Data:** RAW\_recipes.csv (included)

---

📦 Installation

```bash
git clone https://github.com/your-username/recipe-search-engine.git
cd recipe-search-engine
pip install -r requirements.txt
streamlit run app.py
```

📁 Project Structure

recipe-search-engine/
│
├── app.py              # Streamlit app
├── main.py             # Model & data functions
├── RAW_recipes.csv     # Recipe dataset
├── image.jpg           # Background image
├── requirements.txt    # Dependencies
