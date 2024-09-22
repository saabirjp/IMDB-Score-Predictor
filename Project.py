import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import streamlit as st
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load your dataset
df = pd.read_csv('data.csv')

# Select relevant columns
new = df[['director_name', 'duration', 'actor_1_name', 'actor_2_name', 'actor_3_name',
          'gross', 'genres', 'budget', 'imdb_score']].copy()

# Rename the columns
new.columns = ['director', 'runtime', 'star1', 'star2', 'star3', 'revenue', 'genres', 'budget', 'imdb_score']

# Handle missing values
new['runtime'] = new['runtime'].fillna(new['runtime'].mean())
new['revenue'] = new['revenue'].fillna(new['revenue'].median())
new['budget'] = new['budget'].fillna(new['budget'].median())
new['genres'] = new['genres'].fillna(new['genres'].mode()[0])

# Convert 'genres' into lists of genres
new['genres'] = new['genres'].apply(lambda x: x.split('|'))

# Combine cast into a list
new['cast'] = new[['star1', 'star2', 'star3']].values.tolist()

### Director IMDb Score Encoding
# Create a dictionary of director average IMDb scores
director_scores = {}
for index, row in df.iterrows():
    director = row['director_name']
    if director not in director_scores:
        director_scores[director] = []
    director_scores[director].append(row['imdb_score'])

# Calculate the average IMDb score for each director
director_avg_scores = {director: np.mean(scores) for director, scores in director_scores.items()}

# Add a new feature that is the average IMDb score of the director
def average_director_score(director):
    return director_avg_scores.get(director, np.nan)

new['average_director_score'] = new['director'].apply(average_director_score)

### Actor's IMDb Score Encoding
# Create a dictionary of actor average IMDb scores
actor_scores = {}
for index, row in df.iterrows():
    actors = [row['actor_1_name'], row['actor_2_name'], row['actor_3_name']]
    for actor in actors:
        if actor not in actor_scores:
            actor_scores[actor] = []
        actor_scores[actor].append(row['imdb_score'])

# Calculate the average IMDb score for each actor
actor_avg_scores = {actor: np.mean(scores) for actor, scores in actor_scores.items()}

# Add a new feature that is the average IMDb score of the cast
def average_actor_score(cast):
    scores = [actor_avg_scores.get(actor, np.nan) for actor in cast]
    return np.nanmean(scores)

new['average_actor_score'] = new['cast'].apply(average_actor_score)

### Genre IMDb Score Encoding
# Create a dictionary of genre average IMDb scores
genre_scores = {}
for index, row in df.iterrows():
    genres = row['genres']
    for genre in genres:
        if genre not in genre_scores:
            genre_scores[genre] = []
        genre_scores[genre].append(row['imdb_score'])

# Calculate the average IMDb score for each genre
genre_avg_scores = {genre: np.mean(scores) for genre, scores in genre_scores.items()}

# Add a new feature that is the average IMDb score of the genres
def average_genre_score(genres):
    scores = [genre_avg_scores.get(genre, np.nan) for genre in genres]
    return np.nanmean(scores)

new['average_genre_score'] = new['genres'].apply(average_genre_score)

# Drop the original 'genres' and 'cast' columns AFTER applying the functions
new = new.drop(columns=['director', 'genres', 'cast', 'star1', 'star2', 'star3'])

# Separate features (X) and target (y)
X = new.drop(columns=['imdb_score'])
y = new['imdb_score']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Title of the app
st.title("IMDb Score Predictor")

# Prepare options for dropdowns
director_options = list(df['director_name'].unique()) + ["Other"]
actor_options = list(set(df['actor_1_name'].unique()) | set(df['actor_2_name'].unique()) | set(df['actor_3_name'].unique())) + ["Other"]
genre_options = list(set(genre for genres in df['genres'] for genre in genres.split('|'))) + ["Other"]

# User inputs for key features
director = st.selectbox("Director", director_options)
runtime = st.slider("Movie Runtime (in minutes)", min_value=60, max_value=300, step=1, value=120)
revenue = st.number_input("Revenue (in $)", value=1000000)
budget = st.number_input("Budget (in $)", value=5000000)

# Input for actors (up to 3 actors)
actor1 = st.selectbox("Actor 1", actor_options)
actor2 = st.selectbox("Actor 2", actor_options)
actor3 = st.selectbox("Actor 3", actor_options)

# Combine actors into a list
actors = [actor1, actor2, actor3]

# Input for genres (up to 3 genres)
genre1 = st.selectbox("Genre 1", genre_options)
genre2 = st.selectbox("Genre 2", genre_options)
genre3 = st.selectbox("Genre 3", genre_options)

# Combine genres into a list
genres = [genre1, genre2, genre3]

# Average scores for actors and genres
default_avg_actor_score = new['average_actor_score'].mean()  # Default average for actors
default_avg_genre_score = new['average_genre_score'].mean()  # Default average for genres

# Calculate average IMDb score for the selected actors
def get_average_actor_score(actors):
    scores = [actor_avg_scores.get(actor, default_avg_actor_score) for actor in actors if actor != "Other"]
    return np.mean(scores) if scores else default_avg_actor_score

# Calculate average IMDb score for the selected genres
def get_average_genre_score(genres):
    scores = [genre_avg_scores.get(genre, default_avg_genre_score) for genre in genres if genre != "Other"]
    return np.mean(scores) if scores else default_avg_genre_score

# Calculate scores
average_actor_score = get_average_actor_score(actors)
average_genre_score = get_average_genre_score(genres)

# Get the average director score
average_director_score = director_avg_scores.get(director, default_avg_actor_score)

# Create a DataFrame for the user input
input_data = pd.DataFrame({
    'runtime': [runtime],
    'revenue': [revenue],
    'budget': [budget],
    'average_director_score': [average_director_score],
    'average_actor_score': [average_actor_score],
    'average_genre_score': [average_genre_score]
})

# Predict IMDb score using the trained model
if st.button('Predict IMDb Score'):
    imdb_score = rf_model.predict(input_data)[0]
    st.success(f"The predicted IMDb score is: {imdb_score:.2f}")

# Predictions on the test set
y_pred = rf_model.predict(X_test)

# Calculate R-squared
r2 = r2_score(y_test, y_pred)

# Calculate Mean Absolute Error (MAE)
mae = mean_absolute_error(y_test, y_pred)



