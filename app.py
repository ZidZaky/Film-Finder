from flask import Flask, render_template, request
from flask_wtf import FlaskForm
from wtforms import SelectField
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key_here'

# Load movie dataset
movie_data = pd.read_csv('datasets/movies.csv')
print(f"Movies dataset loaded with {movie_data.shape[0]} records")

# Load ratings dataset
ratings_data = pd.read_csv('datasets/ratings.csv')
print(f"Ratings dataset loaded with {ratings_data.shape[0]} records")

# Create a pivot table
ratings_matrix = ratings_data.pivot(index='userId', columns='movieId', values='rating').fillna(0)
print("Ratings matrix created")

# Calculate cosine similarity matrix
cosine_sim_matrix = cosine_similarity(ratings_matrix.T)
print("Cosine similarity matrix calculated")

# Convert to DataFrame for easier lookup
cosine_sim_df = pd.DataFrame(cosine_sim_matrix, index=ratings_matrix.columns, columns=ratings_matrix.columns)

# Flask forms
class MovieForm(FlaskForm):
    movie_choices = [(row['movieId'], row['title']) for index, row in movie_data.iterrows()]
    movie = SelectField('Movie', choices=movie_choices)

# Routes
@app.route('/')
def index():
    form = MovieForm()
    return render_template('index.html', form=form)

@app.route('/recommendation', methods=['POST'])
def recommendation():
    form = MovieForm(request.form)
    movie_id = int(form.movie.data)
    print(f"Selected movie ID: {movie_id}")
    
    # Get movie recommendations based on cosine similarity
    if movie_id in cosine_sim_df:
        similar_movies = cosine_sim_df[movie_id].dropna()
        print(f"Found {similar_movies.shape[0]} similar movies")
        similar_movies = similar_movies.sort_values(ascending=False).head(6)  # Get top 6 for demonstration
        
        # Drop the first entry as it will be the movie itself with similarity score 1.0
        similar_movies = similar_movies.iloc[1:]

        recommended_movie_ids = similar_movies.index
        recommended_movie_titles = movie_data[movie_data['movieId'].isin(recommended_movie_ids)][['movieId', 'title']].set_index('movieId').loc[recommended_movie_ids]
        recommended_movies = [(title, similar_movies[movie_id]) for movie_id, title in recommended_movie_titles['title'].items()]
    else:
        print("No similar movies found")
        recommended_movies = []

    movie_title = movie_data.loc[movie_data['movieId'] == movie_id, 'title'].values[0]
    return render_template('recommendation.html', movie_title=movie_title, recommended_movies=recommended_movies)

@app.route('/examples')
def examples():
    return render_template('examples.html')

if __name__ == '__main__':
    app.run(debug=True)
