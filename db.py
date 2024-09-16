import pandas as pd
import sqlite3

from sklearn.feature_extraction.text import CountVectorizer


class Database:
    def __init__(self):
        self.conn = sqlite3.connect('movieAPI.db')
        self.cursor = self.conn.cursor()

    def __del__(self):
        self.conn.close()

    def create_table_from_data(self):
        movies = pd.read_csv('MovieLensData/movies.csv',sep=';')
        ratings = pd.read_csv('MovieLensData/ratings.csv',sep=';')
        users = pd.read_csv('MovieLensData/users.csv',sep=';')
        movies.to_sql('movies', self.conn, if_exists='replace', index=False)
        ratings.to_sql('ratings', self.conn, if_exists='replace', index=False)
        users.to_sql('users', self.conn, if_exists='replace', index=False)

    def refresh_data(self):
        query = "DROP TABLE IF EXISTS movies"
        self.cursor.execute(query)
        query = "DROP TABLE IF EXISTS ratings"
        self.cursor.execute(query)
        query = "DROP TABLE IF EXISTS users"
        self.cursor.execute(query)
        self.create_table_from_data()
        self.conn.commit()


    def get_movie_id(self, movie_name):
        query = "SELECT movieId FROM movies WHERE title = ?"
        self.cursor.execute(query, (movie_name,))

        result = self.cursor.fetchone()

        if result:
            movie_id = result[0]
            return movie_id
        else:
            return None

    def get_rating_pivot(self):
        query = "SELECT movieId, userId, rating FROM ratings"
        ratings_df = pd.read_sql_query(query, self.conn)

        rating_pivot = ratings_df.pivot_table(
            values='rating',
            columns='userId',
            index='movieId'
        ).fillna(0)
        return rating_pivot



    def get_movies(self):
        query = "SELECT * FROM movies"
        movies_df = pd.read_sql_query(query, self.conn)
        return movies_df

    def get_movies_watched(self,userId):
        query = "SELECT movieId FROM ratings WHERE userId = ?"
        movies_df = pd.read_sql_query(query, self.conn, params=(userId,))
        return movies_df

    def get_movies_unwatched(self,userId):
        query="SELECT m.movieId FROM movies m LEFT JOIN ratings r ON m.movieId = r.movieId AND r.userId = ? WHERE r.movieId IS NULL"
        movies_df = pd.read_sql_query(query, self.conn, params=(userId,))
        return movies_df

    def get_users(self):
        query = "SELECT * FROM users"
        users_df = pd.read_sql_query(query, self.conn)
        return users_df

    def get_ratings(self):
        query = "SELECT * FROM ratings"
        ratings_df = pd.read_sql_query(query, self.conn)
        return ratings_df

    def get_all_movie_ids(self):
        query = "SELECT movieId FROM movies"
        movie_ids_df = pd.read_sql_query(query, self.conn)
        return movie_ids_df['movieId'].tolist()

    def get_movies_contents(self):
        query = "SELECT * FROM movies"
        vectorizer = CountVectorizer(stop_words='english')
        movies = self.get_movies()
        genres = vectorizer.fit_transform(movies.genres).toarray()
        contents = pd.DataFrame(genres,columns=vectorizer.get_feature_names_out())

        return contents

    def get_movie_features_on_id(self, movie_id):

        query = "SELECT genres FROM movies WHERE id = ?"
        movie_data = pd.read_sql_query(query, self.conn, params=(movie_id,))


        if movie_data.empty or 'genres' not in movie_data.columns:
            raise ValueError(f"No movie found with ID {movie_id} or 'genres' column is missing")


        vectorizer = CountVectorizer(stop_words='english')


        all_movies_query = "SELECT genres FROM movies"
        all_movies = pd.read_sql_query(all_movies_query, self.conn)
        vectorizer.fit(all_movies['genres'])

        genres = vectorizer.transform(movie_data['genres']).toarray()


        contents = pd.DataFrame(genres, columns=vectorizer.get_feature_names_out(), index=movie_data.index)


        return contents.iloc[0]


db = Database()
db.refresh_data()

