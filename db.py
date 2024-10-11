import datetime
import re
import pandas as pd
import sqlite3
from sklearn.feature_extraction.text import CountVectorizer


def process_title(title):
    aka_match = re.search(r'\(a\.k\.a\.\s*(.*?)\)', title)
    alternative_title = None
    if aka_match:
        alternative_title = aka_match.group(1).strip()

    year_match = re.search(r'\((\d{4})\)$', title)
    year = year_match.group(1) if year_match else None

    if alternative_title:
        title = alternative_title
    else:
        title = title.split(" (")[0]

    # Trzbea zamienic An, A, The bo API wyszukiwania OMDB sie buguje
    parts = title.split(", ")
    if len(parts) > 1:
        if 'The' in parts[1]:
            title = "The " + parts[0] + ' ' + parts[1].replace('The', '')
        elif 'A' in parts[1]:
            title = "The " + parts[0] + ' ' + parts[1].replace('A', '')
        elif 'An' in parts[1]:
            title = "The " + parts[0] + ' ' + parts[1].replace('An', '')
        else:
            title = parts[0] + ' ' + parts[1]

    return title + f" ({year})"


class Database:
    def __init__(self):
        self.conn = sqlite3.connect('movieAPI.db')
        self.cursor = self.conn.cursor()
        self.create_registered_users_table()

    def __del__(self):
        self.conn.close()

    def create_registered_users_table(self):
        query = """
        CREATE TABLE IF NOT EXISTS registered_users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE,
            email TEXT NOT NULL UNIQUE,
            password TEXT NOT NULL,
            gender TEXT,
            userId INTEGER UNIQUE,
            FOREIGN KEY (userId) REFERENCES users(userId)
             
        )"""
        self.cursor.execute(query)
        self.conn.commit()

    def register_user(self, username: str, email: str, password: str, gender: str):
        """Rejestruje nowego użytkownika w bazie danych."""
        try:
            query = """
            INSERT INTO registered_users (username, email, password, gender)
            VALUES (?, ?, ?, ?)
            """
            self.cursor.execute(query, (username, email, password, gender))
            self.conn.commit()
        except sqlite3.IntegrityError:
            raise ValueError("Username or email already exists")

    def login_user(self, username: str, password: str):
        """Loguje użytkownika, zwracając jego dane, jeśli logowanie się powiodło."""
        query = """
        SELECT * FROM registered_users WHERE username = ? AND password = ?
        """
        self.cursor.execute(query, (username, password))
        return self.cursor.fetchone()

    def create_table_from_data(self):
        movies = pd.read_csv('MovieLensData/movies.csv', sep=';')
        ratings = pd.read_csv('MovieLensData/ratings.csv', sep=';')
        users = pd.read_csv('MovieLensData/users.csv', sep=';')
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

        movies_df['title'] = movies_df['title'].apply(process_title)

        return movies_df

    def get_movies_watched(self, userId):
        query = "SELECT movieId FROM ratings WHERE userId = ?"
        movies_df = pd.read_sql_query(query, self.conn, params=(userId,))
        return movies_df

    def get_movies_unwatched(self, userId):
        query = "SELECT m.movieId FROM movies m LEFT JOIN ratings r ON m.movieId = r.movieId AND r.userId = ? WHERE r.movieId IS NULL"
        movies_df = pd.read_sql_query(query, self.conn, params=(userId,))
        return movies_df

    def get_ratings_info(self, userId):
        query = """
            SELECT m.movieId, m.title, r.rating, r.timestamp
            FROM movies m 
            LEFT JOIN ratings r ON m.movieId = r.movieId AND r.userId = ? 
            WHERE r.movieId IS NOT NULL
        """
        movies_df = pd.read_sql_query(query, self.conn, params=(userId,))

        return movies_df

    def get_movie_titles_unwatched(self, userId):
        query = "SELECT m.movieId, m.title FROM movies m LEFT JOIN ratings r ON m.movieId = r.movieId AND r.userId = ? WHERE r.movieId IS NULL"
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
        contents = pd.DataFrame(genres, columns=vectorizer.get_feature_names_out())

        return contents

    def get_movie_features_on_id(self, movie_id):

        query = "SELECT genres FROM movies WHERE movieId = ?"
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

    def google_login_check(self, email):
        cursor = self.conn.cursor()

        cursor.execute("SELECT * FROM registered_users WHERE email = ?", (email,))
        user = cursor.fetchone()
        return user

    def get_registered_user_by_username(self, username):
        self.cursor.execute("SELECT * FROM registered_users WHERE username = ?", (username,))
        user = self.cursor.fetchone()
        return user

    def get_user_by_id(self, user_id):
        self.cursor.execute("SELECT * FROM users WHERE userId = ?", (user_id,))
        return self.cursor.fetchone()

    def get_user_id_by_username(self, username):
        self.cursor.execute("SELECT userId FROM registered_users WHERE username = ?", (username,))
        return self.cursor.fetchone()

    def update_user(self, user_id, gender, age, occupation, zipcode):
        self.cursor.execute("""
            UPDATE users
            SET gender = ?, age = ?, occupation = ?, "zip-code" = ?
            WHERE userId = ?
        """, (gender, age, occupation, zipcode, user_id))
        return self.cursor.rowcount > 0

    def create_user(self, gender, age, occupation, zipcode):

        self.cursor.execute("SELECT userId FROM users ORDER BY userId DESC LIMIT 1")
        user_id = self.cursor.fetchone()[0] + 1

        self.cursor.execute("""
            INSERT INTO users (userId,gender, age, occupation, "zip-code")
            VALUES (?,?, ?, ?, ?)
        """, (user_id, gender, age, occupation, zipcode))
        return user_id  # Zwróć ID nowo utworzonego użytkownika

    def update_registered_user_userId(self, username, user_id):
        self.cursor.execute("""
            UPDATE registered_users
            SET userId = ?
            WHERE username = ?
        """, (user_id, username))
        return self.cursor.rowcount > 0

    def update_registered_user_email(self, username, email):
        cursor = self.cursor
        query = """
        UPDATE registered_users
        SET email = ?
        WHERE username = ?;
        """
        cursor.execute(query, (email, username))
        return cursor.rowcount > 0

    def add_rating(self, user_id: int, movie_id: int, rating: int):
        timestamp = int(datetime.datetime.now().timestamp())
        self.cursor.execute("""
            INSERT INTO ratings (userId ,movieId , rating ,timestamp)
            VALUES (?, ?, ? , ?)
        """, (user_id, movie_id, rating, timestamp))
        self.conn.commit()
        return self.cursor.rowcount > 0

    def delete_rating(self, userId, movieId):
        query = "DELETE FROM ratings WHERE userId = ? AND movieId = ?"
        self.cursor.execute(query, (userId, movieId))
        self.conn.commit()
        return self.cursor.rowcount > 0

    def get_user_details(self, user_id):
        query = "SELECT * FROM users WHERE userId = ?"
        user_details = pd.read_sql_query(query, self.conn, params=(user_id,))
        return user_details

    def check_user_details(self, username):
        query = "SELECT * FROM registered_users WHERE username = ?"
        user_details = pd.read_sql_query(query, self.conn, params=(username,))
        if user_details['userId'].iloc[0] is None:
            return False
        else:
            return True

    def get_last_user_id(self):
        query = "SELECT userId FROM users ORDER BY userId DESC LIMIT 1;"
        self.cursor.execute(query)
        userId = self.cursor.fetchone()[0]
        return userId
