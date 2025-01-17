import ast
import datetime
import re
import pandas as pd
import sqlite3
import mysql.connector
from mysql.connector import Error
import json
from nltk import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from ast import literal_eval
import numpy as np


def get_director(x):
    for i in x:
        if i['job'] == 'Director':
            return i['name']
    return np.nan


def read_and_adjust_metadata():
    md = pd.read_csv('../Metadata/movies_metadata.csv')
    credits = pd.read_csv('../Metadata/credits.csv')
    keywords = pd.read_csv('../Metadata/keywords.csv')

    keywords['id'] = keywords['id'].astype('int')
    credits['id'] = credits['id'].astype('int')
    md = md.drop([19730, 29503, 35587])
    md['id'] = md['id'].astype('int')

    md = md.merge(credits, on='id')
    md = md.merge(keywords, on='id')

    md['cast'] = md['cast'].apply(literal_eval)
    md['crew'] = md['crew'].apply(literal_eval)
    md['keywords'] = md['keywords'].apply(literal_eval)
    md['cast_size'] = md['cast'].apply(lambda x: len(x))
    md['crew_size'] = md['crew'].apply(lambda x: len(x))

    md['director'] = md['crew'].apply(get_director)
    md['cast'] = md['cast'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
    md['cast'] = md['cast'].apply(lambda x: x[:3] if len(
        x) >= 3 else x)  # Mention Director 3 times to give it more weight relative to the entire cast
    md['keywords'] = md['keywords'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
    md['cast'] = md['cast'].apply(lambda x: [str.lower(i.replace(" ", "")) for i in x])
    md['director'] = md['director'].astype('str').apply(lambda x: str.lower(x.replace(" ", "")))
    md['director'] = md['director'].apply(lambda x: [x, x, x])

    s = md.apply(lambda x: pd.Series(x['keywords']), axis=1).stack().reset_index(level=1, drop=True)
    s.name = 'keyword'
    s = s.value_counts()
    s = s[s > 1]

    def filter_keywords(x):
        words = []
        for i in x:
            if i in s:
                words.append(i)
        return words

    stemmer = SnowballStemmer('english')
    md['keywords'] = md['keywords'].apply(filter_keywords)
    md['keywords'] = md['keywords'].apply(lambda x: [stemmer.stem(i) for i in x])
    md['keywords'] = md['keywords'].apply(lambda x: [str.lower(i.replace(" ", "")) for i in x])

    # Konwersja kolumn zawierających listy/słowniki na JSON strings
    md['genres'] = md['genres'].apply(lambda x: json.dumps(x) if pd.notnull(x) else None)
    md['production_companies'] = md['production_companies'].apply(lambda x: json.dumps(x) if pd.notnull(x) else None)
    md['cast'] = md['cast'].apply(lambda x: json.dumps(x) if isinstance(x, (list, dict)) else None)
    md['crew'] = md['crew'].apply(lambda x: json.dumps(x) if isinstance(x, (list, dict)) else None)
    md['keywords'] = md['keywords'].apply(lambda x: json.dumps(x) if isinstance(x, (list, dict)) else None)
    md['director'] = md['director'].apply(lambda x: json.dumps(x) if isinstance(x, (list, dict)) else None)

    return md


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

    parts = title.split(", ")
    if len(parts) > 1:
        if 'The' in parts[1]:
            title = "The " + parts[0] + ' ' + parts[1].replace('The', '')
        elif 'A' in parts[1]:
            title = "A " + parts[0] + ' ' + parts[1].replace('A', '')
        elif 'An' in parts[1]:
            title = "An " + parts[0] + ' ' + parts[1].replace('An', '')
        else:
            title = parts[0] + ' ' + parts[1]

    return title + f" ({year})"


ssl_options = {
    'ca': '/Users/dawid/PycharmProjects/movierecomendationAPI/cert/ca-cert.pem',
    'cert': '/Users/dawid/PycharmProjects/movierecomendationAPI/cert/client-cert.pem',
    'key': '/Users/dawid/PycharmProjects/movierecomendationAPI/cert/client-key.pem'
}


def create_connection():
    connection = None
    try:
        connection = mysql.connector.connect(
            host="localhost",
            user="root",
            password="root",
            database="MOVIEAPPDATABASE",
            ssl_disabled=False,
            ssl_verify_cert=True,
            ssl_ca=ssl_options['ca'],
            ssl_cert=ssl_options['cert'],
            ssl_key=ssl_options['key']

        )
        if connection.is_connected():
            print("Connected to MySQL Server")
            # Sprawdzenie, czy SSL jest używane
            ssl_status = connection.get_server_info()
            print("Server version:", ssl_status)
            cursor = connection.cursor()
            cursor.execute("SHOW STATUS LIKE 'Ssl_cipher';")
            ssl_info = cursor.fetchone()

            if ssl_info[1]:  # jeśli nie jest pusty, SSL jest używane
                print("SSL is used:", ssl_info[1])
            else:
                print("SSL is not used.")

    except Error as e:
        print(f"The error '{e}' occurred")

    return connection


class Database:
    def __init__(self):
        self.conn = create_connection()
        self.cursor = self.conn.cursor()

    def __del__(self):
        self.conn.close()

    def create_registered_users_table(self):
        query = """
        CREATE TABLE IF NOT EXISTS registered_users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(255) NOT NULL UNIQUE,
    email VARCHAR(255) NOT NULL UNIQUE,
    password VARCHAR(255) NOT NULL,
    gender VARCHAR(50),
    userId INT UNIQUE,
    FOREIGN KEY (userId) REFERENCES users(userId)
);"""
        self.cursor.execute(query)
        self.conn.commit()

    def register_user(self, username: str, email: str, password: str, gender: str):
        """Rejestruje nowego użytkownika w bazie danych."""
        try:
            query = """
            INSERT INTO registered_users (username, email, password, gender)
            VALUES (%s, %s, %s, %s)
            """
            self.cursor.execute(query, (username, email, password, gender))
            self.conn.commit()
        except mysql.connector.errors.IntegrityError:
            raise ValueError("Username or email already exists")

    def login_user(self, username: str, password: str):
        """Loguje użytkownika, zwracając jego dane, jeśli logowanie się powiodło."""
        query = """
        SELECT * FROM registered_users WHERE username = %s AND password = %s
        """
        self.cursor.execute(query, (username, password))
        return self.cursor.fetchone()

    def create_table_from_data(self):
        movies = pd.read_csv('../MovieLensData/movies.csv', sep=';')
        ratings = pd.read_csv('../MovieLensData/ratings.csv', sep=';')
        users = pd.read_csv('../MovieLensData/users.csv', sep=';')
        movies.to_sql('movies', self.conn, if_exists='replace', index=False)
        ratings.to_sql('ratings', self.conn, if_exists='replace', index=False)
        users.to_sql('users', self.conn, if_exists='replace', index=False)

        md = read_and_adjust_metadata()
        md.to_sql('metadata_movies', self.conn, if_exists='replace', index=False)

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
        query = "SELECT movieId FROM movies WHERE title = %s"
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
        query = "SELECT movieId FROM ratings WHERE userId = %s"
        movies_df = pd.read_sql_query(query, self.conn, params=(userId,))
        return movies_df

    def get_movies_unwatched(self, userId):
        query = "SELECT m.movieId FROM movies m LEFT JOIN ratings r ON m.movieId = r.movieId AND r.userId = %s WHERE r.movieId IS NULL"
        movies_df = pd.read_sql_query(query, self.conn, params=(userId,))
        return movies_df

    def get_ratings_info(self, userId):
        query = """
            SELECT m.movieId, m.title, r.rating, r.timestamp
            FROM movies m 
            LEFT JOIN ratings r ON m.movieId = r.movieId AND r.userId = %s 
            WHERE r.movieId IS NOT NULL
        """
        movies_df = pd.read_sql_query(query, self.conn, params=(userId,))

        return movies_df

    def get_movie_titles_unwatched(self, userId):
        query = "SELECT m.movieId, m.title FROM movies m LEFT JOIN ratings r ON m.movieId = r.movieId AND r.userId = %s WHERE r.movieId IS NULL"
        movies_df = pd.read_sql_query(query, self.conn, params=(userId,))
        return movies_df

    def get_users(self):
        query = "SELECT * FROM users"
        users_df = pd.read_sql_query(query, self.conn)
        return users_df

    def get_metadata_movies(self):
        query = "SELECT * FROM metadata_movies"
        df = pd.read_sql_query(query, self.conn)
        md = df
        md['cast'] = md['cast'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else None)
        md['keywords'] = md['keywords'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else None)
        md['director'] = md['director'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else None)
        md['genres'] = md['genres'].apply(lambda x: json.loads(x) if pd.notnull(x) else None)
        md['genres'] = md['genres'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else None)
        md['genres'] = md['genres'].apply(lambda x: [genre['name'] for genre in x] if isinstance(x, list) else None)
        md['soup'] = md['keywords'] + md['cast'] + md['director'] + md['genres']
        md['soup'] = md['soup'].apply(lambda x: ' '.join(x))
        return md

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

        query = "SELECT genres FROM movies WHERE movieId = %s"
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

        cursor.execute("SELECT * FROM registered_users WHERE email = %s", (email,))
        user = cursor.fetchone()
        return user

    def get_registered_user_by_username(self, username):
        self.cursor.execute("SELECT * FROM registered_users WHERE username = %s", (username,))
        user = self.cursor.fetchone()
        return user

    def get_user_by_id(self, user_id):
        self.cursor.execute("SELECT * FROM users WHERE userId = %s", (user_id,))
        return self.cursor.fetchone()

    def get_user_id_by_username(self, username):
        self.cursor.execute("SELECT userId FROM registered_users WHERE username = %s", (username,))
        return self.cursor.fetchone()[0]

    def update_user(self, user_id, gender, age, occupation, zipcode):
        self.cursor.execute("""
            UPDATE users
            SET gender = %s, age = %s, occupation = %s, "zip-code" = %s
            WHERE userId = %s
        """, (gender, age, occupation, zipcode, user_id))
        return self.cursor.rowcount > 0

    def create_user(self, gender, age, occupation, zipcode):

        self.cursor.execute("SELECT userId FROM users ORDER BY userId DESC LIMIT 1")
        user_id = self.cursor.fetchone()[0] + 1

        self.cursor.execute("""
            INSERT INTO users (userId,gender, age, occupation, "zip-code")
            VALUES (%s,%s, %s, %s, %s)
        """, (user_id, gender, age, occupation, zipcode))
        return user_id  # Zwróć ID nowo utworzonego użytkownika

    def update_registered_user_userId(self, username, user_id):
        self.cursor.execute("""
            UPDATE registered_users
            SET userId = %s
            WHERE username = %s
        """, (user_id, username))
        return self.cursor.rowcount > 0

    def update_registered_user_email(self, username, email):
        cursor = self.cursor
        query = """
        UPDATE registered_users
        SET email = %s
        WHERE username = %s;
        """
        cursor.execute(query, (email, username))
        return cursor.rowcount > 0

    def add_rating(self, user_id: int, movie_id: int, rating: int):
        timestamp = int(datetime.datetime.now().timestamp())
        self.cursor.execute("""
            INSERT INTO ratings (userId ,movieId , rating ,timestamp)
            VALUES (%s, %s, %s , %s)
        """, (user_id, movie_id, rating, timestamp))
        self.conn.commit()
        return self.cursor.rowcount > 0

    def delete_rating(self, userId, movieId):
        query = "DELETE FROM ratings WHERE userId = %s AND movieId = %s"
        self.cursor.execute(query, (userId, movieId))
        self.conn.commit()
        return self.cursor.rowcount > 0

    def get_user_details(self, user_id):
        query = "SELECT * FROM users WHERE userId = %s"
        user_details = pd.read_sql_query(query, self.conn, params=(user_id,))
        return user_details

    def check_user_details(self, username):
        query = "SELECT * FROM registered_users WHERE username = %s"
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

    def delete_user_by_username(self, username: str):
        """Usuwa użytkownika z bazy danych na podstawie jego nazwy użytkownika."""
        try:
            query = "DELETE FROM registered_users WHERE username = %s"
            self.cursor.execute(query, (username,))
            self.conn.commit()
        except Exception as e:
            print(f"Error deleting user by username: {e}")
            self.conn.rollback()

    def delete_user_by_email(self, email: str):
        """Usuwa użytkownika z bazy danych na podstawie jego email."""
        try:
            query = "DELETE FROM registered_users WHERE email = %s"
            self.cursor.execute(query, (email,))
            self.conn.commit()
        except Exception as e:
            print(f"Error deleting user by username: {e}")
            self.conn.rollback()
