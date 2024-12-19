import os

import numpy as np
import joblib
from PythonFiles import db, building_models as bm


class Recommender:

    def load_recommendation_model(self, model_path, attempts=0, max_attempts=5):

        if attempts >= max_attempts:
            print(f"Przekroczono maksymalną liczbę prób ładowania modelu: {model_path}")
            return None

        if os.path.exists(model_path):
            try:
                model = joblib.load(model_path)
                print(f"Model załadowany pomyślnie: {model_path}")
                return model
            except Exception as e:
                print(f"Błąd podczas ładowania modelu z {model_path}: {e}")
                return None
        else:
            print(f"Plik modelu nie istnieje: {model_path}")
            if model_path == 'knn_model_CF.pkl':
                bm.build_kNN_CF(self.database)
            elif model_path == 'knn_model_CBF.pkl':
                bm.build_kNN_CBF(self.database)
            elif model_path == 'svd_model.pkl':
                bm.build_SVD(self.database)

            return self.load_recommendation_model(model_path, attempts + 1, max_attempts)

    def __init__(self):
        self.hist = []
        self.database = db.Database()

    # Collaborative Filtering

    def recommend_on_movie_kNN_CF(self, movieId, n_reccomend=5):
        knn_model_CF = self.load_recommendation_model('../knn_model_CF.pkl')
        rating_pivot = self.database.get_rating_pivot()
        movies = self.database.get_movies()
        distance, neighbors = knn_model_CF.kneighbors([rating_pivot.loc[movieId]], n_neighbors=n_reccomend + 1)
        movieids = [rating_pivot.iloc[i].name for i in neighbors[0]]
        recommends = [str(movies[movies['movieId'] == mid]['title']).split('\n')[0].split('  ')[-1] for mid in movieids
                      if mid not in [movieId]]
        return recommends[:n_reccomend]

    def recommend_on_user_history_kNN_CF(self, userId, n_reccomend=5):
        knn_model_CF = self.load_recommendation_model('../knn_model_CF.pkl')
        self.hist = []
        self.hist = self.database.get_movies_watched(userId)['movieId'].tolist()
        rating_pivot = self.database.get_rating_pivot()
        movies = self.database.get_movies()
        history = np.array([list(rating_pivot.loc[mid]) for mid in self.hist])
        distance, neighbors = knn_model_CF.kneighbors([np.average(history, axis=0)],
                                                      n_neighbors=n_reccomend + len(self.hist))
        movieids = [rating_pivot.iloc[i].name for i in neighbors[0]]
        recommends = [str(movies[movies['movieId'] == mid]['title']).split('\n')[0].split('  ')[-1] for mid in movieids
                      if mid not in self.hist]
        return recommends[:n_reccomend]

    # Content Base Filtering

    def recommend_on_movie_kNN_CBF(self, movieId, n_reccomend=5):
        knn_model_CBF = self.load_recommendation_model(
            '/Users/dawid/PycharmProjects/movierecomendationAPI/knn_model_CBF.pkl')
        movies = self.database.get_movies()
        distance, neighbors = knn_model_CBF.kneighbors([self.database.get_movie_features_on_id(movieId)],
                                                       n_neighbors=n_reccomend + 1)
        recommends = []
        for i in neighbors[0]:

            neighbor_movie_id = movies.iloc[i]['movieId']
            if neighbor_movie_id != movieId:
                recommends.append(movies.iloc[i]['title'])
        return recommends[:n_reccomend]

    def recommend_on_history_kNN_CBF(self, userId, n_reccomend=5):
        knn_model_CBF = self.load_recommendation_model(
            '/Users/dawid/PycharmProjects/movierecomendationAPI/knn_model_CBF.pkl')
        self.hist = []
        self.hist = self.database.get_movies_watched(userId)['movieId'].tolist()
        movies = self.database.get_movies()
        history = np.array([list(self.database.get_movie_features_on_id(iloc)) for iloc in self.hist])
        distance, neighbors = knn_model_CBF.kneighbors([np.average(history, axis=0)],
                                                       n_neighbors=n_reccomend + len(self.hist))
        recommends = [movies.iloc[i]['title'] for i in neighbors[0] if i not in self.hist]
        return recommends[:n_reccomend]

    # SVD - Singular Value Decomposition - Collaborative Filtering
    def recommend_on_user_SVD(self, user_id, n_recommendations=5):
        self.hist = []
        watched_movies = self.database.get_movies_watched(user_id)
        self.hist = watched_movies['movieId'].tolist()

        SVD_model = self.load_recommendation_model('/Users/dawid/PycharmProjects/movierecomendationAPI/svd_model.pkl')

        all_movie_ids = self.database.get_all_movie_ids()
        history_movie_ids = self.hist
        movies = self.database.get_movies()
        predictions = [SVD_model.predict(user_id, movie_id) for movie_id in all_movie_ids if
                       movie_id not in history_movie_ids]

        predictions.sort(key=lambda x: x.est, reverse=True)

        top_predictions = predictions[:n_recommendations]
        recommended_movie_ids = [pred.iid for pred in top_predictions]
        recommended_movies_est = [pred.est for pred in top_predictions]
        recommends = [str(movies[movies['movieId'] == mid]['title'].values[0]) for mid in recommended_movie_ids]
        recommends = zip(recommended_movies_est, recommends)

        return recommends
