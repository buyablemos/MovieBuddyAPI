import numpy as np
import joblib
import pandas as pd
import db
import tensorflow as tf

class Recommender:
    def __init__(self):

        self.hist = []
        self.ishist = False
        self.knn_model_CF = joblib.load('knn_model_CF.pkl')
        self.knn_model_CBF = joblib.load('knn_model_CBF.pkl')
        self.SVD_model = joblib.load('svd_model.pkl')
        self.database = db.Database()

#Collaborative Filtering
    def recommend_on_movie_kNN_CF(self,movie,n_reccomend = 5):
        self.ishist = True
        movieid = self.database.get_movie_id(movie)
        rating_pivot=self.database.get_rating_pivot()
        movies=self.database.get_movies()
        self.hist.append(movieid)
        distance,neighbors = self.knn_model_CF.kneighbors([rating_pivot.loc[movieid]],n_neighbors=n_reccomend+1)
        movieids = [rating_pivot.iloc[i].name for i in neighbors[0]]
        recommeds = [str(movies[movies['movieId']==mid]['title']).split('\n')[0].split('  ')[-1] for mid in movieids if mid not in [movieid]]
        return recommeds[:n_reccomend]

    def recommend_on_user_history_kNN_CF(self,n_reccomend = 5):
        if self.ishist == False:
            return print('No history found')
        rating_pivot=self.database.get_rating_pivot()
        movies=self.database.get_movies()
        history = np.array([list(rating_pivot.loc[mid]) for mid in self.hist])
        distance,neighbors = self.model.kneighbors([np.average(history,axis=0)],n_neighbors=n_reccomend + len(self.hist))
        movieids = [rating_pivot.iloc[i].name for i in neighbors[0]]
        recommeds = [str(movies[movies['movieId']==mid]['title']).split('\n')[0].split('  ')[-1] for mid in movieids if mid not in self.hist]
        return recommeds[:n_reccomend]

#Content Base Filtering


    def recommend_on_movie_kNN_CBF(self,movie,n_reccomend = 5):
        self.ishist = True
        movies=self.database.get_movies()
        iloc = movies[movies['title']==movie].index[0]
        self.hist.append(iloc)
        distance,neighbors = self.knn_model_CBF.kneighbors([self.database.get_movie_features_on_id(iloc)],n_neighbors=n_reccomend+1)
        recommeds = [movies.iloc[i]['title'] for i in neighbors[0] if i not in [iloc]]
        return recommeds[:n_reccomend]


    def recommend_on_history_kNN_CBF(self,n_reccomend = 5):
        if self.ishist == False:
            return print('No history found')
        movies=self.database.get_movies()
        history = np.array([list(self.database.get_movie_features_on_id(iloc)) for iloc in self.hist])
        distance,neighbors = self.knn_model_CBF.kneighbors([np.average(history,axis=0)],n_neighbors=n_reccomend + len(self.hist))
        recommeds = [movies.iloc[i]['title'] for i in neighbors[0] if i not in self.hist]
        return recommeds[:n_reccomend]


#SVD - Singular Value Decomposition - Collaborative Filtering
    def recommend_on_user_SVD(self,user_id,n_recommendations = 5):

        all_movie_ids = self.database.get_all_movie_ids()
        history_movie_ids = self.hist
        movies=self.database.get_movies()
        predictions = [self.SVD_model.predict(user_id, movie_id) for movie_id in all_movie_ids if movie_id not in history_movie_ids]

        predictions.sort(key=lambda x: x.est, reverse=True)

        top_predictions = predictions[:n_recommendations]
        recommended_movie_ids = [pred.iid for pred in top_predictions]
        recommeds = [str(movies[movies['movieId'] == mid]['title'].values[0]) for mid in recommended_movie_ids]

        return recommeds








