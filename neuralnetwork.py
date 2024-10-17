import heapq
import os
import time

import keras
import pandas as pd
from keras import Input
from keras.src.callbacks import ModelCheckpoint, EarlyStopping
from keras.src.optimizers import Adam
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Embedding, Flatten, concatenate, Concatenate, \
    Lambda
import numpy as np
from tensorflow.keras.regularizers import l2
from last_user_info import LastUserInfo
from tensorflow.keras import backend as k
from db import Database

all_genres = ['Animation', 'Documentary', 'War', 'Action', 'Crime', 'Western', 'Mystery',
              'Adventure', 'Children\'s', 'Sci-Fi', 'Comedy', 'Fantasy', 'Horror',
              'Film-Noir', 'Romance', 'Musical', 'Drama', 'Thriller']

occupation_dict = {
    "other or not specified": 0,
    "academic/educator": 1,
    "artist": 2,
    "clerical/admin": 3,
    "college/grad student": 4,
    "customer service": 5,
    "doctor/health care": 6,
    "executive/managerial": 7,
    "farmer": 8,
    "homemaker": 9,
    "K-12 student": 10,
    "lawyer": 11,
    "programmer": 12,
    "retired": 13,
    "sales/marketing": 14,
    "scientist": 15,
    "self-employed": 16,
    "technician/engineer": 17,
    "tradesman/craftsman": 18,
    "unemployed": 19,
    "writer": 20
}

@keras.saving.register_keras_serializable()
def rmse(y_true, y_pred):
    y_true = k.cast(y_true, dtype='float32')
    return k.sqrt(k.mean(k.square(y_pred - y_true)))


#Content Base Filtering - podejscie

class Model_NN_CBF:

    def __init__(self):
        self.model_path = 'model_NN_CBF.keras'
        self.model = self.load_trained_model()

    def get_data(self):
        database = Database()
        ratings = database.get_ratings()
        movies = database.get_movies()
        users = database.get_users()

        # Przetwarzanie danych użytkowników
        users['gender'] = users['gender'].map({'F': 0, 'M': 1})  # Zamiana na wartości numeryczne

        # Wybieranie odpowiednich kolumn
        users = users[['userId', 'gender', 'age', 'occupation', 'zip-code']]

        movies['genres'] = movies['genres'].str.split('|')
        all_genres = set(g for genre_list in movies['genres'] for g in genre_list)

        # Słownika dla wszystkich gatunków
        genre_dict = {genre: movies['genres'].apply(lambda x: 1 if genre in x else 0) for genre in all_genres}

        genres_df = pd.DataFrame(genre_dict)

        movies = pd.concat([movies, genres_df], axis=1)

        # Usunięcie oryginalnej kolumny 'genres'
        movies.drop(['genres'], axis=1, inplace=True)

        # Łączenie tabel na podstawie userId i movieId
        data = ratings.merge(users, on='userId').merge(movies, on='movieId')
        return data

    def model_training(self):
        tf.keras.config.enable_unsafe_deserialization()
        data = self.get_data()
        features = data.drop(['rating', 'title', 'userId', 'movieId', 'timestamp'], axis=1)
        demographic_features = features.iloc[:, :4]  # Pierwsze 5 kolumn to cechy demograficzne
        genre_features = features.iloc[:, 4:]  # Pozostałe kolumny to gatunki filmowe
        target = data['rating']

        # Normalizacja cech demograficznych
        #scaler = StandardScaler()
        #demographic_features = scaler.fit_transform(demographic_features)

        # Podział danych na zestawy treningowe i testowe
        X_train_demo, X_test_demo, X_train_genre, X_test_genre, Y_train, Y_test = train_test_split(
            demographic_features, genre_features, target, test_size=0.2, random_state=42
        )

        # Definicja modelu
        # Wejście dla cech demograficznych

        print(X_train_genre)

        demo_input = Input(shape=(X_train_demo.shape[1],), name='demographic_input')
        demo_dense = Dense(128, activation='relu')(demo_input)
        demo_dense = BatchNormalization()(demo_dense)
        demo_dense = Dropout(0.3)(demo_dense)

        # Wejście dla gatunków filmowych
        genre_input = Input(shape=(X_train_genre.shape[1],), name='genre_input')
        genre_dense = Dense(128, activation='relu')(genre_input)
        genre_dense = BatchNormalization()(genre_dense)
        genre_dense = Dropout(0.3)(genre_dense)

        # Łączenie obu gałęzi
        concat = Concatenate()([demo_dense, genre_dense])
        dense = Dense(256, activation='relu')(concat)
        dense = BatchNormalization()(dense)
        dense = Dropout(0.3)(dense)

        dense = Dense(128, activation='relu')(dense)
        dense = BatchNormalization()(dense)
        dense = Dropout(0.3)(dense)

        dense = Dense(64, activation='relu')(dense)
        dense = BatchNormalization()(dense)
        dense = Dropout(0.3)(dense)

        dense = Dense(32, activation='relu')(dense)
        dense = BatchNormalization()(dense)
        dense = Dropout(0.3)(dense)

        output = Dense(1, activation='sigmoid')(dense)
        output = Lambda(lambda x: x * 5)(output)

        # Kompilacja modelu
        model = tf.keras.models.Model(inputs=[demo_input, genre_input], outputs=output)
        model.compile(loss='mean_squared_error', optimizer='adam', metrics=[rmse])

        model.summary()

        # Ustalanie callbacks
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        checkpoint = ModelCheckpoint('best_model_CBF.keras', monitor='val_loss', save_best_only=True)

        # Trenowanie modelu
        history = model.fit(
            x=[X_train_demo, X_train_genre], y=Y_train,
            epochs=10, batch_size=128, validation_split=0.2,
            callbacks=[early_stopping, checkpoint]
        )

        # Wykres strat
        training_loss = history.history['loss']
        test_loss = history.history['val_loss']
        epoch_count = range(1, len(training_loss) + 1)

        plt.plot(epoch_count, training_loss, 'r--')
        plt.plot(epoch_count, test_loss, 'b-')
        plt.legend(['Training Loss', 'Test Loss'])
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.show()

        # Ewaluacja modelu
        loss, rmse_value = model.evaluate([X_test_demo, X_test_genre], Y_test)
        print(f'Final test Loss: {loss}, Final test RMSE: {rmse_value}')

        # Zapis modelu
        model.save(self.model_path)
        LastUserInfo.save_last_trained_user("nn_cbf")

    def print_features_for_model(self):
        data = self.get_data()
        features = data.drop(['rating', 'title', 'userId', 'movieId'], axis=1)
        target = data['rating']
        print(features.columns)
        print(features.shape)
        print(features.head())
        print(target.shape)
        print(target.head())

    def load_trained_model(self):
        tf.keras.config.enable_unsafe_deserialization()
        if os.path.exists(self.model_path):
            model = tf.keras.models.load_model(self.model_path)
            print("Model loaded successfully.")
            return model
        else:
            self.model_training()
            if os.path.exists(self.model_path):
                model = tf.keras.models.load_model(self.model_path)
                print("Model loaded successfully.")
                return model
            else:
                raise FileNotFoundError(f"Model file '{self.model_path}' does not exist and can't create new one.")

    def prepare_input_data(self, gender, age, occupation, zip_code, genres_list):
        # Przekształcenie płci na wartości numeryczne
        gender_numeric = 1 if gender == 'M' else 0

        occupation_nr = occupation_dict.get(occupation, -1)  # Domyślnie -1, jeśli zawód nie istnieje

        # Tworzenie listy wejściowej dla każdego filmu
        input_features = []
        for genres in genres_list:
            # Generowanie wektora gatunków dla danego filmu
            genre_vector = [1 if genre in genres else 0 for genre in all_genres]

            # Tworzenie pojedynczego wektora cech
            feature_vector = np.array([
                                          gender_numeric,
                                          age,
                                          occupation_nr,
                                          zip_code
                                      ] + genre_vector)

            input_features.append(feature_vector)

        return np.array(input_features)

    def get_occupation_number(occupation, occupation_dict):

        if occupation in occupation_dict:
            return occupation_dict[occupation]
        else:
            print(f"Warning: Occupation '{occupation}' not avaiable.")
            return -1

    def get_prediction(self, gender, age, occupation, zip_code, genres):
        input_data = self.prepare_input_data(gender, age, occupation, zip_code, genres)

        input_data = input_data.reshape(1, -1)  # Shape: (1, n_features)
        demo_data = input_data[:, :4]
        genre_data = input_data[:, 4:]

        try:
            prediction = self.model.predict([demo_data, genre_data])
            return prediction[0][0]  # Zwróć tylko pierwszą (i jedyną) wartość przewidywania
        except FileNotFoundError as e:
            print(e)

    def get_predictions_on_all_movies(self, gender, age, occupation, zip_code, n=10):
        # Inicjalizacja połączenia z bazą danych
        database = Database()
        # Pobierz wszystkie filmy z bazy danych
        movies = database.get_movies()

        # Zakładamy, że gatunki są rozdzielone pionowymi kreskami i zamieniamy je na listy
        movies['genres'] = movies['genres'].str.split('|')

        # Przygotowanie danych wejściowych
        input_data = self.prepare_input_data(gender, age, occupation, zip_code, movies['genres'].tolist())
        assert np.issubdtype(input_data.dtype, np.number), "input_data contains non-numeric values"

        # Przewidywanie ocen dla wszystkich filmów na raz
        predictions = self.model.predict([input_data[:, :4], input_data[:, 4:]])
        predictions_df = pd.DataFrame({
            'movieId': movies['movieId'],
            'title': movies['title'],
            'predicted_rating': predictions.flatten()  # Flatten to uzyskanie płaskiej tablicy
        })

        # Sortowanie po najwyższych ocenach
        top_recommendations = predictions_df.sort_values(by='predicted_rating', ascending=False)

        # Zwracanie najlepszych rekomendacji
        return top_recommendations.head(n).reset_index(drop=True)


# Colaborative filtering

class Model_NN_CF:
    def __init__(self):
        self.model_path = 'model_NN_CF.keras'
        self.model = self.load_trained_model()

    def load_trained_model(self):
        if os.path.exists(self.model_path):
            model = tf.keras.models.load_model(self.model_path)
            print("Model loaded successfully.")
            return model
        else:
            self.model_training()
            if os.path.exists(self.model_path):
                model = tf.keras.models.load_model(self.model_path)
                print("Model loaded successfully.")
                return model
            else:
                raise FileNotFoundError(f"Model file '{self.model_path}' does not exist and can't create new one.")

    def get_data(self):
        database = Database()
        ratings = database.get_ratings()
        movies = database.get_movies()
        users = database.get_users()

        users['gender'] = users['gender'].map({'F': 0, 'M': 1})  # Zamiana płci na wartości numeryczne

        users = users[['userId', 'gender', 'age', 'occupation', 'zip-code']]

        movies['genres'] = movies['genres'].str.split('|')
        all_genres = set(g for genre_list in movies['genres'] for g in genre_list)

        # Tworzenie słownika dla wszystkich gatunków
        genre_dict = {genre: movies['genres'].apply(lambda x: 1 if genre in x else 0) for genre in all_genres}

        # Tworzenie DataFrame z gatunkami
        genres_df = pd.DataFrame(genre_dict)

        # Łączenie DataFrame gatunków z oryginalnym DataFrame filmów
        movies = pd.concat([movies, genres_df], axis=1)

        # Usunięcie oryginalnej kolumny 'genres'
        movies.drop(['genres'], axis=1, inplace=True)

        # Łączenie tabel na podstawie userId i movieId
        data = ratings.merge(users, on='userId').merge(movies, on='movieId')
        return data

    def model_training(self):

        data = self.get_data()
        n_users = data['userId'].max() + 1
        n_movies = data['movieId'].max() + 1
        n_dim = 50

        print(n_users, n_movies)

        # Warstwa wejściowa dla użytkowników
        user = Input(shape=(1,), name='user_input')
        U = Embedding(n_users, n_dim, embeddings_regularizer=l2(1e-6))(user)
        U = Flatten()(U)
        U = Dense(128, activation='relu', kernel_regularizer=l2(1e-6))(U)
        U = BatchNormalization()(U)
        U = Dropout(0.3)(U)
        U = Dense(256, activation='relu', kernel_regularizer=l2(1e-6))(U)
        U = BatchNormalization()(U)
        U = Dropout(0.3)(U)

        # Warstwa wejściowa dla filmów
        movie = Input(shape=(1,), name='movie_input')
        M = Embedding(n_movies, n_dim, embeddings_regularizer=l2(1e-6))(movie)
        M = Flatten()(M)
        M = Dense(128, activation='relu', kernel_regularizer=l2(1e-6))(M)
        M = BatchNormalization()(M)
        M = Dropout(0.3)(M)
        M = Dense(256, activation='relu', kernel_regularizer=l2(1e-6))(M)
        M = BatchNormalization()(M)
        M = Dropout(0.3)(M)

        # Połączenie warstw U i M
        x = concatenate([U, M])
        x = Dense(512, activation='relu', kernel_regularizer=l2(1e-6))(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        x = Dense(256, activation='relu', kernel_regularizer=l2(1e-6))(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        x = Dense(128, activation='relu', kernel_regularizer=l2(1e-6))(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        x = Dense(64, activation='relu', kernel_regularizer=l2(1e-6))(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        x = Dense(32, activation='relu', kernel_regularizer=l2(1e-6))(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        x = Dense(16, activation='relu', kernel_regularizer=l2(1e-6))(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)

        # Wyjście
        final = Dense(1, activation='linear', kernel_regularizer=l2(1e-6))(x)

        # Definicja modelu
        model = tf.keras.models.Model(inputs=[user, movie], outputs=final)

        model.compile(optimizer=Adam(0.0001),
                      loss='mean_squared_error',
                      metrics=[rmse])

        model.summary()

        checkpoint = ModelCheckpoint('best_model_CF.keras', monitor='val_loss', verbose=0, save_best_only=True)
        early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

        X_train_user, X_test_user, X_train_movie, X_test_movie, Y_train, Y_test = train_test_split(
            data['userId'], data['movieId'], data['rating'], test_size=0.2, random_state=42
        )

        history = model.fit(
            x=[X_train_user, X_train_movie], y=Y_train,
            epochs=20, batch_size=64, validation_split=0.2,
            callbacks=[checkpoint, early_stopping]
        )

        # Wykres strat
        training_loss = history.history['loss']
        test_loss = history.history['val_loss']
        epoch_count = range(1, len(training_loss) + 1)

        # Rysowanie ykresu strat
        plt.plot(epoch_count, training_loss, 'r--')
        plt.plot(epoch_count, test_loss, 'b-')
        plt.legend(['Training Loss', 'Test Loss'])
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.show()

        loss, rmse_value = model.evaluate([X_test_user, X_test_movie], Y_test)
        print(f'Final test loss: {loss}, Final test RMSE: {rmse_value}')

        model.save(self.model_path)
        LastUserInfo.save_last_trained_user("nn_cbf")

    def get_prediction(self, userId, movieId):
        input_data = [userId, movieId]
        #print(input_data)
        try:
            prediction = self.model.predict(input_data)
            return prediction[0][0]
        except FileNotFoundError as e:
            print(e)

    def get_top_n_recommendations(self, user_id, top_n=10):
        database = Database()
        user_watched_movies = database.get_movies_unwatched(user_id)
        all_movie_ids = database.get_all_movie_ids()

        movies_to_predict = [movie_id for movie_id in all_movie_ids if movie_id not in user_watched_movies]

        user_input = np.array([user_id] * len(movies_to_predict))
        movie_input = np.array(movies_to_predict)

        predictions = self.model.predict([user_input, movie_input])

        movies = database.get_movies()
        movies_to_predict = pd.DataFrame({'movieId': movies_to_predict})
        movies_to_predict = movies_to_predict.merge(movies, on='movieId')

        predictions_df = pd.DataFrame({
            'movieId': movies_to_predict['movieId'],
            'title': movies_to_predict['title'],
            'predicted_rating': predictions.flatten()  # Flatten to uzyskać płaską tablicę
        })

        top_recommendations = predictions_df.sort_values(by='predicted_rating', ascending=False)

        return top_recommendations.head(top_n).reset_index(drop=True)


gender = 'F'
age = 0
occupation = "other or not specified"
zip_code = 0
genres = []

gender = 'M'
age = 45
occupation = "farmer"
zip_code = 0
genres = [all_genres[1], all_genres[9], all_genres[7]]

gender = 'M'
age = 20
occupation = "farmer"
zip_code = 55330
genres = [all_genres[2], all_genres[3]]

gender = 'M'
age = 56
occupation = "retired"
zip_code = 11002
genres = ['Comedy', 'Romance']

gender = 'M'
age = 20
occupation = "retired"
zip_code = 55330
genres = [all_genres[2], all_genres[3]]

# my_model=Model_NN_CBF()
# my_model.model_training()
#
# print(my_model.get_predictions_on_all_movies(gender, age, occupation, zip_code, 10))
#
# print(my_model.get_top_n_recommendations(550,10))
#
#
# my_model = Model_NN_CF()
# my_model.model_training()
