from matplotlib import pyplot as plt
from sklearn.neighbors import NearestNeighbors
import joblib
from sklearn.model_selection import train_test_split
from surprise import Dataset, Reader, SVD, accuracy
from surprise.model_selection import train_test_split
from PythonFiles import db
from PythonFiles.last_user_info import LastUserInfo


class SVDWithMonitoring(SVD):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def fit(self, trainset):
        rmse_values = []  # Lista do przechowywania RMSE po każdej epoce

        for epoch in range(self.n_epochs):
            # Trening modelu
            super().fit(trainset)

            # Przewidywanie na zbiorze testowym
            predictions = self.test(trainset.build_testset())

            # Obliczanie RMSE
            rmse = accuracy.rmse(predictions)
            rmse_values.append(rmse)  # Dodaj RMSE do listy

            # Wypisywanie RMSE po każdej epoce
            print(f'Epoch {epoch + 1}/{self.n_epochs} - RMSE: {rmse}')

        return rmse_values


def build_kNN_CF(database):
    rating_pivot = database.get_rating_pivot()
    print('Shape of this pivot table :', rating_pivot.shape)
    print(rating_pivot.head())

    nn_algo = NearestNeighbors(metric='cosine')
    nn_algo.fit(rating_pivot)

    joblib.dump(nn_algo, '../knn_model_CF.pkl')


def build_kNN_CBF(database):
    contents = database.get_movies_contents()
    print('Shape of the content table :', contents.shape)
    print(contents.head())

    nn_algo = NearestNeighbors(metric='cosine')
    nn_algo.fit(contents)

    joblib.dump(nn_algo, '../knn_model_CBF.pkl')


def build_SVD(database):
    ratings_df = database.get_ratings()

    reader = Reader(line_format='user item rating timestamp', sep=',', rating_scale=(1, 5))
    data = Dataset.load_from_df(ratings_df[['userId', 'movieId', 'rating']], reader)

    trainset, testset = train_test_split(data, test_size=0.2)

    # List to store RMSE values for each iteration
    rmse_values = []
    mse_values = []

    last_model = None

    # Training the model and tracking RMSE
    for epoch in range(1, 21):  # Example: 20 iterations (epochs)
        model = SVD(n_epochs=epoch, lr_all=0.005, reg_all=0.02)
        model.fit(trainset)
        predictions = model.test(testset)
        rmse = accuracy.rmse(predictions)
        mse = accuracy.mse(predictions)
        rmse_values.append(rmse)
        mse_values.append(mse)
        if epoch == 20:
            last_model = model
        print(f"Epoch {epoch}: RMSE = {rmse}")

    # Plotting the RMSE values
    plt.plot(range(1, 21, 1), rmse_values)
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    plt.title('RMSE During Training')
    plt.show()
    # Plotting the MSE values
    plt.plot(range(1, 21, 1), mse_values)
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.title('MSE During Training')
    plt.show()

    # Saving the model
    joblib.dump(last_model, '../svd_model.pkl')


def train_models():
    database = db.Database()
    build_kNN_CF(database)
    LastUserInfo.save_last_trained_user("knn_cf")
    build_kNN_CBF(database)
    LastUserInfo.save_last_trained_user("knn_cbf")
    build_SVD(database)
    LastUserInfo.save_last_trained_user("svd")


database = db.Database()
build_SVD(database)
