import hashlib

from flask import Flask, request, jsonify
from flask_cors import CORS
import time
import PythonFiles.db as db
import PythonFiles.recommender as recommender
from PythonFiles.db import Database
from PythonFiles.last_user_info import LastUserInfo
from PythonFiles.metadata_reccommender import MetadataRecommender
from PythonFiles.neuralnetwork import Model_NN_CF, Model_NN_CBF
import PythonFiles.building_models as building_models

app = Flask(__name__)
CORS(app)


# Talisman(app)


def compute_sha256(input_string: str) -> str:
    sha256_hash = hashlib.sha256()

    sha256_hash.update(input_string.encode('utf-8'))

    return sha256_hash.hexdigest()


@app.route('/')
def hello_world():  # put application's code here
    return 'Hello World!'


@app.route('/train-basic-models')
def train_basic_models():
    building_models.train_models()
    return 'Models trained'


@app.route('/train-nn-models')
def train_nn_models():
    recommender3 = Model_NN_CBF()
    recommender3.model_training()
    recommender2 = Model_NN_CF()
    recommender2.model_training()
    return 'Models trained'


@app.route('/recommend_on_movie_kNN_CF', methods=['GET'])
def recommend_on_movie_kNN_CF():
    start_time = time.time()
    movieId = int(request.args.get('movieId'))
    n_recommend = int(request.args.get('n_recommend', 5))
    reco = recommender.Recommender()
    recommendations = reco.recommend_on_movie_kNN_CF(movieId, n_recommend)
    end_time = time.time()
    print(f"Czas wykonania: {end_time - start_time} sekund")
    return jsonify({'data': recommendations})


@app.route('/recommend_on_history_kNN_CF', methods=['GET'])
def recommend_on_history_kNN_CF():
    start_time = time.time()
    user_id = int(request.args.get('user_id'))
    n_recommend = int(request.args.get('n_recommend', 5))
    reco = recommender.Recommender()
    recommendations = reco.recommend_on_user_history_kNN_CF(user_id, n_recommend)
    end_time = time.time()
    print(f"Czas wykonania: {end_time - start_time} sekund")
    return jsonify({'data': recommendations})


@app.route('/recommend_on_movie_kNN_CBF', methods=['GET'])
def recommend_on_movie_kNN_CBF():
    start_time = time.time()
    movieId = int(request.args.get('movieId'))
    n_recommend = int(request.args.get('n_recommend', 5))
    reco = recommender.Recommender()
    recommendations = reco.recommend_on_movie_kNN_CBF(movieId, n_recommend)
    end_time = time.time()
    print(f"Czas wykonania: {end_time - start_time} sekund")
    return jsonify({'data': recommendations})


@app.route('/recommend_on_movie_metadata', methods=['GET'])
def recommend_on_movie_metadata():
    start_time = time.time()
    title = request.args.get('title')
    n_recommend = int(request.args.get('n_recommend', 5))
    reco = MetadataRecommender()
    recommendations = reco.get_most_similar_movies(title, n_recommend)
    end_time = time.time()
    print(f"Czas wykonania: {end_time - start_time} sekund")
    return jsonify({'data': recommendations})


@app.route('/recommend_on_movie_metadata_improved', methods=['GET'])
def recommend_on_movie_metadata_improved():
    start_time = time.time()
    title = request.args.get('title')
    n_recommend = int(request.args.get('n_recommend', 5))
    reco = MetadataRecommender()
    recommendations = reco.improved_recommendations(title, n_recommend)
    end_time = time.time()
    print(f"Czas wykonania: {end_time - start_time} sekund")
    return jsonify({'data': recommendations})


@app.route('/recommend_on_history_kNN_CBF', methods=['GET'])
def recommend_on_history_kNN_CBF():
    start_time = time.time()
    user_id = int(request.args.get('user_id'))
    n_recommend = int(request.args.get('n_recommend', 5))
    reco = recommender.Recommender()
    recommendations = reco.recommend_on_history_kNN_CBF(user_id, n_recommend)
    end_time = time.time()
    print(f"Czas wykonania: {end_time - start_time} sekund")
    return jsonify({'data': recommendations})


@app.route('/reccomend_on_user_NN_CF', methods=['GET'])
def reccomend_on_user_NN_CF():
    start_time = time.time()
    user_id = int(request.args.get('user_id'))
    n_recommend = int(request.args.get('n_recommend', 5))
    my_model = Model_NN_CF()
    recommendations = my_model.get_top_n_recommendations(user_id, n_recommend)
    rating_title_list = list(zip(recommendations['predicted_rating'], recommendations['title']))
    end_time = time.time()
    print(f"Czas wykonania: {end_time - start_time} sekund")
    return jsonify({'data': rating_title_list})


@app.route('/reccomend_on_user_NN_CBF', methods=['GET'])
def reccomend_on_user_NN_CBF():
    start_time = time.time()
    user_id = int(request.args.get('user_id'))
    user_details = db.Database().get_user_details(user_id)
    gender = user_details['gender'].iloc[0]
    age = user_details['age'].iloc[0]
    occupation = user_details['occupation'].iloc[0]
    zip_code = user_details['zip-code'].iloc[0]
    n_recommend = int(request.args.get('n_recommend', 5))
    my_model = Model_NN_CBF()
    recommendations = my_model.get_predictions_on_all_movies(gender, age, occupation, zip_code, n_recommend)
    rating_title_list = list(zip(recommendations['predicted_rating'], recommendations['title']))
    end_time = time.time()
    print(f"Czas wykonania: {end_time - start_time} sekund")
    return jsonify({'data': rating_title_list})


@app.route('/recommend_on_user_SVD', methods=['GET'])
def recommend_on_user_SVD():
    start_time = time.time()
    user_id = int(request.args.get('user_id'))
    n_recommend = int(request.args.get('n_recommend', 5))
    reco = recommender.Recommender()
    recommendations = reco.recommend_on_user_SVD(user_id, n_recommend)
    recommendations = list(recommendations)
    end_time = time.time()
    print(f"Czas wykonania: {end_time - start_time} sekund")
    return jsonify({'data': recommendations})


@app.route('/register', methods=['POST'])
def register_user():
    db = Database()

    data = request.get_json()

    username = data.get('username')
    email = data.get('email')
    password = data.get('password')
    gender = data.get('gender')

    if not username or not email or not password or not gender:
        return jsonify({'error': 'All fields are required!'}), 400
    hashed_password = compute_sha256(password)
    try:
        db.register_user(username=username, email=email, password=hashed_password, gender=gender)
        print("User registered successfully!")
    except ValueError as e:
        print(e)
        return jsonify({'error': str(e)}), 400

    return jsonify({'message': 'User registered successfully!'}), 201


@app.route('/login', methods=['POST'])
def login_user():
    db = Database()
    data = request.get_json()

    username = data.get('username')
    password = data.get('password')

    if not username or not password:
        return jsonify({'error': 'All fields are required!'}), 400

    hashed_password = compute_sha256(password)

    user_data = db.login_user(username=username, password=hashed_password)
    if user_data:
        print("Login successful! User name:", user_data[1])
        return jsonify({'message': 'User login successfully!',
                        'username': user_data[1]}, ), 201
    else:
        print("Invalid username or password.")
        return jsonify({'error': 'Invalid username or password.'}), 400


@app.route('/login-google', methods=['POST'])
def login_google():
    data = request.get_json()
    email = data.get('email')

    db = Database()
    user = db.google_login_check(email)

    if user:
        return jsonify({'success': True, 'message': 'User logged in successfully', 'username': user[1]})
    else:

        return jsonify({'success': False, 'message': 'User not found. Please complete registration with username.'})


@app.route('/users/<username>', methods=['GET'])
def get_user(username):
    db = Database()

    registered_user = db.get_registered_user_by_username(username)

    if registered_user:
        user_id = registered_user[5]
        user = db.get_user_by_id(user_id)

        if user:
            return jsonify({
                'username': registered_user[1],
                'email': registered_user[2],
                'gender': registered_user[4],
                'age': user[2],
                'occupation': user[3],
                'zipcode': user[4]
            }), 200
        else:
            return jsonify({
                'username': registered_user[1],
                'email': registered_user[2],
                'gender': registered_user[4]
            }), 200
    else:
        return jsonify({'error': 'Registered user not found.'}), 404


@app.route('/users/<username>', methods=['POST'])
def update_user(username):
    db = Database()
    data = request.get_json()

    registered_user = db.get_registered_user_by_username(username)

    if registered_user:
        user_id = registered_user[5]

        try:

            if user_id is None:
                new_gender = registered_user[4]
                new_age = data.get('age')
                new_occupation = data.get('occupation')
                new_zipcode = data.get('zipcode')

                new_user_id = db.create_user(new_gender, new_age, new_occupation, new_zipcode)

                db.update_registered_user_userId(username, new_user_id)
                user_id = new_user_id

            new_email = data.get('email')
            if new_email:
                db.update_registered_user_email(username, new_email)

            db.update_user(user_id, data.get('gender'), data.get('age'), data.get('occupation'), data.get('zipcode'))
            db.conn.commit()
            return jsonify({'message': 'User data updated successfully!'}), 200
        except Exception as e:

            db.conn.rollback()
            return jsonify({'error': str(e)}), 400
    else:
        return jsonify({'error': 'Registered user not found.'}), 404


@app.route('/users/<username>/movies-unwatched', methods=['GET'])
def get_movies_unwatched(username):
    db = Database()
    user_id = db.get_user_id_by_username(username)

    if user_id:
        movies = db.get_movie_titles_unwatched(user_id)
        movies = movies.to_dict(orient='records')
    else:
        movies = db.get_movies()
        movies = movies[['movieId', 'title']]
        movies = movies.to_dict(orient='records')

    return jsonify({'data': movies}), 200


@app.route('/movies', methods=['GET'])
def get_movies():
    db = Database()

    movies = db.get_movies()
    movies = movies[['movieId', 'title']]
    movies = movies.to_dict(orient='records')

    return jsonify({'data': movies}), 200


@app.route('/metadata_movies', methods=['GET'])
def get_metadata_movies():
    db = Database()

    movies = db.get_metadata_movies()
    movies = movies[['id', 'title']]
    movies = movies.to_dict(orient='records')

    return jsonify({'data': movies}), 200


@app.route('/users/<username>/userid', methods=['GET'])
def get_user_id_by_username(username):
    db = Database()
    user_id = db.get_user_id_by_username(username)
    if user_id is not None:
        return jsonify({'userid': user_id}), 200
    else:
        return jsonify({'userid': None}), 405


@app.route('/add-rating', methods=['POST'])
def add_user_rating():
    db = Database()
    try:
        data = request.get_json()

        if 'userId' not in data or 'movieId' not in data or 'rating' not in data:
            return jsonify({'success': False, 'error': 'Missing parameters'}), 400

        userid = data['userId']
        movieid = data['movieId']
        rating = data['rating']

        if userid is None or movieid is None or rating is None:
            return jsonify({'success': False, 'error': 'Missing parameters'}), 400

        done = db.add_rating(userid, movieid, rating)

        if done:
            return jsonify({'success': True, 'message': 'Rating added successfully'}), 200
        else:
            return jsonify({'success': False, 'error': 'Rating not added'}), 200

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/<userId>/last-ratings', methods=['GET'])
def get_user_rating(userId):
    db = Database()

    ratings_df = db.get_ratings_info(userId)
    ratings = ratings_df.to_dict(orient='records')

    if ratings:
        return jsonify({'data': ratings}), 200
    else:
        return jsonify({'success': False, 'error': 'No ratings found'}), 404


@app.route('/check_user_details/<username>', methods=['GET'])
def check_user_details(username):
    db = Database()
    authentication = db.check_user_details(username)
    if authentication:
        return jsonify({'success': True, 'message': 'User details checked successfully'}), 200
    else:
        return jsonify({'success': False, 'error': 'User details not checked'}), 404


@app.route('/check_user_history/<username>', methods=['GET'])
def check_user_history(username):
    db = Database()
    userId = db.get_user_id_by_username(username)
    if userId is None:
        return jsonify({'success': False, 'error': 'User history not checked'}), 404

    authentication = db.get_ratings_info(userId).empty
    if authentication is False:
        return jsonify({'success': True, 'message': 'User history checked successfully'}), 200
    else:
        return jsonify({'success': False, 'error': 'User history not checked'}), 404


@app.route('/check_user_access_model/<username>/<model>', methods=['GET'])
def check_user_access_model(username, model):
    db = Database()

    lastuserId = LastUserInfo.read_last_trained_user(model)
    userId = db.get_user_id_by_username(username)

    if userId is None:
        return jsonify({'success': False, 'error': 'User access not checked - problem with database - > UserId based '
                                                   'on username not found'}), 404
    if lastuserId is None:
        return jsonify({'success': False, 'error': 'User access not checked- problem with file - > last UserId not '
                                                   'found'}), 404

    if userId <= lastuserId:
        return jsonify({'success': True, 'message': 'User access checked successfully'}), 200
    else:
        return jsonify({'success': False, 'error': 'Model not trained'}), 404


@app.route('/<int:userId>/ratings/<int:movieId>', methods=['DELETE'])
def delete_rating(userId, movieId):
    db = Database()

    try:
        deleted = db.delete_rating(userId, movieId)
        if deleted:
            return jsonify({'success': True, 'message': 'Rating deleted successfully'}), 200
        else:
            return jsonify({'success': False, 'message': 'Rating not found'}), 404
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=2118)
