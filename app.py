from flask import Flask, request, jsonify
from flask_cors import CORS
import db
import recommender
import hashlib
from db import Database
from neuralnetwork import Model_NN_CF, Model_NN_CBF

app = Flask(__name__)
CORS(app)


def compute_sha256(input_string: str) -> str:
    sha256_hash = hashlib.sha256()

    sha256_hash.update(input_string.encode('utf-8'))

    return sha256_hash.hexdigest()

@app.route('/')
def hello_world():  # put application's code here
    return 'Hello World!'


@app.route('/recommend_on_movie_kNN_CF', methods=['GET'])
def recommend_on_movie_kNN_CF():
    movie = request.args.get('movie')
    n_recommend = int(request.args.get('n_recommend', 5))
    reco = recommender.Recommender()
    recommendations = reco.recommend_on_movie_kNN_CF(movie, n_recommend)
    return jsonify({'data': recommendations})


@app.route('/recommend_on_user_history_kNN_CF', methods=['GET'])
def recommend_on_user_history_kNN_CF():
    user_id = int(request.args.get('user_id'))
    n_recommend = int(request.args.get('n_recommend', 5))
    reco = recommender.Recommender()
    recommendations = reco.recommend_on_user_history_kNN_CF(user_id, n_recommend)
    return jsonify({'data': recommendations})


@app.route('/recommend_on_movie_kNN_CBF', methods=['GET'])
def recommend_on_movie_kNN_CBF():
    movie = request.args.get('movie')
    n_recommend = int(request.args.get('n_recommend', 5))
    reco = recommender.Recommender()
    recommendations = reco.recommend_on_movie_kNN_CBF(movie, n_recommend)
    return jsonify({'data': recommendations})


@app.route('/recommend_on_history_kNN_CBF', methods=['GET'])
def recommend_on_history_kNN_CBF():
    user_id = int(request.args.get('user_id'))
    n_recommend = int(request.args.get('n_recommend', 5))
    reco = recommender.Recommender()
    recommendations = reco.recommend_on_history_kNN_CBF(user_id, n_recommend)
    return jsonify({'data': recommendations})

@app.route('/reccomend_on_user_NN_CF', methods=['GET'])
def reccomend_on_user_NN_CF():
    user_id = int(request.args.get('user_id'))
    n_recommend = int(request.args.get('n_recommend', 5))
    my_model=Model_NN_CF()
    recommendations=my_model.get_top_n_recommendations(user_id,n_recommend)
    rating_title_list = list(zip(recommendations['predicted_rating'], recommendations['title']))
    return jsonify({'data': rating_title_list})

@app.route('/reccomend_on_user_NN_CBF', methods=['GET'])
def reccomend_on_user_NN_CBF():
    gender = int(request.args.get('gender'))
    age = int(request.args.get('age'))
    occupation = int(request.args.get('occupation'))
    zip_code = int(request.args.get('zip_code'))
    n_reccomend = int(request.args.get('n_reccomend', 5))
    my_model=Model_NN_CBF()
    recommendations=my_model.get_predictions_on_all_movies(gender, age, occupation, zip_code, n_reccomend)
    rating_title_list = list(zip(recommendations['predicted_rating'], recommendations['title']))
    return jsonify({'data': rating_title_list})


@app.route('/recommend_on_user_SVD', methods=['GET'])
def recommend_on_user_SVD():
    user_id = int(request.args.get('user_id'))
    n_reccomend = int(request.args.get('n_reccomend', 5))
    reco = recommender.Recommender()
    recommendations = reco.recommend_on_user_SVD(user_id, n_reccomend)
    return jsonify({'data': recommendations})


@app.route('/register', methods=['POST'])
def register_user():

    db=Database()

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

    db=Database()
    data = request.get_json()

    username = data.get('username')
    password = data.get('password')

    if not username or not password :
        return jsonify({'error': 'All fields are required!'}), 400

    hashed_password = compute_sha256(password)

    user_data = db.login_user(username=username, password=hashed_password)
    if user_data:
        print("Login successful! User name:", user_data[1])
        return jsonify({'message': 'User login successfully!',
                        'username': user_data[1]},), 201
    else:
        print("Invalid username or password.")
        return jsonify({'error': 'Invalid username or password.'}), 400


@app.route('/login-google', methods=['POST'])
def login_google():
    data = request.get_json()
    email = data.get('email')

    db=Database()
    user=db.google_login_check(email)

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


@app.route('/users/<username>', methods=['PUT'])
def update_user(username):
    db = Database()
    data = request.get_json()

    registered_user = db.get_registered_user_by_username(username)

    if registered_user:
        user_id = registered_user[5]

        # Start transaction
        try:
            # Jeśli userId jest NULL, stwórz nowego użytkownika
            if user_id is None:
                new_gender = registered_user[4]  # Zakładam, że gender jest na indeksie 4
                new_age = data.get('age')
                new_occupation = data.get('occupation')
                new_zipcode = data.get('zipcode')

                # Dodaj nowego użytkownika
                new_user_id = db.create_user(new_gender, new_age, new_occupation, new_zipcode)

                # Zaktualizuj userId w registered_users
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
    user_id=db.get_user_id_by_username(username)[0]

    if user_id:
        movies = db.get_movie_titles_unwatched(user_id)
        movies = movies.to_dict(orient='records')
    else:
        movies = db.get_movies()
        movies=movies[['movieId','title']]
        movies = movies.to_dict(orient='records')

    return jsonify({'data': movies}), 200

@app.route('/users/<username>/userid', methods=['GET'])
def get_user_id_by_username(username):
    db = Database()
    user_id=db.get_user_id_by_username(username)
    if user_id:
        return jsonify({'userid': user_id}), 200
    else:
        return jsonify({'userid' : None}), 200


@app.route('/add-rating', methods=['POST'])
def add_user_rating():
    db = Database()
    data = request.get_json()

    userid = data['userId'][0]
    movieid = data['movieId']
    rating = data['rating']

    done = db.add_rating(userid, movieid, rating)

    if done:
        return jsonify({'success': True, 'message': 'Rating added successfully'})
    else:
        return jsonify({'success': False, 'error': 'Rating not added'})


@app.route('/<userId>/last-ratings', methods=['GET'])
def get_user_rating(userId):
    db = Database()

    ratings_df = db.get_ratings_info(userId)
    ratings = ratings_df.to_dict(orient='records')

    if ratings:
        return jsonify({'data': ratings}), 200
    else:
        return jsonify({'success': False, 'error': 'No ratings found'}), 404


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
    app.run()
