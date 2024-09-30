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
    n_reccomend = int(request.args.get('n_reccomend', 5))
    reco = recommender.Recommender()
    recommendations = reco.recommend_on_movie_kNN_CF(movie, n_reccomend)
    return jsonify({'data': recommendations})


@app.route('/recommend_on_user_history_kNN_CF', methods=['GET'])
def recommend_on_user_history_kNN_CF():
    n_reccomend = int(request.args.get('n_reccomend', 5))
    reco = recommender.Recommender()
    recommendations = reco.recommend_on_user_history_kNN_CF(n_reccomend)
    return jsonify({'data': recommendations})


@app.route('/recommend_on_movie_kNN_CBF', methods=['GET'])
def recommend_on_movie_kNN_CBF():
    movie = request.args.get('movie')
    n_reccomend = int(request.args.get('n_reccomend', 5))
    reco = recommender.Recommender()
    recommendations = reco.recommend_on_movie_kNN_CBF(movie, n_reccomend)
    return jsonify({'data': recommendations})


@app.route('/recommend_on_history_kNN_CBF', methods=['GET'])
def recommend_on_history_kNN_CBF():
    n_reccomend = int(request.args.get('n_reccomend', 5))
    reco = recommender.Recommender()
    recommendations = reco.recommend_on_history_kNN_CBF(n_reccomend)
    return jsonify({'data': recommendations})
@app.route('/reccomend_on_user_NN_CF', methods=['GET'])
def reccomend_on_user_NN_CF():
    user_id = int(request.args.get('user_id'))
    n_reccomend = int(request.args.get('n_reccomend', 5))
    my_model=Model_NN_CF()
    recommendations=my_model.get_top_n_recommendations(user_id,n_reccomend)
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
        return jsonify({'message': 'User login successfully!'}), 201
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

        return jsonify({'success': True, 'message': 'User logged in successfully'})
    else:

        return jsonify({'success': False, 'message': 'User not found. Please complete registration with username.'})



if __name__ == '__main__':
    app.run()
