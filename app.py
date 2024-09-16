from flask import Flask, request, jsonify
import recommender

app = Flask(__name__)


@app.route('/')
def hello_world():  # put application's code here
    reco = recommender.Recommender()
    returned_reco = reco.recommend_on_user_SVD(8, 10)
    return returned_reco


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


@app.route('/recommend_on_user_SVD', methods=['GET'])
def recommend_on_user_SVD():
    user_id = int(request.args.get('user_id'))
    n_reccomend = int(request.args.get('n_reccomend', 5))
    reco = recommender.Recommender()
    recommendations = reco.recommend_on_user_SVD(user_id, n_reccomend)
    return jsonify({'data': recommendations})


if __name__ == '__main__':
    app.run()
