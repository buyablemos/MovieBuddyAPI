import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from PythonFiles.db import Database

import warnings

warnings.simplefilter('ignore')


def calculate_similarity(md):
    count = CountVectorizer(analyzer='word', ngram_range=(1, 2), min_df=0.0, stop_words='english')
    count_matrix = count.fit_transform(md['soup'])
    cosine_sim = cosine_similarity(count_matrix, count_matrix)
    np.save('/Users/dawid/PycharmProjects/movierecomendationAPI/cosine_similarity_matrix.npy', cosine_sim)


def weighted_rating(x, m, c):
    """Oblicza ważoną ocenę filmu."""
    v = x['vote_count']
    r = x['vote_average']
    # Ważona ocena
    return (v / (v + m) * r) + (m / (v + m) * c)


class MetadataRecommender:
    def __init__(self):
        self.db = Database()
        self.md = self.db.get_metadata_movies()
        if os.path.exists('/Users/dawid/PycharmProjects/movierecomendationAPI/cosine_similarity_matrix.npy'):
            self.cosine_sim = np.load('/Users/dawid/PycharmProjects/movierecomendationAPI/cosine_similarity_matrix.npy')
        else:
            print("No file for cosine similarity - MetadataRecommender - Creating new one...")
            calculate_similarity(self.md)
            self.cosine_sim = np.load('/Users/dawid/PycharmProjects/movierecomendationAPI/cosine_similarity_matrix.npy')

        self.md = self.md.reset_index()
        self.titles = self.md['title']
        self.indices = pd.Series(self.md.index, index=self.md['title'])

    def get_most_similar_movies(self, title, top_n=10):

        idx = self.titles[self.titles == title].index[0]

        # Pobierz wektor podobieństwa dla danego filmu
        sim_scores = self.cosine_sim[idx]

        # Uzyskaj indeksy posortowane w porządku malejącym
        sorted_indices = np.argsort(sim_scores)[::-1]

        # Weź najpierw indeksy (z pominięciem samego filmu)
        similar_indices = sorted_indices[1:top_n + 1]

        # Zwróć tytuły najbardziej podobnych filmów
        data = list(self.titles.iloc[similar_indices])
        return data

    def improved_recommendations(self, title, top_n=10):
        # Zyskanie indeksu filmu
        idx = self.titles[self.titles == title].index[0]

        # Uzyskiwanie podobieństw
        sim_scores = self.cosine_sim[idx]

        # Sortowanie podobieństw
        sorted_indices = np.argsort(sim_scores)[::-1]

        # Wybieranie 3 razy najbardziej podobnych filmów (pomijając pierwszy, który jest sam w sobie)
        similar_indices = sorted_indices[1:top_n * 3 + 1]

        # Uzyskiwanie informacji o filmach
        movies = self.md.iloc[similar_indices][['title', 'vote_count', 'vote_average', 'release_date']]

        # Przekształcanie głosów na liczby całkowite
        vote_counts = movies[movies['vote_count'].notnull()]['vote_count'].astype('int')
        vote_averages = movies[movies['vote_average'].notnull()]['vote_average'].astype('int')

        c = vote_averages.mean()
        m = vote_counts.quantile(0.60)

        # Wybieranie kwalifikujących się filmów
        qualified = movies[
            (movies['vote_count'] >= m) &
            (movies['vote_count'].notnull()) &
            (movies['vote_average'].notnull())
            ]

        # Konwersja głosów na int
        qualified['vote_count'] = qualified['vote_count'].astype('int')
        qualified['vote_average'] = qualified['vote_average'].astype('int')

        # Obliczanie ważonej oceny
        qualified['wr'] = qualified.apply(lambda x: weighted_rating(x, m=m, c=c), axis=1)

        # Sortowanie według ważonej oceny i wybieranie 10 najlepszych filmów
        qualified = qualified.sort_values('wr', ascending=False).head(top_n)

        data = list(qualified['title'])

        return data
