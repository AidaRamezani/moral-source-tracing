import math
import numpy as np
import pandas as pd

def log_odds(pos_prob, neg_prob):
    if pos_prob == 0 :
        return 0
    if pos_prob == 1:
        return 1
    return math.log(pos_prob / neg_prob)
def get_neutral_words():


    valence_df = pd.read_csv("../../data/valence_rating.csv")

    neutral_df = valence_df.loc[valence_df["V.Mean.Sum"] >= 4.2].loc[valence_df["V.Mean.Sum"] <= 6.2]
    return neutral_df["Word"].to_list()


class CentroidModel():
    """
    Centroid classifier for moral sentiments
    """
    name = 'Centroid'

    def __calc_prob(self, X):
        X_1 = np.exp(X)
        softmax_prob = X_1 / np.sum(X_1, axis=0)
        if np.any(np.isnan(softmax_prob)):
            return [0] * len(X)
        return softmax_prob

    def predict_proba(self, data):
        result = []
        for d in data:
            distances = {k: -np.linalg.norm(d - v) for k, v in self.mean_vectors.items()}
            cat_names = sorted(self.mean_vectors.keys())
            probabilities = self.__calc_prob([distances[k] for k in cat_names])
            x_3 = dict(zip(cat_names, probabilities))
            result.append(x_3)
        return result

    def fit(self, df, label, feature_cols):
        self.h = 1
        self.mean_vectors = {}
        for i in df[label].unique():
            mean_vector = np.mean(np.array(df[df[label] == i].loc[:, feature_cols]), axis = 0)
            self.mean_vectors[i] = mean_vector #np arrays

    def predict(self, data):
        all_guesses = []
        probs_data = self.predict_proba(data)
        for d in probs_data:
            all_guesses.append(max(d.keys(), key=(lambda key: d[key])))
        return all_guesses
