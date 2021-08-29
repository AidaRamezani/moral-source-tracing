import pandas as pd
from influence_function import InfluenceFinder


def find_influence(df, start_point, end_point, m_sentiment, time_feature, k):
    '''
    Randomly removes M documents from time (start_point, end_point] for 1000 times to produce a null baseline

    :param df (dataframe): the original dataset, with rows as documents
    :param start_point: change point time
    :param end_point: end of window
    :param m_sentiment: moral sentiment dimension
    :param time_feature: time scale in the df
    :param k: identifies size of random set
    :return: all the random documents removed from the original dataset for 1000 times.
    '''
    influence_finder = InfluenceFinder(df, 1000, k)

    middle_df, rest_df1, rest_df2 = influence_finder.get_df(start_point, end_point, time_feature)
    all_dfs = pd.DataFrame()
    for b in range(influence_finder.B):
        perturbed_df, keeping_new_df, removing_df = influence_finder.random_perturb(middle_df, rest_df1, rest_df2)
        new_j = influence_finder.get_j(perturbed_df, m_sentiment, start_point, end_point, time_feature)
        removing_df['set number'] = [b] * len(removing_df)
        removing_df['j'] = [new_j] * len(removing_df)
        all_dfs = all_dfs.append(removing_df)

    return all_dfs

