import numpy as np
from changepoint.mean_shift_model import MeanShiftModel
import pandas as pd
from statsmodels.stats.multitest import fdrcorrection
import pickle
model = MeanShiftModel()



categories = ['care.virtue', 'care.vice', 'fairness.virtue',
       'fairness.vice', 'loyalty.virtue', 'loyalty.vice', 'authority.virtue',
       'authority.vice', 'sanctity.virtue', 'sanctity.vice', 'moral relevance', 'moral polarity']

window_size = 7
step_size = 3
threshold = 0

def week_size_standardize(df, time_feature):
    article_week_count = dict(df['week'].value_counts())
    dense_weeks = [x for x , v in article_week_count.items() if v >= threshold]
    df = df[df[time_feature].isin(dense_weeks)]
    return df




def get_window_change_point(window, dates, alpha = 0.05):
    start = window[0]
    end = window[1]
    ts = window[2]
    window_str = [dates[i] for i in range(start, end)]
    stats_ts, pvals, nums = model.detect_mean_shift(ts, B=10000)
    fdr_corrections = fdrcorrection(pvals, alpha=alpha, method='indep', is_sorted=False)
    passed = fdr_corrections[0]
    new_pvals = fdr_corrections[1]
    change_points = [window_str[i] for i in range(len(passed)) if passed[i] == True]
    return change_points, new_pvals


def get_entity_time_series(df, feature, time_feature, entity, has_entity = True):
    if has_entity:
        entity_df = df.loc[df["word"] == entity]
    else:
        entity_df = df
    entity_df = entity_df[[feature, time_feature]]
    entity_df = entity_df.sort_values(by=[time_feature])
    dates = np.array(sorted(entity_df[time_feature].unique()))

    return dates,  np.array(entity_df.groupby(time_feature).mean()[feature])


def get_entity_relevance_score_time_series(df, feature, time_feature, entity, has_entity = True):
    if has_entity:
        entity_df = df.loc[df["word"] == entity]
    else:
        entity_df = df
    entity_df['score'] = entity[feature] > 0
    entity_df = entity_df[['score', time_feature]]
    entity_df = entity_df.sort_values(by=[time_feature])
    dates = np.array(sorted(entity_df[time_feature].unique()))

    return dates,  np.array(entity_df.groupby(time_feature).mean()['score'])


def get_windows(ts, window_size):

    n = len(ts)
    starting_points = list(range(0, n, step_size))
    windows = [(i, i + window_size - 1 if i + window_size <= n else n - 1 ,ts[i:i + window_size]) for i in starting_points]
    return windows


def get_change_point(moral_sentiment, src_dir, time_feature, entity, data_source):



    list_rows = []
    df_raw = pd.read_csv(src_dir)

    if time_feature == 'year_month':
        df_raw['date'] = pd.to_datetime(df_raw[['years', 'months', 'days']])
        df_raw['year_month'] = pd.to_datetime(df_raw['date']).dt.to_period('M')
        df_raw['year_month'] = df_raw['year_month'].astype(str)

    df = df_raw[df_raw['relevance_p_mean'] > 0]
    df = week_size_standardize(df, time_feature)

    if moral_sentiment.endswith('virtue'):
        feature = moral_sentiment + '_mean'
        df = df[df['polarity_p_mean'] > 0.5]
    elif moral_sentiment.endswith('vice'):
        feature = moral_sentiment + '_mean'
        df = df[df['polarity_p_mean'] < 0.5]
    elif moral_sentiment == 'moral polarity':
        feature = 'polarity_p_mean'
    else:
        feature = 'relevance_p_mean'




    point_stats = {}

    dates, time_series_mp = get_entity_time_series(df, feature, time_feature, entity, False)
    windows = get_windows(time_series_mp, window_size)


    stats_ts, pvals, nums = model.detect_mean_shift(time_series_mp, B=10000)
    fdr_corrections = fdrcorrection(pvals, alpha=0.05, method='indep', is_sorted=False)

    passed = fdr_corrections[0]
    new_pvals = fdr_corrections[1]
    change_points = [dates[i] for i in range(len(passed)) if passed[i] == True]

    point_stats[(moral_sentiment, 'articles sharpened', dates[0], dates[- 1], None, None)] = (pvals, new_pvals, change_points)


    row = {'start' : dates[0], 'end': dates[-1], 'change point':dates[np.argmin(pvals)], 'pvalue':  np.min(pvals), 'fine-grained': moral_sentiment}
    list_rows.append(row)



    print('change point detection on', moral_sentiment, 'window based')
    for window in windows:
        start = window[0]
        end = window[1]
        ts = window[2]
        if len(ts) < 3:
            continue
        stats_ts, pvals, nums = model.detect_mean_shift(ts, B=10000)
        change_points, new_pvals = get_window_change_point(window, dates)

        print(dates[np.argmin(pvals) + start])
        point_stats[(moral_sentiment, 'articles sharpened', dates[start], dates[end - 1], window_size, step_size)] = (pvals, new_pvals,change_points )
        row = {'start': dates[start], 'end': dates[end], 'change point': dates[np.argmin(pvals) + start], 'pvalue': np.min(pvals),
               'fine-grained': moral_sentiment}
        list_rows.append(row)
        print('-----------------')

    cp_df = pd.DataFrame(list_rows)

    pickle.dump(point_stats, open('../../data/change_points/' + entity +'_' + moral_sentiment + '_' + data_source + '.pkl', 'wb'))
    return cp_df


def save_changepoint(entities, source_dirs, topic_nums, data_source = 'COVID_news', time_feature = 'week'):

    '''

    :param entities: a list of strings. e.g.: ['trump', 'fauci']
    :param source_dirs: a list of strings, path to each entity moral sentiment df. Generated by moral_sentiment.py
    :param topic_nums: 10
    :param data_source: COVID_news of NYT
    :param time_feature: week or year_month
    :return: saves change point information for each entity
    '''

    for entity, src_dir, n in zip(entities, source_dirs, topic_nums):

        total_df = pd.DataFrame()
        for cat in categories:
            df = get_change_point(cat, src_dir, time_feature, entity, data_source)
            total_df = total_df.append(df)

        total_df.to_csv('../../data/change_points/' + entity + '_' + data_source + '.csv', index=False)

