from argparse import ArgumentParser
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from lightgbm import LGBMRegressor


def load_data(path):
    df = pd.read_csv(path)

    return df


def split_feature_label(Xy):
    y = Xy['count']
    X = Xy.drop(['casual', 'registered', 'count'], axis=1, inplace=False)

    return X, y


def transform_feature(X):
    X['datetime'] = X['datetime'].apply(pd.to_datetime)
    X['year'] = X['datetime'].apply(lambda x: x.year)
    X['month'] = X['datetime'].apply(lambda x: x.month)
    X['day'] = X['datetime'].apply(lambda x: x.day)
    X['hour'] = X['datetime'].apply(lambda x: x.hour)
    X.drop('datetime', axis=1, inplace=True)

    return X


def transform_label(y):
    y = y.apply(np.log1p)

    return y


def train_model(X, y):
    one_hot_encoder = OneHotEncoder(handle_unknown='ignore')
    column_transformer = ColumnTransformer([
        ('one_hot_encoder', one_hot_encoder,
         ['year', 'month', 'day', 'hour', 'holiday', 'workingday', 'season', 'weather'])
    ])
    lgbm_regressor = LGBMRegressor(n_estimators=500)
    pipeline = Pipeline([
        ('column_transformer', column_transformer),
        ('lgbm_regressor', lgbm_regressor)
    ])

    pipeline.fit(X, y)

    return pipeline


def parse_args():
    parser = ArgumentParser(
        description='Generate the submission file for Kaggle Bike Sharing Demand competition.')
    parser.add_argument(
        '--train', type=Path, default='train.csv',
        help='path of train.csv downloaded from the competition')
    parser.add_argument(
        '--test', type=Path, default='test.csv',
        help='path of test.csv downloaded from the competition')

    return parser.parse_args()


def main(args):
    Xy_train = load_data(args.train)
    X_train, y_train = split_feature_label(Xy_train)
    X_train = transform_feature(X_train)
    y_train = transform_label(y_train)
    model = train_model(X_train, y_train)

    X_test = load_data(args.test)
    datetimes = X_test['datetime']
    X_test = transform_feature(X_test)
    y_test = model.predict(X_test)
    y_test = np.expm1(y_test)  # log(1 + x) is applied to the result, so undo it

    submission = {
        'datetime': datetimes,
        'count': y_test
    }
    submission = pd.DataFrame(submission)
    submission.to_csv('submission.csv', index=False)


if __name__ == '__main__':
    sys.exit(main(parse_args()))
