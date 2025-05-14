import numpy as np
import pandas as pd
from catboost import CatBoostRegressor


def main():
    train = pd.read_csv('train.csv')
    X_train = train.drop('y', axis=1)
    y_train = train['y']

    X_test = pd.read_csv('test_x.csv')

    model = CatBoostRegressor(
        verbose=False,
        random_seed=42
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    pd.Series(predictions, name='y').to_csv('test_y.csv', index=False)


if __name__ == '__main__':
    main()
