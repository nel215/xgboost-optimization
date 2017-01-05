# coding:utf-8
import xgboost as xgb
from xgboost import DMatrix
from sklearn.datasets import load_boston
from skopt import gp_minimize


space_list = [
    {
        'keys': ['max_depth', 'min_child_weight'],
        'dimensions': [(3, 10), (1, 6)],
        'x0': [5, 1] ,
    },
    {
        'keys': ['gamma', 'subsample', 'colsample_bytree', 'reg_alpha'],
        'dimensions': [(0.0, 0.5), (0.6, 1.0), (0.6, 1.0), (1e-5, 100, 'log')],
        'x0': [0.0, 0.8, 0.8, 0.1] ,
    },
]


def create_objective(dtrain, base, keys):
    def objective(x):
        params = dict(**base, **dict(zip(keys, x)))
        res = xgb.cv(params=params, dtrain=dtrain, nfold=5, seed=1)
        return res['test-rmse-mean'].iat[-1]

    return objective


def optimize(dtrain):
    params = {
        'n_estimators': 100,
        'silent': 1,
        'seed': 1,
    }

    for space in space_list:
        obj = create_objective(dtrain, params, space['keys'])
        res = gp_minimize(func=obj, dimensions=space['dimensions'], x0=space['x0'], n_calls=20, random_state=1)
        params= dict(**params, **dict(zip(space['keys'], res.x)))

    return params


def main():
    X, y = load_boston(return_X_y=True)
    dtrain = DMatrix(data=X, label=y)
    params = optimize(dtrain)
    print(xgb.cv(params=params, dtrain=dtrain))


if __name__ == '__main__':
    main()
