from utils.data_utils import read_ds
from config import *
from pprint import pprint

from sklearn.metrics import mean_squared_error

# SKLearn Regressors
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor

from sklearn.tree import DecisionTreeRegressor

from sklearn.multioutput import MultiOutputRegressor


def train_and_predict(estimators):
    y_test_predict = dict()
    y_mse = dict()
    for name, estimator in estimators.items():
        estimator.fit(x_train, y_train)  # fit() with instantiated object
        y_test_predict[name] = estimator.predict(x_val)  # Make predictions and save it in dict under key: name
        y_mse[name] = mean_squared_error(y_val, y_test_predict[name])

    return y_test_predict, y_mse


if __name__ == "__main__":
    x_train, y_train = read_ds(TRAIN_DS_DIR)
    x_val, y_val = read_ds(VAL_DS_DIR)

    # Prepare a dictionary of estimators after instantiating each one of them
    estimators = {
        # "Extra trees": ExtraTreesRegressor(n_estimators=10,
        #                                    max_features=32,  # Out of 128
        #                                    random_state=MODELS_RANDOM_SEED),
        # "K-nn": KNeighborsRegressor(random_state=MODELS_RANDOM_SEED),
        # "Linear regression": LinearRegression(),
        # "Ridge": RidgeCV(),
        # "Lasso": Lasso(random_state=MODELS_RANDOM_SEED),
        # "ElasticNet": ElasticNet(random_state=MODELS_RANDOM_SEED),
        # "RandomForestRegressor": RandomForestRegressor(n_estimators=100, max_depth=10, random_state=MODELS_RANDOM_SEED),
        # "Decision Tree Regressor": DecisionTreeRegressor(max_depth=10, random_state=MODELS_RANDOM_SEED),
        # "MultiO/P GBR": MultiOutputRegressor(
        #     GradientBoostingRegressor(n_estimators=15, random_state=MODELS_RANDOM_SEED)),
        "MultiO/P AdaB": MultiOutputRegressor(AdaBoostRegressor(n_estimators=15, random_state=MODELS_RANDOM_SEED))
    }

    _, y_mse = train_and_predict(estimators=estimators)

    pprint(y_mse)
