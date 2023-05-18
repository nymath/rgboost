import numpy as np
from sklearn.tree import DecisionTreeRegressor
from scipy.optimize import minimize
from sklearn.base import BaseEstimator
import copy
from functools import reduce
from itertools import accumulate
from .lossfunc import SquareLoss, KernelBasedLossFunction
from .combined_estimator import EstimatorCollections

class MyGradientBoostingRegressor(object):

    def __init__(self, base_estimator, n_estimators=100,
                 learning_rate=0.1, loss_=None):
        assert hasattr(base_estimator, 'fit')
        assert hasattr(base_estimator, 'predict')
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.eta = learning_rate
        if base_estimator is None:
            self.lf = DecisionTreeRegressor()
        else:
            self.lf = loss_  # 添加一个检查类别
        self.estimators = None
        self.base_prediction = None

    def fit(self, X, y):
        self.estimators = []
        self.base_prediction = self._get_optimal_base_value(y, self.lf.loss)

        current_predictions = self.base_prediction * np.ones(shape=y.shape)
        for i in range(self.n_estimators):
            if isinstance(self.lf, KernelBasedLossFunction):
                pseudo_residuals = self.lf.negative_gradient(y, current_predictions, X)
            else:
                pseudo_residuals = self.lf.negative_gradient(y, current_predictions)

            if not isinstance(self.base_estimator, EstimatorCollections):
                self.estimators.append(copy.deepcopy(self.base_estimator))
                self.estimators[-1].fit(X, pseudo_residuals)
                current_predictions += self.eta * self.estimators[-1].predict(X)
            else:
                # TODO:
                temp_estimators = copy.deepcopy(self.base_estimator)
                temp_estimators.fit(X, pseudo_residuals)
                loss_lst = [self.lf.loss(y, model.predict(X)) for model in temp_estimators.models]
                idx = loss_lst.index(min(loss_lst))
                self.estimators.append(temp_estimators.models[idx])
                current_predictions += self.eta * self.estimators[-1].predict(X)
                pass

    def predict(self, X):

        return self.base_prediction + self.eta * reduce(lambda x, y: x + y,
                                                        [estimator.predict(X) for estimator in self.estimators])

    def get_train_process(self, X):
        res = list(accumulate([self.eta * estimator.predict(X) for estimator in self.estimators]))
        nlt = [x + self.base_prediction for x in res]
        return nlt

    @staticmethod
    def _get_optimal_base_value(y, loss):
        '''Find the optimal initial prediction for the base model.'''
        fun = lambda c: loss(y, c)
        c0 = y.mean()
        return minimize(fun=fun, x0=c0).x[0]


if __name__ == '__main__':
    from sklearn.datasets import load_boston
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error

    from sklearn.ensemble import GradientBoostingRegressor
    import warnings
    from sklearn.metrics import mean_squared_error
    from lossfunc import KernelBasedSquareLoss
    from kernels import rbf_kernel
    from functools import partial
    from estimator import KernelRidge
    warnings.filterwarnings("ignore")

    X, y = load_boston(return_X_y=True)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    gbr = GradientBoostingRegressor(n_estimators=10,
                                    learning_rate=0.5,
                                    max_depth=1,
                                    loss='squared_error',
                                    random_state=42)
    gbr.fit(X_train, y_train)
    y_pred = gbr.predict(X_train)
    mgbr = MyGradientBoostingRegressor(base_estimator=KernelRidge(alpha=1),
                                       n_estimators=10,
                                       learning_rate=0.5,
                                       loss_=KernelBasedSquareLoss(kernel=partial(rbf_kernel, gamma=0.1)))
    mgbr.fit(X_train, y_train)
    my_pred = mgbr.predict(X_train)

    mmgbr = MyGradientBoostingRegressor(base_estimator=KernelRidge(alpha=1),
                                        n_estimators=10,
                                        learning_rate=0.5,
                                        loss_=SquareLoss())
    mmgbr.fit(X_train, y_train)
    mmy_pred = mmgbr.predict(X_train)
    print(y_pred[:10])
    print(my_pred[:10])
    print(mean_squared_error(y_train, y_pred)**0.5)
    print(mean_squared_error(y_train, my_pred)**0.5)
    print(mean_squared_error(y_train, mmy_pred)**0.5)
