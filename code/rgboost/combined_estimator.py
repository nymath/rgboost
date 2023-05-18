class EstimatorCollections(object):
    def __init__(self, init_dict=None):
        assert isinstance(init_dict, dict)
        self.models = list(init_dict.values())

    def fit(self, X, y):
        for estimator in self.models:
            estimator.fit(X, y)

    def predict(self, X):
        return [model.predict(X) for model in self.models]

