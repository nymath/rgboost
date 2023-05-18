import numpy as np

from abc import ABCMeta, abstractmethod


class LossFunction(object):
    __metaclass__ = ABCMeta

    @staticmethod
    @abstractmethod
    def loss(y, preds, **kwargs):
        pass

    @staticmethod
    @abstractmethod
    def negative_gradient(y, preds, **kwargs):
        pass


class KernelBasedLossFunction(object):
    __metaclass__ = ABCMeta

    @staticmethod
    @abstractmethod
    def loss(y, preds):
        pass

    @staticmethod
    @abstractmethod
    def negative_gradient(y, preds, X):
        pass

class SquareLoss(LossFunction):

    @staticmethod
    def loss(y, preds, **kwargs):
        return np.mean((y - preds) ** 2) / 2

    @staticmethod
    def negative_gradient(y, preds, **kwargs):
        return y - preds


class KernelBasedSquareLoss(KernelBasedLossFunction):

    def __init__(self, kernel):
        self.kernel = kernel

    @staticmethod
    def loss(y, preds):
        return np.mean((y - preds) ** 2) / 2

    def negative_gradient(self, y, preds, X):
        assert X.shape[0] == y.shape[0]
        return (y - preds).__matmul__(self.kernel(X))

