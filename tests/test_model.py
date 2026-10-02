# tests/test_model.py
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from model import LogisticRegression

def onehot(y, k):
    Y = np.zeros((y.shape[0], k))
    Y[np.arange(y.shape[0]), y] = 1
    return Y

def test_model_accepts_expected_input():
    """Test 1: model accepts the expected input shape and trains successfully"""
    n_features, k = 10, 4
    X = np.random.rand(20, n_features)
    y = np.random.randint(0, k, size=20)
    Y_onehot = onehot(y, k)

    model = LogisticRegression(k=k, n=n_features, method="batch",
                                alpha=0.001, max_iter=10)
    model.fit(X, Y_onehot)

    assert model.W.shape == (n_features, k)

def test_model_output_shape():
    """Test 2: predict() output has the expected shape and valid class range"""
    n_features, k = 10, 4
    X_train = np.random.rand(30, n_features)
    y_train = np.random.randint(0, k, size=30)
    X_test = np.random.rand(8, n_features)
    Y_train_onehot = onehot(y_train, k)

    model = LogisticRegression(k=k, n=n_features, method="batch",
                                alpha=0.001, max_iter=10)
    model.fit(X_train, Y_train_onehot)
    preds = model.predict(X_test)

    assert preds.shape == (8,)
    assert np.all((preds >= 0) & (preds < k))