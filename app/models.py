import numpy as np
import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt
import numpy as np
import mlflow
from sklearn.model_selection import KFold


class LinearRegression(object):

    kfold = KFold(n_splits=3, shuffle=True, random_state=42)

    def __init__(self, regularization, lr=0.001, method='batch', num_epochs=500,
                 batch_size=50, cv=kfold, init_method='zeros',
                 use_momentum=False, momentum=0.9, max_grad_norm=5.0,
                 patience=10, shuffle=True):
        self.lr             = lr
        self.num_epochs     = num_epochs
        self.batch_size     = batch_size
        self.method         = method
        self.cv             = cv
        self.regularization = regularization

        self.init_method    = init_method
        self.use_momentum   = use_momentum
        self.momentum       = momentum
        self.max_grad_norm  = max_grad_norm
        self.patience       = patience
        self.shuffle        = shuffle

    def mse(self, ytrue, ypred):
        return ((ypred - ytrue) ** 2).sum() / ytrue.shape[0]

    def r2(self, ytrue, ypred):
        ss_res = ((ypred - ytrue) ** 2).sum()
        ss_tot = ((ytrue - ytrue.mean()) ** 2).sum()
        if ss_tot == 0:
            return 0.0
        return 1 - (ss_res / ss_tot)

    def _initialize_theta(self, n_features):
        if self.init_method == 'xavier':
            lower, upper = -(1.0 / np.sqrt(n_features)), (1.0 / np.sqrt(n_features))
            theta = lower + np.random.rand(n_features) * (upper - lower)
        else:  # 'zeros'
            theta = np.zeros(n_features)
        return theta

    def fit(self, X_train, y_train):
        self.kfold_scores = list()
        self.kfold_r2     = list()

        for fold, (train_idx, val_idx) in enumerate(self.cv.split(X_train)):

            X_cross_train = X_train[train_idx]
            y_cross_train = y_train[train_idx]
            X_cross_val   = X_train[val_idx]
            y_cross_val   = y_train[val_idx]

            self.theta = self._initialize_theta(X_cross_train.shape[1])
            self.prev_step = np.zeros(X_cross_train.shape[1])

            best_val_loss = np.inf
            best_theta = self.theta.copy()
            patience_counter = 0

            with mlflow.start_run(run_name=f"Fold-{fold}", nested=True):
                params = {
                    "method": self.method,
                    "lr": self.lr,
                    "reg": type(self.regularization).__name__,
                    "init_method": self.init_method,
                    "use_momentum": self.use_momentum,
                    "momentum": self.momentum if self.use_momentum else None,
                }

                mlflow.log_params(params=params)

                for epoch in range(self.num_epochs):
                    # Create a safely shuffled copy of the data
                    if self.shuffle:
                        perm = np.random.permutation(X_cross_train.shape[0])
                        X_epoch = X_cross_train[perm]
                        y_epoch = y_cross_train[perm]
                    else:
                        X_epoch = X_cross_train
                        y_epoch = y_cross_train

                    # Train using the selected method
                    if self.method == 'sto':
                        for batch_idx in range(X_epoch.shape[0]):
                            X_b = X_epoch[batch_idx].reshape(1, -1)
                            y_b = y_epoch[batch_idx:batch_idx+1]
                            train_loss = self._train(X_b, y_b)

                    elif self.method == 'mini':
                        for batch_idx in range(0, X_epoch.shape[0], self.batch_size):
                            X_b = X_epoch[batch_idx:batch_idx+self.batch_size, :]
                            y_b = y_epoch[batch_idx:batch_idx+self.batch_size]
                            train_loss = self._train(X_b, y_b)

                    else:
                        train_loss = self._train(X_epoch, y_epoch)

                    mlflow.log_metric(key="train_loss", value=train_loss, step=epoch)

                    # Evaluate on the validation set
                    yhat_val = self.predict(X_cross_val)
                    val_loss_new = self.mse(y_cross_val, yhat_val)
                    val_r2_new   = self.r2(y_cross_val, yhat_val)
                    mlflow.log_metric(key="val_loss", value=val_loss_new, step=epoch)
                    mlflow.log_metric(key="val_r2", value=val_r2_new, step=epoch)

                    # Early stopping with patience
                    if val_loss_new < best_val_loss - 1e-5:
                        best_val_loss = val_loss_new
                        best_r2 = val_r2_new
                        best_theta = self.theta.copy()
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        if patience_counter >= self.patience:
                            break

                # Restore the best weights for this fold
                self.theta = best_theta
                self.kfold_scores.append(best_val_loss)
                self.kfold_r2.append(best_r2)
                print(f"Fold {fold}: Best MSE={best_val_loss:.4f}, Best R2={best_r2:.4f}")

    def _train(self, X, y):
        yhat = self.predict(X)
        m    = X.shape[0]

        reg_grad = self.regularization.derivation(self.theta)
        reg_grad[0] = 0.0  # Do not penalize the bias theta[0]

        grad = (1/m) * X.T @ (yhat - y) + reg_grad

        # 1. Global Norm Clipping
        if self.max_grad_norm is not None:
            norm = np.linalg.norm(grad)
            if norm > self.max_grad_norm:
                grad = grad * (self.max_grad_norm / norm)

        # 2. Polyak Momentum
        if self.use_momentum:
            self.prev_step = self.momentum * self.prev_step + self.lr * grad
            self.theta = self.theta - self.prev_step
        else:
            self.theta = self.theta - self.lr * grad

        return self.mse(y, yhat)

    def predict(self, X):
        return X @ self.theta

    def _coef(self):
        return self.theta[1:]

    def _bias(self):
        return self.theta[0]

    def plot_feature_importance(self, feature_names=None):
        import matplotlib.pyplot as plt

        coef = self._coef()
        if feature_names is None:
            feature_names = [f"X{i+1}" for i in range(len(coef))]
        feature_names = np.array(feature_names)

        order = np.argsort(np.abs(coef))
        sorted_coef  = coef[order]
        sorted_names = feature_names[order]
        colors = ['tab:red' if c < 0 else 'tab:green' for c in sorted_coef]

        plt.figure(figsize=(8, max(4, 0.35 * len(coef))))
        plt.barh(sorted_names, sorted_coef, color=colors)
        plt.axvline(x=0, color='black', linewidth=0.8)
        plt.xlabel("Coefficient value")
        plt.title("Feature Importance (model coefficients)")
        plt.tight_layout()
        plt.show()


import sys
import numpy as np
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


# --- Penalty Classes ---
class LassoPenalty:
    def __init__(self, l):
        self.l = l

    def __call__(self, theta):
        return self.l * np.sum(np.abs(theta))

    def derivation(self, theta):
        # Use subgradient: np.sign(theta)
        return self.l * np.sign(theta)


class RidgePenalty:
    def __init__(self, l):
        self.l = l

    def __call__(self, theta):
        return self.l * np.sum(np.square(theta))

    def derivation(self, theta):
        return self.l * 2 * theta


class ElasticPenalty:
    def __init__(self, l=0.1, l_ratio=0.5):
        self.l = l
        self.l_ratio = l_ratio

    def __call__(self, theta):
        l1_contribution = self.l_ratio * self.l * np.sum(np.abs(theta))
        l2_contribution = (1 - self.l_ratio) * self.l * 0.5 * np.sum(np.square(theta))
        return l1_contribution + l2_contribution

    def derivation(self, theta):
        l1_derivation = self.l * self.l_ratio * np.sign(theta)
        l2_derivation = self.l * (1 - self.l_ratio) * theta
        return l1_derivation + l2_derivation


class NoPenalty:
    def __init__(self, l=0):
        self.l = l

    def __call__(self, theta):
        return 0.0

    def derivation(self, theta):
        # FIX 3: Return a zero array with the same shape as theta instead of scalar 0.0
        return np.zeros_like(theta)


# --- Regression Subclasses ---
class Normal(LinearRegression):
    def __init__(self, method='batch', lr=0.001, **kwargs):
        self.regularization = NoPenalty()
        super().__init__(self.regularization, lr=lr, method=method, **kwargs)


class Lasso(LinearRegression):
    def __init__(self, method='batch', lr=0.001, l=0.1, **kwargs):
        self.regularization = LassoPenalty(l)
        super().__init__(self.regularization, lr=lr, method=method, **kwargs)


class Ridge(LinearRegression):
    def __init__(self, method='batch', lr=0.001, l=0.1, **kwargs):
        self.regularization = RidgePenalty(l)
        super().__init__(self.regularization, lr=lr, method=method, **kwargs)


class ElasticNet(LinearRegression):
    def __init__(self, method='batch', lr=0.001, l=0.1, l_ratio=0.5, **kwargs):
        self.regularization = ElasticPenalty(l, l_ratio)
        super().__init__(self.regularization, lr=lr, method=method, **kwargs)


class Polynomial(LinearRegression):
    """
    Polynomial regression:
    - Expand numeric columns according to the specified degree.
    - Standardize the expanded features using StandardScaler.
    - Avoid Data Leakage by using fit_transform on the Training set
      and transform on the Validation/Test set.
    """

    def __init__(self, method='batch', lr=0.001, degree=2, n_numeric=None, **kwargs):
        if n_numeric is None:
            raise ValueError("Polynomial requires n_numeric (number of "
                             "leading numeric columns, excluding bias).")
        self.regularization = NoPenalty()
        self.degree = degree
        self.n_numeric = n_numeric
        self.poly = PolynomialFeatures(degree=self.degree, include_bias=False)
        self.scaler = StandardScaler()
        super().__init__(self.regularization, lr=lr, method=method, **kwargs)

    def _expand(self, X, is_train=False):
        bias_col = X[:, [0]]
        numeric  = X[:, 1:1 + self.n_numeric]
        rest     = X[:, 1 + self.n_numeric:]

        if is_train:
            numeric_poly = self.poly.fit_transform(numeric)
            numeric_poly = self.scaler.fit_transform(numeric_poly)
        else:
            numeric_poly = self.poly.transform(numeric)
            numeric_poly = self.scaler.transform(numeric_poly)

        return np.hstack([bias_col, numeric_poly, rest])

    def fit(self, X_train, y_train):
        # FIX 1: Do not fit_transform the entire X_train here
        # to avoid Data Leakage when K-Fold CV runs inside super().fit()
        # Instead, transform X_train before passing it to training
        X_expanded = self._expand(X_train, is_train=True)
        super().fit(X_expanded, y_train)

    def predict(self, X):
        # If the input matrix X has not been Polynomial-transformed
        # (number of columns does not match theta)
        if X.shape[1] != len(self.theta):
            X = self._expand(X, is_train=False)
        return super().predict(X)


def str_to_class(classname):
    return getattr(sys.modules[__name__], classname)