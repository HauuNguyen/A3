import numpy as np

class LogisticRegression:

    def __init__(self, k, n, method, alpha=0.001, max_iter=5000,
                 use_ridge=False, l=0.1):
        self.k = k
        self.n = n
        self.alpha = alpha
        self.max_iter = max_iter
        self.method = method
        self.use_ridge = use_ridge
        self.l = l

    def fit(self, X, Y):
        self.W = np.random.rand(self.n, self.k)
        self.losses = []

        if self.method == "batch":
            for i in range(self.max_iter):
                loss, grad = self.gradient(X, Y)
                self.losses.append(loss)
                self.W = self.W - self.alpha * grad

        elif self.method == "minibatch":
            batch_size = int(0.3 * X.shape[0])
            for i in range(self.max_iter):
                ix = np.random.randint(0, X.shape[0])
                batch_X = X[ix:ix+batch_size]
                batch_Y = Y[ix:ix+batch_size]
                loss, grad = self.gradient(batch_X, batch_Y)
                self.losses.append(loss)
                self.W = self.W - self.alpha * grad

        elif self.method == "sto":
            list_of_used_ix = []
            for i in range(self.max_iter):
                idx = np.random.randint(X.shape[0])
                while i in list_of_used_ix:
                    idx = np.random.randint(X.shape[0])
                X_train = X[idx, :].reshape(1, -1)
                Y_train = Y[idx]
                loss, grad = self.gradient(X_train, Y_train)
                self.losses.append(loss)
                self.W = self.W - self.alpha * grad
                list_of_used_ix.append(i)
                if len(list_of_used_ix) == X.shape[0]:
                    list_of_used_ix = []
        else:
            raise ValueError('Method must be one of the followings: "batch", "minibatch" or "sto".')

    def gradient(self, X, Y):
        m = X.shape[0]
        h = self.h_theta(X, self.W)
        loss = - np.sum(Y * np.log(h)) / m
        error = h - Y
        grad = self.softmax_grad(X, error)

        if self.use_ridge:
            loss += self.l * np.sum(self.W ** 2)
            grad = grad + 2 * self.l * self.W

        return loss, grad

    def softmax(self, theta_t_x):
        return np.exp(theta_t_x) / np.sum(np.exp(theta_t_x), axis=1, keepdims=True)

    def softmax_grad(self, X, error):
        return X.T @ error

    def h_theta(self, X, W):
        return self.softmax(X @ W)

    def predict(self, X_test):
        return np.argmax(self.h_theta(X_test, self.W), axis=1)

    def accuracy(self, ytrue, ypred):
        return np.sum(ytrue == ypred) / len(ytrue)

    def _confusion_counts(self, ytrue, ypred, c):
        TP = np.sum((ypred == c) & (ytrue == c))
        FP = np.sum((ypred == c) & (ytrue != c))
        FN = np.sum((ypred != c) & (ytrue == c))
        return TP, FP, FN

    def precision(self, ytrue, ypred, c):
        TP, FP, _ = self._confusion_counts(ytrue, ypred, c)
        return TP / (TP + FP) if (TP + FP) > 0 else 0.0

    def recall(self, ytrue, ypred, c):
        TP, _, FN = self._confusion_counts(ytrue, ypred, c)
        return TP / (TP + FN) if (TP + FN) > 0 else 0.0

    def f1_score(self, ytrue, ypred, c):
        p, r = self.precision(ytrue, ypred, c), self.recall(ytrue, ypred, c)
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    def macro_f1(self, ytrue, ypred):
        classes = np.unique(ytrue)
        return np.mean([self.f1_score(ytrue, ypred, c) for c in classes])

    def weighted_f1(self, ytrue, ypred):
        classes = np.unique(ytrue)
        weights = np.array([np.sum(ytrue == c) / len(ytrue) for c in classes])
        f1s = np.array([self.f1_score(ytrue, ypred, c) for c in classes])
        return np.sum(weights * f1s)