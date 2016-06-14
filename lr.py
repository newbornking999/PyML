import numpy as np


def kmeansrnd(n, d, k):
    s = 10
    m = np.random.randn(k, d)
    X = np.random.randn(n, d)
    w = np.random.dirichlet(np.ones(k), 1)
    z = np.random.multinomial(1, w[0], n)
    X += s * z @ m
    return (X, np.argmax(z, 1))


def log1pexp(x):
    seed = 33.3
    y = np.copy(x)
    idx = x < seed
    y[idx] = np.log1p(np.exp(x[idx]))
    return y


# def sigmoid(x):
#     return np.exp(-np.logaddexp(0, -x))

def sigmoid(x):
    return np.exp(-log1pexp(-x))


def lr_grad(X, y, w, beta):
    a = X @ w
    z = sigmoid(a)
    g = (z - y) @ X + beta * w
    return g


def lr_llh(X, y, w, beta):
    n = X.shape[0]
    h = 2 * y - 1
    a = X @ w
    llh = -(np.sum(log1pexp(-h * a)) + 0.5 * beta * np.dot(w, w)) / n
    return llh


def lr_gd(X, y, beta):
    n = X.shape[0]
    x0 = np.ones((n, 1))
    X = np.hstack((x0, X))
    d = X.shape[1]
    iter = 200
    llh = np.full(iter, np.inf)
    w = np.zeros(d)
    for t in range(iter):
        g = lr_grad(X, y, w, beta)
        w = w - g
        llh[t] = lr_llh(X, y, w, beta)
    return (w, llh)

def main():
    n = 200
    d = 2
    k = 2
    (X, y) = kmeansrnd(n, d, k)
    beta = 1e-2
    (w, llh) = lr_gd(X, y, beta)

if __name__ == "__main__":
    main()
