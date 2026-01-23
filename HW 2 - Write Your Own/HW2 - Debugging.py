import numpy as np

# ---------------- activation ----------------
def identity(z):
    return z

def identity_back(xbar, z):
    return xbar

# ---------------- loss ----------------
def mse(yhat, y):
    return np.mean((yhat - y) ** 2)

def mse_back(yhat, y):
    return 2 * (yhat - y) / y.shape[1]

# ---------------- layer ----------------
class Layer:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.activation = identity
        self.activation_back = identity_back

    def forward(self, X):
        self.X = X
        self.Z = self.W @ X + self.b
        return self.Z

    def backward(self, Xbar):
        Zbar = self.activation_back(Xbar, self.Z)

        self.Wbar = Zbar @ self.X.T / self.X.shape[1]
        self.bbar = np.mean(Zbar, axis=1, keepdims=True)

        return self.W.T @ Zbar

# ---------------- main ----------------
if __name__ == "__main__":

    np.random.seed(0)

    ns = 10

    # ---- fixed data ----
    X = np.ones((2, ns))
    y = np.ones((1, ns))

    # ---- fixed weights (from prompt) ----
    l1_W = np.array([
        [0.34710014, 0.45232547],
        [0.38122103, 0.34461438],
        [0.26794282, 0.4084993 ]
    ])
    l1_b = np.zeros((3, 1))

    l2_W = np.array([[0.30942088, 0.63057874, 0.68141247]])
    l2_b = np.zeros((1, 1))

    l1 = Layer(l1_W, l1_b)
    l2 = Layer(l2_W, l2_b)

    # ---- forward ----
    h = l1.forward(X)
    yhat = l2.forward(h)

    L = mse(yhat, y)

    print("yhat =", yhat[:, 0])
    print("loss =", L)

    # ---- backward ----
    Zbar = mse_back(yhat, y)
    print("last Zbar =", Zbar[:, 0])

    hbar = l2.backward(Zbar)
    _ = l1.backward(hbar)

    print("\nl1.Wbar =\n", l1.Wbar)
    print("l1.bbar =\n", l1.bbar)

    print("\nl2.Wbar =\n", l2.Wbar)
    print("l2.bbar =\n", l2.bbar)
