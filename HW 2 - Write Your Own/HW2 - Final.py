import numpy as np
import matplotlib.pyplot as plt

# ---------------- activations ----------------
def relu(z):
    return np.maximum(0, z)

def relu_back(xbar, z):
    return xbar * (z > 0)

def identity(z):
    return z

def identity_back(xbar, z):
    return xbar

# ---------------- initialization ----------------
def initialization(nin, nout):
    # He initialization (fixes dead ReLU problem)
    W = np.random.randn(nout, nin) * np.sqrt(2 / nin)
    b = np.zeros((nout, 1))
    return W, b

# ---------------- loss ----------------
def mse(yhat, y):
    return np.mean((yhat - y) ** 2)

def mse_back(yhat, y):
    return 2 * (yhat - y) / y.shape[1]

# ---------------- layer ----------------
class Layer:
    def __init__(self, nin, nout, activation=identity):
        self.W, self.b = initialization(nin, nout)
        self.activation = activation
        self.activation_back = relu_back if activation == relu else identity_back

    def forward(self, X, train=True):
        Z = self.W @ X + self.b
        Xnew = self.activation(Z)
        if train:
            self.X = X
            self.Z = Z
        return Xnew

    def backward(self, Xbar):
        Zbar = self.activation_back(Xbar, self.Z)
        self.Wbar = Zbar @ self.X.T / self.X.shape[1]
        self.bbar = np.mean(Zbar, axis=1, keepdims=True)
        return self.W.T @ Zbar

# ---------------- network ----------------
class Network:
    def __init__(self, layers, loss):
        self.layers = layers
        self.loss = loss
        self.loss_back = mse_back

    def forward(self, X, y, train=True):
        for layer in self.layers:
            X = layer.forward(X, train)
        yhat = X
        L = self.loss(yhat, y)
        if train:
            self.y = y
            self.yhat = yhat
        return L, yhat

    def backward(self):
        Xbar = self.loss_back(self.yhat, self.y)
        for layer in reversed(self.layers):
            Xbar = layer.backward(Xbar)

# ---------------- optimizer ----------------
class GradientDescent:
    def __init__(self, alpha):
        self.alpha = alpha

    def step(self, network):
        for layer in network.layers:
            layer.W -= self.alpha * layer.Wbar
            layer.b -= self.alpha * layer.bbar

# ======================================================
#                      MAIN
# ======================================================
if __name__ == "__main__":

    # ---------- data ----------
    numeric_data = []
    with open("HW 2 - Write Your Own/auto-mpg.data", "r") as file:
        for line in file:
            cols = line.strip().split()
            if "?" in cols[:8]:
                continue
            numeric_data.append([float(v) for v in cols[:8]])

    data = np.array(numeric_data)

    np.random.seed(0)
    data = data[np.random.permutation(len(data))]

    split = int(0.8 * len(data))
    train, test = data[:split], data[split:]

    Xtrain, ytrain = train[:, 1:], train[:, 0]
    Xtest, ytest = test[:, 1:], test[:, 0]

    Xmean, Xstd = Xtrain.mean(axis=0), Xtrain.std(axis=0)
    ymean, ystd = ytrain.mean(), ytrain.std()

    Xtrain = (Xtrain - Xmean) / Xstd
    Xtest = (Xtest - Xmean) / Xstd
    ytrain = (ytrain - ymean) / ystd
    ytest = (ytest - ymean) / ystd

    Xtrain = Xtrain.T
    Xtest = Xtest.T
    ytrain = ytrain.reshape(1, -1)
    ytest = ytest.reshape(1, -1)

    # ---------- model ----------
    layers = [
        Layer(7, 16, relu),
        Layer(16, 8, relu),
        Layer(8, 1, identity)
    ]

    network = Network(layers, mse)
    optimizer = GradientDescent(alpha=1e-2)  # ← KEY FIX

    # ---------- training ----------
    epochs = 4000
    train_losses = []
    test_losses = []

    for epoch in range(epochs):
        Ltrain, _ = network.forward(Xtrain, ytrain, train=True)
        network.backward()
        optimizer.step(network)

        Ltest, _ = network.forward(Xtest, ytest, train=False)

        train_losses.append(Ltrain)
        test_losses.append(Ltest)

        if epoch % 200 == 0:
            print(f"epoch {epoch}: train={Ltrain:.4f}, test={Ltest:.4f}")

    # ---------- evaluation ----------
    _, yhat = network.forward(Xtest, ytest, train=False)
    yhat = yhat * ystd + ymean
    ytest = ytest * ystd + ymean

    print("avg absolute error (mpg):",
          np.mean(np.abs(yhat - ytest)))

    # ---------- plots ----------
    plt.figure()
    plt.plot(train_losses, label="train")
    plt.plot(test_losses, label="test")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("loss")

    plt.figure()
    plt.plot(ytest.T, yhat.T, "o")
    plt.plot([10, 45], [10, 45], "--")
    plt.xlabel("true mpg")
    plt.ylabel("predicted mpg")

    plt.show()
