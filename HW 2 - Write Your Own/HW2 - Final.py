import numpy as np
import matplotlib.pyplot as plt

# ---------- activations ----------
def relu(z): 
    return np.maximum(0, z)

def relu_back(g, z): 
    return g * (z > 0)

def identity(z): 
    return z

def identity_back(g, z): 
    return g

# ---------- initialization ----------
def init(nin, nout):
    return (
        np.random.randn(nout, nin) * np.sqrt(2 / nin),
        np.zeros((nout, 1))
    )

# ---------- loss ----------
def mse(yhat, y):
    return np.mean((yhat - y) ** 2)

def mse_back(yhat, y):
    return 2 * (yhat - y) / y.shape[1]

# ---------- layer ----------
class Layer:
    def __init__(self, nin, nout, act):
        self.W, self.b = init(nin, nout)
        self.act = act
        self.act_back = relu_back if act == relu else identity_back

    def forward(self, X, train=True):
        Z = self.W @ X + self.b
        if train:
            self.X, self.Z = X, Z
        return self.act(Z)

    def backward(self, g):
        g = self.act_back(g, self.Z)
        self.Wg = g @ self.X.T / self.X.shape[1]
        self.bg = g.mean(axis=1, keepdims=True)
        return self.W.T @ g

# ---------- network ----------
class Network:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, X, y=None, train=True):
        for l in self.layers:
            X = l.forward(X, train)
        return (mse(X, y), X) if y is not None else X

    def backward(self, yhat, y):
        g = mse_back(yhat, y)
        for l in reversed(self.layers):
            g = l.backward(g)

# ---------- optimizer ----------
class GD:
    def __init__(self, lr, wd=0.0):
        self.lr = lr
        self.wd = wd

    def step(self, net):
        for l in net.layers:
            l.W *= (1 - self.lr * self.wd)   # weight decay
            l.W -= self.lr * l.Wg
            l.b -= self.lr * l.bg

# ======================================================
#                      MAIN
# ======================================================
if __name__ == "__main__":

    # ----- load data -----
    data = []
    with open("HW 2 - Write Your Own/auto-mpg.data") as f:
        for line in f:
            cols = line.split()
            if "?" not in cols[:8]:
                data.append([float(c) for c in cols[:8]])

    data = np.array(data)
    np.random.seed(0)
    data = data[np.random.permutation(len(data))]

    split = int(0.8 * len(data))
    train, test = data[:split], data[split:]

    Xtr, ytr = train[:, 1:], train[:, 0]
    Xte, yte = test[:, 1:], test[:, 0]

    Xm, Xs = Xtr.mean(0), Xtr.std(0)
    ym, ys = ytr.mean(), ytr.std()

    Xtr = ((Xtr - Xm) / Xs).T
    Xte = ((Xte - Xm) / Xs).T
    ytr = ((ytr - ym) / ys).reshape(1, -1)
    yte = ((yte - ym) / ys).reshape(1, -1)

    # ----- model -----
    net = Network([
        Layer(7, 32, relu),
        Layer(32, 16, relu),
        Layer(16, 1, identity)
    ])

    opt = GD(lr=1e-2, wd=1e-4)

    # ----- training -----
    epochs = 6000
    train_loss, test_loss = [], []

    for e in range(epochs):
        if e == 3000:
            opt.lr = 3e-3

        Ltr, yhat = net.forward(Xtr, ytr)
        net.backward(yhat, ytr)
        opt.step(net)

        Lte, _ = net.forward(Xte, yte, train=False)

        train_loss.append(Ltr)
        test_loss.append(Lte)

        if e % 200 == 0:
            print(f"{e}: train={Ltr:.4f}, test={Lte:.4f}")

    # ----- evaluation -----
    _, yhat = net.forward(Xte, yte, train=False)
    yhat = yhat * ys + ym
    yte = yte * ys + ym

    print("Mean absolute error (mpg):",
          np.mean(np.abs(yhat - yte)))

    # ----- plots -----
    plt.figure()
    plt.plot(train_loss, label="Train Loss")
    plt.plot(test_loss, label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training and Testing Loss Over Epochs")
    plt.legend()

    plt.figure()
    plt.scatter(yte.T, yhat.T, alpha=0.7)
    plt.plot([10, 45], [10, 45], "r--")
    plt.xlabel("True MPG")
    plt.ylabel("Predicted MPG")
    plt.title("True vs. Predicted MPG on Test Set")

    plt.show()
