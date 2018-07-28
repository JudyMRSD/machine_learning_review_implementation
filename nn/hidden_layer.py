import numpy as np

# sigmoid function
def nonlin(x, deriv=False):
    # backward
    if (deriv == True):
        return x * (1 - x)
    # forward
    return 1 / (1 + np.exp(-x))

# input dataset
X = np.array([[0, 0, 1],
              [0, 1, 1],
              [1, 0, 1],
              [1, 1, 1]])

# output dataset
t = np.array([[0, 0, 1, 1]]).T
# seed random numbers to make calculation
# deterministic (just a good practice)
np.random.seed(1)

# initialize weights randomly with mean 0
w = 2 * np.random.random((3, 1)) - 1

lr = 0.01

for iter in range(10000):
    # forward propagation
    a = np.dot(X, w)
    y = nonlin(a)

    # how much did we miss?
    # l1_error = t - y

    # multiply how much we missed by the
    # slope of the sigmoid at the values in l1
    print("nonlin(y, True)", nonlin(y, True))
    print("X", X)
    print("lr", lr)
    print("w", w)
    w_delta = lr * np.matmul(nonlin(y, True).T, X)
    print(w_delta)
    # update weights
    w = w + w_delta

print ("Output After Training:")
print (a)
