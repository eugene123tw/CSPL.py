from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

def getNaturalKs(xs, ys):
    n = len(xs) - 1
    A = np.zeros((n+1, n+2))

    for i in range(1, n):
        A[i][i - 1] = 1 / (xs[i] - xs[i - 1])
        A[i][i] = 2 * (1 / (xs[i] - xs[i - 1]) + 1 / (xs[i + 1] - xs[i]))
        A[i][i + 1] = 1 / (xs[i + 1] - xs[i])
        A[i][n + 1] = 3*((ys[i] - ys[i - 1])/((xs[i] - xs[i - 1])**2)+(ys[i + 1]-ys[i])/((xs[i + 1] - xs[i])**2))

    A[0][0] = 2 / (xs[1] - xs[0])
    A[0][1] = 1 / (xs[1] - xs[0])
    A[0][n + 1] = 3 * (ys[1] - ys[0]) / ((xs[1] - xs[0])**2)

    A[n][n - 1] = 1 / (xs[n] - xs[n - 1])
    A[n][n] = 2 / (xs[n] - xs[n - 1])
    A[n][n + 1] = 3 * (ys[n] - ys[n - 1]) / ((xs[n] - xs[n - 1])**2)

    ks = np.linalg.solve(A[:, :n+1], A[:,n+1])
    return ks

def evalSpline(x, xs, ys, ks):
    i = 1
    while (xs[i] < x): i+=1
    t = (x - xs[i - 1]) / (xs[i] - xs[i - 1])
    a = ks[i - 1] * (xs[i] - xs[i - 1]) - (ys[i] - ys[i - 1])
    b = -ks[i] * (xs[i] - xs[i - 1]) + (ys[i] - ys[i - 1])
    q = (1 - t) * ys[i - 1] + t * ys[i] + t * (1 - t) * (a * (1 - t) + b * t)
    return q

if __name__ == '__main__':
    X = []
    Y = []
    xs = [-1, 0.5, 2]
    ys = [1.5, 1, 1]

    ks = getNaturalKs(xs, ys)
    for x in np.arange(xs[0], xs[-1], 0.05):
        X.append(x)
        Y.append(evalSpline(x, xs, ys, ks))

    plt.axis("equal")
    for i in range(len(xs)):
        circle = plt.Circle((xs[i], ys[i]), radius=0.05, color='r')
        plt.gcf().gca().add_artist(circle)

    plt.plot(X, Y)
    plt.show()