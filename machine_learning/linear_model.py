import MySQLdb as sql
import numpy as np
from sklearn import linear_model as lin
from sklearn.preprocessing import PolynomialFeatures as Poly
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

action = ['left', 'down', 'right', 'up']
method = ['normal', 'ridge', 'lasso']


def get_Qtable():
    db = sql.connect(user='root', passwd='578449', db='mydb')
    cur = db.cursor()
    data_num = cur.execute("select * from Q_table;")
    Qtable = np.zeros([data_num, 4])
    for i in range(data_num):
        Qtable[i, :] = cur.fetchone()[1:]
    cur.close()
    db.close()
    return Qtable


def plot_Qtable(Qtable):
    dims = Qtable.shape
    x = np.array(range(dims[0]))
    mark = ['cp', 'g*', 'r.', 'b+']

    # plot2d
    fig = plt.figure(1)
    ax1 = fig.add_subplot(2, 1, 1)
    for i in range(dims[1]):
        ax1.plot(x, Qtable[:, i], mark[i], label=action[i])
        ax1.set_xlabel("state")
        ax1.set_ylabel("Qvalue")
    ax1.grid()

    ax2 = fig.add_subplot(2, 1, 2)
    index = np.argmax(Qtable, 1)
    for i in range(dims[0]):
        ax2.plot(x[i], Qtable[i, index[i]], mark[index[i]])
    ax2.set_xlabel("state")
    ax2.set_ylabel("action selected")
    ax2.grid()
    fig.legend()

    # plot3d
    y = np.array(range(dims[1]))
    X, Y = [], []
    X, Y = np.meshgrid(x, y)
    fig = plt.figure(2)
    ax = fig.gca(projection='3d')
#    ax.scatter(X, Y, Qtable.T)
#    ax.plot_wireframe(X, Y, Qtable.T)
    surf = ax.plot_surface(X, Y, Qtable.T, cmap=cm.coolwarm)
    ax.set_xlabel("state")
    ax.set_ylabel("action")
    ax.set_zlabel("Qvalue")
    ax.legend()
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()


def normalLR_1D(Qtable, poly=None):
    dims = np.asarray(Qtable).shape
    x = np.arange(dims[0]).reshape(-1, 1)
    X = x
    if poly:
        X = Poly(degree=10).fit_transform(x)
    xita = []

    fig = plt.figure(1)
    for n, reg in enumerate([lin.LinearRegression(), lin.Ridge(alpha=0.5), lin.Lasso(alpha=0.1)]):
        ax = fig.add_subplot(1, 3, n+1)
        print("\n\n" + method[n])
        for i in range(dims[1]):
            y = Qtable[:, i].reshape(-1, 1)
            out = reg.fit(X, y)
            xita.append(out.coef_[0])

            ax.plot(x, y, '.', label=action[i])
            ax.plot(x, out.predict(X), label=action[i])
            ax.set_xlabel("state")
            ax.set_ylabel("Qvalue")
            ax.legend()
            plt.title("method: {}    R^2 = {}".format(method[n], out.score(X, y)))
            print("action:{}\txita = {}{}".format(action[i], xita[i], out.intercept_))
        ax.grid()
    plt.show()


def normalLR(Qtable, poly=None):
    dims = np.asarray(Qtable).shape
    x1 = np.arange(dims[0])
    x2 = np.arange(dims[1])
    x1, x2 = np.meshgrid(x1, x2)
    x = np.vstack((x1.reshape(1, -1), x2.reshape(1, -1))).T
    X = x
    if poly:
        X = Poly(degree=10).fit_transform(x)
    y = Qtable[:, 0].reshape(-1, 1)
    for i in range(1, dims[1]):
        y = np.vstack((y, Qtable[:, i].reshape(-1, 1)))
    for n, reg in enumerate([lin.LinearRegression(), lin.Ridge(alpha=0.5), lin.Lasso(alpha=0.1)]):
        out = reg.fit(X, y)

        fig = plt.figure(n+1)
        ax = fig.gca(projection='3d')
        ax.scatter(x1, x2, Qtable.T)
        ax.plot_surface(x1, x2, out.predict(X).reshape(4, 16), cmap=cm.coolwarm)
        ax.set_xlabel("state")
        ax.set_ylabel("action")
        ax.set_zlabel("Qvalue")
        fig.suptitle("method: {}    R^2 = {}".format(method[n], out.score(X, y)))
        print("method: {}\t xita = {}{}".format(method[n], out.coef_, out.intercept_))
    plt.show()


if __name__ == '__main__':
    Qtable = get_Qtable()
    # plot_Qtable(Qtable)
    normalLR(Qtable, poly=True)
    '''
    dims = np.asarray(Qtable).shape
    x = np.arange(dims[0]).reshape(-1, 1)
    x = np.asarray([x] * 4).reshape(-1, 1)
    X = Poly(degree=10).fit_transform(x)
    y = Qtable.T.reshape(-1, 1)
    out = lin.LinearRegression().fit(X, y)
    fig = plt.figure(1)
    ax = fig.add_subplot()
    ax.plot(x, y, '.')
    ax.plot(x, out.predict(X), '^')
    plt.show()
    '''
