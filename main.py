import data_loader as dl
from MLPerceptron import MLP
from matplotlib import pyplot as plt
from norm import norm as n
import numpy as np

def plot2D():
    ld = dl.loader(train_percent=80)
    tri = ld.getTrainInp()
    tro = ld.getTrainOut()

    tsi = ld.getTestInp()
    tso = ld.getTestOut()

    xn = n(tri)
    yn = n(tro)
    tri = xn.norm(tri)
    tro = yn.norm(tro)
    tsi = xn.norm(tsi)
    tso = yn.norm(tso)

    # f0 = plt.figure(0)
    # fa0 = f0.add_subplot(1, 1, 1)
    # fa0.plot(tri, tro, "b-")
    # fa0.plot(tsi, tso, "r+")
    # plt.show()

    mlp = MLP(tri, tro, tsi, tso, (15, )) #36,
    e_tr, e_ts = mlp.train(epsilon=1e-4)

    e_ts_x= [i for i in range(1, len(e_ts) + 1)]

    f1 = plt.figure(1)
    fa1 = f1.add_subplot(1, 1, 1)
    out = mlp.predict(tri)
    out = yn.denorm(out)

    tri = xn.denorm(tri)
    tro = yn.denorm(tro)

    fa1.plot(tri, out, "b-")
    fa1.plot(tri, tro, "r+")

    out = mlp.predict(tsi)
    out = yn.denorm(out)

    tsi = xn.denorm(tsi)
    tso = yn.denorm(tso)

    fa1.plot(tsi, out, "gv")
    fa1.plot(tsi, tso, "y+")

    f2 = plt.figure(2)
    fa2 = f2.add_subplot(1, 1, 1)
    fa2.plot(e_ts_x, e_ts, "b-")
    fa2.plot(e_ts_x, e_tr, "r-")

    plt.show()

plot2D()