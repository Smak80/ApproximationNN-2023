import data_loader as dl
from MLPerceptron import MLP
from matplotlib import pyplot as plt

def plot2D():
    ld = dl.loader(trainPercent=85)
    tri = ld.getTrainInp()
    tro = ld.getTrainOut()
    tsi = ld.getTestInp()
    tso = ld.getTestOut()

    mlp = MLP(ld, (12, 30))
    e_tr, e_ts = mlp.learn(epsilon= 25e-4)
    e_ts_x = [i for i in range(1, len(e_ts) + 1)]
    f1 = plt.figure(1)
    fa1 = f1.add_subplot(1, 1, 1)
    out = mlp.calc(tri)
    fa1.plot(tri, out, "bo")
    fa1.plot(tri, tro, "r+")

    out = mlp.calc(tsi)
    fa1.plot(tsi, out, "gv")
    fa1.plot(tsi, tso, "y+")

    f2 = plt.figure(2)
    fa2 = f2.add_subplot(1, 1, 1)
    fa2.plot(e_ts_x, e_ts, "b-")
    plt.show()


plot2D()