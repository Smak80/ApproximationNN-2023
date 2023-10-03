import data_loader as dl
from MLPerceptron import MLP
def plot2D():
    ld = dl.loader(trainPercent=85)
    tri = ld.getTrainInp()
    tro = ld.getTrainOut()
    tsi = ld.getTestInp()
    tso = ld.getTestOut()

    mlp = MLP(ld)
    mlp.learn()

plot2D()