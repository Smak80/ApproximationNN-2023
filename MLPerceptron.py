import data_loader as DataLoader
import numpy as np


class MLP:
    __a = 1
    __b = 1

    def __init__(self,
                 ld: DataLoader.loader,
                 neuronNum: tuple = (2,)):
        # Кол-во слоев в сети
        self.__layers = len(neuronNum) + 2
        inp = ld.getTrainInp()
        out = ld.getTrainOut()
        # Кол-во нейронов на каждом слое
        nN = [len(inp[0]), len(out[0])]
        self.__nN = np.insert(nN, 1, neuronNum)
        self.__inp = np.array(inp)
        self.__out = np.array(out)
        self.__tst_inp = np.array(ld.getTestInp())
        self.__tst_out = np.array(ld.getTestOut())
        self.__w = [np.random.rand(
            self.__nN[i] + 1,
            self.__nN[i + 1] + (0 if i == self.__layers - 2 else 1)
        ) for i in range(self.__layers - 1)]

    def nonLinAct(self, x):
        return np.array(self.__a * np.tanh(self.__b * x))

    def nonLinActDer(self, x):
        return np.array(self.__b / self.__a *
                        (self.__a - self.nonLinAct(x)) *
                        (self.__a + self.nonLinAct(x)))

    def linAct(self, x):
        return np.array(x)

    def linActDer(self, x):
        return np.array(1)

    def learn(self,
              eta = 0.01,
              epoches = 1000,
              epsilon = 0.001):
        e_full_tr = []
        e_full_ts = []

        v = np.array([None for i in range(self.__layers)])

        l = np.array([None for i in range(self.__layers)])

        l_err = np.array([None for i in range(1, self.__layers)])

        l_delta = np.array([None for i in range(1, self.__layers)])

        inp = self.__inp
        out = self.__out

        k = 0  # Счётчик эпох
        while (k < 2 or k < epoches and abs(e_full_ts[k-1] - e_full_ts[k-2]) > epsilon):
            k += 1
            for i in range(len(inp)):
                l[0] = np.array([np.insert(inp[i], 0, 1)])
                #Прямой проход
                for j in range(1, self.__layers-1):
                    v[j] = np.dot(l[j-1], self.__w[j-1])
                    l[j] = self.nonLinAct(v[j])
                v[self.__layers - 1] = np.dot(l[self.__layers-2], self.__w[self.__layers - 2])
                l[self.__layers - 1] = self.linAct(v[self.__layers - 1])
                #Обратный проход
                l_err[self.__layers - 2] = out[i] - l[self.__layers - 1]
                l_delta[self.__layers - 2] = np.array([l_err[self.__layers - 2][0] * self.linActDer(v[self.__layers-1])])
                for j in range(self.__layers - 2, 0, -1):
                    l_err[j-1] = np.dot(l_delta[j], self.__w[j].T)
                    l_delta[j-1] = l_err[j-1]*self.nonLinActDer(v[j])
                deltaW = [eta * np.dot(l_delta[j].T, l[j]) for j in range(self.__layers-1)]
                for j in range(self.__layers - 1):
                    self.__w[j] += deltaW[j].T

                outts = self.calc(self.__tst_inp)
                r_outts = np.array([self.__tst_out[i][0] for i in range(len(self.__tst_out))])
                err_n = np.sum(0.5 * (r_outts - outts) ** 2) / len(outts)
                e_full_ts.append(err_n)
                print("Epoche", k, "Test error=", err_n)
            return e_full_ts