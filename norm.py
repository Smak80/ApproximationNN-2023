import numpy as np

class norm:

    def __init__(self, data):
        self.__min = np.min(data)
        self.__max = np.max(data)
        self.__avg = np.average(data)
        self.__z = 1 / (self.__max - self.__min)


    def norm(self, x):
        return (x - self.__min) * self.__z * 2 - 1

    def denorm(self, y):
        return (y + 1)  / (2 * self.__z) + self.__min
