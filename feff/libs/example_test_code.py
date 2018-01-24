'''
* Created by Zhenia Syryanyy (Yevgen Syryanyy)
* e-mail: yuginboy@gmail.com
* License: this code is under GPL license
* Last modified: 2018-01-24
'''
import numpy as np
import matplotlib.pyplot as plt

class Data:
    def __init__(self):
        self.low = 0
        self.high = 100
        self.number_of_numbers = 1000
        self.number = []
        self.normal_number = []
        self.normal = []

    def generate_numbers(self):
        # self.number = np.random.uniform(self.low, self.high, size=self.number_of_numbers)
        self.number = np.random.randint(self.low, self.high, size=self.number_of_numbers)

    def plot(self):
        # plt.hist(self.number)
        plt.hist(self.normal_number)
        plt.show()

    def create_normal(self):
        sum = np.sum(self.number)
        print(sum)
        self.normal_number = self.number - sum/self.number_of_numbers


if __name__ == '__main__':
    print('-> you run ', __file__, ' file in a main mode')
    ob = Data()
    ob.generate_numbers()
    ob.create_normal()
    ob.plot()
    print('end')