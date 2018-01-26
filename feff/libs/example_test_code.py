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

        # we set the number of vars:
        self.number_of_vars = 10000
        # these vars are contained this number of terms/numbers:
        self.number_of_numbers_in_var = 100
        # the value of terms should be inside the region:
        self.low = 1
        self.high = 10
        # now, if we sum the vars and divide the resulting vector by number of vars (average)
        # we will get a vector with a normal distribution of numbers.

        self.number = []
        self.normal_number = []
        self.normal = []
        self.sum_of_vars = []

    def generate_numbers(self):
        # self.number = np.random.uniform(self.low, self.high, size=self.number_of_numbers)
        self.number = np.random.randint(self.low, self.high, size=(self.number_of_vars, self.number_of_numbers_in_var))

    def plot(self):
        # plt.hist(self.number)
        plt.hist(self.sum_of_vars)
        plt.show()

    def create_normal(self):
        self.sum_of_vars = np.sum(self.number, axis=0)/self.number_of_vars
        print(self.sum_of_vars)
        # self.normal_number = self.number - sum/self.number_of_numbers_in_var


if __name__ == '__main__':
    print('-> you run ', __file__, ' file in a main mode')
    ob = Data()
    ob.generate_numbers()
    ob.create_normal()
    ob.plot()
    print('end')