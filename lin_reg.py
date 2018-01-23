#
# inspired by https://dsgazette.com/2018/01/10/linear-regression-by-hand/
#

import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn import datasets
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


class QuantumParticle:
    def __init__(self,m=2):
        self.M = m
        self.w = np.random.uniform(-1., 1., self.M)
        self.p_w = np.array(self.w)
        self.g_w = np.array(self.w)
        self.c = np.array(self.w)
        self.alpha = 0.7
        self.last_fitness = None
        self.fitness = None

    def compute_weights(self):
        phi = np.random.uniform(0., 1.)
        p = np.add(np.multiply(phi, self.p_w), np.multiply(np.subtract(1., phi), self.g_w))
        u = np.random.uniform(0., 1.)
        for i in range(len(self.w)):
            if np.random.uniform(0., 1.) < 0.5:
                self.w[i] = p[i] + self.alpha * np.abs(self.w[i] - self.c[i]) * np.log(1. / u)
            else:
                self.w[i] = p[i] - self.alpha * np.abs(self.w[i] - self.c[i]) * np.log(1. / u)

    # lin regresion yi = b0+b1*xi1+b2*xi2+...+bn*xin
    def evaluate(self, x):
        # b0
        # x = np.array(x)
        # i, j = np.shape(x)
        # x = np.resize(x,(i,j+1))
        # x[:, 0] = 1.
        return np.dot(x, self.w)

    def compute_fitness(self, data, t):
        y = self.evaluate(data)
        self.fitness = mean_squared_error(t, y)

        if self.last_fitness is None or self.last_fitness > self.fitness:
            self.last_fitness = self.fitness
            self.p_w = self.w


class QPSOLinearRegressor:
    def __init__(self, population_size, iterations, m=2):
        self.M = m
        self.population_size = population_size
        self.iterations = iterations
        self.population = []
        self.fitness_history = []

        for i in range(self.population_size):
            self.population.append(QuantumParticle(self.M))


    def run(self, data, t):

        for i in range(self.iterations):
            sum_of_weights = np.zeros(self.M)
            for p in self.population:
                sum_of_weights = np.add(sum_of_weights, p.p_w)
            c = np.divide(sum_of_weights, float(self.population_size))
            for p in self.population:
                p.c = c
                p.compute_weights()
                p.compute_fitness(data, t)

            self.population.sort(key=lambda particle: particle.fitness)

            for p in self.population:
                p.g_w = self.population[0].w

            self.fitness_history.append(self.population[0].fitness)
            print('iteration(%d) = %f | %s' % (i, self.population[0].fitness, str(self.population[0].w)))

        return self.population[0]


def compute_beta(X, y):
    one = np.matmul(np.transpose(X), X)
    one = np.linalg.inv(one)

    return np.matmul(np.matmul(one, X.T), y)


def predict(b, X):
    return X.dot(b)


def test():
    boston = datasets.load_boston()
    y = boston.target
    # print(len(boston.data[0]))
    # print(boston.data[0])
    # print(boston.target[0])
    b = compute_beta(boston.data, y)
    pred = predict(b, boston.data)

    lin_reg = QPSOLinearRegressor(1000, 300, len(boston.data[0]))
    best = lin_reg.run(boston.data, y)
    plt.plot(lin_reg.fitness_history)
    plt.title('qpso fitness evolution')
    plt.show()

    y = np.reshape(y, (506, 1))
    lr = LinearRegression()
    lr.fit(boston.data, y)
    qpso_pred = best.evaluate(boston.data)
    qpso_error = mean_squared_error(qpso_pred, y)
    lr_pred = lr.predict(boston.data)
    lr_error = mean_squared_error(lr_pred, y)
    matrix_error = mean_squared_error(pred, y)

    print('qpso error:' + str(qpso_error))
    print('sklearn lr error:' + str(lr_error))
    print('matrix computation error:' + str(matrix_error))

    # print(lr.coef_)
    # print(best.w)

    plt.plot(boston.target, 'o', color='green')
    plt.plot(lr_pred, 'b')
    plt.plot(qpso_pred, 'r')
    plt.plot(pred, 'yellow')
    plt.title('Comparison of lin. regression computation')
    plt.legend(['boston data set', 'qpso lin. regresion prediction', 'sklearn lin. regresion prediction',
                'matrix computed lin. regresion prediction'])
    plt.show()


if __name__ == '__main__':
    test()
