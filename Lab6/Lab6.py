import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import stats as stats
import seaborn as sns

N = 10000
n = 100
k = 1.44

def generateNormal():
    return stats.norm.rvs(size = n)

def generateCauchy():
    return stats.cauchy.rvs(size = n)

def generateMix():
    return 0.9 * stats.norm.rvs(size = n) + 0.1 * stats.cauchy.rvs(size = n)

def huber(x):
    phi = lambda elem: elem if abs(elem) < k else k * np.sign(elem)
    return np.mean([phi(elem) for elem in x])

def twoFak(x):
    r = plt.boxplot(x)
    topPoints = r["fliers"][0].get_data()[1]
    b = np.delete(x, np.array([(x[i] in topPoints) for i in range(x.size)]))
    return b.mean()

def monteKarlo(sample, func):
    means = [func(sample) for _ in range(N)]
    return np.mean(means), np.var(means)

if __name__ == "__main__":
    print("N(0, 1):\n")
    sample = generateNormal()
    mean, var = monteKarlo(sample, np.mean)
    print("Mean: mean: " + str(mean) + " , var: " + str(var) + "\n")
    mean, var = monteKarlo(sample, np.median)
    print("Median: mean: " + str(mean) + " , var: " + str(var) + "\n")
    mean, var = monteKarlo(sample, huber)
    print("Huber: mean: " + str(mean) + " , var: " + str(var) + "\n")
    mean, var = monteKarlo(sample, twoFak)
    print("2 fak: mean: " + str(mean) + " , var: " + str(var) + "\n")

    print("C(0, 1):\n")
    sample = generateCauchy()
    mean, var = monteKarlo(sample, np.mean)
    print("Mean: mean: " + str(mean) + " , var: " + str(var) + "\n")
    mean, var = monteKarlo(sample, np.median)
    print("Median: mean: " + str(mean) + " , var: " + str(var) + "\n")
    mean, var = monteKarlo(sample, huber)
    print("Huber: mean: " + str(mean) + " , var: " + str(var) + "\n")
    mean, var = monteKarlo(sample, twoFak)
    print("2 fak: mean: " + str(mean) + " , var: " + str(var) + "\n")

    print("0.9 * N(0,1) + 0.1 * C(0, 1):\n")
    sample = generateCauchy()
    mean, var = monteKarlo(sample, np.mean)
    print("Mean: mean: " + str(mean) + " , var: " + str(var) + "\n")
    mean, var = monteKarlo(sample, np.median)
    print("Median: mean: " + str(mean) + " , var: " + str(var) + "\n")
    mean, var = monteKarlo(sample, huber)
    print("Huber: mean: " + str(mean) + " , var: " + str(var) + "\n")
    mean, var = monteKarlo(sample, twoFak)
    print("2 fak: mean: " + str(mean) + " , var: " + str(var) + "\n")

