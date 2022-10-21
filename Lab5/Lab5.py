import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import stats as stats
import seaborn as sns

def generateData():
    array = stats.norm.rvs(size=195)
    outlyings = np.array([5, -4, 3.3, 2.99, -3])
    array = np.append(array, outlyings)
    return array


# Task 1
# Function to do 3sigma man rule
def task1(sample):
    plt.figure()
    print("Task 2:")
    mean = np.mean(sample)
    var = np.var(sample)
    isOutlying = lambda x: np.abs(x - mean) > 3*var
    outlyings = [val for i, val in enumerate(sample) if isOutlying(val)]
    outlyingsIndexes = [i for i, val in enumerate(sample) if isOutlying(val)]
    sample = np.delete(sample, outlyingsIndexes)
    print("Sigma's outlyings: ", outlyings)
    plt.plot(sample, '*', label = 'sample')
    plt.plot(outlyings, 'o', label = 'outlyings')
    plt.grid()
    plt.legend()


# Task 2
# Fucntion to plot boxplot
def task2(sample):
    plt.figure()
    print("\nTask 3:")
    outlyings = plt.boxplot(sample)
    print("Boxplot's outlyings: ", [x.get_ydata() for x in outlyings["fliers"]])

if __name__ == "__main__":
    sample = generateData()
    task1(sample)
    task2(sample)
    plt.show()