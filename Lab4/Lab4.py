import numpy as np
import pandas as pd
import scipy as sp
from dateutil import parser, rrule
from datetime import datetime, time, date
import scipy.linalg
import csv
from scipy import fft
from matplotlib import pyplot as plt
from hurst import compute_Hc, random_walk
# globals
N = 200
h = 0.02

# Function to calculate prony
def prony(x: np.array, T: float):
    if len(x) % 2 == 1:
        x = x[:len(x)-1]

    p = len(x) // 2

    shift_x = [0] + list(x)
    a = scipy.linalg.solve([shift_x[p+i:i:-1] for i in range(p)], -x[p::])

    z = np.roots([*a[::-1], 1])

    h = scipy.linalg.solve([z**n for n in range(1, p + 1)], x[:p])

    f = 1 / (2 * np.pi * T) * np.arctan(np.imag(z) / np.real(z))
    alfa = 1 / T * np.log(np.abs(z))
    A = np.abs(h)
    fi = np.arctan(np.imag(h) / np.real(h))

    return f, alfa, A, fi  

# Function to generate sample
# Return sample from task 1
def generateSample():
    return np.array([sum([(k * np.exp(-h * i / k) * np.cos(4 * np.pi * k * h * i + np.pi / k)) for k in range(1, 4)]) for i in range(1, N + 1)])

# Function to generate model
def generateModel():
    return np.array([np.sqrt(k * h) for k in range(N)])


# Function to calculate slide median points
# In: sample, slide period
# Out: array of mean points
def slideMedian(sample, m):
    res = []
    for i in range(sample.size):
        # check for edge cases
        if i < m:
            res.append(np.median(sample[0 : 2 * i + 1]))
        elif i >= sample.size - m - 1 :
            res.append(np.median(sample[i - (sample.size - i) : sample.size]))
        # default case
        else:
            res.append(np.median(sample[i - m : i + m + 1]))
    return res

# Function to calculate rotation points
# In: sample
# Out: rotation point array
def calculateRotationPoints(sample):
    res = []
    for i in range(1, len(sample) - 2):
        if (sample[i] > sample[i - 1] and sample[i] > sample[i + 1]) or (sample[i] < sample[i - 1] and sample[i] < sample[i + 1]):
            res.append(sample[i])
    return res

# Function to check randomness with Kandell
# In: sample, trend
def checkKandell(sample, trend):
    #plt.figure()
    #plt.title("Task 4")
    #plt.plot(sample,label = "tail")
    #plt.legend()
    tail = sample - trend
    rotationPoints = calculateRotationPoints(tail)

    pMean = (2.0 / 3.0) * (len(sample) - 2)
    pDisp = (16 * len(sample) - 29) / 90.0
    pSize = len(rotationPoints)

    print("Calculated rotation number's sum: ", pSize)
    if pSize < pMean + pDisp and pSize > pMean - pDisp:
        print("\nRandomness\n")
    elif pSize > pMean + pDisp:
        print("\nRapidly oscillating\n")
    elif pSize < pMean - pDisp:
        print("\nPositively correlated\n")


if __name__ == "__main__":
    # Task 1
    print("Task 1: \n")
    sample = generateSample()
    print(sample)
    plt.figure()
    plt.title("Task 1")
    plt.plot(sample, 'o', color = 'black', label = "sample")
    plt.legend()
    
    n = 128
    # Time vector
    t = np.linspace(0, 1, n, endpoint=True)

    # Amplitudes and freqs
    f1, f2, f3 = 2, 7, 12
    A1, A2, A3 = 5, 1, 3

    # Signal
    x = A1 * np.cos(2*np.pi*f1*t) + A2 * np.cos(2*np.pi*f2*t) + A3 * np.cos(2*np.pi*f3*t)

    f, alfa, A, fi = prony(x, 0.1)
    plt.figure()
    plt.stem(2*A)
    plt.plot()
    plt.grid()

    plt.show()

    # Task 2
    data_raw = pd.read_csv('LONDON.csv')
    # Give the variables some friendlier names and convert types as necessary.
    data_raw['mean_temp'] = data_raw['mean_temp'].astype(float)

    data_raw['date'] = [datetime.strptime(str(d), '%Y%m%d') for d in data_raw['date']]
    
    # Extract out only the data we need.
    data = data_raw.loc[:, ['date', 'mean_temp']]
    print(data)

    # plot all values
    plt.plot(data['mean_temp'], label = 'temp')
    trend = slideMedian(np.array(data['mean_temp']), 55)
    plt.plot(trend, label = 'trend')
    plt.legend() 

    # 3 parts: trend, regular coleb, ostatki
    # kandell says if ostatki is random
    checkKandell(trend, np.array(data['mean_temp']))
    #plt.show()
    plt.figure()
    # regular coleb with fft
    f = np.fft.fft(data['mean_temp'])
    ff = f.real * f.real + f.imag * f.imag
    sz = len(ff)
    ff = ff[0:100]
    plt.plot(ff)
    plt.grid()
    for delta in [1, 8]:
        ff = ff[delta:100]
        print(sz / ((ff.argmax() + delta)), "days")  

    # calculate herst
    # Evaluate Hurst equation
    H, c, data = compute_Hc(trend, kind='random_walk', simplified=False)

    # Plot
    f, ax = plt.subplots()
    ax.plot(data[0], c*data[0]**H, color="deepskyblue")
    ax.scatter(data[0], data[1], color="purple")
    print(H)
    plt.show()
