import numpy as np
import scipy as sc
import matplotlib.pyplot as plt

# globals
N = 1000
h = 0.05

# Function to generate sample
# Return sample from task 1
def generateSample():
    return np.array([np.sqrt(k * h) + np.random.normal() for k in range(N)])

# Function to generate model
def generateModel():
    return np.array([np.sqrt(k * h) for k in range(N)])

# Function to calculate slide mean points
# In: sample, slide period
# Out: array of mean points
def slideMean(sample, m):
    res = []
    for i in range(sample.size):
        # check for edge cases
        if i < m:
            res.append(sum([sample[j] for j in range(0, 2 * i + 2)]) / (2.0 * i + 1))
        elif i >= sample.size - m - 1 :
            res.append(sum([sample[j] for j in range(i - (sample.size - i), sample.size)]) / len(range(i - (sample.size - i), sample.size)))
        # default case
        else:
            res.append(sum(sample[j] for j in range(i - m, i + m + 1)) / (2.0 * m + 1))
    return res

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

    # Task 2
    print("Task 2: \n")
    model = generateModel()
    slide10 = slideMean(sample, 10)
    print(slide10)
    slide25 = slideMean(sample, 25)
    slide55 = slideMean(sample, 55)
    plt.figure()
    plt.title("Task 2")
    plt.plot(sample, 'o', color = 'black', label = "sample")
    plt.plot(model, label = "model")
    plt.plot(slide10, label = "Slide mean, m = 10")
    plt.plot(slide25, label = "Slide mean, m = 25")
    plt.plot(slide55, label = "Slide mean, m = 55")
    plt.legend()

    # Task 3
    print("Task 3: \n")
    slideMed10 = slideMedian(sample, 10)
    print(slide10)
    slideMed25 = slideMedian(sample, 25)
    slideMed55 = slideMedian(sample, 55)
    plt.figure()
    plt.title("Task 3")
    plt.plot(sample, 'o', color = 'black', label = "sample")
    plt.plot(model, label = "model")
    plt.plot(slideMed10, label = "Slide median, m = 10")
    plt.plot(slideMed25, label = "Slide median, m = 25")
    plt.plot(slideMed55, label = "Slide median, m = 55")
    plt.legend()

    # Task 4
    print("Kandell for slide mean 10\n")
    checkKandell(slide10, sample)
    print("Kandell for slide mean 25\n")
    checkKandell(slide25, sample)
    print("Kandell for slide mean 55")
    checkKandell(slide55, sample)
    print("Kandell for slide median 10\n")
    checkKandell(slideMed10, sample)
    print("Kandell for slide median 25\n")
    checkKandell(slideMed25, sample)
    print("Kandell for slide median 55")
    checkKandell(slideMed55, sample)

    plt.show()