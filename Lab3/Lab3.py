import numpy as np
import scipy as sc
import scipy.stats as st
import matplotlib.pyplot as plt

# globals
N = 1000
h = 0.1

# Function to generate sample
# Return sample from task 1
def generateSample():
    return np.array([0.5 * np.sin(k * h) + np.random.normal() for k in range(N)])

# Function to generate model
def generateModel():
    return np.array([0.5 * np.sin(k * h) for k in range(N)])

# Function to calculate slide exp mean points
# In: sample, alpha
# Out: array of mean points
def slideExp(sample, alpha):
    res = []
    res.append(sample[0])
    for i in range(1, sample.size):
        res.append(alpha * sample[i] + (1 - alpha) * res[i - 1])
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
        print("Randomness")
    elif pSize > pMean + pDisp:
        print("Rapidly oscillating")
    elif pSize < pMean - pDisp:
        print("Positively correlated")

    # Check for mean and normal
    print('mean ', tail.mean())
    print('standart devotion ', st.tstd(tail))
    # 0.05 or 0.005
    print('probability of Normal = ', st.normaltest(tail)[1])


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
    slide001 = slideExp(sample, 0.01)
    slide005 = slideExp(sample, 0.05)
    slide01 = slideExp(sample, 0.1)
    slide03 = slideExp(sample, 0.3)
    plt.figure()
    plt.title("Task 2")
    plt.plot(sample, 'o', color = 'black', label = "sample")
    plt.plot(model, label = "model")
    plt.plot(slide001, label = "Slide exp mean, a = 0.01")
    plt.plot(slide005, label = "Slide exp mean, a = 0.05")
    plt.plot(slide01, label = "Slide exp mean, a = 0.1")
    plt.plot(slide03, label = "Slide exp mean, a = 0.3")
    plt.legend()

    # Task 4
    print("Task 4: \n")
    plt.figure()
    plt.title("Task 4")
    f = np.fft.fft(sample)
    ampSpec = 2 / N * np.abs(f[:len(sample) // 2])
    freqs = np.linspace(0, 1 / (2.0), len(sample) // 2)
    plt.plot(freqs, ampSpec)
    freq = freqs[np.argmax(ampSpec)]
    print(freq)

    # Task 5
    print("Task 5: \n")
    print("Kandell for slide exp mean 0.01")
    checkKandell(slide001, sample)
    print("\nKandell for slide exp mean 0.05")
    checkKandell(slide005, sample)
    print("\nKandell for slide exp mean 0.1")
    checkKandell(slide01, sample)
    print("\nKandell for slide exp mean 0.3")
    checkKandell(slide03, sample)

    plt.show()
