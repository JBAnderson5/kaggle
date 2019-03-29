#file loads data from the csv files

import pandas as pd
import numpy as np

verbose = True

#determines whether we should practice by only using the training data or produce real results
practice = True
def practiceTraining():
    global practice
    practice = True
def realTest(): #TODO: implement this after getting practice to run correctly
    global practice
    practice = False


trainingData = np.empty(0)
testData = np.empty(0)
realValues = np.empty(0)

def loadData():
    loadTrainingData()

    if(practice):
        splitTrainingData()
    else:
        loadTestingData()

    return trainingData.copy(),testData.copy()

#loads training data from csv file
def loadTrainingData():
    global trainingData
    fileName = "train.csv"

    trainingData = pd.read_csv(fileName,delimiter = ",").to_numpy()

    if(verbose):
        print(trainingData[0,:])



#splits training data into training and testing for practice
def splitTrainingData():
    global trainingData, testData, realValues
    #shuffle array and split part of it into testing
    np.random.shuffle(trainingData)


    size = trainingData.shape[0]
    testSize = int(0.3 * size)

    testData = trainingData[0:testSize,:]
    realValues = testData[:,0:2]
    testData = np.delete(testData,1,axis=1)
    trainingData = trainingData[testSize:size,:]



    if(verbose):
        print(trainingData)
        print(trainingData.shape)
        print(testData)
        print(realValues)

#loads testing data from csv file
def loadTestingData():
    global testData
    fileName = "test.csv"

    testData = pd.read_csv(fileName,delimiter=",").to_numpy()

    if(verbose):
        print(testData)

def evalResults(predictionMatrix):
    if(practice):
        testResults(predictionMatrix)
    else:
        saveResults(predictionMatix)

def saveResults(predictionMatrix):

    fileName="submission.csv"
    pd.DataFrame(predictionMatrix).to_csv(fileName,header="PassengerID,Survived")

def testResults(predictionMatrix):
    global realValues
    correct = 0
    total = predictionMatrix.shape[0]

    realValues = np.sort(realValues,axis=0)
    predictionMatrix = np.sort(predictionMatrix,axis=0)
    if(verbose):
        print(realValues)
        print(predictionMatrix)
    for i in range(total):
        if(realValues[i,1] == predictionMatrix[i,1]):
            correct+=1

    print(correct/total)


#realTest()
loadData()
arr = np.empty(realValues.shape)
evalResults(arr)
