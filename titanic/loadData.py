#file loads data from the csv files

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

def loadData():
    loadTrainingData()

    if(practice):
        splitTrainingData()
    else:
        loadTestingData()

#loads training data from csv file
def loadTrainingData():
    global trainingData
    fileName = "train.csv"
    dataType ="int16,int8,int8,U,U,int8,int8,int8,U,float16,U,U"

    trainingData = np.genfromtxt(fileName, delimiter = ",",names = True,dtype = dataType)

    print(trainingData)



#splits training data into training and testing for practice
def splitTrainingData():
    global trainingData, testData
    #shuffle array and split part of it into testing
    np.random.shuffle(trainingData)


    size = trainingData.shape[0]
    testSize = int(0.3 * size)

    testData = trainingData[0:testSize,:]
    trainingData = trainingData[testSize:size,:]

    if(verbose):
        print(trainingData)
        print(trainingData.shape)
        print(testData)

#loads testing data from csv file
def loadTestingData():
    print("not implemented yet")



loadData()
