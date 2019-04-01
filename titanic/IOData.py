#file loads data from the csv files

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

verbose = True

#determines whether we should practice by only using the training data or produce real results
practice = True
def practiceTraining():
    global practice
    practice = True
def realTest(): #TODO: implement this after getting practice to run correctly
    global practice
    practice = False


def loadData():
    dataframe = loadTrainingData()

    if(practice):
        train, test, val = splitTrainingData(dataframe)
    else:
        train, val = train_test_split(dataframe, test_size = 0.2)
        test = loadTestingData()

    return train, test, val

#loads training data from csv file
def loadTrainingData():
    fileName = "train.csv"

    dataframe = pd.read_csv(fileName,delimiter = ",")
    dataframe.head()

    dataframe = preprocessData(dataframe)

    if(verbose):
        print(dataframe)

    return dataframe


def preprocessData(dataframe):
    print(dataframe.count())

    var_encodes = ["Ticket","Name","Sex","Cabin","Embarked"]
    labelEncoder = preprocessing.LabelEncoder()
    for i in var_encodes:
        dataframe[i] = labelEncoder.fit_transform(dataframe[i].fillna("0"))

    print(dataframe.count())
    return dataframe

#splits training data into training and testing for practice
def splitTrainingData(dataframe):

    train, test = train_test_split(dataframe, test_size = 0.2)
    train, val = train_test_split(train, test_size = 0.2)

    if(verbose):
        print(len(train),"train examples")
        print(len(val), "validation examples")
        print(len(test), "test examples")

    return train, test, val

#loads testing data from csv file
def loadTestingData():
    global testData
    fileName = "test.csv"

    testDataFrame = pd.read_csv(fileName,delimiter=",")

    testDataFrame = preprocessData(testDataFrame)

    if(verbose):
        print(testDataFrame)

    return testDataFrame

#TODO: rework the eval results, save results and test results later
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
#loadData()
#arr = np.empty(realValues.shape)
#evalResults(arr)
