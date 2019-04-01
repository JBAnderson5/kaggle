
import IOData

import tensorflow as tf
import pandas as pd
import numpy as np

from tensorflow import feature_column
from tensorflow.keras import layers


def __main__():
    IOData.practiceTraining()
    #IOData.realTest()
    train, test, val = IOData.loadData()

    featureLayer = buildFeatures()

    model = buildModel(train,val,featureLayer)

    if IOData.practice:
        testModel(model,test)
    else:
        saveTestResults(model,test)


def buildModel(train,val,featureLayer):
    train_ds = df_to_dataset(train)
    val_ds = df_to_dataset(val, shuffle=False)
    model = tf.keras.Sequential([featureLayer, layers.Dense(128, activation="relu"), layers.Dense(128, activation="relu"), layers.Dense(1, activation="sigmoid")])
    model.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])
    model.fit(train_ds,validation_data = val_ds, epochs = 5)

    return model

def testModel(model,test):
    test_ds = df_to_dataset(test, shuffle=False)
    loss, accuracy = model.evaluate(test_ds)
    print("accuracy", accuracy)

def saveTestResults(model,test):
    test_ds = tf.data.Dataset.from_tensor_slices(dict(test))
    test_ds = ds.batch(32)

    #TODO evaluate and save results
    return test

def df_to_dataset(dataframe, shuffle=True,batch_size=32):
    dataframe = dataframe.copy()
    labels = dataframe.pop("Survived")

    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    if shuffle:
        ds = ds.shuffle(buffer_size = len(dataframe))

    ds = ds.batch(batch_size)
    return ds

def buildFeatures():
    feature_columns = []

    #TODO: which features should we use
    for header in ["PassengerId", "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]:
        feature_columns.append(feature_column.numeric_column(header))

    #TODO: what features should we build

    #TODO: how should we represent name, ticket, cabin, embarked


    feature_layer = tf.keras.layers.DenseFeatures(feature_columns)

    return feature_layer

__main__()
