from array import array
import re
import numpy as np
import pandas as pa
import sklearn
import joblib

target_word = "target"
target_regex = "^(target[0-9]*)$"

class DataHandler:
    """
    Get data from sources
    """


    def __init__(self):
        self.csvfile1 = None
        self.csvfile2 = None
        self.gouped_data = None

class FeatureRecipe:
    """
    Feature processing class
    """
    def __init__(self, data: pa.DataFrame):
        self.data = data
        self.continuous = None
        self.categorical = None
        self.discrete = None
        self.datetime = None

class FeatureExtractor:
    """
    Feature Extractor class
    """
    def __init__(self, data: pa.DataFrame, to_drop: list):
        data = data.drop(array(to_drop), axis=1, inplace=True)

        targets = data.filter(target_regex)
        data.loc[:, ~data.columns.str.startswith('Test')]

        return train_test_split()


    """
        Input : pandas.DataFrame, feature list to drop
        Output : X_train, X_test, y_train, y_test according to sklearn.model_selection.train_test_split
    """

class ModelBuilder:

    model = None
    model_path = "decision_rf.joblib"
    """
    Class for train and print results of ml model
    """
    def __init__(self, model_path: str = None, save: bool = None):
        path = self.model_path
        if model_path :
            path = model_path

        self.model = self.load_model(model_path)

    def __repr__(self):
        return self

    def train(self, X, Y):
        self.model.fit(X,Y)

    def predict_test(self, X:[]) -> np.ndarray:
        return self.model.predict(X)

    def predict_from_dump(self, X) -> np.ndarray:
        pass

    def save_model(self, path:str):
        joblib.dump(self.model,path)

    #formely print_accuracy
    def get_accuracy(self):
        self.model.preditc()

    def load_model(self):
        try:
            model = joblib.load(self.model_path)
        except:
            raise Exception("No model found")