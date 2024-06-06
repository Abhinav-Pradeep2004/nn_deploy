import numpy as np
import os

from src.config import config

class preprocess_data:
    
    def fit(self,X,y=None):
        #all those used in data preprocessing or used to calculate about data calculations over data eg pca varience in this, coss tabulation in this
        self.num_rows = X.shape[0]
        
        if len(X.shape) == 1:
            self.num_input_features = 1
        else:
            self.num_input_features = X.shape[1]
        
        if len(y.shape) == 1:
            self.num_target_feature_dim = 1
        else:
            self.num_target_feature_dim = y.shape[1]
        
    def transform(self,X=None,y=None):
        # this function will change the input data that will be finally take in the data
        self.X = np.array(X).reshape(self.num_rows,self.num_input_features) #reshapes to a matrix
        self.Y = np.array(y).reshape(self.num_rows,) 
        #Y allowed because self se bind karke kuch bhi likh sakthe hai
        
        return self.X, self.Y
        