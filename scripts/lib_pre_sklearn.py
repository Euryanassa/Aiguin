
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np

class ClsLibPreprocessing:
    def __init__(self, data):
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input must be a DataFrame")
        self.data = data


    def scale_data(self):
        """
        This function scales the numeric columns in a Pandas DataFrame.

        Args:
            df: The DataFrame to scale.

        Returns:
            A scaled DataFrame.

        Usage:
            df_pre = pre.ClsLibPreprocessing(df)
            df_pre.scale_data()
            print(df_pre.data)
        """
        # Check for numeric columns
        numeric_columns = self.data.select_dtypes(include=['number']).columns

        # Initialize the scalers
        minmax_scaler = MinMaxScaler()
        standard_scaler = StandardScaler()

        # Scale the numeric columns
        for column in numeric_columns:
            if self.data[column].min() == self.data[column].max():
                self.data[column] = standard_scaler.fit_transform(self.data[column].values.reshape(-1, 1))
            else:
                self.data[column] = minmax_scaler.fit_transform(self.data[column].values.reshape(-1, 1))

    def encode_categorical_columns(self):
        """
        This function encodes categorical columns in a Pandas DataFrame.

        Args:
        df: The DataFrame to encode.

        Returns:
        A encoded DataFrame.
        """

        # Check for categorical columns
        categorical_columns = self.data.select_dtypes(include=['object']).columns

        # Encode the categorical columns
        for column in categorical_columns:
            encoder = OneHotEncoder()
            self.data[column] = encoder.fit_transform(self.data[column].values.reshape(-1, 1)).toarray()

   
