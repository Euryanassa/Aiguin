import xgboost as xgb
import numpy as np
import pandas as pd


class ClsLibMlXGBoost():
    def __init__(self, data):
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input must be a DataFrame")
        self.data = data
        self.target_col = None
        self.model = None

