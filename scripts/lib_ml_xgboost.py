from xgboost import XGBClassifier
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV



class ClsLibMlXGBoostClassifier():
    def __init__(self, data):
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input must be a DataFrame")
        self.data = data
        self.test_size = 0.3
        self.target_column = 'target_column'
        self.train_df, self.test_df = train_test_split(self.data, test_size = self.test_size)
        self.train_df_X = self.train_df.loc[:, self.train_df.columns != 'target_column']
        self.train_df_y = self.train_df[self.target_column]
        self.test_df_X = self.test_df.loc[:, self.test_df.columns != 'target_column']
        self.test_df_y = self.test_df[self.target_column]
        self.eval_metric = 'logloss'
        self.eval_results = None
        self.early_stopping_rounds = 10
        self.params_grid = {
                'n_estimators': [10, 100, 1000],
                'learning_rate': [0.01, 0.1, 1.0],
                'max_depth': [3, 5, 10],
                }
        self.best_params = {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 5,
                }

    def grid_search_xgboost_parameters(self):
        """
        This function performs a grid search over the XGBoost parameters.
        """
        
        # Various hyper-parameters to tune
        xgb1 = XGBClassifier()

        grid_search_xgb = GridSearchCV(xgb1,
                                self.params_grid,
                                cv = 2,
                                n_jobs = 5,
                                verbose = False)

        grid_search_xgb.fit(self.train_df_X,
                            self.train_df_y)

        print(grid_search_xgb.best_score_)
        print(grid_search_xgb.best_params_)

        self.best_model = grid_search_xgb.best_estimator_


    def train_xgboost_model(self):
        """
        This function trains an XGBoost model with all the parameters.
        """

        # Create the XGBoost model
        model = XGBClassifier(**self.best_params)

        # Train the model on the train set
        model.fit(self.train_df_X,
                  self.train_df_y,
                  eval_set = [(self.test_df_X, self.test_df_y)],
                  eval_metric = self.eval_metric,
                  early_stopping_rounds = self.early_stopping_rounds,
                  )

        self.eval_results = model.evals_result()

