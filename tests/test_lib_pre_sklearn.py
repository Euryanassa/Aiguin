import unittest
from pandas.testing import assert_frame_equal
import pandas as pd
import numpy as np

import sys
import os
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(f'{ROOT_DIR}/../scripts')
from lib_pre_sklearn import ClsLibPreprocessing


class TestClsLibPreprocessing(unittest.TestCase):

    def test_init(self):
        data = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        cls_lib_preprocessing = ClsLibPreprocessing(data)

        assert_frame_equal(cls_lib_preprocessing.data, data)

    def test_scale_data(self):
        data = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        cls_lib_preprocessing = ClsLibPreprocessing(data)
        cls_lib_preprocessing.scale_data()

        self.assertEqual(cls_lib_preprocessing.data['a'].dtype, np.float64)
        self.assertEqual(cls_lib_preprocessing.data['b'].dtype, np.float64)

    def test_categorical_columns(self):
        data = pd.DataFrame({'column_a': ['a', 'b', 'c'], 'column_b': [1, 2, 3]})
        
        cls_lib_preprocessing = ClsLibPreprocessing(data)
        cls_lib_preprocessing.encode_categorical_columns()

        self.assertTrue(cls_lib_preprocessing.data['column_a'].dtype == 'category')
        self.assertFalse(cls_lib_preprocessing.data['column_b'].dtype == 'category')


    def test_check_dataframe_info(self):
        data = pd.DataFrame({'a': [1, None, 3, 3], 'b': ['a', 'b', 'c', 'c']})
        cls_lib_preprocessing = ClsLibPreprocessing(data)
        info_df = cls_lib_preprocessing.check_dataframe_info()

        self.assertEqual(info_df['Empty Values'].sum(), 1)
 

    def test_drop_duplicates(self):
        data = pd.DataFrame({'a': [1, 1, 2, 3], 'b': ['a', 'a', 'b', 'c']})
        cls_lib_preprocessing = ClsLibPreprocessing(data)
        df_without_duplicates = cls_lib_preprocessing.drop_duplicates()

        self.assertEqual(df_without_duplicates.shape, (3, 2))

    def test_week_month_feature_columns(self):
        df = pd.DataFrame({'date': ['2023-03-08', '2023-03-09', '2023-03-10']})
        df['date']= pd.to_datetime(df['date'])
        cls_lib_preprocessing = ClsLibPreprocessing(df)
        cls_lib_preprocessing.week_month_feature_columns()
        

        self.assertTrue('day_of_week' in df.columns)
        self.assertTrue('month' in df.columns)
