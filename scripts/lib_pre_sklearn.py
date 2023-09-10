
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
        This function converts categorical columns in a Pandas DataFrame into categories.

        Args:
            df: The DataFrame to convert the columns from.

        Returns:
            A DataFrame with the categorical columns converted to categories.
        """

        # Get the categorical columns
        categorical_columns = self.data.select_dtypes(include=['object']).columns

        # Convert the categorical columns to categories
        #for column in categorical_columns:
        #    self.data[column] = pd.Categorical(self.data[column])

        for column in categorical_columns:
            if column == 'current_date':
                continue
            # Get the unique values in the column
            unique_values = self.data[column].unique()

            # Create a dictionary to map the unique values to labels
            label_map = {value: i for i, value in enumerate(unique_values)}

            # Label the values in the column
            self.data[column] = self.data[column].map(label_map)



   
    def check_dataframe_info(self):
        """
        This function checks a DataFrame for empty values and duplicate rows, and creates an information table out of it.

        Args:
            df: The DataFrame to check.

        Returns:
            A DataFrame with information about the empty values and duplicate rows.
        """

        # Check for empty values
        empty_values_df = self.data.isnull().sum().to_frame(name='Empty Values')
        empty_values_df.index.name = 'Column'

        return empty_values_df

    def drop_duplicates(self):
        """
        This function drops duplicates from a DataFrame.

        Args:
            df: The DataFrame to drop duplicates from.

        Returns:
            A DataFrame without duplicates.
        """

        df_without_duplicates = self.data.drop_duplicates()

        return df_without_duplicates

    def week_month_feature_columns(self, date_column = 'date'):
        """
        This function creates feature columns of day of week and month of the year from a Pandas DataFrame.

        Args:
            df: The Pandas DataFrame to create the feature columns from.

        Returns:
            A DataFrame with the new feature columns.
        """

        # Get the day of week column
        day_of_week_column = self.data[date_column].dt.weekday

        # Create a categorical day of week column
        self.data['day_of_week'] = pd.Categorical(day_of_week_column)

        # Get the month column
        month_column = self.data[date_column].dt.month

        # Create a categorical month column
        self.data['month'] = pd.Categorical(month_column)

    def is_date_column(df, column_name):
        """
        This function checks if a column in a Pandas DataFrame is a date.

        Args:
            df: The Pandas DataFrame to check the column from.
            column_name: The name of the column to check.

        Returns:
            True if the column is a date, False otherwise.
        """

        # Check if the column is of type datetime
        if pd.api.types.is_datetime_dtype(df[column_name]):
            return True
        else:
            return False
            