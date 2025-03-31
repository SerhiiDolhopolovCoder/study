from abc import ABC, abstractmethod

import pandas as pd


# Реалізація патерну проектування Builder
class FeatureBuilder(ABC):
    """Feature builder for dataframes.
    Use property df to get a current dataframe.

    Design pattern builder
    """

    def __init__(self, df: pd.DataFrame):
        self.__df = df
        self.df = self.df.drop_duplicates()

    @property
    def df(self) -> pd.DataFrame:
        """Returns a current dataframe.

        Returns:
            pd.DataFrame: return a dataframe with features
        """
        return self.__df

    @df.setter
    def df(self, df: pd.DataFrame):
        self.__df = df

    @abstractmethod
    def build_by_template(self, df: pd.DataFrame) -> pd.DataFrame:
        """Implemented as build from a clear dataframe via chain template

        Returns:
            pd.DataFrame: _description_
        """
        pass

    def _hot_encode(self, column: str, dtype: str = 'int8', drop_first: bool = False) -> 'FeatureBuilder':
        """Encode column by hot-encoding:

        Args:
            column (str): a column for hot-encoding
            dtype (str, optional): what a data type you want.
            drop_first (bool, optional): True if want to solve dummy trap.

        Examples:
            hot_encode('Embarked' (Values=[A, B, C])) -> pd.DataFrame where ('Embarked' -> ('Embarked_A','Embarked_B', 'Embarked_C'))
            hot_encode('Embarked' (Values=[A, B, C]), drop_first=True) -> pd.DataFrame where ('Embarked' -> ('Embarked_B', 'Embarked_C'))


        Check is column in df included.
        """
        self._check_columns_in_df(column)

        self.df = pd.get_dummies(
            self.df, columns=[column], prefix=column, dtype=dtype, drop_first=drop_first)

        return self

    def _add_bins(self, column: str, n_bins: int, dtype: str = 'int8', drop_old: bool = False) -> 'FeatureBuilder':
        """Create age_bins column from a column.

        Args:
            column (str): column name
            n_bins (int): count of bins
            drop_old (bool, optional): remove old column

        Examples:
            add_bins('Age', n_bins=20, drop_old=True)

        Check is column in df included.
        """
        columns_bins = f'{column}_bins'

        self._check_columns_in_df(column)

        self.df[columns_bins] = pd.qcut(
            self.df[column], n_bins, duplicates='drop').cat.codes.astype(dtype)
        if drop_old:
            self.df.drop(columns=[column])
        return self

    def add_missing(self) -> 'FeatureBuilder':
        """Create missing features"""
        def has_nulls(column: str):
            return self.df[column].isnull().sum() > 0

        for column in [column for column in self.df.columns if has_nulls(column)]:
            self.df[f'{column}_missing'] = self.df[column].isnull() \
                                                          .astype('int8')
        return self

    def _normalization_0_to_1(self, column: str) -> 'FeatureBuilder':
        """Normalize int | float column to [0, 1]

        Args:
            column (str): column name
        """
        self._check_columns_in_df(column)
        if not pd.api.types.is_numeric_dtype(self.df[column]):
            raise TypeError('The column must be an integer or float.\n',
                            f'The current type: {self.df[column].dtype}')

        min_value = self.df[column].min()
        max_value = self.df[column].max()
        self.df[column] = self.df[column].apply(
            lambda x: (x - min_value) / (max_value - min_value))

        return self

    def _drop_irrelevant(self, columns: list[str]) -> 'FeatureBuilder':
        self.df.drop(columns=columns, inplace=True, errors='ignore')
        return self

    def _check_columns_in_df(self, *columns: str | tuple[str]) -> bool:
        """Check if columns is in the dataframe.

        Args:
            columns (str | tuple[str]): Columns name

        Returns:
            bool: True if column is in the dataframe, Raise KeyException otherwise
        """
        if all(column in self.df.columns for column in columns):
            return True
        else:
            raise KeyError(
                f'Columns {', '.join(columns)} not found in dataframe. \n The available columns are: {', '.join(self.df.columns)}')
