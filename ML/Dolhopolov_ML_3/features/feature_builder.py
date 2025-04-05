from abc import ABC, abstractmethod

import pandas as pd
from sklearn.preprocessing import MinMaxScaler


class FeatureBuilder(ABC):
    def __init__(self, df):
        self.df = df
        
    @abstractmethod
    def build_by_templete(self, new_df=None) -> pd.DataFrame:
        """
        Build features by template. If new_df is provided, use it instead of self.df.
        Args:
            new_df: pd.DataFrame
        """
        pass
    
    def get_features_target(self, target: str) -> tuple[pd.DataFrame, pd.Series]:
        """Devide dataframe on features and target.
        Returns:
            tuple[pd.DataFrame, pd.Series]: (Features, Target)

        """
        return (self.df.drop(columns=[target]), self.df[target])
    
    def _hot_encode(self, column: str, dtype: str = 'int8', drop_first: bool = False) -> 'FeatureBuilder':
        """Encode column by hot-encoding:

        Args:
            column (str): a column for hot-encoding
            dtype (str, optional): what a data type you want.
            drop_first (bool, optional): True if want to solve dummy trap.

        Examples:
            hot_encode('Embarked' (Values=[A, B, C])) -> pd.DataFrame where ('Embarked' -> ('Embarked_A','Embarked_B', 'Embarked_C'))
            hot_encode('Embarked' (Values=[A, B, C]), drop_first=True) -> pd.DataFrame where ('Embarked' -> ('Embarked_B', 'Embarked_C'))

        """

        self.df = pd.get_dummies(
            self.df, columns=[column], prefix=column, dtype=dtype, drop_first=drop_first)

        return self
    
    def _normalize_from_0_to_1(self, column: str) -> 'FeatureBuilder':
        """Normalize column from 0 to 1.

        Args:
            column (str): a column for normalization

        """
        self.df[column] = MinMaxScaler().fit_transform(self.df[[column]])

        return self