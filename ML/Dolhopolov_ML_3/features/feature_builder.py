from abc import ABC, abstractmethod

import pandas as pd
from sklearn.preprocessing import MinMaxScaler


class FeatureBuilder(ABC):
    def __init__(self, df: pd.DataFrame, target_column: str):
        self.df = df
        if target_column in self.df.columns:
            self.__target = self.df[target_column]
        self.df.drop(columns=[target_column], inplace=True, errors='ignore')
    
    @property
    def target(self) -> pd.Series:
        return self.__target
            
    @abstractmethod
    def build_by_templete(self, new_df=None) -> pd.DataFrame:
        """
        Build features by template. If new_df is provided, use it instead of self.df.
        Args:
            new_df: pd.DataFrame
        """
        pass
        
    def _hot_encode(self, column: str, categories: list[str], dtype: str = 'int8', drop_first: bool = False) -> 'FeatureBuilder':
        """Encode column by hot-encoding:

        Args:
            column (str): a column for hot-encoding
            dtype (str, optional): what a data type you want.
            drop_first (bool, optional): True if want to solve dummy trap.

        Examples:
            hot_encode('Embarked' (Values=[A, B, C])) -> pd.DataFrame where ('Embarked' -> ('Embarked_A','Embarked_B', 'Embarked_C'))
            hot_encode('Embarked' (Values=[A, B, C]), drop_first=True) -> pd.DataFrame where ('Embarked' -> ('Embarked_B', 'Embarked_C'))
        """
        self.df[column] = pd.Categorical(self.df[column], categories=categories)
        self.df = pd.get_dummies(
            self.df, columns=[column], prefix=column, dtype=dtype, drop_first=drop_first)
        return self
    
    def _label_encode(self, column: str, encode_map: dict[str, int]) -> 'FeatureBuilder':
        """Encode column by label-encoding.
        !!! If value in not in encode_map, it will be replaced Null.

        Args:
            column (str): a column for label-encoding
            encode_map (dict[str, int]): map for encoding with type - mark
        """
        new_column = f'{column}_encoded'
        self.df[new_column] = self.df[column].map(encode_map)
        self.df.fillna({new_column: self._no_info}, inplace=True)
        self.df.drop(columns=[column], inplace=True)
        return self
    
    def _normalize_from_0_to_1(self, column: str) -> 'FeatureBuilder':
        """Normalize positive value of a column from 0 to 1.

        Args:
            column (str): a column for normalization
        """
        ## Номралізує тільки там де значення >= 0, від'ємні значення не нормалізую, щоб вони не рахувались як
        ## нормальне значення, але від'ємне
        positive_mask = self.df[column] >= 0
        self.df[column] = self.df[column].astype(float) 
        self.df.loc[positive_mask, column] = MinMaxScaler().fit_transform(self.df.loc[positive_mask, [column]]) 
        return self
    
    #отримую список де тільки об'єкти з високою кореляцією та сумарно отсортировані, спочатку дропаю з дуже великою
    #сумою, потім по-черзі з меншою
    def get_high_corelation_features(self, theresold: float = 0.75) -> pd.Series:
        """Return sum of high corelation (>= theresold) features only where correlation > 0 in descending order.
    
        Returns:
            pd.Series: Features with sum of high corelation, where low correlation are thrown out
        """
        corr = self.df.corr()
        corr_filter = corr[(abs(corr) >= theresold) & (abs(corr) != 1)]
        corr_df = corr_filter.dropna(how='all').dropna(axis=1, how='all')
        return corr_df.abs().sum().sort_values(ascending=False)
    
    @property
    def _no_info(self) -> float:
        return -1
    
    @property
    def _no_item(self) -> float:
        return -2
    
    