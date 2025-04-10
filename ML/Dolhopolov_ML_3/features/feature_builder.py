from abc import ABC, abstractmethod

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

from features.split_data_type import SplitDataType


class FeatureBuilder(ABC):
    def __init__(self, 
                 df: pd.DataFrame, 
                 split_data_type: SplitDataType, 
                 normalization_scaler: MinMaxScaler, 
                 target_column: str
                 ):
        self.df = df
        self.split_data_type = split_data_type
        self.normalization_scaler = normalization_scaler
        if target_column in self.df.columns:
            self.__target = self.df[target_column]
        self.df.drop(columns=[target_column], inplace=True, errors='ignore')
        
    def get_type(self) -> str:
        return self.split_data_type.value
    
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
    
    def _normalize_from_0_to_1(self) -> 'FeatureBuilder':
        columns_to_scale = [col for col in self.df.columns if self.df[col].max() > 1]

        if self.split_data_type == SplitDataType.TRAIN:
            self.df[columns_to_scale] = self.normalization_scaler.fit_transform(self.df[columns_to_scale])
        else:
            self.df[columns_to_scale] = self.normalization_scaler.transform(self.df[columns_to_scale])

        return self
    
    #отримую список де тільки об'єкти з високою кореляцією та сумарно отсортировані, спочатку дропаю з дуже великою
    #сумою
    def get_high_correlation_features(self, theresold: float = 0.75) -> pd.Series:
        """Return sum of high corelation (>= theresold) features only where correlation > 0 in descending order.
    
        Returns:
            pd.Series: Features with sum of high corelation, where low correlation are thrown out
        """
        corr = self.df.corr()
        corr_filter = corr[(abs(corr) >= theresold) & (abs(corr) != 1)]
        corr_df = corr_filter.dropna(how='all').dropna(axis=1, how='all')
        return corr_df.abs().sum().sort_values(ascending=False)

    def get_VIF_correlation_features(self) -> pd.Series:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.df)
        vif_data = pd.DataFrame()
        vif_data["feature"] = self.df.columns
        vif_data["VIF"] = [variance_inflation_factor(X_scaled, i) for i in range(X_scaled.shape[1])]

        vif_data = vif_data.sort_values(by="VIF", ascending=False)
        return vif_data