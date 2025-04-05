import pandas as pd
from pandas.api.types import is_numeric_dtype
import numpy as np

from features.feature_builder import FeatureBuilder


class HousePriceFeatureBuilder(FeatureBuilder):
    def __init__(self, df: pd.DataFrame):
        super().__init__(df)

    def build_by_templete(self, new_df: pd.DataFrame = None) -> pd.DataFrame:
        if new_df:
            self.df = new_df
        return self.drop_id() \
                   .drop_duplicates() \
                   .devide_miscfeature() \
                   .fill_null_columns() \
                   .encode() \
                   .add_is_alley_access() \
                   .add_is_basement() \
                   .add_is_fireplace() \
                   .add_is_garage() \
                   .add_is_pool() \
                   .add_is_fence() \
                   .normalize() \
                   .df

    def drop_id(self) -> 'HousePriceFeatureBuilder':
        self.df.drop(columns=['Id'], inplace=True)
        return self

    def drop_duplicates(self) -> 'HousePriceFeatureBuilder':
        self.df.drop_duplicates(inplace=True)
        return self


    ## тут 0 де-яких значень, але може потім датафрейм буде оновлятись та будуть значення
    def devide_miscfeature(self) -> 'HousePriceFeatureBuilder':
        self.df['has_any_additional_feature'] = np.where(self.df['MiscFeature'].isnull(), 0, 1)
        self.df['has_elev'] = np.where(self.df['MiscFeature'] == 'Elev', 1, 0) 
        self.df['has_second_garage'] = np.where(self.df['MiscFeature'] == 'Gar2', 1, 0) 
        self.df['has_shed'] = np.where(self.df['MiscFeature'] == 'Shed', 1, 0) 
        self.df['has_tennis_court'] = np.where(self.df['MiscFeature'] == 'TenC', 1, 0) 
        self.df['has_something'] = np.where(self.df['MiscFeature'] == 'Othr', 1, 0)
        self.df.drop(columns=['MiscFeature'], inplace=True)        
        return self
    
    def clear_most_null_columns(self) -> 'HousePriceFeatureBuilder':
        self.df.drop(columns=['MiscFeature'], inplace=True)
        return self

    def fill_null_columns(self) -> 'HousePriceFeatureBuilder':
        self.df['Alley'] = self.df['Alley'].fillna('NoAlley')
        self.df['PoolQC'] = self.df['PoolQC'].fillna('NoPool')
        self.df['Fence'] = self.df['Fence'].fillna('NoFence')
        self.df['MasVnrType'] = self.df['MasVnrType'].fillna('NoMasonry')
        self.df['FireplaceQu'] = self.df['FireplaceQu'].fillna('NoFireplace')
        self.df['GarageQual'] = self.df['GarageQual'].fillna('NoGarage')
        self.df['GarageFinish'] = self.df['GarageFinish'].fillna('NoGarage')
        self.df['GarageType'] = self.df['GarageType'].fillna('NoGarage')
        self.df['GarageCond'] = self.df['GarageCond'].fillna('NoGarage')
        
        # заповню медіаною, щоб не впливало на підрахунки
        median_garageyrblt = self.df['GarageYrBlt'].median()
        self.df['GarageYrBlt'] = self.df['GarageYrBlt'].fillna(median_garageyrblt)

        self.df['BsmtExposure'] = self.df['BsmtExposure'].where(self.df['BsmtExposure'] != 'No', 'NoExposure')
        self.df['BsmtExposure'] = self.df['BsmtExposure'].fillna('NoBasement')
        self.df['BsmtFinType1'] = self.df['BsmtFinType1'].fillna('NoBasement')
        self.df['BsmtFinType2'] = self.df['BsmtFinType2'].fillna('NoBasement')
        self.df['BsmtQual'] = self.df['BsmtQual'].fillna('NoBasement')
        self.df['BsmtCond'] = self.df['BsmtCond'].fillna('NoBasement')
        
        # одне значення та не дуже важливе
        median_masvnrarea = pd.to_numeric(self.df['MasVnrArea'], errors='coerce').mean()
        self.df['MasVnrArea'] = self.df['MasVnrArea'].fillna(median_masvnrarea)
        
        self.df['LotFrontage_missing'] = self.df['LotFrontage'].isnull().astype('int8')
        self.df.drop(columns=['LotFrontage'], inplace=True)
        
        # одне за найбільшим
        self.df['Electrical'] = self.df['Electrical'].fillna('SBrkr')
        return self
    
    def encode(self) -> 'HousePriceFeatureBuilder':
        columns = ['Alley', 'PoolQC', 'Fence', 'MSSubClass', 'MSZoning', 'Street', 'LotShape', 
                   'LandContour', 'Utilities', 'LotConfig',
                   'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle',
                   'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 
                   'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure',
                   'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical',
                   'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish',
                   'GarageQual', 'GarageCond', 'PavedDrive', 'SaleType', 'SaleCondition']
        
        
        for column in columns:
            self._hot_encode(column, drop_first=True)
        
        return self
    
    def add_is_alley_access(self) -> 'HousePriceFeatureBuilder':
        self.df['IsAlleyAccess'] = np.where(self.df['Alley'] == 'NoAlley', 0, 1) 
        return self
    
    def add_is_basement(self) -> 'HousePriceFeatureBuilder':
        self.df['IsBasement'] = np.where(self.df['BsmtQual'] == 'NoBasement', 0, 1) 
        return self
    
    def add_is_fireplace(self) -> 'HousePriceFeatureBuilder':
        self.df['IsFireplace'] = np.where(self.df['FireplaceQu'] == 'NoFireplace', 0, 1) 
        return self
    
    def add_is_garage(self) -> 'HousePriceFeatureBuilder':
        self.df['IsGarage'] = np.where(self.df['GarageType'] == 'NoGarage', 0, 1) 
        return self
    
    def add_is_pool(self) -> 'HousePriceFeatureBuilder':
        self.df['IsPool'] = np.where(self.df['PoolQC'] == 'NoPool', 0, 1) 
        return self
    
    def add_is_fence(self) -> 'HousePriceFeatureBuilder':
        self.df['IsFence'] = np.where(self.df['Fence'] == 'NoFence', 0, 1) 
        return self

    def normalize(self) -> 'HousePriceFeatureBuilder':
        for column in self.df.columns:
            if is_numeric_dtype(self.df[column]):
                max_value = self.df[column].max()
                if max_value > 1:
                    self._normalize_from_0_to_1(column)
        return self
