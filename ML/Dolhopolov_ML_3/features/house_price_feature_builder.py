import json

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.preprocessing import MinMaxScaler

from features import FeatureBuilder
from features.split_data_type import SplitDataType


class HousePriceFeatureBuilder(FeatureBuilder):
    def __init__(self, 
                 df: pd.DataFrame, 
                 split_data_type: SplitDataType, 
                 normalization_scaler: MinMaxScaler):
        super().__init__(df, split_data_type, normalization_scaler, target_column='SalePrice')

    def build_by_templete(self, new_df: pd.DataFrame = None) -> pd.DataFrame:
        if new_df:
            self.__init__(new_df)
        return self.drop_id() \
                   .drop_duplicates() \
                   .fill_null_columns() \
                   .encode() \
                   .normalize() \
                   .drop_high_correlation_features() \
                   .df

    def drop_id(self) -> 'HousePriceFeatureBuilder':
        self.df.drop(columns=['Id'], inplace=True, errors='ignore')
        return self

    def drop_duplicates(self) -> 'HousePriceFeatureBuilder':
        self.df.drop_duplicates(inplace=True)
        return self

    def fill_null_columns(self) -> 'HousePriceFeatureBuilder':
        self.df['Alley'] = self.df['Alley'].fillna('NoAlley')
        self.df['MasVnrType'] = self.df['MasVnrType'].fillna('NoMasonry')
        self.df['GarageType'] = self.df['GarageType'].fillna('NoGarage')
        
        # заповню медіаною, щоб не впливало на підрахунки
        median_garageyrblt = self.df['GarageYrBlt'].median()
        self.df['GarageYrBlt'] = self.df['GarageYrBlt'].fillna(median_garageyrblt)

          
        # Цих даних з Null немає в трейні, але є в тестовому, заповнив 0 по контексту
        self.df['BsmtFullBath'] = self.df['BsmtFullBath'].fillna(0)
        self.df['BsmtHalfBath'] = self.df['BsmtHalfBath'].fillna(0)
        self.df['BsmtUnfSF'] = self.df['BsmtUnfSF'].fillna(0)
        self.df['BsmtFinSF1'] = self.df['BsmtFinSF1'].fillna(0)
        self.df['BsmtFinSF2'] = self.df['BsmtFinSF2'].fillna(0)
        self.df['GarageArea'] = self.df['GarageArea'].fillna(0)
        self.df['TotalBsmtSF'] = self.df['TotalBsmtSF'].fillna(0)
        
        # 0, бо немає обліцовки
        self.df['MasVnrArea'] = self.df['MasVnrArea'].fillna(0)
        
        # важлива х-ка, але мало різних значень, перероблю в HasMiscFeature для фільрування чи є щось особливе ще
        self.df['HasAdditionalFeature'] = np.where(self.df['MiscFeature'].isnull(), 1, 0)
        self.df.drop(columns=['MiscFeature'], inplace=True)
        
        # Перетворю в чи є інфа, бо є інші х-ка для показу розмірів
        self.df['LotFrontage_missing'] = self.df['LotFrontage'].isnull().astype('int8')
        self.df.drop(columns=['LotFrontage'], inplace=True)
        
        # одне за найбільшим
        self.df['Electrical'] = self.df['Electrical'].fillna('SBrkr')
        
        # дуже важлива х-ка, але мало різних значень, перероблю в HasPool
        self.df['HasPool'] = np.where(self.df['PoolQC'].isnull(), 1, 0)
        self.df.drop(columns=['PoolQC'], inplace=True)
        return self

    def encode(self) -> 'HousePriceFeatureBuilder':
        # Через те що дані в train та test не співпадають, наприклад:
        # В train для фічі може бути Good та Bad
        # А в test Bad, Average
        # Використаю ручне заповнення категорій через маппінг
        
        with open('value_dictionary.json', 'r', encoding='utf-8') as file:
            columns = json.load(file)
        for column, categories in columns.items():
            self._hot_encode(column, categories=categories, drop_first=True)
            
        for column in ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC', 'KitchenQual',
                       'FireplaceQu', 'GarageQual', 'GarageCond', 'BsmtExposure', 
                       'BsmtFinType1', 'BsmtFinType2', 'GarageFinish', 'Fence']:
            self.df[f'{column}_positive'] = self.df[column].apply(self.is_positive_mark).astype('int8')
            self.df[f'{column}_negative'] = self.df[column].apply(self.is_negative_mark).astype('int8')
            self.df[f'{column}_neutral'] = self.df[column].apply(self.is_neutral_mark).astype('int8')
            self.df.drop(columns=[column], inplace=True, errors='ignore')
        
        return self
    
    def is_positive_mark(self, value: str | None) -> bool:
        return value in ['Ex', 'Gd', 'GLQ', 'Fin', 'GdPrv', 'GdWo']
    
    def is_neutral_mark(self, value: str | None) -> bool:
        return value in ['Ta', 'Av', 'ALQ', 'Rec', 'RFn']
    
    def is_negative_mark(self, value: str | None) -> bool:
        return value in ['Fa', 'Po', 'Mn', 'BLQ', 'LwQ', 'Unf', 'MnPrv', 'MnWw']
    
    # мінусових значень нема(тільки label enc, але там є більше 1)
    # Пробував нормалізовувати тільки чисельні значення, без label encoding (> 100), але не допомогло
    def normalize(self) -> 'HousePriceFeatureBuilder':
        self._normalize_from_0_to_1()
        return self
    
    def drop_high_correlation_features(self) -> 'HousePriceFeatureBuilder':
        # Кореляція Пірсона
        self.df.drop(columns=['SaleType_New', '2ndFlrSF', 'PoolArea', 'Exterior2nd_VinylSd',
                               'Exterior2nd_MetalSd', 'RoofStyle_Hip', 'Exterior2nd_HdBoard',
                               'GarageCars', 'MSZoning_FV', 'Exterior2nd_Wd Sdng',
                               'Electrical_SBrkr', 'Exterior2nd_AsbShng', 'GrLivArea',
                               'TotalBsmtSF', 'MSZoning_RM', 'Foundation_CBlock',
                               'Exterior2nd_Stucco', 'YearBuilt', 'SaleCondition_Partial',
                               'Exterior2nd_Plywood', 'RoofMatl_Tar&Grv', 'Heating_GasW', 'BldgType_Duplex'
                             ], inplace=True, errors='ignore')
        
        # Стає гірше, тому позбавився
        # VIF
        # self.df.drop(columns=['Exterior1st_Plywood', 'GarageType_Basment', 'Exterior2nd_CBlock',
        #                        'BsmtFinType1_negative', 'GarageFinish_negative', 'HouseStyle_1Story',
        #                        'Neighborhood_NAmes', 'BsmtFinType2_negative', 'Condition2_Norm',
        #                        '1stFlrSF', 'GarageType_Attchd', 'Exterior1st_VinylSd', 'BsmtFinSF1'
        #                        'Exterior2nd_AsphShn'], inplace=True, errors='ignore')
        return self


    # викидання х-к не допомогло
    def drop_uninformed_features(self) -> 'HousePriceFeatureBuilder':
        #Для дерева
        features = ['OverallQual', '1stFlrSF', 'ExterQual_positive', 'GarageArea',
       'FullBath', 'LotArea', 'BsmtQual_positive', 'Fireplaces',
       'KitchenQual_positive', 'BsmtFinSF1', 'TotRmsAbvGrd', 'YearRemodAdd',
       'HalfBath', 'OpenPorchSF', 'GarageYrBlt', 'BedroomAbvGr', 'MasVnrArea',
       'HouseStyle_2Story', 'GarageType_Attchd', 'OverallCond', 'CentralAir_Y',
       'GarageType_Detchd', 'MSZoning_RL', 'HouseStyle_1Story', 'WoodDeckSF',
       'BsmtFullBath', 'GarageFinish_negative', 'Foundation_PConc',
       'FireplaceQu_positive', 'BsmtUnfSF', 'PavedDrive_Y',
       'BsmtExposure_positive', 'HeatingQC_positive', 'SaleCondition_Normal',
       'BsmtFinType1_positive', 'MoSold', 'GarageFinish_positive',
       'BsmtFinType1_negative', 'Neighborhood_Crawfor',
       'Neighborhood_Edwards']
        
        # для лін регрес
        features = ['OverallQual', 'FullBath', 'GarageArea', 'TotRmsAbvGrd', '1stFlrSF',
       'BsmtFinType1_positive', 'HalfBath', 'SaleType_WD', 'OverallCond',
       'BsmtFullBath', 'Fireplaces', 'SaleCondition_Normal', 'YearRemodAdd',
       'GarageType_Attchd', 'BsmtFinType1_neutral', 'MSZoning_RL',
       'Condition1_Norm', 'GarageFinish_positive', 'BsmtExposure_positive',
       'Foundation_PConc', 'WoodDeckSF', 'BsmtFinSF1', 'BsmtQual_positive',
       'Neighborhood_NridgHt', 'GarageYrBlt', 'Neighborhood_NoRidge',
       'HeatingQC_positive', 'BsmtUnfSF', 'Neighborhood_Edwards',
       'BedroomAbvGr', 'BsmtFinType1_negative', 'Neighborhood_Somerst',
       'RoofStyle_Gable', 'KitchenQual_positive', 'GarageFinish_neutral',
       'CentralAir_Y', 'Exterior1st_VinylSd', 'LotConfig_Corner',
       'Neighborhood_OldTown', 'ScreenPorch']
        
        self.df = self.df[features]
        return self