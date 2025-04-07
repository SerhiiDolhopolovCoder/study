import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from . import FeatureBuilder


class HousePriceFeatureBuilder(FeatureBuilder):
    def __init__(self, df: pd.DataFrame):
        super().__init__(df, 'SalePrice')

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

        
        self.df['BsmtExposure'] = self.df['BsmtExposure'].fillna('NoBasement')
        self.df['BsmtFinType1'] = self.df['BsmtFinType1'].fillna('NoBasement')
        self.df['BsmtFinType2'] = self.df['BsmtFinType2'].fillna('NoBasement')
        self.df['BsmtQual'] = self.df['BsmtQual'].fillna('NoBasement')
        self.df['BsmtCond'] = self.df['BsmtCond'].fillna('NoBasement')
        
        
        # Цих даних з Null немає в трейні, але є в тестовому, заповнив 0 по контексту
        self.df['BsmtFullBath'] = self.df['BsmtFullBath'].fillna(0)
        self.df['BsmtHalfBath'] = self.df['BsmtHalfBath'].fillna(0)
        self.df['BsmtUnfSF'] = self.df['BsmtUnfSF'].fillna(0)
        self.df['BsmtFinSF1'] = self.df['BsmtFinSF1'].fillna(0)
        self.df['BsmtFinSF2'] = self.df['BsmtFinSF2'].fillna(0)
        self.df['GarageArea'] = self.df['GarageArea'].fillna(0)
        
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
        
        condition_list = ['Artery', 'Feedr', 'Norm', 'PosA', 'PosN', 'RRAe',
                           'RRNn', 'RRAn', 'RRNe']
        exterior_list = ['Other', 'AsbShng', 'AsphShn', 'BrkComm', 'BrkFace', 'CBlock',
                        'CemntBd', 'HdBoard', 'ImStucc', 'MetalSd', 'Plywood',
                        'PreCast', 'Stone', 'Stucco', 'VinylSd', 'Wd Sdng',
                        'WdShing']
        columns = {
            'Alley': ['NoAlley', 'Grvl', 'Pave'],
            'MSSubClass': ['20', '30', '40', '45', '50', '60', '70', '75', 
                           '80', '85', '90', '120', '150', '160', '180', '190'],
            'MSZoning': ['A', 'C', 'FV', 'I', 'RH', 'RL', 'RP', 'RM'],
            'Street': ['Grvl', 'Pave'],
            'LotShape': ['Reg', 'IR1', 'IR2', 'IR3'],
            'LandContour': ['Lvl', 'Bnk', 'HLS', 'Low'],
            'Utilities': ['AllPub', 'NoSewr', 'NoSeWa', 'ELO'],
            'LotConfig': ['Inside', 'Corner', 'CulDSac', 'FR2', 'FR3'],
            'LandSlope': ['Gtl', 'Mod', 'Sev'],
            'Neighborhood': ['Blmngtn', 'Blueste', 'BrDale', 'BrkSide', 'ClearCr', 
                             'CollgCr', 'Crawfor', 'Edwards', 'Gilbert', 'IDOTRR',
                             'MeadowV', 'Mitchel', 'NAmes', 'NoRidge', 'NPkVill',
                             'NridgHt', 'NWAmes', 'OldTown', 'SWISU', 'Sawyer', 
                             'SawyerW', 'Somerst', 'StoneBr', 'Timber', 'Veenker'],
            'Condition1': condition_list,
            'Condition2': condition_list,
            'BldgType': ['1Fam', '2fmCon', 'Duplex', 'TwnhsE', 'TwnhsI'],
            'HouseStyle': ['1.5Unf', '1Story', '2.5Unf', '2Story', 'SFoyer',
                           'SFLiv', '2.5Fin', '1.5Fin', 'SLvl'],
            'RoofStyle': ['Flat', 'Gable', 'Gambrel', 'Hip', 'Mansard', 'Shed'],
            'RoofMatl': ['ClyTile', 'CompShg', 'Membran', 'Metal', 'Roll', 
                         'Tar&Grv', 'WdShake', 'WdShngl'],
            'Exterior1st': exterior_list,
            'Exterior2nd': exterior_list,
            'MasVnrType': ['NoMasonry', 'BrkCmn', 'BrkFace', 'CBlock', 'Stone'],
            'Foundation': ['BrkTil', 'CBlock', 'PConc', 'Slab', 'Stone', 'Wood'],
            'Heating': ['Floor', 'GasA', 'GasW', 'Grav', 'OthW', 'Wall'],
            'CentralAir': ['N', 'Y'],
            'Electrical': ['Mix', 'FuseA', 'FuseF', 'FuseP', 'SBrkr'],
            'Functional': ['Typ', 'Maj1', 'Maj2', 'Min1', 'Min2', 'Mod', 'Sev', 'Sal'],
            'GarageType': ['NoGarage', 'Attchd', 'Basment', 'BuiltIn', 'CarPort',
                           'Detchd', '2Types'],
            'PavedDrive': ['N', 'P', 'Y'],
            'SaleType': ['Oth', 'Con', 'ConLD', 'ConLI', 'ConLw', 'New', 'WD',
                         'CWD', 'VWD', 'COD'],
            'SaleCondition': ['Abnorml', 'AdjLand', 'Alloca', 'Family', 'Normal',
                              'Partial']
        }
        
        # columns = ['Alley', 'MSSubClass', 'MSZoning', 'Street', 'LotShape', 
        #            'LandContour', 'Utilities', 'LotConfig',
        #            'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle',
        #            'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'Foundation',
        #            'Heating', 'CentralAir', 'Electrical', 'Functional', 'GarageType', 
        #            'PavedDrive', 'SaleType', 'SaleCondition']
        
        for column in columns:
            self._hot_encode(column, drop_first=True)
            
            
        for column in ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC', 'KitchenQual',
                       'FireplaceQu', 'GarageQual', 'GarageCond']:
            self._label_encode(column, self.get_map_mark())
        for column in ['BsmtExposure']:
            self._label_encode(column, self.get_map_mark_range())
        for column in ['BsmtFinType1', 'BsmtFinType2']:
            self._label_encode(column, self.get_map_rating())
        for column in ['GarageFinish']:
            self._label_encode(column, self.get_map_finished())
        self._label_encode('Fence', self.get_map_fence_quality())
        
        return self
    
    def get_map_mark(self) -> dict[str, int]:
        return {
            'NoBasement': -2,
            'NoFireplace': -2,
            'NoPool': -2,
            'NoGarage': -2,
            'No': -1,
            'Po': 0,
            'Fa': 1,
            'TA': 2,
            'Gd': 3,
            'Ex': 4,
        }
        
    def get_map_mark_range(self) -> dict[str, int]:
        return {
            'NoBasement': -2,
            'No': -1,
            'Mn': 0,
            'Av': 1,
            'Gd': 2,
        }
        
    def get_map_rating(self) -> dict[str, int]:
        return {
            'NoBasement': -2,
            'Unf': -1,
            'LwQ': 0,
            'Rec': 1,
            'BLQ': 2,
            'ALQ': 3,
            'GLQ': 4,
        }
        
    def get_map_finished(self) -> dict[str, int]:
        return {
            'NoGarage': -2,
            'Unf': 0,
            'RFn': 1,
            'Fin': 2,
        }
        
    def get_map_fence_quality(self) -> dict[str, int]:
        return {
            'NoFence': -2,
            'MnWw': 0,
            'GdWo': 1,
            'MnPrv': 2,
            'GdPrv': 3,
        }

    # мінусових значень нема(тільки label enc, але там є більше 1)
    def normalize(self) -> 'HousePriceFeatureBuilder':
        for column in self.df.columns:
            if is_numeric_dtype(self.df[column]):
                max_value = self.df[column].max()
                if max_value > 1:
                    self._normalize_from_0_to_1(column)
        return self
    
    def drop_high_correlation_features(self) -> 'HousePriceFeatureBuilder':
        self.df.drop(columns=['BsmtQual_encoded', 'BsmtCond_encoded', 'BsmtFinType2_encoded',
                               'GarageCond_encoded', 'GarageQual_encoded', 'GarageType_NoGarage',
                               'SaleType_New', 'BsmtFinType1_encoded', '2ndFlrSF',
                               'HasPool', 'MSSubClass_190', 'Exterior2nd_VinylSd',
                               'Exterior2nd_CmentBd', 'Exterior1st_MetalSd', 'MSSubClass_80',
                               'RoofStyle_Hip', 'MSSubClass_45', 'FireplaceQu_encoded',
                               'Exterior2nd_HdBoard', 'GarageCars', 'MSZoning_FV',
                               'Exterior2nd_Wd Sdng', 'GrLivArea', 'TotalBsmtSF',
                               'MSZoning_RM', 'MasVnrType_NoMasonry', 'Foundation_CBlock',
                               'Exterior2nd_Stucco', 'MSSubClass_120', 'GarageYrBlt',
                               'SaleCondition_Partial', 'MSSubClass_60', 'Exterior2nd_Plywood',
                               'BsmtExposure_encoded', 'KitchenAbvGr', 'Neighborhood_NPkVill', 'MSSubClass_85'
                              ], inplace=True, errors='ignore')
        return self
