from abc import ABC, abstractmethod
from functools import wraps

import pandas as pd


class ModelTrainer(ABC):
    def __init__(self, 
                 test_df: pd.DataFrame, 
                 train_df: pd.DataFrame,
                 model,
                 *,
                 logging: bool = True):
        self.test_df = test_df
        self.train_df = train_df
        self.logging = logging
        self.model = model
        print(set(self.train_df.columns) - set(self.test_df.columns))
        
    @abstractmethod
    def fit(self):
        pass
    
    @abstractmethod
    def predict(self):
        pass    
    
    def __getattribute__(self, name):
        """Використовується для логування, якщо атр колейбл, то логування"""
        attr = super().__getattribute__(name)
        if callable(attr) and name not in [
            '_ModelTrainer__log', '_ModelTrainer__log_info']:
            return self.__log(attr)
        return attr
    
    def __log(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if self.logging:
                self.__log_info('До:')
                
            result = func(*args, **kwargs)
            
            if self.logging:
                self.__log_info('Після:')
                
            return result
        return wrapper
    
    def __log_info(self, title: str):
        print('\n' + '-' * 100 + '\n')
        print(title)
        
        dataframes = {
            'TRAIN': self.train_df,
            'TEST': self.test_df
        }
        
        for name, df in dataframes.items():
            print("     {name}:".format(name=name))
            null_columns_count = df.columns[df.isnull().any()].size
            print("Nulls columns = {counts}".format(counts=null_columns_count))
            print()
        print('-' * 100 + '\n')
