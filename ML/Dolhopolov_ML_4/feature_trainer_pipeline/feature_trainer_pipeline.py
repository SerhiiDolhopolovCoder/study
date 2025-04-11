from abc import ABC, abstractmethod

from sklearn.preprocessing import OneHotEncoder
from feature_manager import FeatureManager
from model_trainer import ModelTrainer


class FeatureTrainerPipeline:
    def __init__(self):
        pass
        
    def train(self):
        one_hot_encoder = OneHotEncoder(handle_unknown='error')
        
        feature_manager = FeatureManager()
        
        
        
        model_trainer = ModelTrainer() 
        
        feature_manager.dr
        feature_manager.drop_high_correlation()
        feature_manager.fill_null()
        feature_manager.encode()
        feature_manager.normalize()
        
        model_trainer.fit()
        model_trainer.predict()
        