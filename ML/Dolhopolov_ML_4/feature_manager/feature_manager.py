from abc import ABC, abstractmethod


class FeatureManager(ABC):
    @abstractmethod
    def drop_not_needed(self):
        pass
        
    @abstractmethod
    def fill_null(self):
        pass
        
    @abstractmethod
    def encode(self):
        pass
        
    @abstractmethod
    def normalize(self):
        pass
    
    @abstractmethod
    def drop_high_correlation(self):
        pass
    
    
    