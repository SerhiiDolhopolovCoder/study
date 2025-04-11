from sklearn.tree import DecisionTreeRegressor

from model_trainer import ModelTrainer


class TreeRegressorTrainer(ModelTrainer):
    def __init__(self, 
                 test_df, 
                 train_df, 
                 model: DecisionTreeRegressor, 
                 *, 
                 logging: bool = True):
        super().__init__(test_df, 
                         train_df, 
                         model = model,
                         logging=logging)
        
    def fit(self):
        print('Fit')
    
    def predict(self):
        print('Predict')  