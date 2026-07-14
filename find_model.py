from sklearn.metrics import f1_score
import pandas as pd

class FindBestModel:
    def __init__(self,model_list:list):
        self.model_list = model_list
        self.evaluation = {}

    def add_model(self, model):
        self.model_list.append(model)

    def model_history(X_train, y_train, X_dev, y_dev, model_list:list):
        fitted_model = [model.fit(X_train, y_train) for model in model_list]
        
        # training evaluation 
        train_predict = [model.predict(X_train) for model in fitted_model]
        evaluate_train = [f1_score(y_train, i) for i in train_predict]

        # evaluate the model with dev set 
        predicted_model = [model.predict(X_dev) for model in fitted_model]
        evaluation_ = [f1_score(y_dev, i) for i in predicted_model]

        return evaluate_train, evaluation_

    def fit(self, X_train,y_train,X_dev,y_dev):
        train_evaluation, dev_evaluation = self.model_history(X_train,y_train, X_dev, y_dev, self.model_list)
        return train_evaluation, dev_evaluation
    
    def model_train(self,X_train,y_train,X_dev,y_dev):
        train_evaluation, dev_evaluation = self.fit(self, X_train,y_train,X_dev,y_dev)
        evaluation = {
            "model_name":self.model_list,
            "train_f1_score": train_evaluation,
            "dev_f1_score":dev_evaluation
            }
        self.evaluation = evaluation
    
    def print_bet_model_Info(self):
        evaluation_score = pd.DataFrame(self.evaluation)
        print(evaluation_score)

    def get_best_model(self):
        # get the best model
        dev_evaluation = self.evaluation["dev_f1_score"]
        max_value = max(dev_evaluation)
        model_ind = dev_evaluation.index(max_value)
        best_model, best_model_acc = self.model_list[model_ind], dev_evaluation[model_ind]
        return (best_model, best_model_acc)