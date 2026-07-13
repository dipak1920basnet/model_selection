from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import pandas as pd 
from sklearn.base import clone


data = load_breast_cancer()

X = data.data
y = data.target

X_train_dev, X_test, y_train_dev, y_test = train_test_split(
    X,
    y,
    test_size=0.2, 
    random_state=42,
    stratify=y)


X_train, X_dev, y_train, y_dev = train_test_split(
    X_train_dev, 
    y_train_dev, 
    test_size=0.25, 
    random_state=42,
    stratify=y_train_dev)


model_list = [
    LogisticRegression(
        max_iter=1000,
        random_state=42
    ),
    DecisionTreeClassifier(
        random_state=42
    ),
    RandomForestClassifier(
        random_state=42
    ),
    MLPClassifier(
        random_state=42,
        max_iter=1000
    )
]


total_model = len(model_list)

def model_history(X_train, y_train, X_dev, y_dev, model_list:list):
    fitted_model = [model.fit(X_train, y_train) for model in model_list]
    
    # training evaluation 
    train_predict = [model.predict(X_train) for model in fitted_model]
    evaluate_train = [f1_score(y_train, i) for i in train_predict]

    # evaluate the model with dev set 
    predicted_model = [model.predict(X_dev) for model in fitted_model]
    evaluation_ = [f1_score(y_dev, i) for i in predicted_model]

    return evaluate_train, evaluation_

train_evaluation, dev_evaluation = model_history(X_train,y_train, X_dev, y_dev, model_list)

evaluation = {
    "model_name":model_list,
    "train_f1_score": train_evaluation,
    "dev_f1_score":dev_evaluation
}

evaluation_score = pd.DataFrame(evaluation)
print(evaluation_score)


# get the best model
max_value = max(dev_evaluation)
model_ind = dev_evaluation.index(max_value)
best_model, best_model_acc = model_list[model_ind], dev_evaluation[model_ind]

# model, best_model_acc= best_model(X_train,y_train, X_dev, y_dev, model_list)
model = best_model

# Hyper parameter tuning for the best model 
print(model)
print(model.get_params())
print(best_model_acc)

# parameters list for hyperparameter tuning
n_estimators = [50,75,100,125,150]
criterion = ["gini", "entropy", "log_loss"]
max_depth = [None] + list(range(2, 31, 2))
min_sample_split = [2, 5, 10, 20]
min_samples_leaf = [1, 2, 4, 8]
max_features = ["sqrt", "log2", None]
bootstrap= [True, False]


tuning_history = []
# Manual Tuning
for i in n_estimators:
    for j in criterion:
        for k in max_depth:
            # rest of the parameter will be in automated tuning process by libray method 
            training = clone(model)
            training.set_params(
                n_estimators=i,
                criterion=j,
                max_depth=k,
                random_state=42
            )

            training.fit(X_train, y_train)

            training_prediction = training.predict(X_train)
            dev_prediction = training.predict(X_dev)

            train_evaluation = f1_score(y_train, training_prediction)
            dev_evaluation =  f1_score(y_dev, dev_prediction)

            history = {"n_estimators" :i,
                "criterion":j,
                "max_depth":k,
                "f_1_train":train_evaluation,
                "f_1_dev":dev_evaluation
            }

            tuning_history.append(history)
        
        print("....")

history_tune = pd.DataFrame(tuning_history)

best = history_tune.loc[
    history_tune["f_1_dev"].idxmax()
]

print(f"best parameters: {best}")

final_model = clone(model)

# Check if this esitmators is actually a none 
# if none then do nothing else change to integer 
estimators = best["n_estimators"]

if pd.isna(estimators):
    estimators = None
else:
    estimators = int(estimators)

depth = best["max_depth"]
if pd.isna(depth):
    depth = None
else: 
    depth = int(depth)

final_model.set_params(
    n_estimators=estimators,
    criterion=best["criterion"],
    max_depth=depth,
    random_state=42
)

final_train = final_model.fit(X_train_dev, y_train_dev)
final_prediction = final_train.predict(X_test)

final_evaluation = f1_score(final_prediction, y_test)
print(f"final_f1_Score: {final_evaluation}")
