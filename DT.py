
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

def décision_tree(datasetpath):
    data = pd.read_csv(datasetpath)

    X = data.drop('label', axis=1)
    y = data['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = DecisionTreeClassifier()


    model.fit(X_train, y_train)


    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    print(f'Test accuracy: {accuracy}')

"""
#  to save the trained model
import joblib
joblib.dump(model, 'digit_recognition_model_dt.joblib')
"""