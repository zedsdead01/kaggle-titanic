import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

passenger_data = pd.read_csv('train.csv', dtype=object)
passenger_data['Sex'] = passenger_data['Sex'].map({'male': 1, 'female': 0})
passenger_data['Cherbourg'] = passenger_data.apply(lambda row: int(row.Embarked == 'C'), axis=1)
passenger_data['Queenstown'] = passenger_data.apply(lambda row: int(row.Embarked == 'Q'), axis=1)
passenger_data['Southampton'] = passenger_data.apply(lambda row: int(row.Embarked == 'S'), axis=1)

mean_age = int(round(passenger_data["Age"].astype(float).mean()))
passenger_feats = passenger_data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Cherbourg', 'Queenstown', 'Southampton']]
passenger_feats['Age'].fillna(mean_age, inplace=True)
passenger_feats = passenger_feats.astype(float).as_matrix()
print(passenger_feats.shape)

labels = passenger_data['Survived'].astype(int).values

ppln_steps = [
    ('scaler', StandardScaler()),
    # ('clf', LogisticRegression()),
    ('clf', SVC(kernel='rbf'))
]

ppln = Pipeline(ppln_steps)

ppln.fit(passenger_feats, labels)
pred_labels = ppln.predict(passenger_feats)

err_rate = np.sum(labels != pred_labels) / labels.shape[0]
print(err_rate)
