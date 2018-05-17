import pandas as pd

train_data = pd.read_csv('train.csv', dtype=object)
train_data['Sex'] = train_data['Sex'].map({'male': 1, 'female': 0})
train_data['Cherbourg'] = train_data.apply(lambda row: int(row.Embarked == 'C'), axis=1)
print(train_data)
