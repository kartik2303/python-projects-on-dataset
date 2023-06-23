import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

rf = RandomForestClassifier(random_state=1)
lr = LogisticRegression(random_state=1)
gbm = GradientBoostingClassifier(n_estimators=10)
dtc = DecisionTreeClassifier(random_state=1)

df = pd.read_csv("C:/Users/LENOVO/PycharmProjects/pythonProject1/Titanic-Dataset.csv")

le = LabelEncoder()

le.fit(df['Sex'])
df['Sex'] = le.transform(df['Sex'])
le.fit(df['Embarked'])
df['Embarked'] = le.transform(df['Embarked'])
print(df)

X = df.drop('PassengerId', axis=1)
X = X.drop('SibSp', axis=1)
X = X.drop('Parch', axis=1)
X = X.drop('Survived', axis=1)
X = X.drop('Name', axis=1)
X = X.drop('Ticket', axis=1)
X = X.drop('Cabin', axis=1)
Y = df['Survived']

print(X.isnull().sum())
X['Age'].fillna((X['Age'].mean()), inplace=True)
print(X.isnull().sum())
print(X)

model = ExtraTreesClassifier()
model.fit(X, Y)
print(model.feature_importances_)
feat_importance = pd.Series(model.feature_importances_, index=X.columns)
feat_importance.nlargest(5).plot(kind='barh')
plt.show()

from collections import Counter
print(Counter(Y))
from imblearn.over_sampling import SMOTE
sms = SMOTE(random_state=0)
X, Y = sms.fit_resample(X,Y)
print(Counter(Y))

import seaborn as sns

sns.boxplot(df['Fare'])
plt.show()

print(X['Fare'])
Q1 = X['Fare'].quantile(0.25)
Q3 = X['Fare'].quantile(0.75)
IQR = Q3 - Q1
print(IQR)
upper = Q3 + 1.5 * IQR
lower = Q1 - 1.5 * IQR
print(upper)
print(lower)

out1 = X[X['Fare'] < lower].values
out2 = X[X['Fare'] > upper].values

X['Fare'].replace(out1, lower, inplace=True)
X['Fare'].replace(out2, upper, inplace=True)
print(X['Fare'])

sns.boxplot(X['Age'])
plt.show()

print(X['Age'])
Q1 = X['Age'].quantile(0.25)
Q3 = X['Age'].quantile(0.75)
IQR = Q3 - Q1
print(IQR)
upper = Q3 + 1.5 * IQR
lower = Q1 - 1.5 * IQR
print(upper)
print(lower)

out1 = X[X['Age'] < lower].values
out2 = X[X['Age'] > upper].values

X['Age'].replace(out1, lower, inplace=True)
X['Age'].replace(out2, upper, inplace=True)
print(X['Age'])

X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=0, test_size=0.3)

func = [lr, rf, dtc, gbm]
for item in func:
    item.fit(X_train, y_train)
    y_pred = item.predict(X_test)
    print(accuracy_score(y_test, y_pred))