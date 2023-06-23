
## Data Preprocessing & Load
import pandas as pd
import numpy as np

feature_name_df = pd.read_csv('features.txt', sep='\s+', header=None, names=['column_index', 'column_name'])

feature_name = feature_name_df.iloc[:, 1].values.tolist()
print(feature_name[:10])

feature_dup_df = feature_name_df.groupby('column_name').count()
print(feature_dup_df[feature_dup_df['column_index'] > 1].count())
feature_dup_df[feature_dup_df['column_index'] > 1].head()

def get_new_feature_name_df(old_feature_name_df) :
    feature_dup_df = pd.DataFrame(old_feature_name_df.groupby('column_name').cumcount(), columns=['dup_cnt'])
    feature_dup_df = feature_dup_df.reset_index()

    new_feature_df = pd.merge(old_feature_name_df.reset_index(), feature_dup_df, how='outer')
    new_feature_df['column_name'] = new_feature_df[['column_name', 'dup_cnt']].apply(lambda x: str(x[0]) + '_' + str(x[1]) if x[1] > 0 else x[0], axis=1)
    new_feature_df = new_feature_df.drop(['index'], axis=1)

    return new_feature_df

def get_human_dataset() :
    feature_name_df = pd.read_csv('features.txt', sep='\s+', header=None, names=['column_index', 'column_name'])

    new_feature_name_df = get_new_feature_name_df(feature_name_df)

    feature_names = new_feature_name_df['column_name'].values.tolist()

    x_train = pd.read_csv('X_train.txt', sep='\s+', names=feature_names)
    x_test = pd.read_csv('X_test.txt', sep='\s+', names=feature_names)

    y_train = pd.read_csv('y_train.txt', sep='\s+', header=None, names=['action'])
    y_test = pd.read_csv('y_test.txt', sep='\s+', header=None, names=['action'])

    return x_train, x_test, y_train, y_test

x_train, x_test, y_train, y_test = get_human_dataset()

print(x_train.info())

print(y_train['action'].value_counts())

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

dt_clf = DecisionTreeClassifier(random_state=156)
dt_clf.fit(x_train, y_train)
pred = dt_clf.predict(x_test)
accuracy = accuracy_score(pred, y_test)
print('ACCU: {0: .4f}'.format(accuracy))

