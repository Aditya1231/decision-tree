# --------------
# Importing Necessary libraries
import warnings
warnings.filterwarnings("ignore")
from matplotlib import pyplot as plt
plt.rcParams['figure.figsize'] = (10, 8)
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

data_train = pd.read_csv(path)
data_test = pd.read_csv(path1)

print(data_train.head(5))
print(data_test.head(5))

data_test.Target.value_counts(dropna=False)
#print(data_test.shape)
data_test.dropna(subset=['Target'],inplace=True)
print(data_test.shape)

target_test = {' <=50K.':0,' >50K.':1}
target_train = {' <=50K':0,' >50K':1}

data_test.Target.replace(target_test,inplace=True)
data_train.Target.replace(target_train,inplace=True)

def create_plots(data,rows=3,cols=5):
    fig = plt.figure(figsize=(15,15))
    for i,column in enumerate(data.columns):
        ax = fig.add_subplot(rows,cols,i+1)
        ax.set_title(column)
        if data.dtypes[column] == np.object:
            #String
            data[column].value_counts().plot(kind='bar',axes=ax)
        else:
            #Numeric
            data[column].hist(axes=ax)
        plt.xticks(rotation='vertical')
        plt.subplots_adjust(hspace=0.7,wspace=0.2 )

create_plots(data_train) 
create_plots(data_test)
#print(data_train.dtypes)
#print(data_test.dtypes)

dt = data_train.dtypes == data_test.dtypes
print(dt[dt==False])
data_test['Age'] = data_test['Age'].astype('int64')
data_test['fnlwgt'] = data_test['fnlwgt'].astype('int64')
data_test['Education_Num'] = data_test['Education_Num'].astype('int64')
data_test['Capital_Gain'] = data_test['Capital_Gain'].astype('int64')
data_test['Capital_Loss'] = data_test['Capital_Loss'].astype('int64')
data_test['Hours_per_week'] = data_test['Hours_per_week'].astype('int64')

categorical_columns = list(data_train.select_dtypes(include="object").columns)
print(categorical_columns)
numerical_columns = list(data_train.select_dtypes(include="number").columns)
print(numerical_columns)


for c in categorical_columns:
    data_train[c].fillna(data_train[c].mode()[0],inplace=True)
    data_test[c].fillna(data_test[c].mode()[0],inplace=True)


for c1 in numerical_columns:
    data_train[c1].fillna(data_train[c1].median(),inplace=True)
    data_test[c1].fillna(data_test[c1].median(),inplace=True)




le = LabelEncoder()
for c in categorical_columns:
    data_train[c] = le.fit_transform(data_train[c])
    data_test[c] = le.transform(data_test[c])


print(data_train.shape,data_test.shape)
data_train_ohe = pd.get_dummies(data=data_train,columns=categorical_columns)
data_test_ohe = pd.get_dummies(data=data_test,columns=categorical_columns)
print(data_train_ohe.shape,data_test_ohe.shape)

list(set(data_train_ohe.columns) - set(data_test_ohe.columns))

data_test_ohe['Country_14'] = 0
list(set(data_train_ohe.columns) - set(data_test_ohe.columns))


print(data_train_ohe.shape)
print(data_test_ohe.shape)


X_train = data_train_ohe.drop(['Target'],axis=1)
y_train = data_train.Target


X_test = data_test_ohe.drop(['Target'],axis=1)
y_test = data_test_ohe.Target


tree = DecisionTreeClassifier(max_depth=3,random_state=17)
tree.fit(X_train,y_train)
tree_pred = tree.predict(X_test)
print('accuracy score',accuracy_score(y_test,tree_pred))


tree_parms = {"max_depth":range(2,11),
             "min_samples_leaf" : range(10,100,10)}
tuned_tree = GridSearchCV(DecisionTreeClassifier(random_state = 17),
            tree_parms,cv=5)

tuned_tree.fit(X_train,y_train)

print(tuned_tree.best_params_)
print(tuned_tree.best_score_)

final_tuned_tree = DecisionTreeClassifier(max_depth=8,min_samples_leaf=20,random_state=17)
final_tuned_tree.fit(X_train,y_train)
y_pred = final_tuned_tree.predict(X_test)
print('accuracy',accuracy_score(y_test,y_pred))









