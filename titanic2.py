import pandas as pd
import numpy as np
import random as rnd


import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn import preprocessing
import xgboost as xgb



#https://www.kaggle.com/startupsci/titanic-data-science-solutions

train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
combine = [train_df, test_df]


print(train_df.info())
print(test_df.info())

print(train_df.describe(include=['O']))
print(train_df.isnull().sum()) #To check out how many null info are on the dataset, we can use isnull().sum().


#g = sns.FacetGrid(train_df, col='Survived')
#g.map(plt.hist, 'Age', bins=20)
#plt.show()

#grid = sns.FacetGrid(train_df, col='Survived', row='Pclass', size=2.2, aspect=1.6)
#grid.map(plt.hist, 'Age', alpha=.5, bins=20)
#grid.add_legend();
#plt.show()

# WORK W categoical
'''
grid = sns.FacetGrid(train_df, row='Embarked', size=2.2, aspect=1.6)
grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')
grid.add_legend()
plt.show()
'''
#



dict_Title = {      "Capt.":       "Officer",
                    "Col.":        "Officer",
                    "Major.":      "Officer",
                    "Jonkheer.":   "Royalty",
                    "Don.":        "Royalty",
                    "Sir." :       "Royalty",
                    "Dr.":         "Officer",
                    "Rev.":        "Officer",
                    "the Countess.":"Royalty",
                    "Dona.":       "Royalty",
                    "Mme.":        "Mrs",
                    "Mlle.":       "Miss",
                    "Ms.":         "Mrs",
                    "Mr." :        "Mr",
                    "Mrs." :       "Mrs",
                    "Miss." :      "Miss",
                    "Master." :    "Master",
                    "Lady." :      "Royalty",
                    "the" :      "Royalty"
                    }


############################################################ CLEAN UP DATA

for dataset in combine:
    # FILL MISSING
    dataset['Embarked'].fillna(test_df['Embarked'].dropna().mode()[0], inplace=True)
    dataset['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)
    df90 = test_df['Fare'].quantile(0.98)
    print('90 pct...',df90)  
    dataset['Fare'] = dataset['Fare'].apply(lambda x: df90 if x>df90 else  x )
#    dataset['Hasage'] = np.where(dataset['Age'].isnull(), 1, 0) # No improvemnts
#    dataset['Age'].fillna(test_df['Age'].dropna().median(), inplace=True)

    # NEW FEATURES
#    dataset['Ischildren'] = np.where(dataset['Age']<15, 1, 0)     # No improvemnts
    dataset['FamilySize'] = (dataset['SibSp'] + dataset['Parch']) 
    dataset['IsAlone'] = 0
    dataset['IsAlone'].loc[dataset['FamilySize']>0]  = 1
    dataset['Namesize'] = dataset['Name'].str.len() 
    dataset['SurnameFirst'] = dataset.Name.str[0]  # Surname start Alphabetical order
    # Exatrpolate title from name e.g. Mss, Mr, ...
    dataset['Name_Title'] = dataset['Name'].apply(lambda x: x.split(',')[1]).apply(lambda x: x.split()[0]) 
   # dataset['Name_Title'] = dataset['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0] 
    dataset['Name_Title'] = dataset['Name_Title'].map(dict_Title) # La soljuzione sotto con count mi pare piu elegante
#    title_names = (dataset['Name_Title'].value_counts() < 10)
#    dataset['Name_Title'] = dataset['Name_Title'].apply(lambda x: 'Misc' if title_names.loc[x] == True else x)
    
    # Replace null ages with the average of the title name group...
    dict_age = dataset.groupby('Name_Title').Age.mean()
    print(dict_age)
    idx = dataset.Age.isnull()
    dataset.loc[idx,'Age'] = dataset.loc[idx, 'Name_Title'].map(dict_age)
    
    # NUmero di persone con lo stesso ticket
#    dict_ticket_mult = dataset.groupby('Ticket').Ticket.count()
#    dataset['Stessothk'] = dataset['Ticket'].map(dict_ticket_mult)

# No improvemnts  using following features 
#    dataset['FareBin'] = pd.qcut(dataset['Fare'], 4)
#    dataset['AgeBin'] = pd.cut(dataset['Age'].astype(int), 5)
 #   dataset['Cabin_Letter'] = dataset['Cabin'].apply(lambda x: str(x)[0]) 
 #   dataset['Surname'] = dataset.Name.str.split(",", 1).str[0]    
 #   dataset['Cab_Len'] = dataset['Cabin'].apply(lambda x: len(str(x))) 
     #dataset['Ticket_Len'] = dataset['Ticket'].apply(lambda x: len(x))



#print(dataset['Name_Title'])
#print(train_df[['Sex', 'Survived']].groupby('Sex', as_index=False).mean())

# PIVOT 
data1_x = ['Sex','Pclass', 'Embarked', 'Name_Title','SibSp', 'Parch', 'Age', 'Fare', 'FamilySize']
Target = ['Survived']
for x in data1_x:
    if train_df[x].dtype != 'float64' :
        print('Survival Correlation by:', x)
        print(train_df[[x, Target[0]]].groupby(x, as_index=False).mean())
        print('-'*10, '\n')



train_df.to_csv('traindf.csv', index=False)
'''
ax = sns.boxplot(x='Name_Title', y='Age', data=train_df)
plt.show()
'''


      
 # encode categorical variables  
labelEnc=LabelEncoder()
cat_vars=['Embarked','Sex', 'SurnameFirst', 'Name_Title']
for col in cat_vars:
        train_df[col]=labelEnc.fit_transform(train_df[col])
        test_df[col]=labelEnc.fit_transform(test_df[col])



# SCALING VALUES

scale_val = ['Age', 'Namesize', 'Fare']
for dataset in combine:
    std_scale = preprocessing.StandardScaler().fit(dataset[scale_val])
    dataset[scale_val] = std_scale.transform(dataset[scale_val])


#train_df.to_csv('traindf.csv', index=False)


# remove column not used by models
drop_colm = ['Parch', 'SibSp', 'Ticket', 'Name','Cabin']
train_df = train_df.drop(drop_colm, axis=1)
test_df = test_df.drop(drop_colm, axis=1)

combine = [train_df, test_df]
    
print(test_df.info())
print(train_df.info())


'''
cols = train_df.columns
for cat in cols:
    ax = sns.boxplot(x="Survived", y=cat, data=train_df)
    ax = sns.stripplot(x="Survived", y=cat, data=train_df, jitter=True, edgecolor="gray")
    plt.show()
'''

# Age looks not significative from boxplot.... dropit imprrove the score sul test set ma lo score ifinale peggiora...
#########################################################    STart MODELS
X_train = train_df.drop(["Survived", "PassengerId"], axis=1)
Y_ID = train_df["PassengerId"]
Y_train = train_df["Survived"]
X_test  = test_df.drop(["PassengerId"], axis=1).copy()
X_train.shape, Y_train.shape, X_test.shape

# ALL VALUE SCALING
'''
std_scaler = StandardScaler()
X_train = std_scaler.fit_transform(X_train)
X_test = std_scaler.transform(X_test)
'''

#X_train.to_csv('xtrain.csv', index=False)



'''
### Logistic Regression
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)
acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
print(acc_log)

coeff_df = pd.DataFrame(train_df.columns.delete(0))
coeff_df.columns = ['Feature']
coeff_df["Correlation"] = pd.Series(logreg.coef_[0])
print(coeff_df.sort_values(by='Correlation', ascending=False))


# Support Vector Machines
svc = SVC()
svc.fit(X_train, Y_train)
Y_pred = svc.predict(X_test)
acc_svc = round(svc.score(X_train, Y_train) * 100, 2)
print(acc_svc)

# KNN
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)
acc_knn = round(knn.score(X_train, Y_train) * 100, 2)
print(acc_knn)

# Gaussian Naive Bayes
gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)
Y_pred = gaussian.predict(X_test)
acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)
print(acc_gaussian)

# Perceptron
perceptron = Perceptron()
perceptron.fit(X_train, Y_train)
Y_pred = perceptron.predict(X_test)
acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)
print(acc_perceptron)

# Linear SVC
linear_svc = LinearSVC()
linear_svc.fit(X_train, Y_train)
Y_pred = linear_svc.predict(X_test)
acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)
print(acc_linear_svc)

# Stochastic Gradient Descent
sgd = SGDClassifier()
sgd.fit(X_train, Y_train)
Y_pred = sgd.predict(X_test)
acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)
print(acc_sgd)


# Decision Tree
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_pred_dt = decision_tree.predict(X_test)
acc_decision_tree = decision_tree.score(X_train, Y_train) * 100
print('Decision tree', acc_decision_tree)
'''

#  TRY GRID SEARCH
'''
print('Trying GRID SEARCH for KNN algo...')
from sklearn.model_selection import GridSearchCV

n_neighbors = [6,7,8,9,10,11,12,14,16,18,20,22]
algorithm = ['auto']
weights = ['uniform', 'distance']
leaf_size = list(range(1,50,5))
hyperparams = {'algorithm': algorithm, 'weights': weights, 'leaf_size': leaf_size, 
               'n_neighbors': n_neighbors}
gd=GridSearchCV(estimator = KNeighborsClassifier(), param_grid = hyperparams, verbose=True, 
                cv=10, scoring = "roc_auc")
gd.fit(X_train, Y_train)
print(gd.best_score_)
print(gd.best_estimator_)

gd.best_estimator_.fit(X_train, Y_train)
y_pred_knn = gd.best_estimator_.predict(X_test)
acc_knn = gd.best_estimator_.score(X_train, Y_train) * 100
print('KNN e Grid search accuracy', acc_knn)
'''

# SKLearn Neaural net
from sklearn.preprocessing import MinMaxScaler

#X_train_scal = MinMaxScaler().fit_transform(X_train)
#X_test_scal = MinMaxScaler().fit_transform(X_test)

X_train_scal = StandardScaler().fit_transform(X_train)
X_test_scal = StandardScaler().fit_transform(X_test)

from sklearn.neural_network import MLPClassifier

nn = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(80, 40, 2), random_state=1)
nn.fit(X_train_scal, Y_train)
acc_nnet = nn.score(X_train_scal, Y_train) * 100
print('Neural Nnet  accuracy', acc_nnet)
Y_neural_net = nn.predict(X_test_scal)
 
 
 
# Random Forest
print('Random forest algo...')
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_pred_rf = random_forest.predict(X_test)
Y_tr_predict = random_forest.predict(X_train)
random_forest.score(X_train, Y_train)
acc_random_forest = random_forest.score(X_train, Y_train) * 100
print('RF accuracy', acc_random_forest)

myfea = pd.concat((pd.DataFrame(X_train.iloc[:, 1:].columns, columns = ['variable']), 
           pd.DataFrame(random_forest.feature_importances_, columns = ['importance'])), 
          axis = 1).sort_values(by='importance', ascending = False)[:20]

print(myfea)



myout = pd.DataFrame({
        "PassengerId": train_df["PassengerId"],
         "Train" : Y_tr_predict,
        "Orig": Y_train
    })
myout.to_csv('myout.csv', index=False)

# XGBOOST
gbm = xgb.XGBClassifier(max_depth=4, n_estimators=1000, learning_rate=0.05)
gbm.fit(X_train, Y_train)
Y_pred_xgb = gbm.predict(X_test)
acc_xgb = gbm.score(X_train, Y_train) * 100
print('XGB accuracy', acc_xgb)




#DATA SUBMISSION
submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": Y_neural_net
    })

submission.to_csv('titanic_sub.csv', index=False)
