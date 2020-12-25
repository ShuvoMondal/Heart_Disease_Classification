import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import pickle
# from sklearn.linear_model import LogisticRegression


true_count =0
false_count =0
df = pd.read_csv('heart.csv')
print(df)
X = df.drop(columns=['target'], axis=1)
y = df['target']

X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.3, random_state= 33)

rfc = RandomForestClassifier(n_jobs= 7,random_state = 25)
rfc.fit(X_train, y_train)

y_pred = rfc.predict(X_test)
print(y_pred,'\n',np.array(y_test))
for i,j in zip(y_pred,np.array(y_test)):
    if i == j:
        true_count +=1
    else:
        false_count +=1

print(accuracy_score(y_pred,y_test))
rdf_dt = confusion_matrix(y_test, y_pred)
print(rdf_dt)

print(true_count)
print(false_count)
print(X_test)
print(rfc.predict([[19,0,3,94,283,0,1,142,0,0.0,2,0,1]]))
print(rfc.predict([[20,1,3,73,158,0,0,142,0,1.8,2,0,3]]))
print(rfc.predict([[18,1,0,125,212,0,1,168,0,3.0,2,2,3]]))

pickle.dump(rfc,open('model.pkl','wb'))

# logistic_regression = LogisticRegression(random_state=69)
# logistic_regression.fit(X_train, y_train)

# lrpr = logistic_regression.predict(X_test)
# print(accuracy_score(lrpr,y_test))
