import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

"""**Import the csv data**"""

df = pd.read_csv("LoanApprovalPrediction.csv")

df.info()

df.head()

df.shape

df.columns

print(df.value_counts())

df.describe()

df.isnull().sum()

df.duplicated().sum()

"""**Create X and Y vlaues**"""

X = df.drop(["Loan_ID","Loan_Status"],axis = 1)
y = df["Loan_Status"]
X

"""**Test Train Split**"""

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
X_train

X_train.shape,X_test.shape,y_train.shape,y_test.shape

"""**Corelation Graph**"""

plt.figure(figsize=(12,6))

sns.heatmap(df.corr(),cmap='BrBG',fmt=".2g",linewidths=2,annot=True)

"""**Createa a PipeLine**"""

numerical_col = ['Dependents','ApplicantIncome', 'CoapplicantIncome', 'LoanAmount','Loan_Amount_Term', 'Credit_History']
Categorical_col = ['Gender', 'Married','Education','Self_Employed','Property_Area']

from sklearn.impute import SimpleImputer #handle the missing values
from sklearn.preprocessing import OneHotEncoder #encoding
from sklearn.preprocessing import StandardScaler #assign all in single unit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

categ_pipe = Pipeline(steps=[("imputer",SimpleImputer(strategy='most_frequent')),("OneHotEncoder",OneHotEncoder())])
num_pipe = Pipeline(steps=[("imputer",SimpleImputer(strategy="median")),('Scalar',StandardScaler())])

"""**Column Transfer**"""

preprocessor = ColumnTransformer([("categ_transform",categ_pipe,Categorical_col),("num_transform",num_pipe,numerical_col)])

import numpy as np
X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)

"""**Model Evaluation**"""

from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

dtc = DecisionTreeClassifier()
ss = SVC()
ran = RandomForestClassifier()
knc = KNeighborsClassifier()

algo = [dtc,ss,ran,knc]
algo_name = ['DecisionTreeClassifier','SVC','RandomForestClassifier','kNearestClassifier']

"""**Prediction and Accuracy Check**"""

from sklearn.metrics import accuracy_score

for i,j in zip(algo,algo_name):
  i.fit(X_train,y_train)
  y_pred = i.predict(X_test)
  print(j)
  print("Accuarcy Score: ",accuracy_score(y_test,y_pred)*100,'%')
  print('++' *20)
"""***Confusion Matrix***"""
from sklearn.metrics import confusion_matrix,classification_report

sns.heatmap(cm,
            annot=True,
            fmt='g',
            xticklabels=['Yes','No'],
            yticklabels=['Approved','Not Approved'])
plt.ylabel('Prediction',fontsize=13)
plt.xlabel('Actual',fontsize=13)
plt.title('Confusion Matrix',fontsize=17)
plt.show()

"""***Classification Report***"""
cm = confusion_matrix(y_test,y_pred)
print(classification_report(y_test,y_pred))
