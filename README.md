# Ex.No.1---Data-Preprocessing
## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

## REQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

Kaggle :
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

Data Preprocessing:

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

Need of Data Preprocessing :

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
Importing the libraries
Importing the dataset
Taking care of missing data
Encoding categorical data
Normalizing the data
Splitting the data into test and train

## PROGRAM:
```
Developed by: D.vishnu vardhan reddy
Register No: 212221230023
```
```
import pandas as pd
df=pd.read_csv("Churn_Modelling.csv")
df.head()
df.isnull().sum()
df.drop(["RowNumber","Age","Gender","Geography","Surname"],inplace=True,axis=1)
print(df)
x=df.iloc[:,:-1].values
y=df.iloc[:,-1].values
print(x)
print(y)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df1 = pd.DataFrame(scaler.fit_transform(df))
print(df1)
from sklearn.model_selection import train_test_split
xtrain,ytrain,xtest,ytest=train_test_split(x,y,test_size=0.2,random_state=2)
print(xtrain)
print(len(xtrain))
print(xtest)
print(len(xtest))
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
df1 = sc.fit_transform(df)
print(df1)
```
## OUTPUT:
![image](https://user-images.githubusercontent.com/94175324/226857568-9b2197d5-4a4b-4eb9-9edd-b98b0b545a68.png)
![image](https://user-images.githubusercontent.com/94175324/226857713-2d34e88e-21dc-4d21-87db-5b3ea07e0caf.png)
![image](https://user-images.githubusercontent.com/94175324/226857842-2fb7e099-a852-4874-9820-062c07f94a6b.png)
![image](https://user-images.githubusercontent.com/94175324/226857945-62801e50-088e-4b92-9d67-867f8d767049.png)
![image](https://user-images.githubusercontent.com/94175324/226858046-7c4c35aa-c011-4ccc-9468-839b53bfab47.png)
![image](https://user-images.githubusercontent.com/94175324/226858140-a7f63838-d0e0-45cb-9e79-373b55c070bc.png)

## RESULT
Thus the above program for standardizing the given data was implemented successfully.

