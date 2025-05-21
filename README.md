# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the necessary python packages using import statements.

2.Read the given csv file using read_csv() method and print the number of contents to be displayed using df.head().

3.Split the dataset using train_test_split.

4.Calculate Y_Pred and accuracy.

5.Print all the outputs.

6.End the Program.

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: Josiah Immanuel A
RegisterNumber: 212223043003 
*/
```
```
import chardet
file = 'spam.csv'
with open(file, 'rb') as rawdata:
    result = chardet.detect(rawdata.read(100000))
print("Detected Encoding:", result)

import pandas as pd
data = pd.read_csv("spam.csv", encoding='windows-1252')

print(data.head())
print(data.info())
print("Missing values:\n", data.isnull().sum())

x = data["v2"].values
y = data["v1"].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
x_train = cv.fit_transform(x_train)
x_test = cv.transform(x_test)

from sklearn.svm import SVC
svc = SVC()
svc.fit(x_train, y_train)

y_pred = svc.predict(x_test)

from sklearn import metrics
accuracy = metrics.accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
```
## Output:

### Encoding:

![Screenshot 2025-05-21 054102](https://github.com/user-attachments/assets/2c14e10b-6a2a-40f0-8cf2-c94ec948ee21)

### Head():

![Screenshot 2025-05-21 054200](https://github.com/user-attachments/assets/b5227489-f58f-483e-bf71-37bdbeed7e9c)

### Info():

![Screenshot 2025-05-21 054304](https://github.com/user-attachments/assets/d2ea74e9-81f1-4f25-a7b2-b1bff07bd1d1)

### isnull().sum()

![Screenshot 2025-05-21 054352](https://github.com/user-attachments/assets/477d96be-a1a9-4d62-8f25-81dcc5fa96a6)

### Prediction of y:

![Screenshot 2025-05-21 054801](https://github.com/user-attachments/assets/84c01430-0bcc-424a-bffc-12ec714fb863)

### Accuracy

![Screenshot 2025-05-21 054437](https://github.com/user-attachments/assets/88b03c97-c198-4106-a91d-0109206837bf)

## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
