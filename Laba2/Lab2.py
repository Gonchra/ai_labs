import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,  root_mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("dataset/processed_test.csv")

print(df.head(10))

target_reg = "Age"
cols_to_drop=["PassengerId", "Cabin", "Name", target_reg]
X = df.drop(columns=cols_to_drop) 
y = df[target_reg]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=42
)

linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

y_pred_test = linear_model.predict(X_test)

mse = mean_squared_error(y_test, y_pred_test)
print(f"Среднеквадратичная ошибка (MSE): {mse:.4f}")

rmse = root_mean_squared_error(y_test, y_pred_test)
print(f"Корень среднеквадратичной ошибки (RMSE): {rmse:.4f}")

mae = mean_absolute_error(y_test, y_pred_test)
print(f"Средняя абсолютная ошибка (MAE): {mae:.4f}")

target_clf = "CryoSleep"
cols_to_drop = ["PassengerId", "Cabin", "Name", target_clf]
X_clf = df.drop(columns=cols_to_drop)
y_clf = df[target_clf]

X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
    X_clf, y_clf, test_size=0.4, random_state=42
)

logreg_model = LogisticRegression()
logreg_model.fit(X_train_clf, y_train_clf)
y_pred_test_clf = logreg_model.predict(X_test_clf)

accuracy = accuracy_score(y_test_clf, y_pred_test_clf)

print(f"Accuracy score: {accuracy:.4f}" )

cm = confusion_matrix(y_test_clf, y_pred_test_clf)

plt.figure(figsize=(4, 3))
sns.heatmap(cm, annot=True, fmt='d', cmap='bwr')
plt.title('Confusion matrix')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()