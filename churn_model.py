import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
import joblib


df = pd.read_csv('data/Customer-Churn.csv')
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())

df.drop('customerID', axis=1, inplace=True)
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0}) 
df = pd.get_dummies(df, drop_first=True)

X = df.drop(['Churn'], axis=1).values
y = df['Churn'].values
for seed in [42]:
  X_train ,X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
  rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
  rf_model.fit(X_train, y_train)
  y_pred = rf_model.predict(X_test)
  print(f"seed={seed}: {accuracy_score(y_test, y_pred)*100:.2f}%")
  print(classification_report(y_test, y_pred))
  
  scaler = StandardScaler()
  X_train_scaled = scaler.fit_transform(X_train)
  X_test_scaled = scaler.transform(X_test)

  lr_model = LogisticRegression(max_iter=1000, random_state=42)
  lr_model.fit(X_train_scaled, y_train)
  y_pred_lr = lr_model.predict(X_test_scaled)
  
  print(f"LogisticRegression: {accuracy_score(y_test, y_pred_lr)*100:.2f}%")
  print(classification_report(y_test, y_pred_lr))
  joblib.dump(lr_model, 'churn_model.pkl')
  joblib.dump(scaler, 'churn_scaler.pkl')  # save scaler too — needed for predictions
  print("Model saved to churn_model.pkl")