from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

iris = load_iris()
X = iris.data    # Features (sepal length, sepal width, petal length, petal width)
y = iris.target  # Labels (0, 1, or 2)

# print(X)
# X_train, X_test, y_train, y_test = train_test_split(
#     X,
#     y, 
#     test_size=0.2,       # 20% of data goes to the test set, 80% to the training set
#     random_state=42,     # Ensures reproducible results every time you run it
# )

# # 1. Initialize the Random Forest model
# # n_estimators is the number of trees; random_state ensures reproducibility
# rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# # 2. Train the model using the training sets
# rf_model.fit(X_train, y_train)

# y_pred = rf_model.predict(X_test)
# print(y_pred)
# accuracy = accuracy_score(y_test, y_pred)
# print(f"Random Forest Model Accuracy: {accuracy * 100:.2f}%")

for seed in [0, 1, 42, 99, 123]:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)
    print(f"seed={seed}: {accuracy_score(y_test, y_pred)*100:.2f}%")
    
# === ML is better when: ===
# 1. Predicting which coffee product a customer will buy next
#    — you have labeled historical transaction data (coffee_sales.csv)
#    — structured tabular input, needs fast prediction at scale
# 2. Detecting anomalies in sales data (unusually low/high revenue day)
#    — pattern recognition on numbers, not language
# 3. Classifying customer segments (high/medium/low value)
#    — fixed categories, thousands of rows, needs consistency every run

# === LLM is better when: ===
# 1. Answering "why did revenue drop last month?" 
#    — requires reasoning and explanation, not just a number
# 2. Classifying a customer complaint email as urgent/routine
#    — unstructured text input, no labeled training data available
# 3. Your RAG agent answering questions about documents
#    — no training data exists for "what does this README say"