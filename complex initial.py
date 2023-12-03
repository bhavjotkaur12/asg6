# Initial Approach - Binary Classification

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Sample data creation
np.random.seed(0)
data_size = 1000
data = {
    'age': np.random.randint(18, 70, data_size),
    'avg_monthly_spend': np.random.uniform(10, 200, data_size),
    'customer_service_calls': np.random.randint(1, 10, data_size),
    'churn': np.random.randint(0, 2, data_size)
}
df = pd.DataFrame(data)

# Preparing data
X = df.drop('churn', axis=1)
y = df['churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Binary classification model
model = LogisticRegression()
model.fit(X_train, y_train)

# Prediction and evaluation
predictions = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, predictions))
