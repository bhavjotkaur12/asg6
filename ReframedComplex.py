# Reframed Approach: Predicting Customer Lifetime Value (CLV)

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Assume that CLV is a function of the existing features (this is a simplification)
df['clv'] = df['avg_monthly_spend'] * df['age'] / df['customer_service_calls']

# Preparing data for regression
X = df.drop(['churn', 'clv'], axis=1)
y = df['clv']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Regression model
reg_model = RandomForestRegressor()
reg_model.fit(X_train, y_train)

# Prediction and evaluation
reg_predictions = reg_model.predict(X_test)
print("Mean Squared Error:", mean_squared_error(y_test, reg_predictions))
