
from sklearn.ensemble import RandomForestRegressor

# Model
reg = RandomForestRegressor(random_state=42)
reg.fit(X_train, y_train)

# Predict and evaluate using a regression metric
predictions = reg.predict(X_test)
print("Mean Squared Error:", mean_squared_error(y_test, predictions))
