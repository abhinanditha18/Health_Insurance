import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
data = pd.read_csv('insurance.csv')

# Define features and target variable
X = data.drop('charges', axis=1)
y = data['charges']

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['age', 'bmi', 'children']),
        ('cat', OneHotEncoder(), ['sex', 'smoker', 'region'])
    ])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the linear regression model
model_lr = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

model_lr.fit(X_train, y_train)

# Make predictions and evaluate the linear regression model
y_pred_lr = model_lr.predict(X_test)

mae_lr = mean_absolute_error(y_test, y_pred_lr)
mse_lr = mean_squared_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)

print(f"Linear Regression Mean Absolute Error: {mae_lr}")
print(f"Linear Regression Mean Squared Error: {mse_lr}")
print(f"Linear Regression R-squared: {r2_lr}")

# Create and train the random forest model
model_rf = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

model_rf.fit(X_train, y_train)

# Make predictions and evaluate the random forest model
y_pred_rf = model_rf.predict(X_test)

mae_rf = mean_absolute_error(y_test, y_pred_rf)
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print(f"Random Forest Mean Absolute Error: {mae_rf}")
print(f"Random Forest Mean Squared Error: {mse_rf}")
print(f"Random Forest R-squared: {r2_rf}")

# Exploratory Data Analysis (EDA)
plt.figure(figsize=(18, 6))

# Histogram of charges
plt.subplot(1, 3, 1)
sns.histplot(data['charges'], kde=True)
plt.title('Distribution of Insurance Charges')

# Horizontal Bar Chart for average charges by sex
avg_charges_sex = data.groupby('sex')['charges'].mean().sort_values()
plt.subplot(1, 3, 2)
avg_charges_sex.plot(kind='barh', color='skyblue')
plt.title('Average Charges by Sex')
plt.xlabel('Average Charges')

# Pie chart for the proportion of smokers vs non-smokers
smoker_counts = data['smoker'].value_counts()
plt.subplot(1, 3, 3)
plt.pie(smoker_counts, labels=smoker_counts.index, autopct='%1.1f%%', colors=['#ff9999','#66b3ff'])
plt.title('Proportion of Smokers vs Non-Smokers')

plt.tight_layout()
plt.show()

# Pair plot to see relationships between numerical features and charges
sns.pairplot(data[['age', 'bmi', 'children', 'charges']])
plt.show()
