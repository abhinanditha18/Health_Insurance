import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.cluster import KMeans
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

# K-Means Clustering for Exploratory Analysis
kmeans = KMeans(n_clusters=4, random_state=42)  # You can change the number of clusters
X_preprocessed = preprocessor.fit_transform(X)
data['cluster'] = kmeans.fit_predict(X_preprocessed)

# Visualize Clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(x='age', y='bmi', hue='cluster', data=data, palette='viridis')
plt.title('Clusters in the Data')
plt.show()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the decision tree model
model_dt = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', DecisionTreeRegressor(random_state=42))
])

model_dt.fit(X_train, y_train)

# Make predictions and evaluate the decision tree model
y_pred_dt = model_dt.predict(X_test)

mae_dt = mean_absolute_error(y_test, y_pred_dt)
mse_dt = mean_squared_error(y_test, y_pred_dt)
r2_dt = r2_score(y_test, y_pred_dt)

print(f"Decision Tree Mean Absolute Error: {mae_dt}")
print(f"Decision Tree Mean Squared Error: {mse_dt}")
print(f"Decision Tree R-squared: {r2_dt}")
