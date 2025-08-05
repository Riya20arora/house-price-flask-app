import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import pickle

# Load dataset
df = pd.read_csv(r"C:\Users\jy7ky\Downloads\Real estate.csv")

# Drop ID and use relevant features
X = df[['house age', 'distance to the nearest MRT station', 
        'number of convenience stores', 'latitude', 'longitude']]
y = df['house price of unit area']

# Train model
model = RandomForestRegressor()
model.fit(X, y)

# Save model
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)