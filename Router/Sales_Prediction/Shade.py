# predict_shade.py
import pandas as pd
import numpy as np
import torch
import joblib
from sklearn.preprocessing import StandardScaler
from Sales_Prediction.data_loader import df
import torch.nn as nn

scaler = joblib.load(r"sm-forecast-engine\Scaler\shade_scaler.pkl")
class NeuralNet(nn.Module):
    def __init__(self, input_size):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x


df["week_sin"] = np.sin(2 * np.pi * df["week"] / 52)
df["week_cos"] = np.cos(2 * np.pi * df["week"] / 52)
df = pd.get_dummies(df, columns=["Shade"])
  
trained_features = joblib.load("trained_features.pkl")
trained_features = ["week_sin", "week_cos", "ADMSITE_CODE"] + [
    col for col in df.columns if "Shade" in col
]
X = df[trained_features]

# Scale the numerical data using the saved scaler
X_scaled = scaler.transform(X)

# Convert the scaled data into a tensor for prediction
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

# Recreate the model architecture
input_size = X_tensor.shape[1]  # Number of input features
model = NeuralNet(input_size)

# Load the state dictionary into the model
model.load_state_dict(torch.load(r"sm-forecast-engine\NN Model\NN_shade_model.pth"))

# Set the model to evaluation mode
model.eval()

# Predict using the model
with torch.no_grad():
    predictions = model(X_tensor).numpy()  # Predictions as NumPy array

# Add predictions to the DataFrame
df["Predicted_Total_Sales"] = predictions.flatten()  # Flatten predictions to 1D
print(df)
