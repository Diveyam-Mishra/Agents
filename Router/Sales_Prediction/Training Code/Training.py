import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import pickle
import matplotlib.pyplot as plt
import joblib
import torch.nn as nn

filtered_data=pd.read_csv('pechmode_sales_data.csv', delimiter='|')
#Preprocessing Code needs to be added here

shade_data = filtered_data.groupby(['year', 'week', 'Shade','ADMSITE_CODE'], as_index=False).agg({
    'QTY': 'sum'
})
X = shade_data.drop(['Total_Sales', 'year'], axis=1)
y = shade_data['Total_Sales']
X = pd.get_dummies(X, columns=['Shade'])
X['week_sin'] = np.sin(2 * np.pi * X['week'] / 52)
X['week_cos'] = np.cos(2 * np.pi * X['week'] / 52)
X.drop('week', axis=1, inplace=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
shade_model = RandomForestRegressor(random_state=42)
shade_model.fit(X_train, y_train)
y_pred = shade_model.predict(X_test)
r2 = r2_score(y_test, y_pred)
efficiency = r2 * 100
print(f"Shade Model Efficiency: {efficiency:.2f}%")
with open("Regressor_shade_model.pkl", "wb") as f:
    pickle.dump(shade_model, f)
# Shade Model Efficiency: 82.82%
class NeuralNet(nn.Module):
    def __init__(self, input_size):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.fc3=nn.Linear(64,32)
        self.fc4 = nn.Linear(32, 1)
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
X_train_tensor, y_train_tensor = X_train_tensor.to(device), y_train_tensor.to(device)
X_test_tensor, y_test_tensor = X_test_tensor.to(device), y_test_tensor.to(device)

input_size = X_train_tensor.shape[1]
shade_model2 = NeuralNet(input_size)
shade_model2.to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(shade_model2.parameters(), lr=0.001)
epochs = 25 #42 is best 
for x in range (epochs):
    for epoch in range(x):
        shade_model2.train()
        for batch_X, batch_y in train_loader:
            predictions = shade_model2(batch_X)
            loss = criterion(predictions, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    shade_model2.eval()
    with torch.no_grad():
        y_pred = shade_model2(X_test_tensor)
        mse = criterion(y_pred, y_test_tensor).item()
        r2_score = 1 - mse / torch.var(y_test_tensor).item()
        efficiency = r2_score * 100
    print(f"Neural Network Model Efficiency of{x}] is: {efficiency:.2f}%")
torch.save(shade_model2.state_dict(), "NN_shade_model.pth")
print("Model saved successfully.")
joblib.dump(scaler, "shade_scaler.pkl")
df={}
filtered_data = df[(df["ENTTYPE"] == "SAL")]
filtered_data['ENTDT'] = pd.to_datetime(filtered_data['ENTDT'])
filtered_data['week'] = filtered_data['ENTDT'].dt.isocalendar().week
filtered_data['year'] = filtered_data['ENTDT'].dt.year
Category_data = filtered_data.groupby(['year', 'week', 'Category','ADMSITE_CODE'], as_index=False).agg({
    'QTY': 'sum'
})
Category_data.rename(columns={'QTY': 'Total_Sales', 'rsp': 'Avg_Price', 'Closing Stock': 'Avg_Closing_Stock'}, inplace=True)
X = Category_data.drop(['Total_Sales', 'year'], axis=1)
y = Category_data['Total_Sales']
X = pd.get_dummies(X, columns=['Category'])
X['week_sin'] = np.sin(2 * np.pi * X['week'] / 52)
X['week_cos'] = np.cos(2 * np.pi * X['week'] / 52)
X.drop('week', axis=1, inplace=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
X_train_tensor, y_train_tensor = X_train_tensor.to(device), y_train_tensor.to(device)
X_test_tensor, y_test_tensor = X_test_tensor.to(device), y_test_tensor.to(device)

input_size = X_train_tensor.shape[1]
Category_model = NeuralNet(input_size)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(Category_model.parameters(), lr=0.001)
epochs = 50
for x in range (24,epochs):
    for epoch in range(x):
        Category_model.train()
        for batch_X, batch_y in train_loader:
            predictions = Category_model(batch_X)
            loss = criterion(predictions, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    Category_model.eval()
    with torch.no_grad():
        y_pred = Category_model(X_test_tensor)
        mse = criterion(y_pred, y_test_tensor).item()
        r2_score = 1 - mse / torch.var(y_test_tensor).item()
        efficiency = r2_score * 100
    print(f"Neural Network Model Efficiency of is: {efficiency:.2f}%")
torch.save(Category_model.state_dict(), "NN_Category_model.pth")
print("Model saved successfully.")
joblib.dump(scaler, "Category_scaler.pkl")
filtered_data = df[(df["ENTTYPE"] == "SAL")]
filtered_data['ENTDT'] = pd.to_datetime(filtered_data['ENTDT'])
filtered_data['week'] = filtered_data['ENTDT'].dt.isocalendar().week
filtered_data['year'] = filtered_data['ENTDT'].dt.year
Material_Category_data = filtered_data.groupby(['year', 'week', 'Material_Category','ADMSITE_CODE'], as_index=False).agg({
    'QTY': 'sum'
})
Material_Category_data.rename(columns={'QTY': 'Total_Sales', 'rsp': 'Avg_Price', 'Closing Stock': 'Avg_Closing_Stock'}, inplace=True)
X = Material_Category_data.drop(['Total_Sales', 'year'], axis=1)
y = Material_Category_data['Total_Sales']
X = pd.get_dummies(X, columns=['Material_Category'])
X['week_sin'] = np.sin(2 * np.pi * X['week'] / 52)
X['week_cos'] = np.cos(2 * np.pi * X['week'] / 52)
X.drop('week', axis=1, inplace=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
input_size = X_train_tensor.shape[1]
Material_model = NeuralNet(input_size)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(Material_model.parameters(), lr=0.001)
epochs = 115
for epoch in range(epochs):
    Material_model.train() 
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        
        predictions = Material_model(batch_X)
        loss = criterion(predictions, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")
    Material_model.eval()
    with torch.no_grad():
        y_pred = Material_model(X_test_tensor)
        mse = criterion(y_pred, y_test_tensor).item()
        r2_score = 1 - mse / torch.var(y_test_tensor).item()
        efficiency = r2_score * 100
    print(f"Neural Network Model Efficiency: {efficiency:.2f}%")
torch.save(Material_model.state_dict(), "material_model.pth")
print("Model saved successfully.")
joblib.dump(scaler, "Material_scaler.pkl")