import numpy as np 
# import os
# import xlrd
import matplotlib.pyplot as plt 
import pandas as pd 
from sklearn.metrics import mean_squared_error 
from sklearn.ensemble import RandomForestRegressor 
from sklearn.metrics import r2_score 
import torch 
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import pickle
import matplotlib.pyplot as plt
import joblib
# value1 = input("Enter ADMSITE_CODE: ")
# value2 = input("Enter hl3name: ")
# value3 = input("Enter week: ")
# value4 = input("Enter cname4: ")
# value5= input("Enter cname5: ")
# df = pd.DataFrame({
#     'ADMSITE_CODE': [value1],
#     'hl3name': [value2],
#     'cname4':[value4],
#     'week': [value3],
#     'cname5':[value5]
# })
df=pd.DataFrame({'ADMSITE_CODE':['2'],'hl3name':['Dress'],'cname4':['White'],'week':[1],'cname5':['Net']})
# predictions = Regressor_shade_model.predict(X_test)
categories = {
    'Western': ['Dress', 'Dress Material', 'Gown', 'Top', 'Top Bottom Set'],
    'Daily Wear Indian': ['Kaftan', 'Tunic', 'Salwar Suit', 'Dupatta', 'Kurti', 'Night Suit'],
    'Festive Wear Indian': ['Saree', 'Lehengas'],
    'Dump': ['Store items']
}

def map_category(hl3name):
    for category, items in categories.items():
        if hl3name in items:
            return category
    return 'Dump'


def map_shade(color):
    if color in ['Beige', 'Brown', 'Cream', 'Chikoo', 'Offwhite', 'Grey']:
        return 'Neutral'
    elif color in ['White', 'Cream & Green', 'Cream & Wine', 'Peach', 'Lavender', 'Mauve']:
        return 'Pastel'
    elif color in ['Black', 'Navy Blue', 'Maroon', 'Burgundy', 'RUST']:
        return 'Dark'
    elif color in ['Pink', 'Magenta', 'Wine', 'Maroon', 'Peach']:
        return 'RedPink'
    elif color in ['Blue', 'Navy Blue', 'Teal', 'Turquoise', 'Green', 'Olive', 'Rama Green', 'Firozi']:
        return 'BlueGreen'
    elif color in ['Yellow', 'Gold', 'Mustard','Orange', 'Rust']:
        return 'YellowGold'
    elif color in ['Multicolor']:
        return 'Multicolor'
    else:
        return 'RedPink'

# def map_season(month):
#     if month in [5, 6, 7]:
#         return 'Summer'
#     elif month in [8]:
#         return 'Rainy'
#     elif month in [9, 10, 11]:
#         return 'Festival'
#     elif month in [12, 1, 2]:
#         return 'Winter'
#     elif month in [3, 4]:
#         return 'Spring'
# df['season'] = df['ENTDT'].dt.month.apply(map_season)

def map_material(material):
    if material in ['Cotton', 'Muslin', 'Pure Cotton', 'Mulmul Cotton', 'Cotton Silk', 'Cotton Blend', 'Giza Cotton']:
        return 'Natural-Cotton'
    elif material in ['Pure Silk', 'Banarasi Silk', 'Kanjivaram Silk', 'Tussar Silk', 'Assam Silk', 'Raw Silk', 
                      'Vichitra Silk', 'Chanderi Silk', 'Bhagalpuri Silk', 'Satin Silk', 'Paper Silk', 'Tapetta Silk', 
                      'Art Silk', 'Jacquard', 'Tissue Silk', 'Banglori Silk', 'Kanjivaran Silk']:
        return 'Natural-Silk'
    elif material in ['Rayon', 'Modal']:
        return 'Synthetic-Rayon'
    elif material in ['Poly Cotton']:
        return 'Synthetic-Polyester'
    elif material in ['Satin', 'Crepe']:
        return 'Synthetic-SatinCrepe'
    elif material in ['Net', 'Organza', 'Oraganza', 'Kota Orgnaza', 'Kota Doria']:
        return 'Synthetic-NetOrganza'
    elif material in ['Lycra']:
        return 'Synthetic-Lycra'
    elif material in ['Georgette', 'Chiffon', 'Chinon', 'Mulmul Cotton', 'Tissue', 'Brasso', 'Dola Silk']:
        return 'Lightweight'
    elif material in ['Banarasi Silk', 'Kanjivaram Silk', 'Chanderi Silk', 'Jacquard', 'Khadi Silk', 'Tussar Silk', 
                      'Bhagalpuri Silk', 'Vichitra Silk', 'Banglori Silk']:
        return 'Heavy-Traditional'
    elif material in ['Khadi', 'Khaadi', 'Chanderi', 'Kota Doria', 'Assam Silk', 'Bhagalpuri Silk']:
        return 'Handloom'
    elif material in ['Cottton Blend', 'Art Silk', 'Poly Cotton']:
        return 'Blended'
    else:
        return 'Other'
df['Material_Category'] = df['cname5'].apply(map_material)
df['Category'] = df['hl3name'].apply(map_category)
df['Shade'] = df['cname4'].apply(map_shade)
df = df.drop(['cname5','hl3name','cname4'], axis=1)
print(df)