import pandas as pd
import numpy as np
from datetime import datetime
import holidays
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error
import shap
from Training import df
df = df.rename(columns={'QTY': 'Sale_QTY'})
df=df[df['Sale_QTY']<2]
class ClothingSalesPredictor:
    def __init__(self):
        self.label_encoders = {}
        self.model = None
        self.feature_names = None
        
    def _create_time_features(self, df):
        df['date'] = pd.to_datetime(df['ENTDT'])
        df['month'] = df['date'].dt.month
        df['day_of_week'] = df['date'].dt.dayofweek
        df['quarter'] = df['date'].dt.quarter
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['season'] = pd.cut(df['month'], 
                            bins=[0, 2, 5, 8, 11, 12], 
                            labels=['Winter', 'Spring', 'Summer', 'Fall', 'Winter'],
                             right=False,
                             ordered=False)
        in_holidays = holidays.India()
        df['is_holiday'] = df['date'].apply(lambda x: x in in_holidays).astype(int)
        
        return df
    
    def _create_price_features(self, df):
        df['price_bracket'] = pd.qcut(df['rsp'], q=5, labels=['Budget', 'Value', 'Mid', 'Premium', 'Luxury'],duplicates='drop')
        print(df['price_bracket'])
        print(df['price_bracket'].nunique())
        category_avg_price = df.groupby('Category')['rsp'].transform('mean')
        print(category_avg_price)
        df['price_vs_category_avg'] = df['rsp'] / category_avg_price
        return df
    
    def _create_product_features(self, df):
        df['material_shade'] = df['Material_Category'] + '_' + df['Shade']
        df['category_popularity'] = df.groupby('Category')['Sale_QTY'].transform('sum')
        df['shade_popularity'] = df.groupby('Shade')['Sale_QTY'].transform('sum')
        df['material_popularity'] = df.groupby('Material_Category')['Sale_QTY'].transform('sum')
        
        return df
    
    def _encode_categorical_features(self, df, train=True):
        categorical_columns = ['Category', 'Shade', 'Material_Category', 
                             'season', 'material_shade']
        
        if train:
            for col in categorical_columns:
                self.label_encoders[col] = LabelEncoder()
                df[col + '_encoded'] = self.label_encoders[col].fit_transform(df[col])
        else:
            for col in categorical_columns:
                unique_labels = set(self.label_encoders[col].classes_)
                df[col] = df[col].map(lambda x: list(unique_labels)[0] if x not in unique_labels else x)
                df[col + '_encoded'] = self.label_encoders[col].transform(df[col])
                
        return df
    
    def prepare_features(self, df, train=True):
        """Prepare all features"""
        df = self._create_time_features(df)
        # df = self._create_price_features(df)
        df = self._create_product_features(df)
        df = self._encode_categorical_features(df, train)
        self.feature_names = [
            'month', 'day_of_week', 'quarter', 'is_weekend', 'is_holiday',
            'rsp',
            'category_popularity', 'shade_popularity', 'material_popularity',
            'Category_encoded', 'Shade_encoded', 'Material_Category_encoded',
            'season_encoded', 'material_shade_encoded'
        ]
        
        return df
    
    def train(self, df):
        df = self.prepare_features(df, train=True)
        X = df[self.feature_names]
        y = df['Sale_QTY']
        
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        params = {
            'objective': 'regression',
            'metric': 'mae',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': 0
        }
        
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_val, label=y_val)
        
        self.model = lgb.train(
            params,
            train_data,
            valid_sets=[valid_data],
            num_boost_round=1000,
            callbacks=[lgb.early_stopping(stopping_rounds=50)]
        )
        
        return self
    
    def predict(self, df):
        
        df = self.prepare_features(df, train=False)
        return self.model.predict(df[self.feature_names])
    
    def analyze_feature_importance(self):
        """Analyze feature importance using SHAP values"""
        import shap
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(pd.DataFrame(columns=self.feature_names))
        
        # Get feature importance
        importance_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': np.abs(shap_values).mean(0)
        })
        
        return importance_df.sort_values('Importance', ascending=False)