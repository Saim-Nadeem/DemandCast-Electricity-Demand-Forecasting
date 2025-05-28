import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import xgboost as xgb
import warnings
import joblib
import os
import time
warnings.filterwarnings('ignore')

def load_and_prepare_data():
    """Load and prepare data for predictive modeling"""
    print("\nLoading and preparing data...")
    
    # Load the preprocessed data
    df = pd.read_csv('preprocessed_data2.csv')
    print(f"Initial data shape: {df.shape}")
    
    # Convert day_of_week to numeric if it's string
    day_mapping = {
        'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3,
        'Friday': 4, 'Saturday': 5, 'Sunday': 6
    }
    if 'day_of_week' in df.columns and df['day_of_week'].dtype == 'object':
        df['day_of_week_num'] = df['day_of_week'].map(day_mapping)
    
    # Convert month to numeric if it's string
    month_mapping = {
        'January': 1, 'February': 2, 'March': 3, 'April': 4,
        'May': 5, 'June': 6, 'July': 7, 'August': 8,
        'September': 9, 'October': 10, 'November': 11, 'December': 12
    }
    if 'month' in df.columns and df['month'].dtype == 'object':
        df['month_num'] = df['month'].map(month_mapping)
    
    # Create weekend feature
    if 'day_of_week_num' in df.columns:
        df['is_weekend'] = df['day_of_week_num'].isin([5, 6]).astype(int)
    
    # Add more engineered features for better predictions
    if 'hour' in df.columns:
        # Hour of day as cyclical features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24.0)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24.0)
    
    if 'day_of_week_num' in df.columns:
        # Day of week as cyclical features
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week_num'] / 7.0)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week_num'] / 7.0)
    
    if 'month_num' in df.columns:
        # Month as cyclical feature
        df['month_sin'] = np.sin(2 * np.pi * df['month_num'] / 12.0)
        df['month_cos'] = np.cos(2 * np.pi * df['month_num'] / 12.0)
    
    # Create lagged features for demand
    if 'demand' in df.columns and 'hour' in df.columns:
        df['demand_lag24'] = df.groupby(['city', 'hour'])['demand'].shift(24)  # Previous day same hour
        df['demand_lag48'] = df.groupby(['city', 'hour'])['demand'].shift(48)  # Two days ago same hour
        df['demand_lag168'] = df.groupby(['city', 'hour'])['demand'].shift(168)  # One week ago same hour
        
        # Create rolling mean features with city grouping
        df['demand_rolling_mean_24h'] = df.groupby('city')['demand'].rolling(window=24).mean().reset_index(level=0, drop=True)
        df['demand_rolling_mean_7d'] = df.groupby('city')['demand'].rolling(window=168).mean().reset_index(level=0, drop=True)
        
        # Create city-specific statistics
        df['city_demand_mean'] = df.groupby('city')['demand'].transform('mean')
        df['city_demand_std'] = df.groupby('city')['demand'].transform('std')
        
        # Normalize demand within each city
        df['demand_normalized'] = (df['demand'] - df['city_demand_mean']) / df['city_demand_std']
        
        # Create target variable (next day's demand)
        df['target'] = df.groupby(['city', 'hour'])['demand'].shift(-24)
        
        # Create normalized target (for training city-agnostic models)
        df['target_normalized'] = (df['target'] - df['city_demand_mean']) / df['city_demand_std']
    
    # Drop rows with NaN values
    df_before = df.shape[0]
    df = df.dropna()
    print(f"Number of rows dropped due to NaN: {df_before - df.shape[0]}")
    
    return df

def prepare_features(df, city_specific=False):
    """Prepare features for modeling with additional feature engineering"""
    print("\nPreparing features for modeling...")
    print(f"City-specific modeling: {city_specific}")
    
    # Get unique cities for city-specific modeling
    cities = df['city'].unique()
    print(f"Found {len(cities)} unique cities: {cities}")
    
    # Define required features
    required_features = [
        'temperature', 'humidity', 'pressure', 'windSpeed', 'cloudCover', 
        'demand', 'hour', 'month_num', 'day_of_week_num', 'is_weekend',
        'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos',
        'demand_lag24', 'demand_lag48', 'demand_lag168',
        'demand_rolling_mean_24h', 'demand_rolling_mean_7d',
        'demand_normalized', 'city_demand_mean', 'city_demand_std'
    ]
    
    # Check if all required features are present
    missing_features = [f for f in required_features if f not in df.columns]
    if missing_features:
        raise ValueError(f"Missing required features: {missing_features}")
    
    # Separate numerical and categorical columns
    numerical_features = required_features
    categorical_features = ['season']
    
    # If we're not doing city-specific models, include city as a categorical feature
    if not city_specific:
        categorical_features.append('city')
    
    print(f"Number of numerical features: {len(numerical_features)}")
    print(f"Number of categorical features: {len(categorical_features)}")
    
    if city_specific:
        # For city-specific approach, prepare separate models for each city
        city_preprocessors = {}
        city_X_y_data = {}
        
        for city in cities:
            print(f"\nProcessing city: {city}")
            city_df = df[df['city'] == city].copy()
            
            # Create preprocessing pipelines for this city
            numeric_transformer = Pipeline(steps=[
                ('scaler', StandardScaler())
            ])
            
            categorical_transformer = Pipeline(steps=[
                ('onehot', OneHotEncoder(drop='first', sparse_output=False))
            ])
            
            # Combine preprocessing steps
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numeric_transformer, numerical_features),
                    ('cat', categorical_transformer, categorical_features)
                ])
            
            # Prepare feature matrix X and target vector y
            city_X = city_df[numerical_features + categorical_features]
            
            # Choose between regular target or normalized target
            city_y = city_df['target']  # Using regular target for city-specific models
            
            print(f"City {city} feature matrix shape before transformation: {city_X.shape}")
            
            # Fit and transform the data
            city_X_transformed = preprocessor.fit_transform(city_X)
            
            # Get feature names after transformation
            feature_names = []
            
            # Add numerical feature names
            feature_names.extend(numerical_features)
            
            # Add categorical feature names
            for i, feature in enumerate(categorical_features):
                if hasattr(preprocessor.named_transformers_['cat'].named_steps['onehot'], 'categories_'):
                    cats = preprocessor.named_transformers_['cat'].named_steps['onehot'].categories_[i][1:]
                    feature_names.extend([f"{feature}_{cat}" for cat in cats])
            
            print(f"City {city} feature matrix shape after transformation: {city_X_transformed.shape}")
            
            # Store preprocessor and data
            city_preprocessors[city] = {
                'preprocessor': preprocessor,
                'feature_cols': feature_names,
                'numerical_features': numerical_features,
                'categorical_features': categorical_features,
                'city_mean': city_df['city_demand_mean'].mean(),
                'city_std': city_df['city_demand_std'].mean()
            }
            
            city_X_y_data[city] = (city_X_transformed, city_y)
        
        return city_preprocessors, city_X_y_data
    
    else:
        # Global model approach
        # Create preprocessing pipelines for numerical and categorical data
        numeric_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(drop='first', sparse_output=False))
        ])
        
        # Combine preprocessing steps
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numerical_features),
                ('cat', categorical_transformer, categorical_features)
            ])
        
        # Prepare feature matrix X and target vector y
        X = df[numerical_features + categorical_features]
        
        # Choose between regular target or normalized target
        # Use normalized target for global model to handle different city scales
        y = df['target_normalized']  
        
        print(f"Feature matrix shape before transformation: {X.shape}")
        
        # Fit and transform the data
        X_transformed = preprocessor.fit_transform(X)
        
        # Get feature names after transformation
        feature_names = []
        
        # Add numerical feature names
        feature_names.extend(numerical_features)
        
        # Add categorical feature names
        for i, feature in enumerate(categorical_features):
            if hasattr(preprocessor.named_transformers_['cat'].named_steps['onehot'], 'categories_'):
                cats = preprocessor.named_transformers_['cat'].named_steps['onehot'].categories_[i][1:]
                feature_names.extend([f"{feature}_{cat}" for cat in cats])
        
        print(f"Feature matrix shape after transformation: {X_transformed.shape}")
        
        # Save the preprocessor components
        preprocessor_components = {
            'feature_cols': feature_names,
            'numerical_features': numerical_features,
            'categorical_features': categorical_features,
            'preprocessor': preprocessor,
            'city_stats': df.groupby('city')[['city_demand_mean', 'city_demand_std']].mean().to_dict('index')
        }
        
        return preprocessor_components, (X_transformed, y)

def train_test_validation_split(X, y, test_size=0.2, val_size=0.1):
    """Split data into train, validation, and test sets, maintaining chronological order"""
    # First split into temp and test
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)
    
    # Then split temp into train and validation
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=val_size_adjusted, shuffle=False)
    
    print(f"Train set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def train_linear_models(X_train, y_train):
    """Train linear regression models"""
    print("Training linear models...")
    # Linear Regression
    lr = LinearRegression(n_jobs=4)
    lr.fit(X_train, y_train)
    
    # Ridge Regression with fixed alpha
    ridge = Ridge(alpha=10.0)
    ridge.fit(X_train, y_train)
    
    return lr, ridge

def train_tree_models(X_train, y_train):
    """Train tree-based models with CPU"""
    print("Training Random Forest...")
    # Random Forest
    rf = RandomForestRegressor(
        n_estimators=300,
        max_depth=20,
        min_samples_split=5,
        random_state=42,
        n_jobs=8
    )
    rf.fit(X_train, y_train)
    
    print("Training XGBoost...")
    # XGBoost with CPU
    params = {
        'n_estimators': 2000,
        'max_depth': 8,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'n_jobs': 8,
        'tree_method': 'hist'  # Use CPU-based histogram method
    }
    
    xgb_model = xgb.XGBRegressor(**params)
    xgb_model.fit(
        X_train, 
        y_train,
        eval_set=[(X_train[:1000], y_train[:1000])]
    )
    
    return rf, xgb_model

def evaluate_models(models, X_test, y_test, city_stats=None, normalized_target=False):
    """Evaluate models using multiple metrics"""
    results = {}
    
    for name, model in models.items():
        try:
            y_pred = model.predict(X_test)
            
            # If we're using normalized targets, denormalize predictions
            if normalized_target and city_stats is not None:
                # We need to know which city each prediction belongs to
                # This should be provided alongside X_test in a practical implementation
                # Here I'll show the concept assuming we have city information
                city_means = np.array([city_stats[city]['city_demand_mean'] for city in test_cities])
                city_stds = np.array([city_stats[city]['city_demand_std'] for city in test_cities])
                
                # Denormalize predictions
                y_pred = y_pred * city_stds + city_means
                
                # Similarly for true values if they're normalized
                y_true = y_test * city_stds + city_means
            else:
                y_true = y_test
            
            results[name] = {
                'MAE': mean_absolute_error(y_true, y_pred),
                'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
                'MAPE': mean_absolute_percentage_error(y_true, y_pred) * 100
            }
        except Exception as e:
            print(f"Error evaluating {name} model: {str(e)}")
            results[name] = {
                'MAE': float('nan'),
                'RMSE': float('nan'),
                'MAPE': float('nan')
            }
    
    return results

def save_models(models, preprocessor_components, city_specific=False):
    """Save trained models and preprocessing components"""
    # Create models directory if it doesn't exist
    models_dir = 'models_city_specific' if city_specific else 'models'
    os.makedirs(models_dir, exist_ok=True)
    
    if city_specific:
        # Save city-specific models
        for city, city_models in models.items():
            city_dir = os.path.join(models_dir, city)
            os.makedirs(city_dir, exist_ok=True)
            
            # Save each model for this city
            for name, model in city_models.items():
                joblib.dump(model, os.path.join(city_dir, f"{name.lower().replace(' ', '_')}.joblib"))
            
            # Save preprocessor components for this city
            joblib.dump(preprocessor_components[city], os.path.join(city_dir, 'preprocessor_components.joblib'))
    else:
        # Save global models
        for name, model in models.items():
            joblib.dump(model, os.path.join(models_dir, f"{name.lower().replace(' ', '_')}.joblib"))
        
        # Save global preprocessor components
        joblib.dump(preprocessor_components, os.path.join(models_dir, 'preprocessor_components.joblib'))

def train_city_specific_models(city_preprocessors, city_X_y_data):
    """Train separate models for each city"""
    city_models = {}
    
    for city, (X, y) in city_X_y_data.items():
        print(f"\nTraining models for city: {city}")
        
        # Split data for this city
        X_train, X_val, X_test, y_train, y_val, y_test = train_test_validation_split(X, y)
        
        # Train models for this city
        print(f"Training linear models for {city}...")
        lr, ridge = train_linear_models(X_train, y_train)
        
        print(f"Training tree models for {city}...")
        rf, xgb_model = train_tree_models(X_train, y_train)
        
        # Store models for this city
        city_models[city] = {
            'Linear Regression': lr,
            'Ridge Regression': ridge,
            'Random Forest': rf,
            'XGBoost': xgb_model
        }
        
        # Evaluate on validation set
        print(f"\nEvaluating models for {city} on validation set...")
        val_results = evaluate_models(city_models[city], X_val, y_val)
        
        # Print validation results
        print(f"\nValidation Results for {city}:")
        for model_name, metrics in val_results.items():
            print(f"{model_name}:")
            for metric_name, value in metrics.items():
                print(f"  {metric_name}: {value:.4f}")
        
        # Evaluate on test set
        print(f"\nEvaluating models for {city} on test set...")
        test_results = evaluate_models(city_models[city], X_test, y_test)
        
        # Print test results
        print(f"\nTest Results for {city}:")
        for model_name, metrics in test_results.items():
            print(f"{model_name}:")
            for metric_name, value in metrics.items():
                print(f"  {metric_name}: {value:.4f}")
    
    return city_models

def load_models(city_specific=False, city=None):
    """Load trained models and preprocessing components
    
    Args:
        city_specific (bool): Whether to load city-specific models
        city (str, optional): City name if loading city-specific models
        
    Returns:
        tuple: (models_dict, preprocessor_components)
    """
    models_dir = 'models_city_specific' if city_specific else 'models'
    
    if not os.path.exists(models_dir):
        raise FileNotFoundError(f"Models directory {models_dir} not found")
    
    if city_specific:
        if city is None:
            raise ValueError("City name must be provided when loading city-specific models")
            
        city_dir = os.path.join(models_dir, city)
        if not os.path.exists(city_dir):
            raise FileNotFoundError(f"Directory for city {city} not found in {models_dir}")
        
        # Load preprocessor components
        preprocessor_components = joblib.load(os.path.join(city_dir, 'preprocessor_components.joblib'))
        
        # Load models
        models = {}
        model_files = {
            'Linear Regression': 'linear_regression.joblib',
            'Ridge Regression': 'ridge_regression.joblib',
            'Random Forest': 'random_forest.joblib',
            'XGBoost': 'xgboost.joblib'
        }
        
        for model_name, file_name in model_files.items():
            model_path = os.path.join(city_dir, file_name)
            if os.path.exists(model_path):
                models[model_name] = joblib.load(model_path)
            else:
                print(f"Warning: Model file {file_name} not found for city {city}")
    
    else:
        # Load global preprocessor components
        preprocessor_components = joblib.load(os.path.join(models_dir, 'preprocessor_components.joblib'))
        
        # Load global models
        models = {}
        model_files = {
            'Linear Regression': 'linear_regression.joblib',
            'Ridge Regression': 'ridge_regression.joblib',
            'Random Forest': 'random_forest.joblib',
            'XGBoost': 'xgboost.joblib'
        }
        
        for model_name, file_name in model_files.items():
            model_path = os.path.join(models_dir, file_name)
            if os.path.exists(model_path):
                models[model_name] = joblib.load(model_path)
            else:
                print(f"Warning: Model file {file_name} not found")
    
    return models, preprocessor_components

def main():
    # Start timer
    start_time = time.time()
    
    # Define which approach to use: 'global', 'city_specific', or 'both'
    approach = 'both'  
    
    try:
        # Load and prepare data
        data = load_and_prepare_data()
        
        # Approach 1: Global model
        if approach in ['global', 'both']:
            print("\n=== Training Global Model ===")
            
            # Prepare features for global model
            preprocessor_components, (X, y) = prepare_features(data, city_specific=False)
            
            # Split data
            X_train, X_val, X_test, y_train, y_val, y_test = train_test_validation_split(X, y)
            
            # Train models
            print("\nTraining global linear models...")
            lr, ridge = train_linear_models(X_train, y_train)
            
            print("\nTraining global tree models...")
            rf, xgb_model = train_tree_models(X_train, y_train)
            
            # Store models
            global_models = {
                'Linear Regression': lr,
                'Ridge Regression': ridge,
                'Random Forest': rf,
                'XGBoost': xgb_model
            }
            
            # Save models
            print("\nSaving global models...")
            save_models(global_models, preprocessor_components, city_specific=False)
            
            
        # Approach 2: City-specific models
        if approach in ['city_specific', 'both']:
            print("\n=== Training City-Specific Models ===")
            
            # Prepare features for city-specific models
            city_preprocessors, city_X_y_data = prepare_features(data, city_specific=True)
            
            # Train city-specific models
            city_models = train_city_specific_models(city_preprocessors, city_X_y_data)
            
            # Save city-specific models
            print("\nSaving city-specific models...")
            save_models(city_models, city_preprocessors, city_specific=True)
        
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        import traceback
        traceback.print_exc()
    
    # Print total time
    print(f"\nTotal execution time: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()