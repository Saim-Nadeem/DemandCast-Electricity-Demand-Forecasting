import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
import joblib
import traceback
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import sys
from plotly.subplots import make_subplots

# Disable Streamlit's file watcher for PyTorch modules
os.environ['STREAMLIT_SERVER_WATCH_DIRS'] = 'false'

# Add local directory to path to make imports work
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the model loading functions
from predictive_modeling import load_and_prepare_data, prepare_features, load_models

# Page configuration
st.set_page_config(
    page_title="Demand Forecasting App",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS for styling
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
        padding: 1rem;
    }
    .stApp {
        max-width: 100%;
        margin: 0 auto;
    }
    h1, h2, h3 {
        color: #2c3e50;
    }
    .stButton button {
        background-color: #4285f4;
        color: white;
    }
    .stSelectbox, .stDateInput {
        margin-bottom: 15px;
    }
    .plot-container {
        width: 100%;
        height: 100%;
    }
    .stPlotlyChart {
        width: 100%;
        height: 100%;
    }
    </style>
    """, unsafe_allow_html=True)

# Global variables to store models and data
@st.cache_resource
def initialize_models():
    """Initialize models and load data"""
    try:
        st.write("Loading saved models...")
        models, preprocessor_components = load_models()  # Properly unpack the tuple
        if models is None:
            st.error("Failed to load models. Please run predictive_modeling.py first to train and save the models.")
            return None, None, None
        
        st.write("Loading data...")
        data = load_and_prepare_data()
        
        st.success("Models loaded successfully")
        return models, data, preprocessor_components
    except Exception as e:
        st.error(f"Error initializing models: {str(e)}")
        traceback.print_exc()  # Add this to see the full error trace
        return None, None, None

# Initialize models and data
models, data, preprocessor_components = initialize_models()

# Define page structure with sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Forecast", "Cluster Analysis", "Demand Analysis", "Model Information"])

# Title of the app
st.title("Demand Forecasting Application")

if page == "Forecast":
    st.header("Demand Forecast")
    
    # Sidebar inputs
    st.sidebar.header("Forecast Parameters")
    
    # Get available cities
    if data is not None:
        cities = data['city'].unique().tolist()
        
        # City selection
        city = st.sidebar.selectbox("Select City", cities)
        
        # Date range selection
        min_date = pd.to_datetime(data['time'].min())
        max_date = pd.to_datetime(data['time'].max())
        
        date_range = st.date_input(
            "Select Date Range",
            value=(min_date.date(), max_date.date()),
            min_value=min_date.date(),
            max_value=max_date.date()
        )
        
        if len(date_range) == 2:
            start_date = pd.to_datetime(date_range[0])
            end_date = pd.to_datetime(date_range[1])
        else:
            start_date = min_date
            end_date = max_date
        
        # Model selection
        available_models = list(models.keys())
        selected_models = st.multiselect("Select Models for Comparison", 
                                        available_models,
                                        default=available_models[:2] if len(available_models) >= 2 else available_models)
        
        # Advanced options
        with st.expander("Advanced Options"):
            show_feature_importance = st.checkbox("Show Feature Importance", True)
            show_error_analysis = st.checkbox("Show Error Analysis", True)
            
        st.markdown("---")
        st.write("Developed using models from predictive_modeling.py")
        
        # Generate forecast button
        if st.sidebar.button("Generate Forecast"):
            if not selected_models:
                st.warning("Please select at least one model to generate forecasts.")
            else:
                st.write(f"Generating forecast for {city} from {start_date} to {end_date}")
                
                # Filter data for selected city and date range
                mask = (data['city'] == city) & (pd.to_datetime(data['time']) >= start_date) & (pd.to_datetime(data['time']) <= end_date)
                filtered_data = data[mask]
                
                if len(filtered_data) == 0:
                    st.error('No data available for selected parameters')
                else:
                    try:
                        with st.spinner("Calculating predictions..."):
                            # Get the original feature columns before transformation
                            numerical_features = preprocessor_components['numerical_features']
                            categorical_features = preprocessor_components['categorical_features']
                            
                            # Prepare the input data with original features
                            X = filtered_data[numerical_features + categorical_features]
                            
                            # Transform features using the preprocessor
                            X_transformed = preprocessor_components['preprocessor'].transform(X)
                            
                            # Get predictions from selected models
                            predictions = {}
                            model_status = {}
                            
                            # Get city statistics for denormalization
                            city_mean = filtered_data['city_demand_mean'].iloc[0]
                            city_std = filtered_data['city_demand_std'].iloc[0]
                            
                            print(f"Debug - City stats for {city}: mean={city_mean:.2f}, std={city_std:.2f}")
                            print(f"Debug - Raw demand range: {filtered_data['demand'].min():.2f}-{filtered_data['demand'].max():.2f}")
                            
                            for model_name in selected_models:
                                if model_name in models:
                                    try:
                                        if model_name == 'Ensemble':
                                            ensemble_predictions = []
                                            used_models = []
                                            for submodel_name, submodel in models.items():
                                                if (
                                                    submodel_name not in ['Ensemble']
                                                    and hasattr(submodel, 'predict')
                                                    and callable(getattr(submodel, 'predict', None))
                                                ):
                                                    try:
                                                        if hasattr(submodel, 'n_features_in_'):
                                                            expected_features = submodel.n_features_in_
                                                            if X_transformed.shape[1] != expected_features:
                                                                X_model = X_transformed[:, :expected_features]
                                                            else:
                                                                X_model = X_transformed
                                                        else:
                                                            X_model = X_transformed
                                                        pred_sub = submodel.predict(X_model)
                                                        # Denormalize predictions
                                                        pred_sub = pred_sub * city_std + city_mean
                                                        ensemble_predictions.append(pred_sub)
                                                        used_models.append(submodel_name)
                                                    except Exception as sub_e:
                                                        st.warning(f"Skipping {submodel_name} in ensemble due to error: {sub_e}")
                                            if ensemble_predictions:
                                                pred = np.mean(ensemble_predictions, axis=0)
                                                pred = np.array(pred)
                                                if pred.size == 0 or np.all(np.isnan(pred)):
                                                    model_status[model_name] = 'Ensemble returned all NaN or empty predictions.'
                                                else:
                                                    model_status[model_name] = f'Ensemble used models: {used_models}'
                                                predictions[model_name] = pred.tolist()
                                            else:
                                                pred = np.full(len(filtered_data), np.nan)
                                                model_status[model_name] = 'No valid submodels for ensemble prediction.'
                                                predictions[model_name] = pred.tolist()
                                        elif model_name == 'XGBoost':
                                            # Set device to CPU for prediction to avoid GPU issues
                                            if hasattr(models[model_name], 'set_params'):
                                                models[model_name].set_params(device='cpu')
                                            
                                            if hasattr(models[model_name], 'n_features_in_'):
                                                expected_features = models[model_name].n_features_in_
                                                if X_transformed.shape[1] != expected_features:
                                                    X_model = X_transformed[:, :expected_features]
                                                else:
                                                    X_model = X_transformed
                                            else:
                                                X_model = X_transformed
                                            
                                            try:
                                                # Make prediction
                                                pred = models[model_name].predict(X_model)
                                                # Denormalize predictions
                                                pred = pred * city_std + city_mean
                                                pred = np.array(pred)
                                                if pred.size == 0 or np.all(np.isnan(pred)):
                                                    model_status[model_name] = 'XGBoost returned all NaN or empty predictions.'
                                                else:
                                                    model_status[model_name] = 'ok'
                                                predictions[model_name] = pred.tolist()
                                            except Exception as e:
                                                st.error(f"Error in XGBoost prediction: {str(e)}")
                                                model_status[model_name] = f'Error: {str(e)}'
                                                predictions[model_name] = np.full(len(filtered_data), np.nan).tolist()
                                        else:
                                            # For other models (Linear Regression, Ridge, Random Forest)
                                            if hasattr(models[model_name], 'n_features_in_'):
                                                expected_features = models[model_name].n_features_in_
                                                if X_transformed.shape[1] != expected_features:
                                                    X_model = X_transformed[:, :expected_features]
                                                else:
                                                    X_model = X_transformed
                                            else:
                                                X_model = X_transformed
                                            
                                            pred = models[model_name].predict(X_model)
                                            # Denormalize predictions
                                            pred = pred * city_std + city_mean
                                            pred = np.array(pred)
                                            if pred.size == 0 or np.all(np.isnan(pred)):
                                                model_status[model_name] = f'{model_name} returned all NaN or empty predictions.'
                                            else:
                                                model_status[model_name] = 'ok'
                                            predictions[model_name] = pred.tolist()
                                            
                                            # Debug print for predictions
                                            print(f"Debug - {model_name} predictions range: {pred.min():.2f}-{pred.max():.2f}")
                                            
                                    except Exception as e:
                                        st.error(f"Error in {model_name} prediction: {str(e)}")
                                        model_status[model_name] = f'Error: {str(e)}'
                                        predictions[model_name] = np.full(len(filtered_data), np.nan).tolist()
                            
                            # Create a DataFrame with predictions
                            results_df = pd.DataFrame({
                                'time': filtered_data['time'],
                                'actual': filtered_data['demand']
                            })
                            
                            # Add predictions from each model
                            for model_name in selected_models:
                                if model_name in predictions:
                                    results_df[f'{model_name}_pred'] = predictions[model_name]
                            
                            # Display model status
                            st.subheader("Model Status")
                            for model_name, status in model_status.items():
                                if status == 'ok':
                                    st.success(f"{model_name}: {status}")
                                else:
                                    st.warning(f"{model_name}: {status}")
                            
                            # Plot predictions
                            st.subheader("Forecast Results")
                            
                            # Use the same color palette as cluster analysis for model predictions
                            cluster_colors = [
                                "#4db6ac",  # Teal (softer)
                                "#81c784",  # Light Green
                                "#ffd54f",  # Soft Yellow
                                "#64b5f6",  # Light Blue
                                "#ffb74d",  # Light Orange
                                "#ba68c8",  # Soft Purple
                                "#4dd0e1",  # Cyan / Aqua
                                "#f06292",  # Soft Pink
                                "#aed581",  # Light Lime
                                "#ffccbc",  # Light Salmon / Peach
                            ]
                            
                            # Create figure
                            fig = go.Figure()
                            
                            # Add actual values
                            fig.add_trace(go.Scatter(
                                x=results_df['time'],
                                y=results_df['actual'],
                                name='Actual',
                                line=dict(color='red', width=2)
                            ))
                            
                            # Add predictions for each model with cluster colors
                            for i, model_name in enumerate(selected_models):
                                if model_name in predictions:
                                    fig.add_trace(go.Scatter(
                                        x=results_df['time'],
                                        y=results_df[f'{model_name}_pred'],
                                        name=f'{model_name} Prediction',
                                        line=dict(color=cluster_colors[i % len(cluster_colors)], width=1.5, dash='solid')
                                    ))
                            
                            # Update layout
                            fig.update_layout(
                                title=f'Demand Forecast for {city}',
                                xaxis_title='Time',
                                yaxis_title='Demand',
                                hovermode='x unified',
                                showlegend=True,
                                height=600,
                                autosize=True,
                                margin=dict(l=50, r=50, t=50, b=50),
                                yaxis=dict(
                                    autorange=True,
                                    showgrid=True,
                                    zeroline=True,
                                    showline=True,
                                    showticklabels=True
                                )
                            )
                            
                            # Show plot
                            st.plotly_chart(fig, use_container_width=True, config={'responsive': True})
                            
                            # Show error analysis if requested
                            if show_error_analysis:
                                st.subheader("Error Analysis")
                                
                                # Calculate error metrics for each model
                                error_metrics = {}
                                for model_name in selected_models:
                                    if model_name in predictions:
                                        pred = results_df[f'{model_name}_pred']
                                        actual = results_df['actual']
                                        
                                        # Calculate metrics
                                        mae = np.mean(np.abs(pred - actual))
                                        rmse = np.sqrt(np.mean((pred - actual) ** 2))
                                        mape = np.mean(np.abs((actual - pred) / actual)) * 100
                                        
                                        error_metrics[model_name] = {
                                            'MAE': mae,
                                            'RMSE': rmse,
                                            'MAPE': mape
                                        }
                                
                                # Display error metrics
                                error_df = pd.DataFrame(error_metrics).T
                                st.dataframe(error_df.style.format({
                                    'MAE': '{:.2f}',
                                    'RMSE': '{:.2f}',
                                    'MAPE': '{:.2f}%'
                                }))
                            
                            # Show feature importance if requested
                            if show_feature_importance:
                                st.subheader("Feature Importance")
                                
                                # Get feature names
                                feature_names = preprocessor_components['feature_cols']
                                
                                # Calculate feature importance for each model
                                for model_name in selected_models:
                                    if model_name in models:
                                        model = models[model_name]
                                        
                                        if hasattr(model, 'feature_importances_'):
                                            # For tree-based models
                                            importances = model.feature_importances_
                                        elif hasattr(model, 'coef_'):
                                            # For linear models
                                            importances = np.abs(model.coef_)
                                        else:
                                            continue
                                        
                                        # Show only the top 3 most important features
                                        feature_importance_pairs = list(zip(feature_names[:len(importances)], importances))
                                        # Sort by importance descending
                                        feature_importance_pairs.sort(key=lambda x: x[1], reverse=True)
                                        top_features = feature_importance_pairs[:3]
                                        
                                        if not top_features:
                                            st.info(f"No important features found for {model_name}.")
                                            continue
                                        
                                        st.markdown(f"**Top 3 Significant Features for {model_name}:**")
                                        for fname, imp in top_features:
                                            # Creative, human-readable interpretations
                                            interpretation = ""
                                            fname_lower = fname.lower()
                                            if 'weekend' in fname_lower:
                                                interpretation = "Demand is higher on weekends."
                                            elif 'night' in fname_lower or (fname_lower == 'hour' or 'hour' in fname_lower):
                                                interpretation = "Demand changes significantly with the hour of the day. Higher values may indicate increased demand at night or day, depending on the sign."
                                            elif 'temp' in fname_lower:
                                                interpretation = "Temperature has a strong effect on demand."
                                            elif 'humidity' in fname_lower:
                                                interpretation = "Humidity strongly influences demand."
                                            elif 'demand_lag' in fname_lower:
                                                interpretation = "Past demand (lag features) is a strong predictor of current demand."
                                            elif 'rolling' in fname_lower:
                                                interpretation = "Recent demand trends (rolling mean) are important for forecasting."
                                            elif 'month' in fname_lower:
                                                interpretation = "Seasonality (month) has a strong effect on demand."
                                            elif 'day' in fname_lower:
                                                interpretation = "Day of week or day features are significant for demand."
                                            elif 'cloud' in fname_lower:
                                                interpretation = "Cloud cover is a significant factor for demand."
                                            elif 'wind' in fname_lower:
                                                interpretation = "Wind speed is a significant factor for demand."
                                            else:
                                                interpretation = f"{fname} is a significant feature for demand."
                                            st.markdown(f"- **{fname}** (importance: {imp:.2f}): {interpretation}")
                            
                            # Download results
                            st.subheader("Download Results")
                            csv = results_df.to_csv(index=False)
                            st.download_button(
                                label="Download Predictions as CSV",
                                data=csv,
                                file_name=f'forecast_{city}_{start_date.date()}_{end_date.date()}.csv',
                                mime='text/csv'
                            )
                            
                    except Exception as e:
                        st.error(f"Error generating forecast: {str(e)}")
                        st.error(traceback.format_exc())

elif page == "Cluster Analysis":
    st.header("Cluster Analysis")
    
    if data is not None:
        # Sidebar inputs for cluster analysis
        st.sidebar.header("Cluster Analysis Parameters")
        
        # Number of clusters
        n_clusters = st.sidebar.slider("Number of Clusters", 2, 10, 4)
        
        # Features for clustering
        numerical_features = [
            'temperature', 'humidity', 'pressure', 'windSpeed', 'cloudCover',
            'demand', 'hour', 'month_num', 'day_of_week_num', 'is_weekend',
            'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos',
            'demand_lag24', 'demand_lag48', 'demand_lag168',
            'demand_rolling_mean_24h', 'demand_rolling_mean_7d'
        ]
        
        selected_features = st.multiselect(
            "Select Features for Clustering",
            numerical_features,
            default=['demand', 'temperature', 'humidity', 'hour']
        )
        
        # Sample size control
        sample_size = st.sidebar.slider(
            "Sample Size (% of data)", 
            min_value=1, 
            max_value=100, 
            value=20,
            help="Lower values = faster performance, higher values = more accurate"
        )
        
        # Data filtering options
        with st.sidebar.expander("Data Filtering Options"):
            filter_by_city = st.checkbox("Filter by City", False)
            if filter_by_city:
                cities = data['city'].unique().tolist()
                selected_city = st.selectbox("Select City", cities)
        
        @st.cache_data(ttl=3600, show_spinner=False)
        def perform_cluster_analysis(data_input, selected_features, n_clusters, sample_size, filter_city=None):
            """Cached function for performing cluster analysis"""
            # Apply city filter if selected
            if filter_city:
                filtered_data = data_input[data_input['city'] == filter_city].copy()
            else:
                filtered_data = data_input.copy()
            
            # Sample data for better performance
            if sample_size < 100:
                data_sample = filtered_data.sample(frac=sample_size/100, random_state=42)
            else:
                data_sample = filtered_data
                
            # Prepare data for clustering - only select columns we need
            X = data_sample[selected_features].copy()
            
            # Check for missing values and handle them
            if X.isna().any().any():
                X = X.fillna(X.mean())
            
            # Scale the features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Perform PCA for visualization
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_scaled)
            
            # Perform clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(X_scaled)
            
            # Create results DataFrame with only necessary columns
            results_df = pd.DataFrame({
                'PC1': X_pca[:, 0],
                'PC2': X_pca[:, 1],
                'Cluster': clusters
            })
            
            # Add only selected features to the results dataframe
            for feature in selected_features:
                results_df[feature] = data_sample[feature].values
                
            # Get PCA feature importance
            feature_importance = np.abs(pca.components_)
            feature_importance = feature_importance / np.sum(feature_importance, axis=1, keepdims=True)
            
            importance_df = pd.DataFrame(
                feature_importance,
                columns=selected_features,
                index=['PC1', 'PC2']
            )
            
            # Calculate cluster statistics directly
            cluster_stats = {}
            for cluster_id in range(n_clusters):
                cluster_data = results_df[results_df['Cluster'] == cluster_id][selected_features]
                cluster_stats[cluster_id] = {
                    'mean': cluster_data.mean().to_dict(),
                    'std': cluster_data.std().to_dict(),
                    'size': len(cluster_data)
                }
                
            return results_df, importance_df, cluster_stats, pca.explained_variance_ratio_
        
        if st.sidebar.button("Perform Cluster Analysis"):
            try:
                with st.spinner("Performing cluster analysis..."):
                    # Check if we have enough features
                    if not selected_features:
                        st.error("Please select at least one feature for clustering.")
                        st.stop()
                        
                    # Get city filter if enabled
                    filter_city = selected_city if filter_by_city else None
                    
                    # Perform cluster analysis with caching
                    results_df, importance_df, cluster_stats, explained_variance = perform_cluster_analysis(
                        data, selected_features, n_clusters, sample_size, filter_city
                    )
                    
                    # Display variance explained by PCA components
                    col1, col2 = st.columns(2)
                    with col1:
                        st.info(f"PC1 explains {explained_variance[0]*100:.1f}% of variance")
                    with col2:
                        st.info(f"PC2 explains {explained_variance[1]*100:.1f}% of variance")
                    
                    # Assign solid, distinct colors for clusters
                    cluster_colors = [
                        "#4db6ac", "#81c784", "#ffd54f", "#64b5f6", "#ffb74d",
                        "#ba68c8", "#4dd0e1", "#f06292", "#aed581", "#ffccbc"
                    ]
                    
                    # Plot clusters
                    st.subheader("Cluster Visualization")
                    
                    # Convert the Cluster column to a categorical type
                    results_df['Cluster'] = results_df['Cluster'].astype(str)
                    
                    # Create the scatter plot with categorical color mapping
                    fig = px.scatter(
                        results_df, 
                        x='PC1', 
                        y='PC2', 
                        color='Cluster',
                        color_discrete_sequence=cluster_colors[:n_clusters],
                        hover_data=selected_features,
                        title="PCA Cluster Visualization",
                        category_orders={"Cluster": [str(i) for i in range(n_clusters)]}  # Ensures proper order
                    )
                    
                    # Simpler layout updates
                    fig.update_layout(
                        height=600,
                        margin=dict(l=50, r=50, t=50, b=50),
                    )
                    
                    # Use renderer that better handles larger datasets
                    st.plotly_chart(fig, use_container_width=True, config={'responsive': True})
                    
                    # Show feature importance with simpler plotting
                    st.subheader("Feature Importance in Principal Components")
                    
                    fig_importance = px.bar(
                        importance_df.T.reset_index(),
                        x='index',
                        y=['PC1', 'PC2'],
                        barmode='group',
                        labels={'index': 'Feature', 'value': 'Importance'},
                        color_discrete_sequence=[cluster_colors[0], cluster_colors[1]]
                    )
                    
                    fig_importance.update_layout(
                        height=500,
                        margin=dict(l=50, r=50, t=50, b=50),
                        xaxis_tickangle=-45
                    )
                    
                    st.plotly_chart(fig_importance, use_container_width=True)
                    
                    # Show simplified cluster statistics with tabs
                    st.subheader("Cluster Statistics")
                    
                    # Create tabs for each cluster
                    tabs = st.tabs([f"Cluster {i} ({cluster_stats[i]['size']} points)" for i in range(n_clusters)])
                    
                    for i, tab in enumerate(tabs):
                        with tab:
                            # Create a more structured dataframe for this cluster
                            stats_df = pd.DataFrame({
                                'Feature': selected_features,
                                'Mean': [cluster_stats[i]['mean'][f] for f in selected_features],
                                'Std Dev': [cluster_stats[i]['std'][f] for f in selected_features]
                            })
                            
                            # Display without complex styling
                            st.dataframe(stats_df.set_index('Feature').round(2))
                    
                    # Simplified download option
                    st.download_button(
                        label="Download Cluster Analysis Results",
                        data=results_df.to_csv(index=False),
                        file_name='cluster_analysis_results.csv',
                        mime='text/csv'
                    )
                    
            except Exception as e:
                st.error(f"Error performing cluster analysis: {str(e)}")
                st.error(traceback.format_exc())

elif page == "Demand Analysis":
    st.header("Demand Analysis")
    
    if data is not None:
        # Sidebar inputs for analysis
        st.sidebar.header("Analysis Parameters")
        
        # City selection
        cities = data['city'].unique().tolist()
        city = st.sidebar.selectbox("Select City", cities)
        
        # Prepare model names and plotting variables
        model_names = [name for name in models.keys() if name in ['Linear Regression', 'Ridge Regression', 'Random Forest', 'XGBoost']]
        bar_width = 0.15
        offsets = np.linspace(-2*bar_width, 2*bar_width, 5)
        colors = ['#ffd54f', '#4db6ac', '#81c784', '#ba68c8', '#4dd0e1']
        
        # Filter data for selected city
        city_data = data[data['city'] == city].copy()
        
        # Ensure prediction columns for all models exist before any groupby/plotting
        numerical_features = preprocessor_components['numerical_features']
        categorical_features = preprocessor_components['categorical_features']
        X = city_data[numerical_features + categorical_features].copy()
        X_transformed = preprocessor_components['preprocessor'].transform(X)
        city_mean = city_data['city_demand_mean'].iloc[0]
        city_std = city_data['city_demand_std'].iloc[0]
        for model_name in model_names:
            try:
                pred = models[model_name].predict(X_transformed)
                pred = pred * city_std + city_mean
                city_data[f'{model_name}_pred'] = pred
                st.write(f"{model_name} prediction: min={np.nanmin(pred)}, max={np.nanmax(pred)}, NaNs={np.isnan(pred).sum()}")
            except Exception as e:
                st.error(f"Error predicting with {model_name}: {e}")
                city_data[f'{model_name}_pred'] = np.nan
        
        # Ensure 'Actual' is always included in bar charts
        all_bar_names = ['Actual'] + model_names
        
        # Convert time to datetime if not already
        city_data['time'] = pd.to_datetime(city_data['time'])
        
        # Add month name and season
        city_data['month_name'] = city_data['time'].dt.strftime('%B')
        city_data['season'] = city_data['time'].dt.month.map({
            12: 'Winter', 1: 'Winter', 2: 'Winter',
            3: 'Spring', 4: 'Spring', 5: 'Spring',
            6: 'Summer', 7: 'Summer', 8: 'Summer',
            9: 'Autumn', 10: 'Autumn', 11: 'Autumn'
        })
        
        # Create tabs for different analyses
        tab1, tab2, tab3, tab4 = st.tabs([
            "Seasonal Analysis", 
            "Temperature Impact", 
            "Time of Day Analysis",
            "Weather Factors"
        ])
        
        with tab1:
            st.subheader("Seasonal Demand Patterns")
            # Grouped bar chart for actual and model predictions
            seasonal_group = city_data.groupby('season')
            seasonal_demand = pd.DataFrame({
                'Actual': seasonal_group['demand'].mean()
            })
            for model_name in model_names:
                seasonal_demand[model_name] = seasonal_group[f'{model_name}_pred'].mean()
            seasonal_demand = seasonal_demand.reset_index()
            seasonal_demand = seasonal_demand.sort_values('Actual', ascending=False)
            # Bar chart
            fig = go.Figure()
            x = np.arange(len(seasonal_demand['season']))
            for i, name in enumerate(all_bar_names):
                fig.add_trace(go.Bar(
                    x=x + offsets[i],
                    y=seasonal_demand[name],
                    name=name,
                    width=bar_width,
                    marker_color=colors[i],
                    offsetgroup=i
                ))
            fig.update_layout(
                title=f'Seasonal Demand Patterns in {city}',
                xaxis=dict(
                    tickmode='array',
                    tickvals=x,
                    ticktext=seasonal_demand['season']
                ),
                yaxis_title='Average Demand',
                barmode='group',
                height=500,
                margin=dict(l=50, r=50, t=50, b=50)
            )
            st.plotly_chart(fig, use_container_width=True)
            # Key Insights
            highest_season = seasonal_demand.iloc[0]
            lowest_season = seasonal_demand.iloc[-1]
            st.markdown("### Key Insights:")
            st.markdown(f"""
            - **Highest Demand**: {highest_season['season']} (Average: {highest_season['Actual']:.2f})
            - **Lowest Demand**: {lowest_season['season']} (Average: {lowest_season['Actual']:.2f})
            - **Seasonal Variation**: {highest_season['Actual'] - lowest_season['Actual']:.2f} units between highest and lowest seasons
            """)
        
        with tab2:
            st.subheader("Temperature Impact on Demand")
            city_data['temp_bin'] = pd.cut(
                city_data['temperature'],
                bins=11,
                labels=[f'{i}°C' for i in range(-10, 41, 5)]
            )
            temp_group = city_data.groupby('temp_bin')
            temp_demand = pd.DataFrame({
                'Actual': temp_group['demand'].mean()
            })
            for model_name in model_names:
                temp_demand[model_name] = temp_group[f'{model_name}_pred'].mean()
            temp_demand = temp_demand.reset_index()
            # Grouped bar chart
            fig_bar = go.Figure()
            x = np.arange(len(temp_demand['temp_bin']))
            for i, name in enumerate(all_bar_names):
                fig_bar.add_trace(go.Bar(
                    x=x + offsets[i],
                    y=temp_demand[name],
                    name=name,
                    width=bar_width,
                    marker_color=colors[i],
                    offsetgroup=i
                ))
            fig_bar.update_layout(
                title=f'Temperature vs Demand in {city}',
                xaxis=dict(
                    tickmode='array',
                    tickvals=x,
                    ticktext=temp_demand['temp_bin'].astype(str)
                ),
                yaxis_title='Average Demand',
                barmode='group',
                height=500,
                margin=dict(l=50, r=50, t=50, b=50),
                xaxis_tickangle=-45
            )
            st.plotly_chart(fig_bar, use_container_width=True)
            # Line chart for trend
            fig_line = go.Figure()
            for i, name in enumerate(all_bar_names):
                fig_line.add_trace(go.Scatter(
                    x=temp_demand['temp_bin'].astype(str),
                    y=temp_demand[name],
                    mode='lines+markers',
                    name=name,
                    line=dict(width=2, color=colors[i])
                ))
            fig_line.update_layout(
                title=f'Temperature vs Demand Trend in {city}',
                xaxis_title='Temperature Range',
                yaxis_title='Average Demand',
                height=500,
                margin=dict(l=50, r=50, t=50, b=50),
                xaxis_tickangle=-45
            )
            st.plotly_chart(fig_line, use_container_width=True)
        
        with tab3:
            st.subheader("Time of Day Analysis")
            hour_group = city_data.groupby('hour')
            hour_demand = pd.DataFrame({
                'Actual': hour_group['demand'].mean(),
                'Std_Actual': hour_group['demand'].std()
            })
            for model_name in model_names:
                hour_demand[model_name] = hour_group[f'{model_name}_pred'].mean()
                hour_demand[f'Std_{model_name}'] = hour_group[f'{model_name}_pred'].std()
            hour_demand = hour_demand.reset_index()
            # Line chart with error bars for actual and all models
            fig = go.Figure()
            # Actual
            fig.add_trace(go.Scatter(
                x=hour_demand['hour'],
                y=hour_demand['Actual'],
                mode='lines+markers',
                name='Actual',
                line=dict(width=3, color=colors[0]),
                error_y=dict(type='data', array=hour_demand['Std_Actual'], visible=True, color=colors[0], thickness=1.5, width=0),
                showlegend=True
            ))
            # Models
            for i, model_name in enumerate(model_names):
                fig.add_trace(go.Scatter(
                    x=hour_demand['hour'],
                    y=hour_demand[model_name],
                    mode='lines+markers',
                    name=model_name,
                    line=dict(width=2, color=colors[i+1]),
                    error_y=dict(type='data', array=hour_demand[f'Std_{model_name}'], visible=True, color=colors[i+1], thickness=1.5, width=0),
                    showlegend=True
                ))
            fig.update_layout(
                title=f'Hourly Demand Pattern in {city}',
                xaxis_title='Hour of Day',
                yaxis_title='Average Demand',
                height=500,
                margin=dict(l=50, r=50, t=50, b=50),
                xaxis=dict(tickmode='linear', tick0=0, dtick=1)
            )
            st.plotly_chart(fig, use_container_width=True)
            # Peak hour analysis
            peak_hour = hour_demand.loc[hour_demand['Actual'].idxmax()]
            off_peak_hour = hour_demand.loc[hour_demand['Actual'].idxmin()]
            st.markdown("### Peak Hours Analysis:")
            st.markdown(f"""
            - **Peak Hour**: {int(peak_hour['hour'])}:00 (Average: {peak_hour['Actual']:.2f} ± {peak_hour['Std_Actual']:.2f})
            - **Off-Peak Hour**: {int(off_peak_hour['hour'])}:00 (Average: {off_peak_hour['Actual']:.2f} ± {off_peak_hour['Std_Actual']:.2f})
            - **Peak to Off-Peak Ratio**: {peak_hour['Actual']/off_peak_hour['Actual']:.2f}x
            """)
        
        with tab4:
            st.subheader("Weather Factors Impact")
            from plotly.subplots import make_subplots
            weather_factors = ['temperature', 'humidity', 'pressure', 'windSpeed', 'cloudCover']
            
            # Prepare subplot grid for line charts
            fig = make_subplots(rows=1, cols=len(weather_factors), shared_yaxes=True, subplot_titles=[f.title() for f in weather_factors])
            
            # Define formatting based on feature type
            format_rules = {
                'temperature': '{:.0f}',      # No decimal for temperature
                'humidity': '{:.0%}',         # Percentage format for humidity
                'pressure': '{:.1f}',         # 1 decimal for pressure
                'windSpeed': '{:.1f}',        # 1 decimal for wind speed
                'cloudCover': '{:.0%}'        # Percentage format for cloud cover
            }
            
            for col, factor in enumerate(weather_factors, 1):
                bins = pd.qcut(city_data[factor], q=10, duplicates='drop') if city_data[factor].nunique() > 10 else city_data[factor]
                group = city_data.groupby(bins)
                actual = group['demand'].mean()
                line_data = pd.DataFrame({'Actual': actual})
                
                for model_name in model_names:
                    line_data[model_name] = group[f'{model_name}_pred'].mean()
                
                line_data = line_data.reset_index()
                
                # Format bin labels robustly
                if hasattr(bins, 'categories'):
                    intervals = list(bins.categories)
                    bin_labels = [
                        f"{format_rules[factor].format(interval.left)}-{format_rules[factor].format(interval.right)}"
                        for interval in intervals
                    ]
                else:
                    values = line_data[bins.name if hasattr(bins, 'name') else factor]
                    bin_labels = []
                    for val in values:
                        if hasattr(val, 'left') and hasattr(val, 'right'):
                            # It's an interval
                            bin_labels.append(f"{format_rules[factor].format(val.left)}-{format_rules[factor].format(val.right)}")
                        else:
                            # It's a float/int
                            bin_labels.append(format_rules[factor].format(float(val)))
                
                # Add traces for each model
                for i, name in enumerate(all_bar_names):
                    fig.add_trace(
                        go.Scatter(
                            x=bin_labels,
                            y=line_data[name],
                            mode='lines+markers',
                            name=name if col == 1 else None,  # Only show legend once
                            line=dict(width=2, color=colors[i]),
                            showlegend=(col == 1)
                        ),
                        row=1, col=col
                    )
                
                # Update x-axis labels
                fig.update_xaxes(
                    tickangle=-45,
                    ticktext=bin_labels,
                    row=1, col=col
                )
            
            fig.update_layout(
                title=f'Weather Factors Impact on Demand in {city}',
                yaxis_title='Average Demand',
                height=600,
                margin=dict(l=50, r=50, t=50, b=100),
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Weather Impact Analysis
            st.markdown("### Weather Impact Analysis:")
            weather_corr = city_data[weather_factors + ['demand']].corr()['demand'].sort_values(ascending=False)
            for factor, corr in weather_corr.items():
                if factor != 'demand':
                    impact = "strongly increases" if corr > 0.5 else "moderately increases" if corr > 0.2 else "slightly increases" if corr > 0 else "slightly decreases" if corr > -0.2 else "moderately decreases" if corr > -0.5 else "strongly decreases"
                    st.markdown(f"- **{factor.title()}**: {impact} demand (correlation: {corr:.2f})")
        # Add download option for the analysis
        st.download_button(
            label="Download Analysis Data",
            data=city_data.to_csv(index=False),
            file_name=f'demand_analysis_{city}.csv',
            mime='text/csv'
        )
        
    else:
        st.warning("No data available for analysis. Please ensure data is properly loaded.")

else:  # Model Information page
    st.header("Model Information")
    
    if models is not None:
        # Display information about each model
        for model_name, model in models.items():
            st.subheader(model_name)
            
            # Model type
            st.write(f"Model Type: {type(model).__name__}")
            
            # Model parameters
            if hasattr(model, 'get_params'):
                st.write("Model Parameters:")
                params = model.get_params()
                for param, value in params.items():
                    st.write(f"- {param}: {value}")
            
            # Feature importance if available
            if hasattr(model, 'feature_importances_'):
                st.write("Feature Importance:")
                importances = model.feature_importances_
                feature_names = preprocessor_components['feature_cols']
                
                importance_df = pd.DataFrame({
                    'Feature': feature_names[:len(importances)],
                    'Importance': importances
                })
                
                importance_df = importance_df.sort_values('Importance', ascending=False)
                
                fig = px.bar(
                    importance_df.head(10),
                    x='Importance',
                    y='Feature',
                    orientation='h',
                    title=f'Top 10 Feature Importance - {model_name}'
                )
                fig.update_layout(
                    height=500,
                    autosize=True,
                    margin=dict(l=50, r=50, t=50, b=50)
                )
                st.plotly_chart(fig, use_container_width=True, config={'responsive': True})
            
            st.markdown("---")
    else:
        st.warning("No models loaded. Please run predictive_modeling.py first to train and save the models.")

# Footer
st.markdown("""
---
Developed with Streamlit - © 2025
""")

def make_prediction(data, model_name, model, preprocessor_components):
    """Make prediction using the selected model"""
    try:
        # Prepare features
        numerical_features = preprocessor_components['numerical_features']
        categorical_features = preprocessor_components['categorical_features']
        
        # Filter available features from data
        available_num_features = [f for f in numerical_features if f in data.columns]
        available_cat_features = [f for f in categorical_features if f in data.columns]
        
        # Select features from data
        X = data[available_num_features + available_cat_features]
        
        # Transform features
        X_transformed = preprocessor_components['preprocessor'].transform(X)
        
        if model_name == 'XGBoost':
            # Set device to CPU for prediction
            model.set_param({'device': 'cpu'})
            predictions = model.predict(X_transformed)
        else:
            # Standard scikit-learn model prediction
            predictions = model.predict(X_transformed)
        
        return predictions
    
    except Exception as e:
        st.error(f"Error making prediction with {model_name}: {str(e)}")
        traceback.print_exc()
        return np.full(len(data), np.nan)

def load_models():
    """Load trained models and preprocessing components"""
    models = {}
    
    # Load preprocessor components first to get feature dimensions
    preprocessor_components = joblib.load('models/preprocessor_components.joblib')
    input_size = len(preprocessor_components['feature_cols'])
    print(f"Loading models with input size: {input_size}")
    
    # Load scikit-learn models
    for model_file in os.listdir('models'):
        if model_file.endswith('.joblib') and not model_file.startswith('preprocessor_components'):
            try:
                model_name = model_file.replace('.joblib', '').replace('_', ' ').title()
                model = joblib.load(f'models/{model_file}')
                
                # Set XGBoost to CPU mode if it's an XGBoost model
                if model_name == 'XGBoost':
                    import xgboost as xgb
                    model.set_param({'device': 'cpu'})
                
                models[model_name] = model
                print(f"Loaded {model_name} model")
            except Exception as e:
                print(f"Error loading {model_file}: {str(e)}")

    
    return models, preprocessor_components
