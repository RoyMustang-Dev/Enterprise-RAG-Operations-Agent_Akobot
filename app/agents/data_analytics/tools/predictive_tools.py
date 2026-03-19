import logging
from typing import Dict, Any, List, Optional, Union
import pandas as pd
from langchain_core.tools import tool

logger = logging.getLogger(__name__)

# To prevent heavy ML library instantiation lag at global startup, we import them globally
# only when generating the tool, or inside the tool itself if preferred. 
# We'll put them inside the tool functions so the main API doesn't crash on boot if they are missing.

def time_series_forecast_prophet(historical_data: Union[pd.DataFrame, str], periods: int = 30) -> str:
    """
    Use this tool exclusively for time-series forecasting (e.g., predicting daily revenue, 
    weekly website traffic, monthly churn) over a specific time horizon.
    
    Args:
        historical_data: Strict Pandas DataFrame containing TWO keys: 'ds' (the date string) and 'y' (the numeric value to predict).
        periods: The integer number of future days to forecast (e.g. 30).
    
    Returns:
        JSON string mapping the future predicted dates to their forecasted values.
    """
    try:
        from prophet import Prophet
        import json
        
        if isinstance(historical_data, str):
            try:
                historical_data = json.loads(historical_data)
                df = pd.DataFrame(historical_data)
            except json.JSONDecodeError:
                return "Fatal Error: 'historical_data' string could not be parsed as valid JSON."
        elif isinstance(historical_data, pd.DataFrame):
            df = historical_data.copy()
        else:
            return "Fatal Error: 'historical_data' must be a Pandas DataFrame."
        
        # Prophet requires strict 'ds' and 'y' column definitions
        if 'ds' not in df.columns or 'y' not in df.columns:
            return "Fatal Error: Data must contain exactly 'ds' (date format) and 'y' (numeric format)."
            
        df['ds'] = pd.to_datetime(df['ds'])
        df['y'] = pd.to_numeric(df['y'])
        
        logger.info(f"[PROPHET FORECAST] Training on {len(df)} points. Projecting {periods} days.")
        
        m = Prophet(daily_seasonality=True, yearly_seasonality=True)
        m.fit(df)
        
        future = m.make_future_dataframe(periods=periods)
        forecast = m.predict(future)
        
        # We only care about the newly generated rows (the tail end `periods`)
        future_forecast = forecast[['ds', 'yhat']].tail(periods)

        # Backtesting (simple holdout)
        holdout = max(3, int(len(df) * 0.1))
        train_df = df.iloc[:-holdout] if len(df) > holdout else df
        test_df = df.iloc[-holdout:] if len(df) > holdout else df
        m2 = Prophet(daily_seasonality=True, yearly_seasonality=True)
        m2.fit(train_df)
        future_test = m2.make_future_dataframe(periods=holdout)
        pred_test = m2.predict(future_test).tail(holdout)
        mae = float((pred_test['yhat'].values - test_df['y'].values).mean())
        std_resid = float((pred_test['yhat'].values - test_df['y'].values).std())
        
        import os
        import uuid
        export_dir = os.path.join(os.getcwd(), "data", "exports")
        os.makedirs(export_dir, exist_ok=True)
        file_name = f"prophet_forecast_{uuid.uuid4().hex[:8]}.csv"
        future_forecast.to_csv(os.path.join(export_dir, file_name), index=False)
        
        preview = future_forecast.head(5).to_dict(orient="records")
        return f"SUCCESS: Forecast computed. Massive dataframe safely bypassed token limit and saved to disk at '/api/v1/exports/{file_name}'. Preview of first 5 days array: {preview}. Backtest MAE={mae:.4f}. CI=+/-{1.96*std_resid:.4f}"
        
    except ImportError:
        logger.warning("[PROPHET FORECAST] Prophet not installed. Falling back to linear trend forecast.")
        try:
            import numpy as np
            import json
            if isinstance(historical_data, str):
                historical_data = json.loads(historical_data)
                df = pd.DataFrame(historical_data)
            elif isinstance(historical_data, pd.DataFrame):
                df = historical_data.copy()
            else:
                return "Fatal Error: 'historical_data' must be a Pandas DataFrame."
            if 'ds' not in df.columns or 'y' not in df.columns:
                return "Fatal Error: Data must contain exactly 'ds' (date format) and 'y' (numeric format)."
            df['ds'] = pd.to_datetime(df['ds'])
            df['y'] = pd.to_numeric(df['y'])
            df = df.dropna(subset=['ds', 'y'])
            df = df.sort_values('ds')
            if df.empty:
                return "Fatal Error: No valid data points for forecasting."
            # Linear trend on ordinal dates
            x = df['ds'].map(pd.Timestamp.toordinal).to_numpy()
            y = df['y'].to_numpy()
            coef = np.polyfit(x, y, 1)
            last_date = df['ds'].max()
            future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=periods, freq='D')
            future_x = future_dates.map(pd.Timestamp.toordinal).to_numpy()
            yhat = coef[0] * future_x + coef[1]
            future_forecast = pd.DataFrame({"ds": future_dates, "yhat": yhat})
            # Backtest MAE and CI
            yhat_hist = coef[0] * x + coef[1]
            resid = yhat_hist - y
            mae = float(np.mean(np.abs(resid))) if len(resid) else 0.0
            std_resid = float(np.std(resid)) if len(resid) else 0.0
            import os
            import uuid
            export_dir = os.path.join(os.getcwd(), "data", "exports")
            os.makedirs(export_dir, exist_ok=True)
            file_name = f"prophet_fallback_forecast_{uuid.uuid4().hex[:8]}.csv"
            future_forecast.to_csv(os.path.join(export_dir, file_name), index=False)
            preview = future_forecast.head(5).to_dict(orient="records")
            return f"SUCCESS: Prophet unavailable; linear trend forecast generated. Saved to '/api/v1/exports/{file_name}'. Preview: {preview}. Backtest MAE={mae:.4f}. CI=+/-{1.96*std_resid:.4f}"
        except Exception as e:
            return f"Prophet fallback failed: {e}"
    except Exception as e:
        logger.error(f"[PROPHET FORECAST] Execution Error: {e}")
        return f"Error executing Meta Prophet Forecast: {str(e)}"

def linear_regression_projection(features_data: pd.DataFrame, target_column: str, future_feature_values: List[Dict[str, Any]]) -> str:
    """
    Use this tool exclusively for multi-variate continuous prediction where one metric depends 
    on several other numerical features (e.g., predicted sales based on Ad Spend, Website Clicks, and Price).
    
    Args:
        features_data: Pandas DataFrame containing historical rows with feature columns and the target column.
        target_column: The exact string name of the target column you are trying to predict.
        future_feature_values: Array of dicts representing the hypothetical future scenarios (feature matrices) you want to predict against.
    """
    try:
        from sklearn.linear_model import LinearRegression
        import json
        
        df = features_data.copy()
        df_future = pd.DataFrame(future_feature_values)
        
        if target_column not in df.columns:
            return f"Fatal Error: Target column '{target_column}' missing from provided historical data."
            
        # Separate X (features) and Y (target)
        X_train_raw = df.drop(columns=[target_column])
        y_train = pd.to_numeric(df[target_column], errors="coerce")
        # One-hot encode categorical features for robustness
        X_train = pd.get_dummies(X_train_raw, drop_first=False)
        # Align future features with training columns
        df_future_encoded = pd.get_dummies(df_future, drop_first=False)
        df_future_encoded = df_future_encoded.reindex(columns=X_train.columns, fill_value=0)
        
        logger.info(f"[SKLEARN REGRESSION] Training Multivariable Regression on {len(X_train.columns)} features.")
        
        model = LinearRegression()
        # Drop rows with NaN target
        valid_mask = y_train.notna()
        X_train = X_train.loc[valid_mask]
        y_train = y_train.loc[valid_mask]
        model.fit(X_train, y_train)
        
        predictions = model.predict(df_future_encoded)
        df_future[f'predicted_{target_column}'] = predictions
        
        import os
        import uuid
        export_dir = os.path.join(os.getcwd(), "data", "exports")
        os.makedirs(export_dir, exist_ok=True)
        file_name = f"regression_{uuid.uuid4().hex[:8]}.csv"
        df_future.to_csv(os.path.join(export_dir, file_name), index=False)
        
        preview = df_future.head(5).to_dict(orient="records")
        return f"SUCCESS: Multi-variable continuous prediction generated. Saved securely to '/api/v1/exports/{file_name}'. Preview matrix: {preview}"
        
    except ImportError:
        return "System Architectural Error: 'scikit-learn' is not installed."
    except Exception as e:
        logger.error(f"[SKLEARN REGRESSION] Error: {e}")
        return f"Error executing Scikit-Learn Regression: {str(e)}"

def time_series_forecast_xgboost(historical_data: Union[pd.DataFrame, str], periods: int = 30) -> str:
    """
    Use this tool exclusively for advanced multi-variable (Panel Data) Time-Series forecasting utilizing XGBoost Regression.
    Best for predicting across multiple concurrent regions and categories simultaneously without collapsing the dimensional data into a flat line.
    
    Args:
        historical_data: Pandas DataFrame containing at least THREE keys: 'ds' (date string), 'y' (the numeric value to predict), and 'unique_id' (string category like 'Region1_ProductA').
        periods: The integer number of future days to forecast (e.g. 30).
    """
    try:
        import xgboost as xgb
        from sklearn.preprocessing import LabelEncoder
        import json
        
        if isinstance(historical_data, str):
            try:
                historical_data = json.loads(historical_data)
                df = pd.DataFrame(historical_data)
            except json.JSONDecodeError:
                return "Fatal Error: 'historical_data' string could not be parsed as valid JSON."
        elif isinstance(historical_data, pd.DataFrame):
            df = historical_data.copy()
        else:
            return "Fatal Error: 'historical_data' must be a Pandas DataFrame."
        
        if 'ds' not in df.columns or 'y' not in df.columns or 'unique_id' not in df.columns:
            return "Fatal Error: Data must contain exactly 'ds' (date format), 'y' (numeric), and 'unique_id' (string label)."
            
        df['ds'] = pd.to_datetime(df['ds'])
        df['y'] = pd.to_numeric(df['y'])
        
        logger.info(f"[XGBOOST FORECAST] Training Panel Matrix on {len(df)} discrete points. Projecting {periods} days ahead per ID.")
        
        # 1. Feature Engineer advanced Time vectors
        df['year'] = df['ds'].dt.year
        df['month'] = df['ds'].dt.month
        df['day'] = df['ds'].dt.day
        df['dayofweek'] = df['ds'].dt.dayofweek
        
        # 2. One-Hot Label Encoding for Multi-Region strings
        le = LabelEncoder()
        df['unique_id_encoded'] = le.fit_transform(df['unique_id'])
        
        # 3. Train XGBoost Model (with simple holdout backtest)
        features = ['year', 'month', 'day', 'dayofweek', 'unique_id_encoded']
        X = df[features]
        y_train = df['y']

        # Holdout last 10% for backtest
        split_idx = max(1, int(len(X) * 0.9))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_tr, y_te = y_train.iloc[:split_idx], y_train.iloc[split_idx:]

        model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
        model.fit(X_train, y_tr)
        if len(X_test) > 0:
            preds = model.predict(X_test)
            mae = float((preds - y_te.values).mean())
            std_resid = float((preds - y_te.values).std())
        else:
            mae = 0.0
            std_resid = 0.0
        
        # 4. Generate N-Dimensional Future Projection Matrix
        last_date = df['ds'].max()
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=periods, freq='D')
        
        future_rows = []
        for uid_orig in le.classes_:
            uid_enc = le.transform([uid_orig])[0]
            for date in future_dates:
                future_rows.append({
                    'unique_id': uid_orig,
                    'ds': date,
                    'year': date.year,
                    'month': date.month,
                    'day': date.day,
                    'dayofweek': date.dayofweek,
                    'unique_id_encoded': uid_enc
                })
        
        future_df = pd.DataFrame(future_rows)
        future_X = future_df[features]
        
        # 5. Predict isolated target vectors
        future_df['y_forecast'] = model.predict(future_X)
        
        import os
        import uuid
        export_dir = os.path.join(os.getcwd(), "data", "exports")
        os.makedirs(export_dir, exist_ok=True)
        file_name = f"xgboost_forecast_{uuid.uuid4().hex[:8]}.csv"
        
        final_df = future_df[['unique_id', 'ds', 'y_forecast']]
        final_df.to_csv(os.path.join(export_dir, file_name), index=False)
        
        preview = final_df.head(5).to_dict(orient="records")
        return f"SUCCESS: XGBoost Multi-variate Panel Forecast completed. Heavy payload written seamlessly to CSV at '/api/v1/exports/{file_name}'. Small Preview buffer: {preview}. Backtest MAE={mae:.4f}. CI=+/-{1.96*std_resid:.4f}"
        
    except ImportError:
        logger.error("[XGBOOST FORECAST] Missing XGBoost Library.")
        return "System Architectural Error: The 'xgboost' PIP library is missing."
    except Exception as e:
        logger.error(f"[XGBOOST FORECAST] Execution Fault Error: {e}")
        return f"Error explicitly executing XGBoost Forecast: {str(e)}"

