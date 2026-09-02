
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_excel("EAT Case Interview - Data - Aug26.xlsx")

# Converting 'week' (YYYYWW format) into a real date
df['week_str'] = df['week'].astype(str)
df['year'] = df['week_str'].str[:4].astype(int)
df['week_num'] = df['week_str'].str[4:].astype(int)
df['date'] = pd.to_datetime(df['year'].astype(str) + df['week_num'].astype(str) + '1', format='%G%V%u')

df = df.sort_values(['city', 'cuisine', 'product_type', 'date']).reset_index(drop=True)

print(df.shape)
print(df.isnull().sum().sum(), "total nulls")
print("Duplicates:", df.duplicated(subset=['city', 'cuisine', 'product_type', 'week']).sum())

# revenue trend by city over time
revenue_by_city = df.groupby(['date', 'city'])['revenue'].sum().reset_index()

fig, ax = plt.subplots(figsize=(14, 6))
for city in revenue_by_city['city'].unique():
    subset = revenue_by_city[revenue_by_city['city'] == city]
    ax.plot(subset['date'], subset['revenue'], label=city)

ax.set_title('Revenue by city over time')
ax.set_xlabel('Fecha')
ax.set_ylabel('Revenue')
ax.legend()
plt.show()

# Checking for seasonality each month or each trimester
from statsmodels.tsa.seasonal import seasonal_decompose
agg_city = df.groupby(['city', 'date'])['revenue'].sum().reset_index()
for i in ['HOUSTON', 'SAN FRANCISCO', 'SPOKANE']:

    serie = agg_city[agg_city['city'] == i].set_index('date')['revenue']

    result = seasonal_decompose(serie, model='additive', period=4)

    fig = result.plot()
    fig.suptitle(f'Descomposición - {i}', y=1.02)
    fig.set_size_inches(12, 8)
    plt.tight_layout()
    plt.show()

for i in ['HOUSTON', 'SAN FRANCISCO', 'SPOKANE']:

    serie = agg_city[agg_city['city'] == i].set_index('date')['revenue']

    result = seasonal_decompose(serie, model='additive', period=13)

    fig = result.plot()
    fig.suptitle(f'Descomposición - {i}', y=1.02)
    fig.set_size_inches(12, 8)
    plt.tight_layout()
    plt.show()

# detect outliers at CITY level (aggregate), then impute at ROW level ---
agg_city = df.groupby(['city', 'date'])['revenue'].sum().reset_index()
agg_city['z_score'] = agg_city.groupby('city')['revenue'].transform(lambda x: (x - x.mean()) / x.std())
agg_city['is_outlier'] = agg_city['z_score'].abs() > 3

outlier_events = agg_city[agg_city['is_outlier']][['city', 'date']].to_dict('records')
print("Outlier events detected:", outlier_events)

# impute each outlier row using the average of the prior and following week --
cols_to_impute = ['revenue', 'cases', 'cost']

for event in outlier_events:
    city, date = event['city'], event['date']
    mask_outlier = (df['city'] == city) & (df['date'] == date)

    for idx, row in df[mask_outlier].iterrows():
        mask_prev = ((df['city'] == row['city']) & (df['cuisine'] == row['cuisine']) &
                     (df['product_type'] == row['product_type']) & (df['date'] == date - pd.Timedelta(days=7)))
        mask_next = ((df['city'] == row['city']) & (df['cuisine'] == row['cuisine']) &
                     (df['product_type'] == row['product_type']) & (df['date'] == date + pd.Timedelta(days=7)))

        for col in cols_to_impute:
            prev_val = df.loc[mask_prev, col].values
            next_val = df.loc[mask_next, col].values
            if len(prev_val) > 0 and len(next_val) > 0:
                df.loc[idx, col] = (prev_val[0] + next_val[0]) / 2

print("Imputation complete.")

#re-check z-scores after imputation
agg_city_check = df.groupby(['city', 'date'])['revenue'].sum().reset_index()
agg_city_check['z_score'] = agg_city_check.groupby('city')['revenue'].transform(lambda x: (x - x.mean()) / x.std())
print("Remaining outliers:", (agg_city_check['z_score'].abs() > 3).sum())

# Revenue trend by city over time
revenue_by_city = df.groupby(['date', 'city'])['revenue'].sum().reset_index()

fig, ax = plt.subplots(figsize=(14, 6))
for city in revenue_by_city['city'].unique():
    subset = revenue_by_city[revenue_by_city['city'] == city]
    ax.plot(subset['date'], subset['revenue'], label=city)

ax.set_title('Revenue by city over time')
ax.set_xlabel('Fecha')
ax.set_ylabel('Revenue')
ax.legend()
plt.show()

## Feature engineering
df['month'] = df['date'].dt.month
df['quarter'] = df['date'].dt.quarter
df['week_of_year'] = df['date'].dt.isocalendar().week.astype(int)

# Lags and rolling features (per combination of city + cuisine + product_type) for 1 week, 4 weeks, and 13 weeks (1 week, 1 month, 3 months)
group_cols = ['city', 'cuisine', 'product_type']

for lag in [1, 4, 13]:
    df[f'revenue_lag_{lag}'] = df.groupby(group_cols)['revenue'].shift(lag)

df['revenue_roll_mean_4'] = df.groupby(group_cols)['revenue'].transform(
    lambda x: x.shift(1).rolling(window=4).mean()
)
df['revenue_roll_std_4'] = df.groupby(group_cols)['revenue'].transform(
    lambda x: x.shift(1).rolling(window=4).std()
)

# one-hot encode categoricals
df_model = pd.get_dummies(df, columns=['city', 'cuisine', 'product_type'], drop_first=False)

# drop rows without enough data for the longest lag
df_model = df_model.dropna(subset=['revenue_lag_13'])

print(df_model.shape)
print(df_model.columns.tolist())

# columns to exclude from features for modeling due to irrelevance
cols_exclude = ['customer_count', 'cases', 'cost', 'revenue',
                'week', 'week_str', 'year', 'week_num', 'date']

feature_cols = [c for c in df_model.columns if c not in cols_exclude]

# last 12 weeks for testing, rest for training
cutoff_date = df_model['date'].max() - pd.Timedelta(weeks=12)

train = df_model[df_model['date'] <= cutoff_date]
test = df_model[df_model['date'] > cutoff_date]

X_train, y_train = train[feature_cols], train['revenue']
X_test, y_test = test[feature_cols], test['revenue']

print("Train:", X_train.shape, " Test:", X_test.shape)
print("Test weeks:", test['date'].nunique())
print("Train range:", train['date'].min(), "-", train['date'].max())
print("Test range:", test['date'].min(), "-", test['date'].max())

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error

def evaluate_model(name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    print(f"{name:20s} | MAE: {mae:,.0f} | RMSE: {rmse:,.0f} | MAPE: {mape:.2f}%")
    return {'model': name, 'MAE': mae, 'RMSE': rmse, 'MAPE': mape}

results = []

# Base line model: Naive (lag_1)
y_pred_naive = X_test['revenue_lag_1'].values
results.append(evaluate_model("Naive (lag_1)", y_test.values, y_pred_naive))

# Linear regression
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
results.append(evaluate_model("Linear Regression", y_test.values, y_pred_lr))

# time series cross-validation for hyperparameter tuning
tscv = TimeSeriesSplit(n_splits=4)

# Random Forest Grid Search
rf_param_grid = {
    'n_estimators': [200, 300, 500],
    'max_depth': [4, 6, 8, 10],
    'min_samples_leaf': [1, 5, 10]
}

rf_grid = GridSearchCV(
    estimator=RandomForestRegressor(random_state=50, n_jobs=-1),
    param_grid=rf_param_grid,
    cv=tscv,
    scoring='neg_mean_absolute_percentage_error',
    n_jobs=-1,
    verbose=1
)
rf_grid.fit(X_train, y_train)

print("Best Random Forest params:", rf_grid.best_params_)
rf = rf_grid.best_estimator_
y_pred_rf = rf.predict(X_test)
results.append(evaluate_model("Random Forest (tuned)", y_test.values, y_pred_rf))

# XGBoost Grid Search
xgb_param_grid = {
    'n_estimators': [200, 300, 500],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.7, 0.8, 1.0]
}

xgb_grid = GridSearchCV(
    estimator=XGBRegressor(random_state=50, colsample_bytree=0.8),
    param_grid=xgb_param_grid,
    cv=tscv,
    scoring='neg_mean_absolute_percentage_error',
    n_jobs=-1,
    verbose=1
)
xgb_grid.fit(X_train, y_train)

print("Best XGBoost params:", xgb_grid.best_params_)
xgb = xgb_grid.best_estimator_
y_pred_xgb = xgb.predict(X_test)
results.append(evaluate_model("XGBoost (tuned)", y_test.values, y_pred_xgb))

results_df = pd.DataFrame(results)
print("\nModel comparison:\n", results_df)

def walk_forward_validation(df, feature_cols, target_col, model_fn, n_splits=4, test_weeks=12):
    max_date = df['date'].max()
    fold_results = []

    for i in range(n_splits, 0, -1):
        split_cutoff = max_date - pd.Timedelta(weeks=test_weeks * i)
        test_end = split_cutoff + pd.Timedelta(weeks=test_weeks)

        fold_train = df[df['date'] <= split_cutoff]
        fold_test = df[(df['date'] > split_cutoff) & (df['date'] <= test_end)]

        if len(fold_test) == 0 or len(fold_train) == 0:
            continue

        X_tr, y_tr = fold_train[feature_cols], fold_train[target_col]
        X_te, y_te = fold_test[feature_cols], fold_test[target_col]

        model = model_fn()
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)

        mae = mean_absolute_error(y_te, y_pred)
        rmse = np.sqrt(mean_squared_error(y_te, y_pred))
        mape = np.mean(np.abs((y_te - y_pred) / y_te)) * 100

        fold_results.append({
            'fold': n_splits - i + 1,
            'train_end': split_cutoff.date(),
            'test_end': test_end.date(),
            'n_train': len(fold_train),
            'MAE': mae, 'RMSE': rmse, 'MAPE': mape
        })

    return pd.DataFrame(fold_results)

rf_wf = walk_forward_validation(
    df_model, feature_cols, 'revenue',
    model_fn=lambda: RandomForestRegressor(n_estimators=300, max_depth=8, random_state=42, n_jobs=-1),
    n_splits=4, test_weeks=12
)
print("Random Forest - Walk-forward:\n", rf_wf)
print("Avg MAPE:", rf_wf['MAPE'].mean())

xgb_wf = walk_forward_validation(
    df_model, feature_cols, 'revenue',
    model_fn=lambda: XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05,
                                    subsample=0.8, colsample_bytree=0.8, random_state=42),
    n_splits=4, test_weeks=12
)
print("\nXGBoost - Walk-forward:\n", xgb_wf)
print("Avg MAPE:", xgb_wf['MAPE'].mean())

importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

print(importance.to_string())

# Refit final model on all the data
final_model = RandomForestRegressor(n_estimators=300, max_depth=8, random_state=42, n_jobs=-1)
final_model.fit(df_model[feature_cols], df_model['revenue'])

# Use the pre-dummy version of df (has city/cuisine/product_type as plain columns)
history = df.copy()
group_cols = ['city', 'cuisine', 'product_type']
combinations = history[group_cols].drop_duplicates()

last_date = history['date'].max()
future_dates = [last_date + pd.Timedelta(weeks=i) for i in range(1, 13)]

forecast_rows = []

for _, combo in combinations.iterrows():
    combo_history = history[
        (history['city'] == combo['city']) &
        (history['cuisine'] == combo['cuisine']) &
        (history['product_type'] == combo['product_type'])
    ].sort_values('date').copy()

    for future_date in future_dates:
        new_row = {
            'city': combo['city'],
            'cuisine': combo['cuisine'],
            'product_type': combo['product_type'],
            'date': future_date,
            'month': future_date.month,
            'quarter': future_date.quarter,
            'week_of_year': future_date.isocalendar().week,
        }

        new_row['revenue_lag_1'] = combo_history['revenue'].iloc[-1]
        new_row['revenue_lag_4'] = combo_history['revenue'].iloc[-4] if len(combo_history) >= 4 else np.nan
        new_row['revenue_lag_13'] = combo_history['revenue'].iloc[-13] if len(combo_history) >= 13 else np.nan
        new_row['revenue_roll_mean_4'] = combo_history['revenue'].iloc[-4:].mean()
        new_row['revenue_roll_std_4'] = combo_history['revenue'].iloc[-4:].std()

        row_df = pd.DataFrame([new_row])
        row_df = pd.get_dummies(row_df, columns=['city', 'cuisine', 'product_type'])
        row_df = row_df.reindex(columns=feature_cols, fill_value=False)

        pred_revenue = final_model.predict(row_df[feature_cols])[0]
        new_row['revenue'] = pred_revenue
        forecast_rows.append(new_row)

        combo_history = pd.concat([combo_history, pd.DataFrame([new_row])], ignore_index=True)

forecast_df = pd.DataFrame(forecast_rows)
forecast_by_city = forecast_df.groupby(['city', 'date'])['revenue'].sum().reset_index()
print(forecast_by_city)

forecast_df.to_csv("forecast_results_Sysco.csv", index=False)

hist_by_city = history.groupby(['city', 'date'])['revenue'].sum().reset_index()

fig, ax = plt.subplots(figsize=(14, 6))

for city in forecast_by_city['city'].unique():
    hist_city = hist_by_city[hist_by_city['city'] == city]
    fcst_city = forecast_by_city[forecast_by_city['city'] == city]

    line = ax.plot(hist_city['date'], hist_city['revenue'], label=f'{city} - Historical')
    ax.plot(fcst_city['date'], fcst_city['revenue'], linestyle='--',
            color=line[0].get_color(), label=f'{city} - Forecast')

ax.axvline(last_date, color='gray', linestyle=':', label='Forecast start')
ax.set_title('Historical Revenue and 12-Week Forecast by City')
ax.set_xlabel('Date')
ax.set_ylabel('Revenue')
ax.legend()
plt.tight_layout()
plt.show()