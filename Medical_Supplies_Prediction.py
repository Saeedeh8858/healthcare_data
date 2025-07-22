# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import os
import pycountry
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor

dataset_urls = {
    "population_country": "https://raw.githubusercontent.com/Saeedeh8858/healthcare_data/refs/heads/main/Total_Population_Country",
    "population+65_country": "https://raw.githubusercontent.com/Saeedeh8858/healthcare_data/refs/heads/main/Population_65_plus.csv",
    "GDP_current_us": "https://raw.githubusercontent.com/Saeedeh8858/healthcare_data/refs/heads/main/GDP_current_us.csv",
    "GDP_per_capital": "https://raw.githubusercontent.com/Saeedeh8858/healthcare_data/refs/heads/main/GDP_per_capita.csv"
}

value_names  = {
    "population_country": "population_country",
    "population+65_country": "population+65_country",
    "GDP_current_us": "GDP_current_us",
    "GDP_per_capital": "GDP_per_capita"
}

def preprocess_wide_to_long(path, output_path, value_name):
    df = pd.read_csv(path)
    id_vars = ["Country Name"]
    if "Country Code" in df.columns:
        id_vars.append("Country Code")
    if "Indicator Name" in df.columns:
        id_vars.append("Indicator Name")
    if "Indicator Code" in df.columns:
        id_vars.append("Indicator Code")
    df_melted = df.melt(
        id_vars=id_vars,
        var_name="Year",
        value_name=value_name
    )
    df_melted = df_melted.dropna(subset=[value_name])
    df_melted = df_melted[["Country Name", "Year", value_name]]
    df_melted.to_csv(output_path, index=False)

for key, value_name in value_names.items():
    url = dataset_urls[key]
    output_file = f"{key}.csv"
    preprocess_wide_to_long(url, output_file, value_name)

local_files = {
    "population_country": "population_country.csv",
    "population+65_country": "population+65_country.csv",
    "GDP_current_us": "GDP_current_us.csv",
    "GDP_per_capital": "GDP_per_capital.csv"
}

remote_files = {
    "diabetes": "https://raw.githubusercontent.com/Saeedeh8858/healthcare_data/main/Diabetes.csv",
    "Diabetes_treatment": "https://raw.githubusercontent.com/Saeedeh8858/healthcare_data/refs/heads/main/Diabetestreatment.csv",
    "hospitaldensity": "https://raw.githubusercontent.com/Saeedeh8858/healthcare_data/main/Hospital_density.csv",
    "inactivity": "https://raw.githubusercontent.com/Saeedeh8858/healthcare_data/main/inactivity.csv",
    "obesity": "https://raw.githubusercontent.com/Saeedeh8858/healthcare_data/main/obesity.csv",
    "smoking": "https://raw.githubusercontent.com/Saeedeh8858/healthcare_data/refs/heads/main/death-rate-smoking.csv",
    "tobocoprimarycare": "https://raw.githubusercontent.com/Saeedeh8858/healthcare_data/main/tobocoprimarycare.csv",
    "air-Polution": "https://raw.githubusercontent.com/Saeedeh8858/healthcare_data/refs/heads/main/AIR_11.csv.csv",
    "NCD_management": "https://raw.githubusercontent.com/Saeedeh8858/healthcare_data/refs/heads/main/NCD_management.csv",
    "nursingper10000": "https://raw.githubusercontent.com/Saeedeh8858/healthcare_data/refs/heads/main/nursingper10000.csv"
}

datasets = {
    "NCD_management":{
        "keep_columns": ['Location', 'Period', 'FactValueTranslationID'],
        "rename_cols": {'Location': 'country', 'Period': 'year', 'FactValueTranslationID': 'NCD_Status'}
    },
    "Diabetes_treatment":{
        "keep_columns": ['Location', 'Period', 'Dim1', 'Dim2' ,'FactValueNumeric'],
        "rename_cols": {'Location': 'country', 'Period': 'year', 'Dim1': 'gender' , 'Dim2':'age' , 'FactValueNumeric':'Diabetes_Treatment'}
    },
    "nursingper10000":{
        "keep_columns": ['Location', 'Period', 'Value'],
        "rename_cols": {'Location': 'country', 'Period': 'year', 'Value': 'nursingper10000'}
    },
    "GDP_per_capital":{
        "keep_columns": ['Country Name', 'Year', 'GDP_per_capita'],
        "rename_cols": {'Country Name': 'country', 'Year': 'year', 'GDP_per_capita': 'GDP_per_capita'}
    },
    "GDP_current_us":{
        "keep_columns": ['Country Name', 'Year', 'GDP_current_us'],
        "rename_cols": {'Country Name': 'country', 'Year': 'year', 'GDP_current_us': 'GDP_current_us'}
    },
    "population_country":{
        "keep_columns": ['Country Name', 'Year', 'population_country'],
        "rename_cols": {'Country Name': 'country', 'Year': 'year', 'population_country': 'population_country'}
    },
    "population+65_country":{
        "keep_columns": ['Country Name', 'Year', 'population+65_country'],
        "rename_cols": {'Country Name': 'country', 'Year': 'year', 'population+65_country': 'population+65_country'}
    },
    "tobocoprimarycare":{
        "keep_columns": ['Location', 'Period', 'Value'],
        "rename_cols": {'Location': 'country', 'Period': 'year', 'Value': 'TobaccoCare'}
    },
    "inactivity": {
        "keep_columns": ['Location', 'Period', 'FactValueNumeric'],
        "rename_cols": {'Location': 'country', 'Period': 'year', 'FactValueNumeric': 'Inactivity_Prevalence'}
    },
    "obesity": {
        "keep_columns": ['Location', 'Period', 'FactValueNumeric'],
        "rename_cols": {'Location': 'country', 'Period': 'year', 'FactValueNumeric': 'Obesity_Prevalence'}
    },
    "smoking": {
        "keep_columns": ['Entity', 'Year', 'Smoking mortality'],
        "rename_cols": {'Entity': 'country', 'Year': 'year', 'Smoking mortality': 'Smoking_Prevalence'}
    },
    "hospitaldensity": {
        "keep_columns": ['Location', 'Period', 'Value'],
        "rename_cols": {'Location': 'country', 'Period': 'year', 'Value': 'Beds_Per10000'}
    },
    "diabetes": {
        "keep_columns": ['SpatialDim', 'TimeDim', 'NumericValue'],
        "rename_cols": {'SpatialDim': 'country', 'TimeDim': 'year', 'NumericValue': 'Diabetes_Prevalence'}
    },
    "air-Polution": {
        "keep_columns": ["SpatialDim", "TimeDim", "NumericValue"],
        "rename_cols": {"SpatialDim": "country", "TimeDim": "year", "NumericValue": 'AirPolution_Prevalence' }
    }
}

special_cases = {
    'XKX': 'Kosovo',
    'ROM': 'Romania',
    'ZAR': 'Congo (Kinshasa)'
}

invalid_region_codes = {'AFR', 'SSA', 'EAS', 'WPR', 'EUR', 'AMR', 'EMR', 'SEAR'}

country_name_map = {
    "netherlands" : "netherlands (kingdom of the)",
    "oecd members": "oecd countries",
    "occupied palestinian territory, including east jerusalem": "palestine",
    "palestine, state of": "palestine",
    "russia": "russian federation",
    "tÃ¼rkiye": "turkey",
    "turkiye": "turkey",
    "venezuela, bolivarian republic of": "venezuela, rb",
    "yemen": "yemen, rep.",
    "bahamas": "bahamas, the",
    "bolivia (plurinational state of)": "bolivia, plurinational state of",
    "brunei": "brunei darussalam",
    "cote d'ivoire": "côte d'ivoire",
    "cÃ´te d'ivoire": "côte d'ivoire",
    "egypt": "egypt, arab rep.",
    "gambia": "gambia, the",
    "iran (islamic republic of)": "iran, islamic republic of",
    "iran, islamic rep.": "iran, islamic republic of",
    "korea, dem. people's rep.": "north korea",
    "korea, democratic people's republic of": "north korea",
    "korea, rep.": "south korea",
    "korea, republic of": "south korea",
    "republic of korea": "south korea",
    "lao pdr": "lao people's democratic republic",
    "micronesia (federated states of)": "micronesia",
    "micronesia, fed. sts.": "micronesia",
    "micronesia, federated states of": "micronesia",
    "micronesia (country)": "micronesia"
}
non_countries_keywords = [
    "region", "income", "world", "global", "g20", "who", "dividend",
    "classification", "group", "states", "area", "situation", "small states",
    "excluded", "early", "late", "fragile", "affected", "high income", "low income",
    "middle income", "upper middle income", "lower middle income", "wb", "ida", "ibrd"
]

def fix_encoding_issues(name):
    try:
        return name.encode('latin1').decode('utf-8')
    except:
        return name

def is_valid_country(name):
    if pd.isnull(name):
        return False
    name = name.strip().lower()
    try:
        pycountry.countries.lookup(name)
        return True
    except LookupError:
        return False

def convert_country_code_to_name(code):
    if pd.isnull(code):
        return code
    code = str(code).strip()
    upper_code = code.upper()
    if upper_code in invalid_region_codes:
        return None
    if len(upper_code) == 3 and upper_code.isalpha():
        if upper_code in special_cases:
            return special_cases[upper_code]
        country = pycountry.countries.get(alpha_3=upper_code)
        if country:
            return country.name
    try:
        country = pycountry.countries.lookup(code)
        return country.name
    except LookupError:
        return code

def standardize_country_column(series):
    series = series.apply(convert_country_code_to_name)
    series = series.str.lower().str.strip()
    series = series.apply(fix_encoding_issues)
    series = series.replace(country_name_map)
    return series

def clean_dataset(df, keep_columns=None, drop_fully_nan=True, rename_cols=None,
                  parse_dates=None, save_path=None, dataset_name=None,
                  year_filter=None):
    if drop_fully_nan:
        df = df.dropna(axis=1, how='all')
        df = df.dropna(axis=0, how='all')
    if keep_columns:
        df = df[keep_columns]
    if rename_cols:
        df = df.rename(columns=rename_cols)
    if dataset_name == "smoking" and 'Smoking_Prevalence' in df.columns:
        df['Smoking_Prevalence'] = (
            df['Smoking_Prevalence']
            .astype(str)
            .str.replace('.', '', regex=False)
            .str.replace(',', '', regex=False)
        )
        df['Smoking_Prevalence'] = pd.to_numeric(df['Smoking_Prevalence'], errors='coerce')
    if 'country' in df.columns:
        df['country'] = standardize_country_column(df['country'])
        df = df[df['country'].apply(is_valid_country)]
        df = df.drop_duplicates(subset=["country", "year"])
    value_map = {
        "No": 0,
        "Yes in some": 1,
        "Yes in most": 2
    }
    if 'TobaccoCare' in df.columns:
        df['TobaccoCare'] = df['TobaccoCare'].astype(str).str.strip()
        mapped = df['TobaccoCare'].map(value_map)
        unmapped = df['TobaccoCare'][mapped.isna()]
        numeric_converted = pd.to_numeric(unmapped, errors='coerce')
        df.loc[mapped.notna(), 'TobaccoCare'] = mapped.dropna()
        df.loc[mapped.isna(), 'TobaccoCare'] = numeric_converted
        df.loc[~df['TobaccoCare'].isin([0, 1, 2]), 'TobaccoCare'] = np.nan
        df['TobaccoCare'] = df['TobaccoCare'].fillna(0).astype(int)
    if dataset_name == "hospitaldensity" and 'Beds_Per10000' in df.columns:
        df['Beds_Per10000'] = pd.to_numeric(df['Beds_Per10000'], errors='coerce') / 10
    if parse_dates:
        for col in parse_dates:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
    if year_filter and 'year' in df.columns:
        try:
            df['year'] = pd.to_numeric(df['year'], errors='coerce')
            df = df[(df['year'] >= year_filter[0]) & (df['year'] <= year_filter[1])]
        except:
            pass
    if 'country' in df.columns and 'year' in df.columns:
        df.sort_values(["country", "year"], inplace=True)
    if save_path:
        df.to_csv(save_path, index=False)
    return df

all_files = {**local_files, **remote_files}
cleaned_data = {}

for name, path in all_files.items():
    if name not in datasets:
        continue
    try:
        df = pd.read_csv(path, encoding='utf-8-sig')
        rules = datasets[name]
        save_path = f"{name}_cleaned.csv"
        cleaned_df = clean_dataset(df, save_path=save_path, dataset_name=name, year_filter=(2012, 2025), **rules)
        cleaned_data[name] = cleaned_df
    except Exception as e:
        pass

from functools import reduce

dfs = list(cleaned_data.values())
master_df = reduce(lambda left, right: pd.merge(left, right, on=['country', 'year'], how='outer'), dfs)
master_df.to_csv("master_cleaned.csv", index=False, encoding="utf-8-sig")

def fill_missing_extended(df, threshold, min_valid_rows_per_country):
    df = df.copy()
    cols_to_check = ['Beds_Per10000', 'TobaccoCare', 'Inactivity_Prevalence',
                     'Smoking_Prevalence', 'AirPolution_Prevalence', 'NCD_Status',
                     'Obesity_Prevalence', 'Diabetes_Prevalence','Diabetes_Treatment','gender',
                     'GDP_current_us', 'GDP_per_capita','nursingper10000',
                     'population+65_country', 'population_country']
    numeric_cols = ['Beds_Per10000', 'Inactivity_Prevalence', 'Smoking_Prevalence','nursingper10000','Diabetes_Treatment',
                    'AirPolution_Prevalence', 'Obesity_Prevalence', 'Diabetes_Prevalence',
                    'GDP_current_us', 'GDP_per_capita', 'population+65_country', 'population_country']
    categorical_cols = ['TobaccoCare', 'NCD_Status']
    for col in numeric_cols:
        df[col] = df.groupby("country")[col].transform(lambda x: x.interpolate(method='linear', limit_direction='both'))
        df[col] = df.groupby("country")[col].transform(lambda x: x.fillna(x.mean()))
    for col in categorical_cols:
        df[col] = df.groupby("country")[col].transform(lambda x: x.fillna(x.mode().iloc[0]) if not x.mode().empty else x)
    df = df[df[cols_to_check].isnull().mean(axis=1) < threshold]
    valid_counts = df.groupby("country").apply(
        lambda x: x[cols_to_check].notnull().all(axis=1).sum()
    )
    valid_countries = valid_counts[valid_counts >= min_valid_rows_per_country].index
    df = df[df["country"].isin(valid_countries)]
    df = df.dropna(subset=cols_to_check)
    return df

master_df = fill_missing_extended(master_df, threshold=0.5, min_valid_rows_per_country=5)

import missingno as msno
msno.matrix(master_df)
msno.bar(master_df)

import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

df = master_df.copy()
df['Ratio_65plus'] = df['population+65_country'] / df['population_country']
df['Demand_per_million'] = df['NCD_Status'] / df['population_country'] * 1_000_000
threshold = df['Demand_per_million'].quantile(0.2)
df['Shortage'] = (df['Demand_per_million'] < threshold).astype(int)
feature_cols = [
    'GDP_per_capita', 'Ratio_65plus', 'Beds_Per10000', 'TobaccoCare',
    'Obesity_Prevalence', 'Inactivity_Prevalence', 'Smoking_Prevalence'
]
df = df.dropna(subset=feature_cols + ['Shortage'])
X = df[feature_cols]
y = df['Shortage']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
clf = xgb.XGBClassifier(scale_pos_weight=scale_pos_weight, n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1-score:", f1_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Shortage', 'Shortage'], yticklabels=['No Shortage', 'Shortage'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
xgb.plot_importance(clf)
plt.title("Feature Importance for Shortage Prediction")
plt.show()
print(df['Shortage'].value_counts())

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
df = master_df.copy()
df['Ratio_65plus'] = df['population+65_country'] / df['population_country']
feature_cols = ['Ratio_65plus', 'GDP_per_capita', 'Beds_Per10000', 'TobaccoCare']
df['Health_Index'] = df['NCD_Status']
df = df.dropna(subset=feature_cols + ['Health_Index'])
X = df[feature_cols]
y = df['Health_Index']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"MSE (Loss): {mse:.4f}")
print(f"R2 Score: {r2:.4f}")
xgb.plot_importance(model)
plt.show()
y_baseline = np.full_like(y_test, y_train.mean())
baseline_rmse = np.sqrt(mean_squared_error(y_test, y_baseline))
baseline_r2 = r2_score(y_test, y_baseline)
print(f"Baseline RMSE (mean prediction): {baseline_rmse:.4f}")
print(f"Baseline R2 (mean prediction): {baseline_r2:.4f}")
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, y_train)
y_lr_pred = lr.predict(X_test)
lr_rmse = np.sqrt(mean_squared_error(y_test, y_lr_pred))
lr_r2 = r2_score(y_test, y_lr_pred)
print(f"Linear Regression RMSE: {lr_rmse:.4f}")
print(f"Linear Regression R2: {lr_r2:.4f}")
importances = model.feature_importances_
feature_importance_df = pd.DataFrame({'feature': X.columns, 'importance': importances})
print(feature_importance_df.sort_values('importance', ascending=False))

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
df = master_df.copy()
scaler = StandardScaler()
df['Ratio_65plus'] = df['population+65_country'] / df['population_country']
feature_cols = ['Inactivity_Prevalence', 'Smoking_Prevalence', 'AirPolution_Prevalence', 'Obesity_Prevalence', 'Diabetes_Prevalence','Ratio_65plus']
df['Health_Index'] = df[['NCD_Status','Beds_Per10000']].mean(axis=1)
df = df.dropna(subset=feature_cols + ['Health_Index'])
X = df[feature_cols]
y = df['Health_Index']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print(f"RMSE: {rmse:.3f}")
print(f"R²: {r2:.3f}")
import matplotlib.pyplot as plt
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, alpha=0.7, color='teal')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect Prediction')
plt.xlabel('Actual Health Index', fontsize=14)
plt.ylabel('Predicted Health Index', fontsize=14)
plt.title('Actual vs Predicted Health Index', fontsize=16, fontweight='bold')
plt.legend()
plt.tight_layout()
plt.show()

df = master_df.copy()
scaler = StandardScaler()
df['Ratio_65plus'] = df['population+65_country'] / df['population_country']
feature_cols = ['Ratio_65plus','GDP_per_capita','Beds_Per10000','TobaccoCare']
health_cols = ['NCD_Status']
df['Health_Index'] = master_df[health_cols].mean(axis=1)
df = df.dropna(subset=feature_cols + ['Health_Index'])
X = df[feature_cols]
y = df['Health_Index']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
print(f"RMSE: {rmse:.4f}")
print(f"R2 Score: {r2:.4f}")
df_avg = df.groupby('country')[feature_cols].mean().reset_index()
latest_pop = df[df['year'] == df['year'].max()][['country', 'population_country']]
df_avg = df_avg.merge(latest_pop, on='country', how='left')
X_avg = df_avg[feature_cols]
y_pred_avg = model.predict(X_avg)
df_avg['Predicted_Demand'] = y_pred_avg
df_avg['Demand_per_million'] = df_avg['Predicted_Demand'] / df_avg['population_country'] * 1_000_000
top10_avg = df_avg.sort_values('Predicted_Demand', ascending=False).head(10)
print(top10_avg[['country', 'Predicted_Demand', 'Demand_per_million']])
sns.set(style="whitegrid")
plt.figure(figsize=(12, 6))
top10_total = df_avg.sort_values('Predicted_Demand', ascending=False).head(10)
plt.subplot(1, 2, 1)
sns.barplot(data=top10_total, x='Predicted_Demand', y='country', palette='Blues_d')
plt.title("Top 10 Countries by Total Predicted Demand")
plt.xlabel("Predicted Demand")
plt.ylabel("Country")
top10_per_million = df_avg.sort_values('Demand_per_million', ascending=False).head(10)
plt.subplot(1, 2, 2)
sns.barplot(data=top10_per_million, x='Demand_per_million', y='country', palette='Greens_d')
plt.title("Top 10 Countries by Demand per Million")
plt.xlabel("Demand per Million")
plt.ylabel("Country")
plt.tight_layout()
plt.show()
sns.barplot(
    data=top10_total,
    x='Predicted_Demand',
    y='country',
    hue='country',
    palette='Blues_d',
    legend=False
)
sns.barplot(
    data=top10_per_million,
    x='Demand_per_million',
    y='country',
    hue='country',
    palette='Greens_d',
    legend=False
)

df = master_df.copy()
scaler = StandardScaler()
df['Ratio_65plus'] = df['population+65_country'] / df['population_country']
df['Ratio_young'] = 1 - df['Ratio_65plus']
df['Beds_per_Elderly'] = df['Beds_Per10000'] / df['population+65_country']
df['Smoking_GDP_Interaction'] = df['Smoking_Prevalence'] * df['GDP_per_capita']
df['Health_Access_Index'] = df[['Beds_Per10000','nursingper10000']].mean(axis=1)
df['Health_Index'] = df['Health_Access_Index']
feature_cols = ['Ratio_65plus', 'GDP_per_capita', 'Ratio_young','Smoking_GDP_Interaction','Inactivity_Prevalence','Obesity_Prevalence']
df = df.dropna(subset=feature_cols + ['Health_Index'])
X = df[feature_cols]
y = df['Health_Index']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
print(f"RMSE: {rmse:.4f}")
print(f"R2 Score: {r2:.4f}")
xgb.plot_importance(model)
plt.show()
df_avg = df.groupby('country')[feature_cols].mean().reset_index()
latest_pop = df[df['year'] == df['year'].max()][['country', 'population_country']]
df_avg = df_avg.merge(latest_pop, on='country', how='left')
X_avg = df_avg[feature_cols]
y_pred_avg = model.predict(X_avg)
df_avg['Predicted_Demand'] = y_pred_avg
df_avg['Demand_per_million'] = df_avg['Predicted_Demand'] / df_avg['population_country'] * 1_000_000
top10_avg = df_avg.sort_values('Predicted_Demand', ascending=False).head(10)
print(top10_avg[['country', 'Predicted_Demand', 'Demand_per_million']])
sns.set(style="whitegrid")
plt.figure(figsize=(12, 6))
top10_total = df_avg.sort_values('Predicted_Demand', ascending=False).head(10)
plt.subplot(1, 2, 1)
sns.barplot(data=top10_total, x='Predicted_Demand', y='country', palette='Blues_d')
plt.title("Top 10 Countries by Total Predicted Demand")
plt.xlabel("Predicted Demand")
plt.ylabel("Country")
top10_per_million = df_avg.sort_values('Demand_per_million', ascending=False).head(10)
plt.subplot(1, 2, 2)
sns.barplot(data=top10_per_million, x='Demand_per_million', y='country', palette='Greens_d')
plt.title("Top 10 Countries by Demand per Million")
plt.xlabel("Demand per Million")
plt.ylabel("Country")
plt.tight_layout()
plt.show()

df = master_df.copy()
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df['Ratio_65plus'] = df['population+65_country'] / df['population_country']
df['Ratio_young'] = 1 - df['Ratio_65plus']
df['Health_Access_Index'] = df[['Diabetes_Treatment']].mean(axis=1)
df['Health_Index'] = df['Health_Access_Index']
df = pd.get_dummies(df, columns=['gender'])
feature_cols = ['GDP_per_capita', 'Diabetes_Prevalence','Ratio_young','Ratio_65plus']
feature_cols += [col for col in df.columns if col.startswith('gender_')]
df = df.dropna(subset=feature_cols + ['Health_Index'])
X = df[feature_cols]
y = df['Health_Index']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
print(f"RMSE: {rmse:.4f}")
print(f"R2 Score: {r2:.4f}")
import matplotlib.pyplot as plt
xgb.plot_importance(model, height=0.6)
plt.title("Feature Importance")
plt.show()

df = master_df.copy()
scaler = StandardScaler()
df['Ratio_65plus'] = df['population+65_country'] / df['population_country']
df['Ratio_young'] = 1 - df['Ratio_65plus']
df['Health_Access_Index'] = df[['Diabetes_Treatment','Beds_Per10000']].mean(axis=1)
df['Health_Index'] = df['Health_Access_Index']
df = pd.get_dummies(df, columns=['gender'])
feature_cols = ['GDP_per_capita', 'Diabetes_Prevalence','Ratio_65plus','Obesity_Prevalence','Inactivity_Prevalence' ]
feature_cols += [col for col in df.columns if col.startswith('gender_')]
df = df.dropna(subset=feature_cols + ['Health_Index'])
X = df[feature_cols]
y = df['Health_Index']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print(f"RMSE: {rmse:.4f}")
print(f"R2 Score: {r2:.4f}")
df_avg_year = df.groupby(['year', 'country'])['Health_Index'].mean().reset_index()
top10_2015 = df_avg_year[df_avg_year['year'] > 2021].sort_values(by='Health_Index', ascending=False).head(10)
print(top10_2015)
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(10,6))
sns.barplot(data=top10_2015, x='Health_Index', y='country', palette='viridis')
plt.title('Top 10 Countries by Health Index in 2015')
plt.xlabel('Average Health Index')
plt.ylabel('Country')
plt.show()

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
df = master_df.copy()
df = pd.get_dummies(df, columns=['gender'])
def infer_gender(row):
    if row['gender_Male'] == 1:
        return 'Male'
    elif row['gender_Female'] == 1:
        return 'Female'
    elif row['gender_Both sexes'] == 1:
        return 'Both'
    else:
        return 'Unknown'
df['gender'] = df.apply(infer_gender, axis=1)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df['Ratio_65plus'] = df['population+65_country'] / df['population_country']
df['Ratio_young'] = 1 - df['Ratio_65plus']
df['Smoking_Prevalence_Percent'] = df['Smoking_Prevalence'] / df['population_country'] * 100
df['Smoking_x_Young'] = df['Smoking_Prevalence_Percent'] * df['Ratio_young']
df['Smoking_x_Eldery'] = df['Smoking_Prevalence_Percent'] * df['Ratio_65plus']
df['Ratio_young_to_elderly'] = df['Ratio_young'] / (df['Ratio_65plus'] + 1e-6)
df['GDP_per_capita_per_Youth'] = df['GDP_per_capita'] / df['Ratio_young']
feature_cols = ['GDP_per_capita' , 'population_country' , 'Smoking_x_Young' , 'Beds_Per10000' , 'Ratio_young_to_elderly']
feature_cols += [col for col in df.columns if col.startswith('gender_')]
df['Health_Index'] = df['TobaccoCare']
df = df.dropna(subset=feature_cols + ['Health_Index'])
X = df[feature_cols]
y = df['Health_Index']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = xgb.XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print(f"XGBoost RMSE: {rmse:.4f}")
print(f"XGBoost R2: {r2:.4f}")
import matplotlib.pyplot as plt
xgb.plot_importance(model)
plt.show()
df_grouped = df.groupby(['country', 'gender'])['Smoking_Prevalence'].mean().reset_index()
top_countries = df_grouped.groupby('country')['Smoking_Prevalence'].mean().nlargest(10).index
df_top = df_grouped[df_grouped['country'].isin(top_countries)]
plt.figure(figsize=(15,8))
sns.barplot(data=df_top, x='country', y='Smoking_Prevalence', hue='gender')
plt.xticks(rotation=45)
plt.ylabel('Smoking Prevalence (%)')
plt.xlabel('Country')
plt.legend(title='Gender')
plt.show()

import matplotlib.pyplot as plt
import pandas as pd
tasks = [
    ("Data Collection & Cleaning", '2025-04-22', '2025-05-15'),
    ("Feature Engineering & EDA", '2025-05-16', '2025-05-31'),
    ("Modeling & Evaluation", '2025-06-01', '2025-06-15'),
    ("Interpretation & Visualization", '2025-06-16', '2025-06-30'),
    ("Writing & Reporting", '2025-07-01', '2025-07-24'),
    ("Thesis Registration Date", '2025-04-15', '2025-07-24'),
]
df = pd.DataFrame(tasks, columns=["Task", "Start", "Finish"])
df["Start"] = pd.to_datetime(df["Start"])
df["Finish"] = pd.to_datetime(df["Finish"])
df["Duration"] = (df["Finish"] - df["Start"]).dt.days
fig, ax = plt.subplots(figsize=(10, 5))
colors = plt.cm.Pastel1(range(len(df)))
for i, row in df.iterrows():
    ax.barh(row["Task"], row["Duration"], left=row["Start"], color=colors[i], edgecolor='k')
ax.set_xlabel('Date')