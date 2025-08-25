import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pickle
#from plot import plotting



base_path = Path(__file__).parent
print(base_path)

csv_path = base_path / "Crop_recommendation.csv"
df = pd.read_csv(csv_path)


#analisi dei dati
print(df.shape)

print(df.isnull().sum())

#plotting(df,"Saved_models/plots")
print(df.head())

#eliminazione degli outliers

# Copia del dataset originale
df_clean = df.copy()

# Rimuove outlier da tutte le colonne numeriche
"""
for col in df_clean.drop(columns='label').columns:
    Q1 = df_clean[col].quantile(0.25)
    Q3 = df_clean[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
"""
print(df_clean.head())

#plotting(df_clean)

#definizione di encoder e scaler
le =LabelEncoder()
scaler = StandardScaler()

scaling_columns = ['N','P','K','temperature','humidity','ph','rainfall']

# label encoding
df_clean['label'] = le.fit_transform(df_clean['label'])

X = df_clean.drop("label", axis=1)
y = df_clean["label"]

# split prima dello scaling
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# scaler
x_train[scaling_columns] = scaler.fit_transform(x_train[scaling_columns])  # fit solo sul train per calcolare media e deviazione
x_test[scaling_columns] = scaler.transform(x_test[scaling_columns])


with open("../clean_df.pkl", "wb") as f:
    pickle.dump(df_clean, f)
