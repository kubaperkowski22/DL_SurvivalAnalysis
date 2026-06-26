import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

def prepare_small_intestine_cancer_data(filepath):
   
    na_values = [
        "Blank(s)", 
        "Unable to calculate", 
        "Unknown or size unreasonable (includes any tumor sizes 401-989)",
        "None/Unknown",
        "No/Unknown",
        "Unknown",
        "Recode scheme not yet available"
    ]

    df = pd.read_csv(filepath, sep=";", na_values=na_values, low_memory=False)
    print(f"Dane wczytane. Rozmiar początkowy: {df.shape}")

    # WyodrębnienieCzas i Status
    df['duration'] = pd.to_numeric(df['Survival months'], errors='coerce')
    df['event'] = df['Vital status recode (study cutoff used)'].apply(
        lambda x: 1 if pd.notna(x) and 'Dead' in str(x) else 0
    )
    
    df = df.dropna(subset=['duration'])

    # Czyszczenie niepotrzebnych kolumn
    columns_to_drop = [
        'Patient ID', 
        'Survival months', 
        'Vital status recode (study cutoff used)'
    ]
    df = df.drop(columns=columns_to_drop, errors='ignore')

    # Feature Engineering

    # Zmienna porządkowa: Wiek
    def parse_age(age_str):
        if pd.isna(age_str): return np.nan
        age_str = str(age_str).lower()
        if '90+' in age_str: return 90.0
        if '<1' in age_str or '00 years' in age_str: return 0.0
        try:
            return float(int(age_str.split('-')[0]) + 2)
        except:
            return np.nan

    if 'Age recode with <1 year olds and 90+' in df.columns:
        df['Age'] = df['Age recode with <1 year olds and 90+'].apply(parse_age)
        df = df.drop(columns=['Age recode with <1 year olds and 90+'])

    # Zmienne liczbowe
    num_cols = [
        "Year of diagnosis",
        "Time from diagnosis to treatment in days recode",
        "Tumor Size Over Time Recode (1988+)",
        "CS tumor size (2004-2015)",
        "Regional nodes examined (1988+)",
        "Regional nodes positive (1988+)"
    ]

    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            # W kolumnach dotyczących węzłów chłonnych i klasyfikacji CS, kody od 98 wzwyż to zazwyczaj braki
            if 'nodes' in col.lower() or 'tumor size' in col.lower():
                df.loc[df[col] >= 98, col] = np.nan

    # One-Hot Encoding
    cat_cols = [
        "Race recode (White, Black, Other)",
        "Sex",
        "Site recode - rare tumors",
        "SEER historic stage A (1973-2015)",
        "Derived AJCC T, 6th ed (2004-2015)",
        "Derived AJCC N, 6th ed (2004-2015)",
        "Derived AJCC M, 6th ed (2004-2015)",
        "T value - based on AJCC 3rd (1988-2003)",
        "N value - based on AJCC 3rd (1988-2003)",
        "M value - based on AJCC 3rd (1988-2003)",
        "AJCC stage 3rd edition (1988-2003)",
        "Radiation recode",
        "Chemotherapy recode (yes, no/unk)",
        "RX Summ--Surg Prim Site (1998-2022)"
    ]

    cat_cols_present = [col for col in cat_cols if col in df.columns]
    
    for col in cat_cols_present:
        df[col] = df[col].astype(str)

    df = pd.get_dummies(df, columns=cat_cols_present, dummy_na=False, drop_first=True)

    # usuwanie pustych kolumn
    threshold = 0.5 
    min_non_na = int(df.shape[0] * threshold)
    
    df = df.dropna(axis=1, thresh=min_non_na)
    

    # Podział na zbiory Train, Val, Test    
    try:
        df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
        df_train, df_val = train_test_split(df_train, test_size=0.25, random_state=42)
    except ValueError:
        print("za mało danych")
        df_train = df_test = df_val = df

    def get_x_y(data_frame):
        y_time = data_frame['duration'].values.astype('float32')
        y_event = data_frame['event'].values.astype('int32')
        x_features = data_frame.drop(columns=['duration', 'event'])
        return x_features, (y_time, y_event)

    x_train_df, y_train = get_x_y(df_train)
    x_val_df, y_val = get_x_y(df_val)
    x_test_df, y_test = get_x_y(df_test)


    # Imputacja i Skalowanie
    imputer = SimpleImputer(strategy='median')
    x_train_imputed = imputer.fit_transform(x_train_df)
    x_val_imputed = imputer.transform(x_val_df)
    x_test_imputed = imputer.transform(x_test_df)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train_imputed).astype('float32')
    x_val = scaler.transform(x_val_imputed).astype('float32')
    x_test = scaler.transform(x_test_imputed).astype('float32')

    val_data = (x_val, y_val)

    print(f"\nRozmiar X_train: {x_train.shape}")
    
    return x_train, y_train, val_data, x_test, y_test