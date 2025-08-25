import pickle
import pandas as pd
import numpy as np
from dataset.DataFrame import scaler, le
import os



def predict_with_model(model_path):
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    scaling_columns = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']


    values = []
    for column in scaling_columns:
        val = float(input(f"Inserisci il valore per {column}: "))
        values.append(val)

    values_df = pd.DataFrame([values], columns=scaling_columns)
    scaled_values = scaler.transform(values_df)

    print(model.feature_names_in_)
    prediction = model.predict(scaled_values)
    print("\nBest crop is:", le.inverse_transform(prediction)[0])


def make_prediction():
    while True:
        os.system('cls')
        print("\n--- MENU PREDIZIONI ---")
        print("1. Usa Random Forest")
        print("2. Usa SVC")
        print("3. Usa KNN")
        print("4. Usa Logistic Regression")
        print("0. Torna indietro")
        choice = input("Scegli un modello: ")

        if choice == "0":
            break
        elif choice == "1":
            predict_with_model('Saved_models/rm_model.pkl')
        elif choice == "2":
            predict_with_model("Saved_models/svc_model.pkl")
        elif choice == "3":
            predict_with_model("Saved_models/knn_model.pkl")
        elif choice == "4":
            predict_with_model("Saved_models/log_reg.pkl")