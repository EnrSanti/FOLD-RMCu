import pandas as pd
import numpy as np

def clean_flight_dataset(input_path, output_path):
    print(f"Caricamento di {input_path}...")
    
    # Leggiamo il CSV
    # Nota: low_memory=False evita avvisi su tipi di dati misti in colonne grandi
    df = pd.read_csv(input_path, low_memory=False)
    initial_rows = len(df)
    
    # 1. Convertiamo i placeholder testuali come '?' in NaN reali
    # Aggiungi altri simboli se il tuo dataset ne usa di diversi (es. 'N/A', 'None')
    df = df.replace('?', np.nan)
    
    # 2. Rimuoviamo ogni riga che ha ALMENO un valore mancante (NaN)
    # axis=0: opera sulle righe
    # how='any': basta un solo valore nullo per eliminare la riga
    df_cleaned = df.dropna(axis=0, how='any')
    
    # 3. Reset dell'indice per "compattare" il dataset
    # drop=True evita di creare una vecchia colonna 'index'
    df_cleaned = df_cleaned.reset_index(drop=True)
    
    final_rows = len(df_cleaned)
    removed_rows = initial_rows - final_rows
    
    print(f"Pulizia completata.")
    print(f"Righe originali: {initial_rows}")
    print(f"Righe rimosse:   {removed_rows} ({(removed_rows/initial_rows)*100:.2f}%)")
    print(f"Righe finali:    {final_rows}")
    
    # Salvataggio del dataset pulito
    df_cleaned.to_csv(output_path, index=False)
    print(f"File salvato in: {output_path}")

# Esempio d'uso
clean_flight_dataset('Combined_Flights_2022.csv', 'Combined_Flights_2022_cleaned.csv')