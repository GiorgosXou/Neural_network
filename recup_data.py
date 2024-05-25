import pandas as pd


df = pd.read_excel('data_2.xlsx')


start_row_1 = 2  
end_row_1 = 756  




start_col_1 = 1 
end_col_1 = 101  


df_1 = df.iloc[start_row_1:end_row_1, start_col_1:end_col_1]


df_1 = df_1.fillna(0)


transposed_data_1 = df_1.values.T.tolist()


TrainData = [list(col) for col in transposed_data_1]

print("trainData len:", len(TrainData[0]))





# Définir les limites de la deuxième plage (par exemple, de DH3 à DI779)
start_row_2 = 2  # index de la première ligne (0 pour la première ligne)
end_row_2 = 756  # index de la dernière ligne + 1 558
start_col_2 = 102  # index de la première colonne (104 pour DH, 105 pour DI)
end_col_2 = 104  # index de la dernière colonne + 1 (106 pour DI inclus)

# Sélectionner la deuxième plage de cellules
df_2 = df.iloc[start_row_2:end_row_2, start_col_2:end_col_2]

# Vérifiez que les données sont bien sélectionnées
print(df_2.head()) 


# Remplacer les cellules vides par 0
df_2 = df_2.fillna(0)


transposed_data_2 = df_2.values.T.tolist()

# Formater les données pour correspondre au format souhaité
Test_data = [list(col) for col in transposed_data_2]

# Afficher le résultat




print("testData len:", len(TrainData[0]))


# Enregistrer les données dans un fichier test_humain_test.py
with open('train_data_humain.py', 'w') as f:
    f.write(f'inputs_data = {TrainData}\n')
    
with open('test_data_humain.py', 'w') as f:
    f.write(f'testData = {Test_data}\n')