#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Atividade para trabalhar o pré-processamento dos dados.

Criação de modelo preditivo para diabetes e envio para verificação de peformance
no servidor.

@author: Aydano Machado <aydano.machado@gmail.com>
"""
#%%
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import requests
# %%
print('\n - Lendo o arquivo com o dataset sobre diabetes')
data = pd.read_excel('diabetes_dataset.xlsx')
# %%
# Criando X and y par ao algorítmo de aprendizagem de máquina.\
print(data.head(15))
#%%
print(data.info())
#%%
print(data.describe())
# %%
data.isna().sum()
#%%
print(' - Criando X e y para o algoritmo de aprendizagem a partir do arquivo diabetes_dataset')
# Caso queira modificar as colunas consideradas basta alterar o array a seguir.
# colunas com mtos NAs retiradas
cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
data_modified = data[cols]
#%%
# retiradas linhas com NAs restantes
data_modified = data_modified.dropna()
data_modified.isna().sum()
#%%
feature_cols =['Pregnancies', 'Glucose', 'BloodPressure', 'BMI', 'DiabetesPedigreeFunction', 'Age']
X = data_modified[feature_cols]
y = data_modified.Outcome


#%%
# Ciando o modelo preditivo para a base trabalhada
print(' - Criando modelo preditivo')
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X, y)

#realizando previsões com o arquivo de
print(' - Aplicando modelo e enviando para o servidor')
data_app = pd.read_excel('diabetes_app.xlsx')
data_app = data_app[feature_cols]
y_pred = neigh.predict(data_app)

# Enviando previsões realizadas com o modelo para o servidor
URL = "https://aydanomachado.com/mlclass/01_Preprocessing.php"

#TODO Substituir pela sua chave aqui
DEV_KEY = "Machine Losers"

# json para ser enviado para o servidor
data = {'dev_key':DEV_KEY,
        'predictions':pd.Series(y_pred).to_json(orient='values')}

# Enviando requisição e salvando o objeto resposta
r = requests.post(url = URL, data = data)

# Extraindo e imprimindo o texto da resposta
pastebin_url = r.text
print(" - Resposta do servidor:\n", r.text, "\n")