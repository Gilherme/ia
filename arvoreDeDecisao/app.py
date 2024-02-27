import numpy as np 
import itertools
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from six import StringIO
# import pydotplus
# from IPython.display import Image 


# IMPORTANDO O DATASET PARA O DATAFRAME
df = pd.read_csv('arvoreDeDecisao/dataset_einstein.csv', delimiter=';')

# MOSTRANDO AS PRIMEIRAS CINCO LINHAS
print(df.head(5))

count_row = df.shape[0]  # PEGANDO OS NÚMEROS DE REGISTROS
count_col = df.shape[1]  # PEGANDO OS NUMEROS DE COLUNAS

# REMOVENDO OS REGISTROS NOS QUAIS PELO MENOS UM CAMPO ESTÁ EM BRANCO (NAN) 
df = df.dropna()

print(df.head(10))

print('Quantidade de colunas: ', df.shape[1])
print('Quantidade de linhas: ', df.shape[0])

#VAMOS VERIFICAR SE O BANCO DE DADOS ESTÁ BALANCEADO OU DESBALANCEADO
print ('Total de registros negativos: ', df[df['SARS-Cov-2 exam result'] =='negative'].shape[0])
print ('Total de registros positivos: ', df[df['SARS-Cov-2 exam result'] =='positive'].shape[0])

"""Precisamos converter o Dataframe para um Array Numpy, que é o tipo de dados que 
iremos usar no treinamento. Também iremos já separar o Dataset em dois. Um com as
 features de entrada, e outro com os labels (etiquetas, rótulos do registro).   
Neste caso, estamos tentando fazer um classificador para o teste do Covid, neste caso,
 queremos treinar o nosso modelo com a etiqueta presente no campo 'SARS-Cov-2 exam result'"""

# ARMAZENAR OS RESULTADOS (ETIQUETAS) EM UMA VARIAVÉL Y
Y = df['SARS-Cov-2 exam result'].values
print(Y)

# ARMAZENAR AS FEATURES EM UMA VAVIÁVEL X

X = df[['Hemoglobin', 'Leukocytes', 'Basophils','Proteina C reativa mg/dL']].values

print(X)

X_treino, X_teste, Y_treino, Y_teste = train_test_split(X, Y, test_size=0.2, random_state=3)

# CRIAR UM ALGORTIMO QUE SERÁ DO TIPO DE ÁRVORE DE DECISÃO

algortimo_arvore = DecisionTreeClassifier(criterion='entropy', max_depth=5)
# AGORA EM MINHA_ARVORE EU TENHO ASSOCIADA A ELA O ALGORITMO DE TREINAMENTO, 
# BASICAMENTE A RECEITA QUE VIMOS NA PARTE TÉORICA. 

#AGORA PRECISAMOS TREINÁ-LA
modelo = algortimo_arvore.fit(X_treino, Y_treino)

#PODEMOS MOSTRAR A FEATURE MAIS IMPORTANTE (WHITE BOX?)
print (modelo.feature_importances_)

nome_features = ['Hemoglobin', 'Leukocytes', 'Basophils','Proteina C reativa mg/dL']
nome_classes = modelo.classes_

