import pandas as pd

train_validation=pd.read_csv('CSV/Responses_train+validation.csv')
test=train_validation.drop(0)
test=train_validation.apply(pd.to_numeric,errors='coerce')
columns=test.columns

#Soluzione Medie Colonne
df2 = test.mean(axis = 0, skipna = True)
df2.to_csv('TXT/MediaColonne.txt', sep=' ')

#Soluzione Medie Righe
df2 = test.mean(axis = 1, skipna = True)
df2.to_csv('TXT/MediaRighe.txt', sep=' ')