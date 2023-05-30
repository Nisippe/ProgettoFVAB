import pandas as pd

train_validation=pd.read_csv('C:/Users/drugo/PycharmProjects/ProgettoFVAB/CSV/Responses_train+validation.csv')
train_validation=train_validation.apply(pd.to_numeric,errors='coerce')
train_validation=train_validation.drop('Gait ID', axis=1)

#Soluzione Medie Colonne
df2 = train_validation.mean(axis = 0, skipna = True)
df2=pd.DataFrame(df2)
df2.to_csv('TXT/MediaColonne.txt',sep=',')









