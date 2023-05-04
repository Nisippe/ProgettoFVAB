import pandas as pd

train_validation=pd.read_csv('CSV/Responses_train+validation.csv')
#test=train_validation.drop(0)
test=train_validation.apply(pd.to_numeric,errors='coerce')
test=test.drop('Gait ID', axis=1)
print(test.head())
columns=test.columns
list5 = []
for cols in columns:
    count=test[cols].value_counts()
    #print(count)
    count5=(test[cols]==5).sum()
    list5.append(count5)
    print(count5)


#Soluzione Medie Colonne
df2 = test.mean(axis = 0, skipna = True)
df2=pd.DataFrame(df2)
print(df2)
df2=df2.assign(num5=list5)
df2.to_csv('TXT/MediaColonne.txt',sep=',')


