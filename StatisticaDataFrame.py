import pandas as pd

rows=pd.read_csv('TXT/MediaRighe.txt')
cols=pd.read_csv('TXT/MediaColonne.txt')
train_validation=pd.read_csv('CSV/Responses_train+validation.csv')
test=train_validation.apply(pd.to_numeric,errors='coerce')
test=test.drop('Gait ID', axis=1)

medieRighe=rows['mediaRighe']
medieColonne=cols['mediaColonne']
col1=cols['num1']
col2=cols['num2']
col3=cols['num3']
col4=cols['num4']
col5=cols['num5']

etichette_perc=pd.read_csv('TXT/Etichette_Percentuali.txt')


def check_big_col(col1,col2,col3,col4,col5):
    list=[col1,col2,col3,col4,col5]
    return list.index(max(list))

val = 0
for i in range(60):
    n=0
    list_perc = []
    list=[]
    for j in range(0,4):
        list_perc.append(round(medieColonne[val+j],1))
        n+=list_perc[j]
    list.append('Vid'+str(i))
    for j in range(0,4):
        list.append(int(round((list_perc[j]/5)*100,1)))
    etichette_perc.loc[i]=list
    etichette_perc.to_csv('TXT/Etichette_Percentuali.txt',sep=',')
    val+=4
