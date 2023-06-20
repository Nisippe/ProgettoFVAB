import pandas as pd

test=pd.read_csv('C:/Users/drugo/PycharmProjects/ProgettoFVAB/CSV/responses_test.csv')
test=test.apply(pd.to_numeric,errors='coerce')
test=test.drop('Gait ID', axis=1)

#Soluzione Medie Colonne
df2 = test.mean(axis = 0, skipna = True)
df2=pd.DataFrame(df2)
df2.rename(columns = {0:'mediaColonne'}, inplace = True)
print(df2)
df2.to_csv('C:/Users/drugo/PycharmProjects/ProgettoFVAB/TXT/MediaColonneTest.txt',sep=',')

cols=pd.read_csv('C:/Users/drugo/PycharmProjects/ProgettoFVAB/TXT/MediaColonneTest.txt')
medieColonne=cols['mediaColonne']
etichette_perc=pd.read_csv('C:/Users/drugo/PycharmProjects/ProgettoFVAB/TXT/Etichette_Percentuali_Test.txt')


def create_etichette_perc(medieColonne,etichette_perc):
    """
    Crea un file txt dove associa per ogni video le etichette (in percentuali)
    :param medieColonne: colonna medie colonne
    :param etichette_perc: file csv etichette_percentuali
    """
    val = 0
    for i in range(15):
        list_perc = []
        list=[]
        for j in range(0,4):
            list_perc.append(round(medieColonne[val+j],2))
        list.append('Vid'+str(i))
        for j in range(0,4):
            list.append(round(float((list_perc[j]/5)*100),1))
        print(list)
        etichette_perc.loc[i]=list
        etichette_perc.to_csv('C:/Users/drugo/PycharmProjects/ProgettoFVAB/TXT/Etichette_Percentuali_Test.txt',sep=',')
        val+=4

create_etichette_perc(medieColonne,etichette_perc)