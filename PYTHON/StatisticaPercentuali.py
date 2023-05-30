import pandas as pd

cols=pd.read_csv('../TXT/MediaColonne.txt')
medieColonne=cols['mediaColonne']
etichette_perc=pd.read_csv('../TXT/Etichette_Percentuali.txt')
def create_etichette_perc(medieColonne,etichette_perc):
    """
    Crea un file txt dove associa per ogni video le etichette (in percentuali)
    :param medieColonne: colonna medie colonne
    :param etichette_perc: file csv etichette_percentuali
    """
    val = 0
    for i in range(61):
        n=0
        list_perc = []
        list=[]
        for j in range(0,4):
            list_perc.append(round(medieColonne[val+j],2))
            n+=list_perc[j]
        list.append('Vid'+str(i))
        for j in range(0,4):
            list.append(round(float((list_perc[j]/5)*100),1))
        etichette_perc.loc[i]=list
        etichette_perc.to_csv('../TXT/Etichette_Percentuali.txt',sep=',')
        val+=4



create_etichette_perc(medieColonne,etichette_perc)