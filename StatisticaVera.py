import pandas as pd

rows=pd.read_csv('TXT/MediaRighe.txt')
cols=pd.read_csv('TXT/MediaColonne.txt')

medieRighe=rows['mediaRighe']
medieColonne=cols['mediaColonne']
col5=cols['num5']

val = 0
for i in range(60):
    df = medieColonne[val:val + 4]
    listdf = df.tolist()
    max(listdf)
    index = listdf.index(max(listdf))
    print(index)
    colNeutra = medieColonne[val + 4]

    for j in range(87):
        partecipante=medieRighe[j]



    val+=4