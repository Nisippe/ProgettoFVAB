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

def check_big_col(col1,col2,col3,col4,col5):
    list=[col1,col2,col3,col4,col5]
    return list.index(max(list))

val = 0
for i in range(60):
    df = medieColonne[val:val + 4]
    listdf = df.tolist()
    maxMedia=max(listdf)
    index = listdf.index(maxMedia)
    print(index)
    index_n=check_big_col(col1,col2, col3, col4, col5)
    for j in range(87):
        partecipante=medieRighe[j]
        if (df[0].isnull().values.any()):
            if(index==3):
                if(index_n==4):
                    '''
                    Soluzione neutra completa:
                    3-3-3-5
                    '''
                    test=test[j,val]=3
                    test=test[j,val+1]=3
                    test=test[j,val+2]=3
                    test=test[j,val+3]=5
                elif(index_n==3):
                    '''
                    Soluzione neutra:
                    Media partecipanti con neutro 4
                    ?-?-?-4
                    '''
                else:
                    '''
                    Soluzione strana:
                    Media partecipanti con neutro 3 o minore
                    ?-?-?-3
                    '''
            else:
                for i in range(0,4):
                    '''
                    Soluzione prime 3 colonne:
                    -Se max value = 5 allora:
                    gli altri vanno a 1
                    5-1-1-1
                    1-5-1-1
                    1-1-5-1
                    
                    if index_n==5:
                        if i is index:
                            test[val+i]=5
                        else:
                            test[val+i]=1
                    elif index_n==4:
                        if i is index:
                            test[val+i]=4
                        else:
                    elif index_n==3:
                    '''
    val+=4