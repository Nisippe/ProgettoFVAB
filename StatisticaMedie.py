import pandas as pd

train_validation=pd.read_csv('CSV/Responses_train+validation.csv')
test=train_validation.apply(pd.to_numeric,errors='coerce')
test=test.drop('Gait ID', axis=1)

columns=test.columns
list1 = []
list2 = []
list3 = []
list4 = []
list5 = []
def count_n_video_column(df,col,value):
    return (df[col]==value).sum()


for col in columns:
    count1 = count_n_video_column(test, col, 1)
    count2 = count_n_video_column(test, col, 2)
    count3 = count_n_video_column(test, col, 3)
    count4 = count_n_video_column(test, col, 4)
    count5 = count_n_video_column(test,col,5)
    list1.append(count1)
    list2.append(count2)
    list3.append(count3)
    list4.append(count4)
    list5.append(count5)

#Soluzione Medie Colonne
df2 = test.mean(axis = 0, skipna = True)
df2=pd.DataFrame(df2)


df2=df2.assign(num1=list1,num2=list2,num3=list3,num4=list4,num5=list5)
df2.to_csv('TXT/MediaColonne.txt',sep=',')









