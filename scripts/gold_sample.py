import pandas as pd
inputfile=open('../data/gold.csv',encoding='UTF-8')
df1=pd.read_csv(inputfile)
pos_len=len(df1.loc[df1.loc[:,'class']=='pos',:])          #calculate the positve cases number in original dataset
neg_len=len(df1.loc[df1.loc[:,'class']=='neg',:])          #calculate the negative cases number in original dataset
pos_num=int(1000*pos_len/(len(df1)))                       #calculate the positive proportion in original dataset

pos_sample=df1.loc[df1.loc[:,'class']=='pos',:].sample(n=pos_num, random_state = 1)          #generate the positive samples with the specified seed number  n
neg_sample=df1.loc[df1.loc[:,'class']=='neg',:].sample(n=1000-pos_num,random_state = 1)      #generate the negative samples with the specified seed number (1000-n)
m_sample=pd.concat([pos_sample,neg_sample],axis=0,ignore_index=True)                         #combine the postive and negative cases together as new training set
m_sample.to_csv('../data/gold_sample.csv',index=False)                                       #Export to CSV file
