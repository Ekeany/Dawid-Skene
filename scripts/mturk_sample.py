from gold_sample import *       #Import all objects and functions used in gol_sample file
from train_gold import *        #Import all objects and functions used in train_gold file
import pandas as pd


m_sample=m_sample.sort_values(by=['id'])

inputf=open('../data/mturk.csv',encoding='UTF-8')
mturk=pd.read_csv(inputf)        #Read the file
mturk_sample=mturk.loc[mturk.loc[:,'id'].isin(m_sample.loc[:,'id']),:]      #exteact the rows in mturk dataset which also exist in m_sample dataset built in task 1
mturk_sample=mturk_sample.sort_values(by=['id'])      #Sort all cases by ID
mturk_sample.to_csv('../data/mturk_sample.csv',index=False)    #Export to CSV file

