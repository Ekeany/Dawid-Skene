from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,f1_score
from gold_sample import *
from mturk_sample import *
from train_gold import *
from train_mv import *
from io import StringIO
import pandas as pd
import numpy as np
#---------------------------------------------------------------------------------------------
def Big_confusion_matrix(worker, df):
    cm = np.zeros((2, 2))
    # calculate strange confusion matrix
    # essentially folows the classic confusion matirx
    # how ever if the worker has any false positive or false negatives
    # these should be updated by the positive and negative polarities
    # instead of a simple count.
    for index, row in df.iterrows():
        if row[worker] == row.c_l_a_s_s:
            cm[int(row[worker])][int(row.c_l_a_s_s)] += 1
        elif row[worker] > row.c_l_a_s_s:
            cm[int(row[worker])][int(row.c_l_a_s_s)] += row.n_e_g
        else:
            cm[int(row[worker])][int(row.c_l_a_s_s)] += row.p_o_s

    cm[0][0], cm[1][1] = cm[1][1], cm[0][0]
    # normalize rows
    # if statement is used to remove the normalization
    # dividing by zero.
    rows = cm.sum(axis=1)
    if rows[0] > 0 and rows[1] > 0:
        cm[0] = cm[0] / rows[0]
        cm[1] = cm[1] / rows[1]
    elif rows[1] > 0:
        cm[1] = cm[1] / rows[1]
    elif rows[0] > 0:
        cm[0] = cm[0] / rows[0]
    return (cm)

#-----------Above Function declaration Part-----------------------------
df2 = mturk_sample
# map the class to one hot encoded
df2['class'] = df2['class'].map({'pos': 1, 'neg': 0})
#subset dataframe and preform majority vote
newdf = df2[['id', 'class']]
newdf = newdf.groupby(['id']).agg(lambda x: x.value_counts().index[0])
# create pos and neg values to store the updated polarities 
newdf['pos'] = newdf['class'].map(lambda x: 1 if x == 1 else 0)
newdf['neg'] = newdf['class'].map(lambda x: 1 if x == 0 else 0)
# reformat the dataframe wger each worker is a column and each review is a row
newdf2 = df2[['id', 'annotator', 'class']]
newdf2 = newdf2.pivot(index='id', columns='annotator')
newdf2 = pd.concat([newdf2, newdf], axis=1)
# pivit table changes the column names to a tuple so replace this with a single value
newdf2.columns = list(map("_".join, newdf2.columns))
my_column_names = list(newdf2.columns.values)

# Temp value for convergence rate
convergee = 10
while (convergee > 0.2):
    CMs = {}
    Total = newdf2['p_o_s'].sum()
    # calculate the confusion matrix for each worker by filtering dataframe based on worker.
    for i in range(len(my_column_names)):
        CMs[my_column_names[i]] = Big_confusion_matrix(my_column_names[i], newdf2[newdf2[my_column_names[i]].notnull()])

    # iterate through each row and filter by the workers that
    # contributed to the actual calculation of the th polarity
    # then update the polarity as shown in the lecture slides
    for index, row in newdf2.iterrows():
        worker_names = row.dropna().index
        row_workers = row.dropna().values
        pos_ = 0
        neg_ = 0
        for i in range(len(row_workers)):
            if row_workers[i] == 1:
                pos_ += CMs[worker_names[i]][0][0]
                neg_ += CMs[worker_names[i]][1][0]
            elif row_workers[i] == 0:
                pos_ += CMs[worker_names[i]][0][1]
                neg_ += CMs[worker_names[i]][1][1]
        # normalise the columns
        newdf2.loc[index, 'p_o_s'] = pos_ / (pos_ + neg_)
        newdf2.loc[index, 'n_e_g'] = neg_ / (pos_ + neg_)
    # delete the column and apply argmin to update the
    # true value of the review then calculate the difference between the previous iteration
    # if this value is below the convergence value of 0.2
    # then the loop is broken.
    newTotal = newdf2['p_o_s'].sum()
    del newdf2['c_l_a_s_s']
    newdf2['c_l_a_s_s'] = newdf2.apply(lambda row: np.argmin(np.array([row.p_o_s, row.n_e_g])), axis=1)
    convergee = abs(newTotal - Total)


#  drop duplicates
df2 = df2.drop_duplicates(subset='id')
# final = pd.concat([df2, newdf2['c_l_a_s_s']], axis=1)
df2['class'] = np.array(newdf2.loc[:, 'c_l_a_s_s'])

# create classifier
c = DecisionTreeClassifier(min_samples_split=200, random_state=0)  # Building decision tree model
features = list(df2.columns[2:-1])

x_ds_train = df2.loc[:, features]
y_ds_train = df2.loc[:, "class"]

model = c.fit(x_ds_train, y_ds_train)
x_ds_test=testing_set.loc[:,features]
y_ds_test=testing_set.loc[:,"class"]
y_ds_pred=c.predict(x_ds_test)
y_ds_pred=pd.DataFrame(data=y_ds_pred,columns=['pred'])
y_ds_pred=y_ds_pred['pred'].map({1:'pos',0:'neg'})
sen_id=testing_set.loc[:,'id']


generate_prediction_file(y_ds_test,y_ds_pred,sen_id,'../results/train_ds',"David & Skene")
