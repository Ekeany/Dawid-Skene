from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,f1_score
from gold_sample import *
from mturk_sample import *
from train_gold import *
import pandas as pd
from io import StringIO

grouped = mturk_sample.groupby('id')  # Group by each cases by id
index_list = []
tag_list = []

for name, group in grouped:  # Apply the major voting method
    count_pos = 0;
    count_neg = 0;
    final_tag = ''
    for k in group.loc[:, 'class']:
        if k == 'pos':
            count_pos = count_pos + 1;
        else:
            count_neg = count_neg + 1;

    if count_pos > count_neg:  # If the number of positive cases is greater than number of negative cases, the final label would be labeled with positive, otherwise negative
        final_tag = 'pos'
    else:
        final_tag = 'neg'

    index_list.append(name)
    tag_list.append(final_tag)

features = m_sample.columns[1:-1]  # get columns (TOPIC 0,1,2.....) as the features
new_m_sample = mturk_sample.loc[:, features]  # Only get the feature columns
new_m_sample = new_m_sample.drop_duplicates(subset=features)  # Remove the duplicated rows based on features
new_m_sample.insert(0, 'id', index_list)  # Add a new column 'id' which is used to concanate with class_table

Data = {'id': index_list, 'class': tag_list}

new_class = pd.DataFrame(Data, columns=['id',
                                        'class'])  # A dataframe which store the target label for each sentence after majority voting

new_final_sample=new_m_sample.merge(new_class,on='id')
cx=DecisionTreeClassifier(min_samples_split=100,random_state=0)                               #Create the new decision tree classifier


x_new_train=new_final_sample.loc[:,features]     #New Training set, features set
y_new_train=new_final_sample.loc[:,"class"]      #New training set, labels set

new_model=cx.fit(x_new_train,y_new_train)
x_new_test=testing_set.loc[:,features]           #New testing set, features set
y_new_test=testing_set.loc[:,"class"]            #New testing set, labels set
labl_test=testing_set.loc[:,"id"]

y_new_pred=cx.predict(x_new_test)

generate_prediction_file(y_new_test,y_new_pred,sen_id,'../results/train_mv','Major Voting')

