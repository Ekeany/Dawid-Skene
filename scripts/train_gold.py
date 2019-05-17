from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,f1_score
from gold_sample import m_sample
from io import StringIO
import pandas as pd
import numpy as np

def generate_prediction_file(y_test, y_pred, sen_id,file_name,approach_name):            #Decllared function to generate the F-score, Accuracy and Prediction probabilites table
    f = open(file_name+'.txt', 'w')                                                      #Open a file as writing
    score = accuracy_score(y_test, y_pred) * 100                                         #Get the accuracy
    generate_confusionMatrix(y_test,y_pred,file_name)                                    #Generate the confusion matrix
    f_score = f1_score(y_test, y_pred, average='binary', pos_label='pos')                #Calculate the F-score
    probability_tb = c.predict_proba(x_test)                                             #Calculate the probabilites for pos and neg classes
    label_tb = c.predict(x_test)                                                         #c is the decision tree model declared in bottom
    result = approach_name+":   The accuracy is :" + str(round(score, 1)) + "%\n\n"
    result += "\n\nThe F-Score is:" + str(f_score)+"\n"
    print(result)
    f.write(result)
    pos_prob = list(probability_tb[:, 1])
    neg_prob = list(probability_tb[:, 0])
    Data = {
        'Neg_Prob': neg_prob,
        'Pos_Prob': pos_prob,
        'Pre_Label': label_tb,
        'Sentence ID': sen_id
    }

    mdf = pd.DataFrame(Data, columns=['Sentence ID', 'Neg_Prob', 'Pos_Prob', 'Pre_Label'])

    f.write(mdf.to_string())                                                            #Ouput to file
    f.close();


def generate_confusionMatrix(y_test,y_pred,file_name):                                  #generate the confustion matrix funciton
    TP=0
    TN=0
    FP=0
    FN=0
    for m in range(len(y_test)):
      if (y_test[m]=='pos')& (y_pred[m]=='pos'):
        TP=TP+1
      if (y_test[m]=='pos')&(y_pred[m]=='neg'):
        FN=FN+1
      if (y_test[m]=='neg')&(y_pred[m]=='pos'):
        FP=FP+1
      if (y_test[m]=='neg')&(y_pred[m]=='neg'):
        TN=TN+1
    print("TP: ",TP,"TN: ",TN,"FP: ",FP,"FN: ",FN)


#----------------------------------------------------------------------------------------
training_set=m_sample
c=DecisionTreeClassifier(min_samples_split=100,random_state=0)     #Building decision tree model
features=list(training_set.columns[1:-1])

x_train=training_set.loc[:,features]
y_train=training_set.loc[:,"class"]

model=c.fit(x_train,y_train)

test=open('../data/test.csv',encoding='UTF-8')   #Import testing set to predict
testing_set=pd.read_csv(test)

x_test=testing_set.loc[:,features]              #split the features in testing set
y_test=testing_set.loc[:,"class"]

y_pred=c.predict(x_test)#IShowing result and export result into file

sen_id=testing_set.loc[:,'id']
generate_prediction_file(y_test,y_pred,sen_id,'../results/train_gold',"Gold Sample")

