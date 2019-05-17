# Dawid-Skene

## Introduction.

The objective of this assignment involves the sentiment classification of a selection of movie reviews collected from the Rotten Tomatoes website. This is a supervised task where the polarity labels were provided by crowd sourced workers on the Amazon Mechanical Turk platform. There is high uncertainty about the quality of the dataset that will be returned when dealing with a crowd sourced platform. As some of the workers will be lazy or uninterested and just submit random answers, some workers will have the right intentions but still submit an incorrect answer, and others will perform as expected. Therefore in order to obtain the actual true result for each review, methods for extracting the signal from the noise by evaluating the quality of the individual workers will have to be implemented. Three datasets were provided in the assignment their details can be found below. 
 
Gold_csv: This file contains 5000 rows and 1202 columns. The first column represents the review id and last column is the polarity of the review. The polarity of this dataset was not crowd sourced but instead it was obtained by extracting the rotten tomatoes internal rating score. The remaining 1200 columns represented features extracted from each review by using Latent Semantic analysis. As this dataset was not produced by crowdsourcing it’s labels are assumed to be correct as human error can be omitted.  
 
Mturk.csv: this dataset contains the crowd sourced dataset which contains the polarity labels for each review as given by crowd sourced workers. This file contains 27,746 rows and 1203 columns. The extra column identifies which crowd sourced worker reviewed what sentence. The remaining columns are the same as the gold standard.  
 
Test.csv: This dataset is similar to the gold standard were the sentiment classification is assumed to be correct. This file was used to measure the accuracy of the models produced in part one, two and three. The file contained approximately 5000 rows and the same column structure as the gold standard.

## Part One. 

The distribution of labels in the gold.csv file was found to be approximately equal as seen below.


