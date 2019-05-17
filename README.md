# Dawid-Skene

## Introduction.

The objective of this assignment involves the sentiment classification of a selection of movie reviews collected from the Rotten Tomatoes website. This is a supervised task where the polarity labels were provided by crowd sourced workers on the Amazon Mechanical Turk platform. There is high uncertainty about the quality of the dataset that will be returned when dealing with a crowd sourced platform. As some of the workers will be lazy or uninterested and just submit random answers, some workers will have the right intentions but still submit an incorrect answer, and others will perform as expected. Therefore in order to obtain the actual true result for each review, methods for extracting the signal from the noise by evaluating the quality of the individual workers will have to be implemented. Three datasets were provided in the assignment their details can be found below. 
 
Gold_csv: This file contains 5000 rows and 1202 columns. The first column represents the review id and last column is the polarity of the review. The polarity of this dataset was not crowd sourced but instead it was obtained by extracting the rotten tomatoes internal rating score. The remaining 1200 columns represented features extracted from each review by using Latent Semantic analysis. As this dataset was not produced by crowdsourcing it’s labels are assumed to be correct as human error can be omitted.  
 
Mturk.csv: this dataset contains the crowd sourced dataset which contains the polarity labels for each review as given by crowd sourced workers. This file contains 27,746 rows and 1203 columns. The extra column identifies which crowd sourced worker reviewed what sentence. The remaining columns are the same as the gold standard.  
 
Test.csv: This dataset is similar to the gold standard were the sentiment classification is assumed to be correct. This file was used to measure the accuracy of the models produced in part one, two and three. The file contained approximately 5000 rows and the same column structure as the gold standard.

## Part One. 

The distribution of labels in the gold.csv file was found to be approximately equal as seen below.

<p align="center">
  <img width="351" height="317" src="/Images_/image1.PNG">
</p>

Despite this a stratified sampling technique was implemented. This technique divides the population (by their label) into two separate groups, called strata. Then, a standard random sampling technique from pandas was used to extract 500 reviews from each strata. Both of these separate samples were then combined together to produce the final result which was then stored in the gold_sample.csv file. As seen below the gold_sample contains an equal split of 500 ironic and non-ironic randomly selected reviews. 

<p align="center">
  <img width="560" height="234" src="/Images_/image2.PNG">
</p>

## Model One. 
 
A simple decision tree classifier was implemented using the Scikit learn package in python. The model was trained on a sample sub section of the gold.csv file created previously in the assignment. The minimum number of instances needed to cause a split was heuristically set to 100 as this produced the best accuracy, this parameter essentially stops the tree from over fitting the training dataset. The accuracy of the model was found by predicting the labels for the test.csv file provided at the beginning of the assignment. Using the Scikit learn Package the performance of the model on the test.csv file was measured by calculating its corresponding accuracy and F1 score. As required the resulting F1-score, accuracy and prediction probabilities were stored in a text file named train_gold.txt.

<p align="center">
  <img width="582" height="346" src="/Images_/image3.PNG">
</p>

## Model Two. 

The same classifier for part one was re-implemented in part two. However the training data used to train the model differed. Using the review id’s obtained from the gold_sample.csv file, the corresponding rows were extracted from crowd sourced dataset. As each review contains a number of polarity estimates from a number of workers there is an inherent amount of noise in the system stemming from human error. There are two approaches when dealing with noise, the "most frequent" approach, in which we use the "majority" of the votes, or a "Bayesian" approach in which we consider the labels to be inherently uncertain. For this assignment the most frequent approach was implemented only, therefore the data was essentially treated as if it was noise-free, letting the existing decision tree algorithm to work without any modification. This was elegantly implemented in Python by using the group by function on the review ID and setting the aggregation method to most frequent element in this subgroup. Again the accuracy and performance of the model was measured using the test.csv file and the results stored in the train_mv.txt file. 

<p align="center">
  <img width="517" height="37" src="/Images_/image4.PNG">
</p>

The mturk_sample contained approximately 5000 thousand rows, where each row corresponded to a movie review whose polarity was provided by a specific worker. The figure below represents the distribution of workers per review. On average 5.6 workers commented on a single review with the maximum and minimum values being eight and four respectively. 
 
 <p align="center">
  <img width="439" height="338" src="/Images_/image5.PNG">
</p>

The number of workers within the sample was found to be approximately 185. The contribution of each of these workers differed drastically as one single worker provided 831 polarity estimates whereas another provided only a single estimate. On average each worker provided 29 polarity estimates, the plot below describes the distribution of workers throughout the entire sample. 

<p align="center">
  <img width="385" height="375" src="/Images_/image6.PNG">
</p>

It was noticed that for the purpose of this assignment ten “FAKE” workers were introduced to the original file. The contribution of these workers varied, however in total they contributed to approximately ten percent of the polarity estimates. These workers obviously introduced more noise to the system and had a direct effect on the performance of the classifier chosen for part two. The figures below visualize the contributions of these “FAKE” workers. 

<p align="center">
  <img width="399" height="396" src="/Images_/image7.PNG">
</p>
<p align="center">
  <img width="705" height="208" src="/Images_/image8.PNG">
</p>

## Model Three.  
 
The original decision tree model was also implemented in task three. However the polarities of the reviews in the Mturk_sample were generated by implementing the Dawid & Skene method.  The Dawid-Skene model (1979) was one of the first models developed that could discover the true polarity of an item when combining data from multiple noisy sources, and has since become a classical label crowdsourcing model. Essentially this model is predicated upon an expression for the probability of a worker mislabeling the polarity of a review. Therefore each worker has a latent confusion matrix describing the joint probability distribution over the true and reported labels. An example of such a confusion matrix for the case of binary classification is shown in Figure 2.2. The worker with the confusion matrix shown on this figure will label negative instances correctly with probability 0.8, positive instances with 0.7. Other workers may have completely different confusion matrices.  To produce each workers confusion matrix the relationship between the true polarity and the workers label are required. As we do not implicitly know the true nature of the polarity they are estimated using the majority vote method. The implementation used within this model started by: 
 
1. Compute the joint probability matrix for each worker individually, these values were then stored in a dictionary data structure for easy access.  
2. For each review in the sample data find the number of workers who individually contributed to their polarity. 
3. Update the polarity by summing all the probability matrices of each worker that contributed to their polarity. 
4. Repeat steps one to three until the polarity stabilize or converge to their real values. 
 
### Code.
The big confusion matrix function calculates the joint probability confusion matrix for a worker given their labels and the true value. This function deviates from a traditional confusion matrix as if a worker predicts a false positive then the false positive cell is updated with the negative probability value instead of a simple count increment. Similarly if the worker produces a false negative then the positive probability will be added. Also as this is a probability matrix each value in the matrix is normalized by its row. 
The Mturk dataset has been transposed therefore the annotators represent the column values and the reviews represent the columns the values in each cell represent the label estimated by that particular worker for that review. The true label acquired from the majority vote is stored in the last column and two temporary columns representing the probability of the true value being positive or negative were added. A basic example of this table format is presented below. 

<p align="center">
  <img width="641" height="362" src="/Images_/image9.PNG">
</p>

The algorithm first computes the confusion matrix for every worker by selecting their respective column and removing all the NA values (as not every worker has contributed to every review). Each matrix is then stored in a dictionary using the column name as a key. Next the algorithm iterates through every review and only selects the workers that have contributed to its polarity. It then takes the probabilities from all the workers and sums them up and then normalizes before updating the probability columns for that review this process is completed for every review in the dataset. The true class label is then updated by computing the argmin between these two columns. The sum of the positive probability column is computed and compared to the value from the previous iteration if this value is less than a certain threshold (0.01) the function breaks as it has converged otherwise these processes run again. The resultant polarity labels were then used to replace the majority voted labels in the training data. The classifier was trained on this data and its performance was again measured using the test.csv file. 
 
<p align="center">
  <img width="618" height="852" src="/Images_/image10.PNG">
</p> 

The graph below visualizes the change in the distribution of labels from the majority vote sample to the Dawid and Skene method in total 9 reviews changed polarity. 

<p align="center">
  <img width="678" height="297" src="/Images_/image11.PNG">
</p> 

## Discussion. 
 
All three models performed poorly on the task at hand, with no model achieving an accuracy above 56%.These low accuracy scores could be attributed the high number of features in the dataset approximately 1200. As the dimensionality was rather large the volume of the feature space was also large thus causing the small number of available training data to become sparse. Shockingly the performance of each model was essentially ranked on their order of completion with the gold_sample.csv achieving a mere 51.8% accuracy the lowest of the three. This result was counter intuitive as this sample was taken from a dataset named gold essentially inferring that it contained the “gold standard” or the correct polarity of each review. The majority vote sample improved upon this accuracy by one percent achieving an average accuracy of 52.7% and the Dawid and Skene method again improved upon this with an average accuracy of 54.9%. As expected the Dawid and Skene method outperformed the majority vote method as it essentially reduced the noise in the system coming from the “FAKE” workers. A total of 9 workers polarity flipped from a positive sentiment to a negative. Due to the sensitivity of decision trees this small change led to the creation of a much more simple and efficient tree structure. As seen in the appendices below

<p align="center">
  <img width="661" height="97" src="/Images_/image12.PNG">
</p> 

Below are the corresponding confusion matrices for the Gold sample, Majority Vote and Dawid and Skene method.

<p align="center">
  <img width="760" height="244" src="/Images_/image13.PNG">
</p> 

## References.

1. https://rss.onlinelibrary.wiley.com/doi/abs/10.2307/2346806 
2. https://nuigalway.blackboard.com/bbcswebdav/pid-1679883-dt-content-rid12611442_1/courses/1819-CT5103/Zhang_2016.pdf 
3. https://nuigalway.blackboard.com/bbcswebdav/pid-1669303-dt-content-rid12536733_1/courses/1819-CT5103/CT5103_2019%20Assignment%202%281%29.pdf 

## Appendices.

All images represent the decision tree structure for each model in order of completion. Gold Sample

<p align="center">
  <img width="761" height="759" src="/Images_/image14.PNG">
</p> 

### Majority Vote:

<p align="center">
  <img width="721" height="845" src="/Images_/image15.PNG">
</p> 

### Dawid & Skene:

<p align="center">
  <img width="679" height="473" src="/Images_/image16.PNG">
</p> 



