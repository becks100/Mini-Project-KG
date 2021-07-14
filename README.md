<h1><center>Mini-Project  -  Foundations of Knowledge Graphs</center></h1> 
<h2><center>Classification of the remaining individuals from carcinogenesis</center></h2> 
<h3><center>Knowledge Group LRJ</center></h3> 
<center><b>Team members (IMT user name):</b></center> 
   <br>
   <center>Jonas Thorben Becker (becks100)</center> 
   <br>
   <center>Lukas Kneilmann (lukn)</center> 
   <br>
   <center>Rupesh Sapkota (rupezzz) (now deregistered)</center> 

<br>
<br>
<br>

<h3>Description of our Approach</h3>

For each learning problem, the workflow consists of 7 consecutive steps:

1.  Extraction of the negative and positive instances of the learning problem from the graph

2.  Identification of the missing instances by comparing them to a complete list of all instances

3.  Conversion of the instances in the learning problem into the training data set (X_train, y_train) and the test data set (X_test) by collecting the corresponding pretrained embeddings   

4.  In preparation for the training, the dataset is standardized and resampled into a more balanced data set using svmsmote sampling

5.  A linear SVM is adapted to the now newly sampled training data set    

6.  The customized SVM is used to predict the classes of the instances in the X_test record
    
7.  The instances and the corresponding predictions are saved to later convert them into a graph
    
<br>
<br>

<h3>Motivation of our approach</h3>

Bringing the given data into a format we are already familiar with is our first step in tackling the Learning Problems. Since all of us have some experience with Machine-Learning-Algorithms our plan was to convert the Ontology into matrices/tensors, which are compatible with the common range of ML and DL methods. With the help of ‘OWL2Vec*’, a random walk and word embedding based ontology embedding method by Chen, Jiaoyan et al. (2021), embeddings for the carcinogenesis ontology were learned. For further procedure, these embeddings contain the features for each carcinogen and can be searched trough by a unique key.

To determine the most suitable method for classifying the carcinogens, many combinations of sampling methods and models for supervised machine learning were tested on the data set 'kg-mini-project-train'.

Since the individual learning problems are very unbalanced, different sampling methods of the 'imbalanced learn' package for over-, under- and a combination of over- and undersampling were tried out in order to resample more balanced data sets.

Logistic regression, linear and polynomial SVMs, naive Bayes, ensemble methods such as random forrests, adaboost and smaller neural networks were considered as possible models as classifiers. To ensure comparability and consistent results, the 'GridSearchCV' method of the 'sklearnPackage' was used, a combination of cross-validation with an exhaustive search for the optimal hyperparameters of the respective model. The result of this method are the models with the optimal hyperparameters for the respective learning problem in relation to the F1 score. This enabled the respective machine learining models to be compared with one another in their optimal configuration.

After comparing the combinations of different samplers and classifiers for all learning problems in the 'kg-mini-project-train' data set, the composition of SVMSMOTE, a SMOTE variant, and linear SVM turned out to be the most suitable. This combination was consistently one of the best for all learning problems (with regards to the F1-score), but with varying regularization parameters. Since SVM algorithms are not scale-invariant, the data was standardized beforehand.

These findings were then transferred to the processing of the data set 'kg-mini-project-grading': After the features have been standardized, a training data set is sampled from the data set with an SVMSMOTE sampler. The 'GridSearchCV' method is then used to determine the optimal regularization parameter for the SVM model, and then the final model is fitted to the training dataset with these optimal parameters. This model is then used to classify the instances missing in the LP.

The last step consists of bringing the results into the predefined data format and outputting it as a turtle file.

*Chen, J., Hu, P., Jeminez-Ruiz, E., Holter, O. M., Antonyrajah, D., & Horrocks, I. (2021). OWL2Vec∗ : Embedding of OWL Ontologies.*

<br>
<br>
<h3>How to run</h3>


**Reproduce our results:**

-   Run the Jupyter Notebook *‘grading_classification’*
	-   the generated file which contains the graph with the classification results is *'grading.ttl'*
    

**Create embeddings:**

1.  Run the *OWL2VEC_Standalone.py* script, which can be found in */Knowledge Graphs Project/OWL2Vec-Star-master*.
	  -   *createEmbeddings.ipynb*  includes the run command in the first cell.
    -   *default.cfg* contains parameter, which can be used for different adjustments, e.g., embedding size.
    
2.  Copy the trained model from the *Knowledge Graphs Project/OWL2Vec-Star-master/cache* directory into */Knowledge Graphs Project*.
     -   The model is called output
    -   If the size of the model is too big, additional .npy files are created, which need to be copied into */Knowledge Graphs Project* as well (*output.trainables.syn1neg.npy* and *output.wv.vectors.npy*)
    

**How we determined our model:**
 -    The *'BigGrid'* notebook is an example of how we tried different combinations as described in “Motivation for Our Approach” to find the best model for our task.

