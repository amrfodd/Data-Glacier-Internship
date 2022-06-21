## Week 5

### Cloud and API deployment

Task:

1. Select any toy data (simple data) ( You are allowed to use data set of week 4)

2. Save the model ( You are allowed to use model of week 4)

3. Deploy the model on any open source cloud eg Heroku (Deployment should be API based as well as web app)

4. Create pdf document (Name, Batch code, Submission date, Submitted to ) which should contain snapshot of each step of deployment)

5. Upload the document to Github

6. Submit the URL of the uploaded document.



In this project, I willl apply supervised learning (classification) algorithms on spam dataset to build spam classifier 
We used a public SMS Spam dataset, which is not purely clean dataset. The data consists of two different columns (features), such as context, and class. The column context is referring to SMS. The column class may take a value that can be either spam or ham corresponding to related SMS context.

Before applying any supervised learning methods, we applied a bunch of data cleansing operations to get rid of messy and dirty data since it has some broken and messy context.

After obtaining cleaned dataset, we created tokens and lemmas of SMS corpus seperately by using Spacy, and then, we generated bag-of-word and TF-IDF of SMS corpus, respectively. In addition to these data transformations, we also performed SVD, SVC, PCA to reduce dimension of dataset.

To manage data transformation in training and testing phase effectively and avoid data leakage, we used Sklearn's Pipeline class. So, we added each data transformation step (e.g. bag-of-word, TF-IDF, SVC) and classifier (e.g. Naive Bayesian, SVM, Random Forest Classifier) into an instance of class Pipeline.

After applying those supervised learning methods, we also perfomed deep learning. Our deep learning architecture we used is based on LSTM. To perform LSTM approching in Keras (Tensorflow), we needed to create an embedding matrix of our corpus. So, we used Gensim's Word2Vec approach to obtain embedding matrix, rather than TF-IDF.

At the end of each processing by different classifier, we plotted confusion matrix to compare which one the best classifier for filtering SPAM SMS.

