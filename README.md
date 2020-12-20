# Multiclass-Classification
**Problem Statement**:\
Use Machine Learning & NLP To Predict The News Category

**System Specification:** <br>
**Processor:** i3 7th gen <br>
**RAM:** 4GB <br>
**Graphics:** 2GB <br>
**OS:** Windows/Linux <br>
**Disk Space:** 2GB <br>
**Tools:** Anaconda, Jupyter, Python <br>

**Components:** <br>
**NLTK:-** We have used the NLTK (Natural Language Toolkit) for the preprocessing of data. We have used the stop word bag provided by NLTK by which we can remove the stop words present in the news heading. After this preprocessing of data, we get the news heading in a form from which we can efficiently predict their category. <br>
**SKlearn:-** We used different classification models from this library for classifying our data and predicting the category of newly entered data. <br>
**Numpy:-** We used this library to give the statistics information present in the dataset such as column count, rows in dataset, count of record in each category, etc. <br>
**Pandas:-** This library is used for loading the dataset in the model and manipulating the dataset. <br>
**Matplotlib:-** We have used matplotlib for data visualization. We have generated a bar graph on the category column in the dataset to analyze consistency in the dataset. <br>

**Design:**
	We have designed a program that can classify the News Article headlines in one of the four mentioned categories.

1. Politics
2. Technology
3. Sport
4. Entertainment
5. Business

We have tried several models for news category classification, in which the **SVC (Support Vector Classifier)** is given the most efficient classification with an **accuracy of 99%**.
Firstly we are performing some **data preprocessing **by which the model can efficiently give its intended category. For this purpose, we are using Natural Language Toolkit or **NLTK** library. 
We are firstly **removing all punctuations** present in the sentence, after this step, we perform the **stop word removal** by which the sentence is cleaned from all the stop words which are nothing but the most common words of the language after all this preprocessing we finally perform **Tokenization** by which we get individual word of a sentence as a token.
After the step of tokenization, we are using **TFIDF Vectorizer** the first part of it is **The Term Frequency** which is nothing but the number of times a word appears in a document divided by the total number of words in the document and the other part is the** Inverse Data Frequency** which is The log of the number of documents divided by the number of documents that contain the word and the TFIDF is just the product of these two which gives us a statistical measure used to evaluate the importance of the word in the document of the corpus. After this, we use svc for predicting the category of the news headline. We have to perform TFIDF Vectorization and **SVC prediction** on each and every row of testing data so for this we are using the Sklearn pipeline which encapsulates all functions models provided to it as a single object after that we can perform our task of category classification.

**Output:**
[![](NLP  OUTPUT)](https://drive.google.com/file/d/1twF8YYlWxUoy4L8y2hdBZS1I4snfOupB/view?usp=sharing)
**Conclusion:**
We have trained various models for multiclass classification to predict the right category of the given news description. Given below are the results observed:

| Algorithm  | Accuracy   |
| :------------: | :------------: |
|  RandomForestClassifier | 95  |
|  KNeighborsClassifier | 92  |
| MultinomialNaiveBayes  |  97 |
|LinearSVC  | 99  |

So, here we conclude that LinearSVC is the best model fit for the selected dataset.

**Future scope:**
	We can add Sentimental analysis as a module to this project and can also add some user interface which will make the project easy to use.

**Reference:**
[Data Science Hackathon: Use Machine Learning & NLP To Predict The News Category & Win Exciting Prizes (analyticsindiamag.com)](https://analyticsindiamag.com/data-science-hackathon-predict-news-category/ "Data Science Hackathon: Use Machine Learning & NLP To Predict The News Category & Win Exciting Prizes (analyticsindiamag.com)")


