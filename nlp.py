import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import cross_validation
from sklearn.naive_bayes import GaussianNB

#Reading our datasets
dataframe = pd.read_csv('Restaurant_Reviews.tsv',delimiter = '\t',quoting = 3)

#Cleaning the text part
cleaned_data = []
for i in range(0,1000):# because to iterate through each row of my datasets
    cleanse = re.sub('[^a-zA-Z]',' ',dataframe['Review'][i])#This line of code will help us in only taking the letters and will eliminate all the unnecessary Things
    cleanse = cleanse.lower()
    cleanse = cleanse.split()
    port_stemmer = PorterStemmer()

    cleanse = [port_stemmer.stem(word) for word in cleanse if not word in set(stopwords.words('english'))]

    cleanse = ' '.join(cleanse)
    cleaned_data.append(cleanse)

#Now we need to create the bag of words model
bag = CountVectorizer()
features = bag.fit_transform(cleaned_data).toarray()
labels = dataframe['Liked']

#Splitting our training and testing datasets

features_train,features_test,labels_train,labels_test = cross_validation.train_test_split(features,labels,test_size = 0.20)

#Applying our classifier algorithm
classifier = GaussianNB()
classifier.fit(features_train,labels_train)
Accuracy = classifier.score(features_test,labels_test)

print 'Accuracy is:',Accuracy
