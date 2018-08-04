#-----------------------------------------Import The libraries------------------------------------------------------------
import pandas as pd
import numpy as np
import re,string
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
#%matplotlib inline
import warnings
warnings.filterwarnings('ignore')
#%config InlineBackend.figure_format = 'retina'
import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn import svm
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.feature_extraction.text import CountVectorizer
#nltk.download('stopwords')
#------------------------------------------Import the dataset----------------------------------------------------- 
data = pd.read_csv("spam.csv",encoding='latin-1') 
data1= data.copy(deep=True)  #making a copy of the data incase I miss up something 
data2 = data.copy(deep = True)
#print(data.head())

just_emails = data1['v2']
def Process(emails):
	for i in range(len(just_emails)) :
	    just_emails[i] = just_emails[i].lower()

	    just_emails[i] = re.sub(r'[0-9]+','number',just_emails[i])
	    #now for punctuation 
	    just_emails[i] = re.sub(r'<[^<>]+>','',just_emails[i])
	    #handling URLS 
	    just_emails[i] = re.sub(r'(http|https): //[^\s]*','httpaddr',just_emails[i])
	    #Emails 
	    just_emails[i] = re.sub(r'[^\s]+@[^\s]+','emailaddr',just_emails[i])
	    #dollar sign 
	    just_emails[i] = re.sub(r'[$]+','dollar ',just_emails[i])

	    just_emails[i] = re.sub(r'[^a-zA-Z0-9]', ' ',just_emails[i])  # remove none alphapatical symbols




Process(just_emails)
#print(data1.tail())


to_drop = ["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"]  # dropping useless columns 
data1 = data1.drop(to_drop, axis=1)
#print(data1.head())
data1=data1.rename(columns={"v1":"label" , "v2":"text"}) 
data1['label_num'] = data1.label.map({'ham':0,'spam':1}) # numerating the y axis  0 -> no spam , 1->spam 
#print(data1.tail())
#print(data1.head())


	
# -----------------------------------------Split the dataset!!---------------------------------------------------------- 
X_train,X_test,Y_train,Y_test = train_test_split(data1['text'],data1['label'],test_size=.10,random_state=0) 
	
#------------------------------------------Text Transformation---------------------------------------------------------- 

vect = CountVectorizer()  # Here we use this function to create The bag of words to get a dict with the freauency of the words in each email 
vect.fit(X_train) # applying it 
X_train_df = vect.transform(X_train)
X_test_df = vect.transform(X_test)

#------------------------------------------Visualization--------------------------------------------------------------------

spam_words = ''
ham_words  = ''

spam =  data1[data1.label_num == 1]
ham  =  data1[data1.label_num == 0]

#Preprocessing and preparing the spam emails for visualization 
for txt in spam.text:
	text = txt.lower() # lower casing them 
	tokens = nltk.word_tokenize(text) # tokenizing them 
	tokens = [word for word in tokens if word not in stopwords.words('english')] #removing stop words

	for words in tokens: 
		spam_words = spam_words+words+' ' 
    

#Preprocessing and preparing the ham emails for visualization 
for txt in ham.text:
	text = txt.lower()
	tokens = nltk.word_tokenize(text)
	tokens = [word for word in tokens if word not in stopwords.words('english')]

	for words in tokens: 
		ham_words = ham_words+words+' '

    
# Generate a word cloud image
spam_wordcloud = WordCloud(width=600, height=400).generate(spam_words)
ham_wordcloud = WordCloud(width=600, height=400).generate(ham_words)

#Spam Word cloud
"""
plt.figure( figsize=(10,8), facecolor='k')
plt.imshow(spam_wordcloud)
plt.axis("off")
plt.tight_layout(pad=0)
plt.show()

#Ham word cloud
plt.figure( figsize=(10,8), facecolor='k')
plt.imshow(ham_wordcloud)
plt.axis("off")
plt.tight_layout(pad=0)
plt.show()
"""


#----------------------------------------------------Model Training---------------------------------------------------------------


#now we train the model 
prediction = dict()

model = MultinomialNB(alpha=0.1)
model.fit(X_train_df,Y_train)
prediction['Multinomial'] = model.predict(X_test_df)
score = accuracy_score(Y_test,prediction['Multinomial'])
print(score) 
#-------------------------------------
model2 = RandomForestClassifier(3) 
model2.fit(X_train_df,Y_train)
prediction['Forest'] = model2.predict(X_test_df)
print(accuracy_score(Y_test,prediction['Forest'])) 	
#--------------------------------------
model3 = XGBClassifier(n_estimators = 1000,learning_rate = 0.05)
model3.fit(X_train_df,Y_train)
prediction['XGB'] = model3.predict(X_test_df)
print(accuracy_score(Y_test,prediction['XGB']))
#---------------------------------------------
model4 = svm.SVC()
model4.fit(X_train_df,Y_train)
prediction['Svm'] = model4.predict(X_test_df)
print(accuracy_score(Y_test,prediction['Svm']))

print(classification_report(Y_test, prediction['Multinomial'], target_names = ["Ham", "Spam"]))
conf_mat = confusion_matrix(Y_test, prediction['Multinomial'])
conf_mat_normalized = conf_mat.astype('float') / conf_mat.sum(axis=1)[:, np.newaxis]

sns.heatmap(conf_mat_normalized)
plt.ylabel('True label')
plt.xlabel('Predicted label')
print(conf_mat)



#print(X_test[Y_test < prediction["Multinomial"] ])

#print(X_test[Y_test > prediction["Multinomial"] ])
#count = 0 
#count1 = []
#arr=[just_emails[0],just_emails[1],just_emails[8],just_emails[9],just_emails[11]]
#for i in arr:
	#for word in i :
		#count+=1
	#count1.append(count)
	#count = 0	
#print(count1)
 ## This model as of now can achieve 99.4% Accuracy using multinomialNB
