import pandas as pd
import numpy as np
import collections
import requests
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from io import StringIO
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from nltk.stem.snowball import SnowballStemmer
from sklearn.svm import LinearSVC
import time

class StemmedCountVectorizer(CountVectorizer):
    def build_analyzer(self):
        analyzer = super(StemmedCountVectorizer, self).build_analyzer()
        return lambda doc: ([stemmer.stem(w) for w in analyzer(doc)])

def get_drugs(cond):
    cond_dff = pd.DataFrame()
    cond_dff = dff[(dff.condition == cond)]
    cond_dff = cond_dff.drop(columns=['condition']).reset_index()
    cond_dff = cond_dff.drop(columns=['index'])
    if len(cond_dff) < 10:
        cond_dff = cond_dff.head(len(cond_dff))
    else:
        cond_dff = cond_dff.head(10) 
    print(cond_dff)

def classify_now(x):
    y=text_clf_svm.predict(tfidf_transformer.transform(count_vect.transform([x])))
    y= ", ".join( repr(e) for e in y)
    return y
print("\n\n*****************************************[STARTING SYSTEM]*****************************************")
#cleaning and formatting
dataset=pd.read_csv('/home/vikmachine/BD_final/drug_analysis.csv', error_bad_lines=False)
df = dataset[~dataset["condition"].str.contains("</span> users found this comment helpful.", na=False)]
df = dataset[~dataset["condition"].str.contains("</span> users found this comment helpful.", na=False)]
col = ['condition', 'review']
df = df[col]
df = df[pd.notnull(df['review'])]
df.columns = ['condition', 'review']
df=df.dropna()
df['condition'] = df['condition'].str.replace('eve','fever')
df['category_id'] = df['condition'].factorize()[0]

#declaring stemmer
stemmer = SnowballStemmer("english", ignore_stopwords=True)

#split into two sets
X_train, X_test, y_train, y_test = train_test_split(df['review'], df['condition'], random_state = 0)

#preprocessing starts
print("\nPREPROCESSING STARTED......")
start = time.time()
count_vect = StemmedCountVectorizer("english")
X_train_counts = count_vect.fit_transform(X_train)

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
print("\nPREPROCESSING DONE")
print('Conditions | Features')
print(X_train_tfidf.shape)
end = time.time()
print("\nTotal time taken for preprocessing = "+str(end-start))

#Training SVM
print("\n\nSVM TRAINING STARTED......")
start = time.time()

text_clf_svm= LinearSVC().fit(X_train_tfidf, y_train)
predicted = text_clf_svm.predict(X_train_tfidf)

end = time.time()
print("\nSVM TRAINING DONE Accuracy = ")
print(np.mean(predicted == y_train))
print("\nTotal time taken for for training = "+str(end-start))

#formatting datasets
print("\nFORMATTING DATASETS")
f3=pd.read_csv('/home/vikmachine/BD_final/drug_analysis.csv', error_bad_lines=False)

d=f3['drugName']
c=f3['condition']
r=f3['rating']
datalist =[]

for l in range(0,len(f3)):
    datalist.append({d[l],c[l]})
ulist=[]
for n in range(0,len(f3)):
    if datalist[n] not in ulist:
        ulist.append(datalist[n])

m = 0
result = {}
while m < len(f3):
        d3 = d[m]
        c3 = c[m]
        rating = r[m]
        if {d3,c3} in ulist:
            #result.append({d3,c3,rating})
            try:
                result[d3,c3].append(rating)
            except KeyError:
                result[d3,c3] = [rating]
            
        m = m+1

result2 = {}
for key in result:
    result2[key] = sum(result[key])

leng={}
for key in result:
        leng[key] = len(result[key])

result3 = {}
for key in result:
    result3[key] = (sum(result[key]))/(len(result[key]))


keys = [ k for k in result3 ]
vals = list_values = [ v for v in result3.values() ]

dd=[]
cc=[]
for (i,m) in keys:
    dd.append(i)
    cc.append(m)

dff = pd.DataFrame()

dff['Drug'] = dd
dff['condition'] = cc
dff['Average_Rating'] = vals

dff = dff[~dff["condition"].str.contains("</span> users found this comment helpful.", na=False)]
dff = dff.dropna()
dff = dff.reset_index()

dff = dff.drop(['index'], axis=1)
#dff.to_csv('new.csv', sep=',')

dff=dff.sort_values(by=['condition','Average_Rating'], ascending=[True,False])
dff = dff[~dff["condition"].str.contains("</span> users found this comment helpful.", na=False)]

print("_____________________________________SYSTEM IS READY_____________________________________")
while(1):
	y=input("\nENTER YOUR PAST OR CURRENT REVIEW\n")
	y=classify_now(y).strip('\'')
	print("\nThe Summary of your condition was/is [ "+str(y)+" ]")
	print("\n\tTop results are:\n_________________________________________________________________")
	get_drugs(y)
	print("_________________________________________________________________")





