from twitterscraper import query_tweets_from_user
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd 
import datetime as dt
begin_date=dt.date(2020,4,1)
end_date=dt.date(2020,4,2)

limit=500
lang="english"
user="guardian"    
#tweets=query_tweets_from_user("guardian", begindate=begin_date, enddate=end_date, limit=limit, lang=lang)
tweets=query_tweets_from_user("cnn", limit=limit)
df=pd.DataFrame(t.__dict__ for t in tweets)   #forming a database of all the tweets
lengthrows=len(df.index)
# print("row len =",lengthrows)
df=df.drop(['screen_name','username', 'user_id', 'tweet_id', 'tweet_url',
       'timestamp', 'timestamp_epochs', 'text_html',
       'hashtags', 'has_media', 'img_urls', 'video_url', 
       'replies', 'is_replied', 'is_reply_to', 'parent_tweet_id',
       'reply_to_users'], axis=1)   #removing all the information we do not need and keeping only text, retweets and likes

for i in range(0,lengthrows):
		if(len(df.at[i,'links'])!=1):
			if(len(df.at[i,'links'])==0):
				df=(df.drop(i))
			else:
				print("oh shit",len(df.at[i,'links']))
				print(df.at[i,'links'])
				print(df.at[i,'text'])
				print(i)
		else:
			s=df.at[i,'text'].replace(df.at[i,'links'][0],"")
			df.at[i,'text']=s
		#print(df.at[i,'text'])
df=df.reset_index()
lengthrows=len(df.index)
df=df.drop("links",axis=1)
#print(df)
textlist=[]
sharesandlikeslist=[]
#print(textlist)
for i in range(0,lengthrows):
		textlist.append(df.at[i,'text'])
		textlist[i] = textlist[i].replace(u'\xa0', u' ')
		sharesandlikeslist.append([df.at[i,'likes'],df.at[i,'retweets']])
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(textlist)
XX=X.todense()
XXX=np.array(XX)
Y=sharesandlikeslist
a,b=XX.shape
print(" a and b are")
print(a,b) 
ls,lowlike=1000,100   #observing the highest and the lowest value of likes and shares to prdict virality
hs,highlike=0,0
for i in range(0, len(Y)):
	if(Y[i][0]<lowlike):
		lowlike=Y[i][0]
	if(Y[i][0]>highlike):
		highlike=Y[i][0]
	if(Y[i][1]<ls):
		ls=Y[i][1]
	if(Y[i][1]>hs):
		hs=Y[i][1]
print("stats are")
print(lowlike,highlike,ls,hs)
viralityindex= 50.0/100.0* (highlike+hs)
print(viralityindex)

X_train, X_test, y_train, y_test = train_test_split(XXX, Y, test_size=0.25, random_state=42)
#print(X.toarray())
#print(vectorizer.get_feature_names())
c,d=X_train.shape
print(type(X_train))
print(len(vectorizer.get_feature_names()))
a,b,=X.shape
print(b)

model = RandomForestRegressor(n_estimators=1000)


# model = Sequential()
# model.add(Dense(350, input_dim=d, activation='relu'))
# model.add(Dense(300, activation='relu'))
# model.add(Dense(200, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(200, activation='relu'))
# model.add(Dense(150, activation='relu'))
# model.add(Dense(50, activation='relu'))
# model.add(Dense(2, activation='relu'))
# model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
# model.fit(np.array(X_train), np.array(y_train), epochs=30, verbose=1, validation_split=0.2)

model.fit(np.array(X_train), np.array(y_train))
ypredicted=model.predict(X_test) #stores the predicted values of the shares and likes
print(ypredicted,y_test)
yactual=[]
yactualviralornot=[]
ypredictedvalues=[]
ypredviralornot=[]
for i in range(0,len(y_test)):
	yactual.append(y_test[i][0]+ y_test[i][1])
	if(yactual[i]>=viralityindex):
		yactualviralornot.append("Viral")
	else:
		yactualviralornot.append("Not Viral")
for i in range(0,len(y_test)):
	ypredictedvalues.append(y_test[i][0]+ y_test[i][1])
	if(ypredictedvalues[i]>=viralityindex):
		ypredviralornot.append("Viral")
	else:
		ypredviralornot.append("Not Viral")


print(ypredviralornot)
print(yactualviralornot)

# loss=0
# for i in range(0,len(y_test)):
# 	if(ypredviralornot[i]!=yactualviralornot[i]):
# 		loss=loss+1
# accuracy=(len(y_test)-loss)/len(y_test)*100
# print("accuracy is", accuracy)






