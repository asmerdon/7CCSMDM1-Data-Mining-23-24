# If for whatever reason this code does not work with the structure for marking, I have left in my (commented out) testing code under
# every function, which gives a working example of the output.

# Imports
import pandas as pd
from collections import Counter
import requests
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Part 3: Text mining.

# Return a pandas dataframe containing the data set.
# Specify a 'latin-1' encoding when reading the data.
# data_file will be populated with a string 
# corresponding to a path containing the wholesale_customers.csv file.
def read_csv_3(data_file):
    dataframe = pd.read_csv(data_file, encoding='latin-1') #latin-1
    return dataframe

"""data_file = 'coronavirus_tweets.csv'
df = read_csv_3(data_file)
print(df.head())"""

# Return a list with the possible sentiments that a tweet might have.
def get_sentiments(df):
    sentiments = df['Sentiment'].unique().tolist() #extracts unique values from 'Sentiment' column of df, adds to list
    return sentiments

"""sentiments_list = get_sentiments(df)
print(sentiments_list)"""

# Return a string containing the second most popular sentiment among the tweets.
def second_most_popular_sentiment(df):
    sentiment_counts = df['Sentiment'].value_counts() #count occurrences of each sentiment
    sorted_sentiments = sentiment_counts.sort_values(ascending=False) #order by desc
    second_most_popular= sorted_sentiments.index[1] #get second from list (indicated by [1])
    return second_most_popular

"""second_most_popular = second_most_popular_sentiment(df)
print(second_most_popular)"""

# Return the date (string as it appears in the data) with the greatest number of extremely positive tweets.
def date_most_popular_tweets(df):
    extremely_positive_tweets = df[df['Sentiment'] == 'Extremely Positive'] # filter df to include only extremely positive tweets
    date_counts = extremely_positive_tweets['TweetAt'].value_counts() #count dates
    most_popular_date = date_counts.idxmax() # idx max returns most common val in a list (so most popular date)
    return most_popular_date

"""most_popular_date = date_most_popular_tweets(df)
print(most_popular_date)"""

# Modify the dataframe df by converting all tweets to lower case. 
def lower_case(df):
    df['OriginalTweet'] = df['OriginalTweet'].str.lower()

"""lower_case(df)
print(df.head())"""

# Modify the dataframe df by replacing each characters which is not alphabetic or whitespace with a whitespace.
def remove_non_alphabetic_chars(df):
    df['OriginalTweet'] = df['OriginalTweet'].str.replace('[^a-zA-Z\s]', ' ', regex=True) #use regex to replace all non alphabetic chars with whitespace 

"""remove_non_alphabetic_chars(df)
print(df.head())"""

# Modify the dataframe df with tweets after removing characters which are not alphabetic or whitespaces.
def remove_multiple_consecutive_whitespaces(df):
    df['OriginalTweet'] = df['OriginalTweet'].str.replace('\s+', ' ', regex=True) #\s+ regex matches multiple whitespaces, replace with single ' '

"""remove_multiple_consecutive_whitespaces(df)
print(df.head())"""

# Given a dataframe where each tweet is one string with words separated by single whitespaces,
# tokenize every tweet by converting it into a list of words (strings).
def tokenize(df):
    df['TokenizedTweet'] = df['OriginalTweet'].str.split() #splits to single word
    return df
    
# Given dataframe tdf with the tweets tokenized, return the number of words in all tweets including repetitions.
def count_words_with_repetitions(tdf):
    all_words = [] #get all words into a single list
    for tweet in tdf['TokenizedTweet']:
        for word in tweet: #iterate over each word
            all_words.append(word)
    total_words_count = len(all_words) #count words (including repititions)
    return total_words_count

"""tdf = tokenize(df)
print(tdf.head())
total_words_count = count_words_with_repetitions(tdf)
print("Total number of words including repetitions:", total_words_count)"""

# Given dataframe tdf with the tweets tokenized, return the number of distinct words in all tweets.
def count_words_without_repetitions(tdf):
    unique_words = set() #create a set of unique words from tweets
    for tweet in tdf['TokenizedTweet']:
        for word in tweet: # iterate over each word in tokenized tweet
            unique_words.add(word) #add to set
    distinct_words_count = len(unique_words) #count
    return distinct_words_count

"""distinct_words_count = count_words_without_repetitions(tdf)
print("Total number of distinct words:", distinct_words_count)"""

# Given dataframe tdf with the tweets tokenized, return a list with the k distinct words that are most frequent in the tweets.
def frequent_words(tdf,k):
    all_words = [] #flatten the list of tokenized tweets into single list
    for tweet in tdf['TokenizedTweet']:
        for word in tweet:
            all_words.append(word)
    word_counts = Counter(all_words) #count the frequency of each word (dict)
    frequent_words_list = [] #get k distinct words that are most frequent
    for word, _ in word_counts.most_common(k): #.most_common(k) returns a list of tuples
        frequent_words_list.append(word) #add to list
    return frequent_words_list

"""k = 10
top_k_words = frequent_words(tdf, k)
print(f"The {k} most frequent words are:", top_k_words)"""

# Given dataframe tdf with the tweets tokenized, remove stop words and words with <=2 characters from each tweet.
# The function should download the list of stop words via:
# https://raw.githubusercontent.com/fozziethebeat/S-Space/master/data/english-stop-words-large.txt
def remove_stop_words(tdf):
    stop_words_url = "https://raw.githubusercontent.com/fozziethebeat/S-Space/master/data/english-stop-words-large.txt" #download list of stop words
    response = requests.get(stop_words_url)
    stop_words = set(response.text.split()) #split up each stop word
    def filter_words(tweet):
        filtered_tweet = []
        for word in tweet:
            if word not in stop_words and len(word) > 2: #add non-stopwords to list
                filtered_tweet.append(word)
        return filtered_tweet
    tdf['TokenizedTweet'] = tdf['TokenizedTweet'].apply(filter_words) #apply filtering function (which returns non-stop words) to each tweet in df
    
"""remove_stop_words(tdf)
print(tdf.head())"""

# Given dataframe tdf with the tweets tokenized, reduce each word in every tweet to its stem.
def stemming(tdf):
    stemmer = PorterStemmer() #initialise porter stemmer function 
    def stem_words(tweet): #function takes in tweet and stems each word
        stemmed_tweet = []
        for word in tweet:
            stemmed_word = stemmer.stem(word) #apply stemmer to word
            stemmed_tweet.append(stemmed_word) #append to list
        return stemmed_tweet #return full tweet
    tdf['TokenizedTweet'] = tdf['TokenizedTweet'].apply(stem_words) #appply function to each tweet in 'TokenizedTweet' column of tdf
    
"""stemming(tdf)
k = 10
top_k_words_modified = frequent_words(tdf, k)
print(f"The {k} most frequent words after stemming and removing stop words are:", top_k_words_modified)"""

# Before stemming the most frequent words are:
# ['the', 'to', 't', 'co', 'and', 'https', 'covid', 'of', 'a', 'in']
# After stemming they are:
# ['http', 'covid', 'coronaviru', 'price', 'store', 'supermarket', 'food', 'groceri', 'peopl', 'consum']
# Something appears to be cutting off the final letters of some words.

# Given a pandas dataframe df with the original coronavirus_tweets.csv data set,
# build a Multinomial Naive Bayes classifier. 
# Return predicted sentiments (e.g. 'Neutral', 'Positive') for the training set
# as a 1d array (numpy.ndarray). 
def mnb_predict(df):
    vectorizer = CountVectorizer(max_features=None, ngram_range=(1, 3)) #these were the parameters I found worked the best. increasing the max_features always seemed to improve (so I set to None), however this may cause overfit. parameters not specified are automatically set to defualt
    X = vectorizer.fit_transform(df['OriginalTweet']) #convert text data into term-document matrix
    clf = MultinomialNB() #build and train MNB classifier
    clf.fit(X, df['Sentiment']) #target attribute is 'Sentiment'
    y_pred = clf.predict(X) # predict sentiments for the training set
    return y_pred

# Given a 1d array (numpy.ndarray) y_pred with predicted labels (e.g. 'Neutral', 'Positive') 
# by a classifier and another 1d array y_true with the true labels, 
# return the classification accuracy rounded in the 3rd decimal digit.
def mnb_accuracy(y_pred,y_true):
    accuracy = accuracy_score(y_true, y_pred) #calculate classification accuracy
    rounded_accuracy = round(accuracy, 3) #round to 3dp
    return rounded_accuracy

# training accuracy of 0.986 and a testing accuracy of 0.996 (with coronavirus_tweets.csv)

"""data_file = 'coronavirus_tweets.csv' #read csv
df = read_csv_3(data_file)
X_train, X_test, y_train, y_test = train_test_split(df['OriginalTweet'], df['Sentiment'], test_size=0.2, random_state=28) #split
y_pred_train = mnb_predict(pd.DataFrame({'OriginalTweet': X_train, 'Sentiment': y_train})) #training
y_pred_test = mnb_predict(pd.DataFrame({'OriginalTweet': X_test, 'Sentiment': y_test}))
print("y_pred_train:", y_pred_train)
print("y_pred_test:", y_pred_test)
accuracy_train = mnb_accuracy(y_pred_train, y_train)
accuracy_test = mnb_accuracy(y_pred_test, y_test)
print("Training accuracy:", accuracy_train) # training accuracy of 0.986 and a testing accuracy of 0.996
print("Testing accuracy:", accuracy_test)"""
