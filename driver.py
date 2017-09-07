import pandas as pd
import glob
import csv
import string
import nltk.data
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn import linear_model
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score

train_path = "../aclImdb/train/" # source data
test_path = "../imdb_te.csv" # test data for grade evaluation. 
stopwords = []
stop = open("../stopwords.en.txt", 'r')
for line in stop:
    stopwords.append(line.replace('\n', ''))

def imdb_data_preprocess(inpath, outpath="./", name="imdb_tr.csv", mix=False):
    '''Implement this module to extract
    and combine text files under train_path directory into 
    imdb_tr.csv. Each text file in train_path should be stored 
    as a row in imdb_tr.csv. And imdb_tr.csv should have three 
    columns, "row_number", "text" and label'''
    count = 0
    files1 = glob.glob(inpath + "pos/" + "*.txt")
    files0 = glob.glob(inpath + "neg/" + "*.txt")
    with open(name, "w") as outfile:
        cw1 = csv.writer(outfile)
        cw1.writerow(['row_number' ,'text','polarity'])
        for r in files1:
            with open(r, "r") as infile:
                for row in csv.reader(infile):
                    pos = [word for word in ''.join(row).translate(string.maketrans(string.punctuation,' '*len(string.punctuation))).split() if word.lower() not in stopwords]
                    cw1.writerow([count, ' '.join(pos), 1])
                    count += 1
    with open(name, "a") as append:
        cw0 = csv.writer(append)
        for r in files0:
            with open(r, "r") as infile:
                for row in csv.reader(infile):
                    neg = [word for word in ''.join(row).translate(string.maketrans(string.punctuation,' '*len(string.punctuation))).split() if word.lower() not in stopwords]
                    cw0.writerow([count, ' '.join(neg), 0])
                    count += 1
imdb_data_preprocess(train_path)
if __name__ == "__main__":
    with open(test_path, "r") as f:
        with open("imdb_te.csv", "w") as overwrite:
            cw = csv.writer(overwrite)
            for row in csv.reader(f.read().splitlines()):
                words = [word for word in ''.join(row[1]).translate(string.maketrans(string.punctuation,' '*len(string.punctuation))).split() if word.lower() not in stopwords]
                cw.writerow([row[0], ' '.join(words)])

    train = pd.read_csv("imdb_tr.csv")
    train_data = train.text
    test = pd.read_csv("imdb_te.csv", encoding = "ISO-8859-1")
    test_data = test.text

    '''train a SGD classifier using unigram representation,
    predict sentiments on imdb_te.csv, and write output to
    unigram.output.txt'''
    vectorizer = CountVectorizer()
    train_vec = vectorizer.fit_transform(train_data)
    test_vec = vectorizer.transform(test_data)
    clf = linear_model.SGDClassifier(loss ='hinge', penalty = 'l1')
    clf.fit(train_vec, train.polarity)
    uni = clf.predict(test_vec)
    #scores = cross_val_score(clf, train_vec, train.polarity, cv = 5, scoring = 'accuracy')
    #print scores.mean()
    with open("unigram.output.txt", "w") as unigram:
        for num in uni:
            unigram.write('{}\n'.format(num))

    '''train a SGD classifier using bigram representation,
    predict sentiments on imdb_te.csv, and write output to
    unigram.output.txt'''
    vectorizer2 = CountVectorizer(ngram_range=(2,2))
    train_vec2 = vectorizer2.fit_transform(train_data)
    test_vec2 = vectorizer2.transform(test_data)
    clf2 = linear_model.SGDClassifier(loss ='hinge', penalty = 'l1')
    clf2.fit(train_vec2, train.polarity)
    bi = clf2.predict(test_vec2)
    #scores = cross_val_score(clf2, train_vec2, train.polarity, cv = 5, scoring = 'accuracy')
    #print scores.mean()
    with open("bigram.output.txt", "w") as bigram:
        for num in bi:
            bigram.write('{}\n'.format(num))

    '''train a SGD classifier using unigram representation
     with tf-idf, predict sentiments on imdb_te.csv, and write 
     output to unigram.output.txt'''
    transformer = TfidfVectorizer(smooth_idf=False)
    tfidf_train = transformer.fit_transform(train_data)
    tfidf_test = transformer.transform(test_data)
    tfidf = linear_model.SGDClassifier(loss ='hinge', penalty = 'l1')
    tfidf.fit(tfidf_train, train.polarity)
    uni_tf = tfidf.predict(tfidf_test)
    #scores = cross_val_score(tfidf, tfidf_train, train.polarity, cv = 5, scoring = 'accuracy')
    #print scores.mean()
    with open("unigramtfidf.output.txt", "w") as unigram_tf:
        for num in uni_tf:
            unigram_tf.write('{}\n'.format(num))

    '''train a SGD classifier using bigram representation
     with tf-idf, predict sentiments on imdb_te.csv, and write 
     output to unigram.output.txt'''
    transformer2 = TfidfVectorizer(smooth_idf=False, ngram_range=(2,2))
    tfidf_train2 = transformer.fit_transform(train_data)
    tfidf_test2 = transformer.transform(test_data)
    tfidf2 = linear_model.SGDClassifier(loss ='hinge', penalty = 'l1')
    tfidf2.fit(tfidf_train2, train.polarity)
    bi_tf = tfidf.predict(tfidf_test2)
    #scores = cross_val_score(tfidf2, tfidf_train2, train.polarity, cv = 5, scoring = 'accuracy')
    #print scores.mean()
    with open("bigramtfidf.output.txt", "w") as bigram_tf:
        for num in bi_tf:
            bigram_tf.write('{}\n'.format(num))



