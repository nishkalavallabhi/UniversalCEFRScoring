#Purpose: Build a scorer with POS N-grams. Use it on another language.

import pprint
import os
import collections
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn import linear_model
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score,cross_val_predict,StratifiedKFold
from sklearn.metrics import f1_score,classification_report,accuracy_score,confusion_matrix, mean_absolute_error

def makePOSsentences(conllufilepath):
    fh =  open(conllufilepath)
    everything_POS = []

    pos_sentence = []
    sent_id = 0
    for line in fh:
        if line == "\n":
            pos_string = " ".join(pos_sentence) + "\n"
            everything_POS.append(pos_string)
            pos_sentence = []
            sent_id = sent_id+1
        elif not line.startswith("#"):
            pos_tag = line.split("\t")[3]
            pos_sentence.append(pos_tag)
    fh.close()
    return " ".join(everything_POS) #Returns a string which contains one sentence as POS tag sequence per line

def getLangData(dirpath):
    files = os.listdir(dirpath)
    fileslist = []
    posversionslist = []
    for file in files:
        if file.endswith(".txt"):
            pos_version_of_file = makePOSsentences(os.path.join(dirpath,file))
            fileslist.append(file)
            posversionslist.append(pos_version_of_file)
    return fileslist, posversionslist

#Get categories from filenames  -Classification
def getcatlist(filenameslist):
    result = []
    for name in filenameslist:
        #result.append(name.split("_")[3].split(".txt")[0])
        result.append(name.split(".txt")[0].split("_")[-1])
    return result

#Get numbers from filenames - Regression
def getnumlist(filenameslist):
    result = []
    mapping = {"A1":1, "A2":2, "B1":3, "B2":4, "C1":5, "C2":6}
    for name in filenameslist:
        #result.append(mapping[name.split("_")[3].split(".txt")[0]])
        result.append(mapping[name.split(".txt")[0].split("_")[-1]])
    return result

#Training on one language data, Stratified 10 fold CV
def train_onelang_classification(train_labels,train_data):
    uni_to_tri_vectorizer =  CountVectorizer(analyzer = "word", tokenizer = None, preprocessor = None, stop_words = None, ngram_range=(1,3), min_df=10, max_features = 500)
    vectorizers = [uni_to_tri_vectorizer]
    classifiers = [GradientBoostingClassifier()] #Add more.
    k_fold = StratifiedKFold(10)
    for vectorizer in vectorizers:
        for classifier in classifiers:
            print("Printing results for: " + str(classifier) + str(vectorizer))
            train_vector = vectorizer.fit_transform(train_data).toarray()
            cross_val = cross_val_score(classifier, train_vector, train_labels, cv=k_fold, n_jobs=1)
            print(cross_val)
            print(sum(cross_val)/float(len(cross_val)))
            print(vectorizer.get_feature_names())
            print("SAME LANG EVAL DONE")

def train_onelang_regression(train_scores,train_data):
    uni_to_tri_vectorizer =  CountVectorizer(analyzer = "word", tokenizer = None, preprocessor = None, stop_words = None, ngram_range=(1,3), min_df=10, max_features = 500)
    vectorizers = [uni_to_tri_vectorizer]
    regressors = [linear_model.LinearRegression(), RandomForestRegressor()] #GradientBoostingRegressor()]
    k_fold = StratifiedKFold(10)
    for vectorizer in vectorizers:
        for regressor in regressors:
            train_vector = vectorizer.fit_transform(train_data).toarray()
            print("Printing results for: " + str(regressor) + str(vectorizer))
            cross_val = cross_val_score(regressor, train_vector, train_scores, cv=k_fold, n_jobs=1)
            predicted = cross_val_predict(regressor, train_vector, train_scores, cv=k_fold)
            print(cross_val)
            print(sum(cross_val)/float(len(cross_val)))
            print(vectorizer.get_feature_names())
            predicted[predicted < 0] = 0
            n = len(predicted)
            print("RMSLE: ", np.sqrt((1/n) * sum(np.square(np.log10(predicted +1) - (np.log10(train_scores) +1)))))
            print("MAE: ", mean_absolute_error(train_scores,predicted))
    print("SAME LANG EVAL DONE")

def cross_lang_testing_classification(train_labels,train_data, test_labels, test_data):
    uni_to_tri_vectorizer =  CountVectorizer(analyzer = "word", tokenizer = None, preprocessor = None, stop_words = None, ngram_range=(1,3), min_df=10, max_features = 500)
    vectorizers = [uni_to_tri_vectorizer]
    classifiers = [RandomForestClassifier()] #GradientBoostingClassifier()] #Side note: gradient boosting needs a dense array. Testing fails for that. Should modifiy the pipeline later to account for this.
    #Check this discussion for handling the sparseness issue: https://stackoverflow.com/questions/28384680/scikit-learns-pipeline-a-sparse-matrix-was-passed-but-dense-data-is-required
    for vectorizer in vectorizers:
        for classifier in classifiers:
            print("Printing results for: " + str(classifier) + str(vectorizer))
            text_clf = Pipeline([('vect', vectorizer), ('clf', classifier)])
            text_clf.fit(train_data,train_labels)
            predicted = text_clf.predict(test_data)
            print(vectorizer.get_feature_names())
            print(np.mean(predicted == test_labels,dtype=float))
            print(confusion_matrix(test_labels, predicted, labels=["A1","A2","B1","B2", "C1", "C2"]))

            print("CROSS LANG EVAL DONE")

def cross_lang_testing_regression(train_scores, train_data, test_scores, test_data):
    uni_to_tri_vectorizer =  CountVectorizer(analyzer = "word", tokenizer = None, preprocessor = None, stop_words = None, ngram_range=(1,3), min_df=10, max_features = 500)
    vectorizers = [uni_to_tri_vectorizer]
    regressors = [linear_model.LinearRegression(), RandomForestRegressor()]
    for vectorizer in vectorizers:
        for regressor in regressors:
            train_vector = vectorizer.fit_transform(train_data).toarray()
            print("Printing results for: " + str(regressor) + str(vectorizer))
            text_clf = Pipeline([('vect', vectorizer), ('clf', regressor)])
            text_clf.fit(train_data,train_scores)
            predicted = text_clf.predict(test_data)
            predicted[predicted < 0] = 0
            n = len(predicted)
            print("RMSLE: ", np.sqrt((1/n) * sum(np.square(np.log10(predicted +1) - (np.log10(test_scores) +1)))))
            print("MAE: ", mean_absolute_error(test_scores,predicted))

def main():
    itdirpath = "/Users/sowmya/Research/CrossLing-Scoring/CrossLingualScoring/Datasets/IT-Parsed"
    fileslist,itposdata = getLangData(itdirpath)
    itlabels = getcatlist(fileslist)
    itscores = getnumlist(fileslist)
    print("IT data details: ", len(fileslist), len(itposdata))

    dedirpath = "/Users/sowmya/Research/CrossLing-Scoring/CrossLingualScoring/Datasets/DE-Parsed"
    defiles,deposdata = getLangData(dedirpath)
    delabels = getcatlist(defiles)
    descores = getnumlist(defiles)
    print("DE data details: ", len(delabels), len(deposdata))

    czdirpath = "/Users/sowmya/Research/CrossLing-Scoring/CrossLingualScoring/Datasets/CZ-Parsed"
    czfiles,czposdata = getLangData(czdirpath)
    czlabels = getcatlist(czfiles)
    czscores = getnumlist(czfiles)
    print("CZ data details: ", len(czlabels), len(czposdata))

    """
    train_onelang_classification(delabels,deposdata)
    cross_lang_testing_classification(delabels,deposdata, itlabels, itposdata)
    print(collections.Counter(itlabels)) #get basic stats
    cross_lang_testing_classification(delabels,deposdata, czlabels, czposdata)
    print(collections.Counter(czlabels)) #get basic stats
    #print(collections.Counter(getcatlist(fileslist))) #get basic stats
    """
    train_onelang_regression(descores,deposdata)
    cross_lang_testing_regression(descores,deposdata,itscores,itposdata)
    cross_lang_testing_regression(descores,deposdata,czscores,czposdata)


if __name__ == "__main__":
    main()
