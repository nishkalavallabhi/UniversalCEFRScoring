#Purpose: Build a scorer with POS N-grams. Use it on another language.

import pprint
import os
import collections
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score,cross_val_predict,StratifiedKFold
from sklearn.metrics import f1_score,classification_report,accuracy_score,confusion_matrix, mean_absolute_error
from xgboost import XGBClassifier, XGBRegressor

from scipy.stats import spearmanr, pearsonr

import language_check

'''
convert a text into its POS form. i.e., each word is replaced by its POS
'''
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

def makeTextOnly(conllufilepath):
    fh =  open(conllufilepath)
    allText = []
    this_sentence = []
    sent_id = 0
    for line in fh:
        if line == "\n":
            word_string = " ".join(this_sentence) + "\n"
            allText.append(word_string)
            this_sentence = []
            sent_id = sent_id+1
        elif not line.startswith("#"):
            word = line.split("\t")[1]
            this_sentence.append(word)
    fh.close()
    return " ".join(allText) #Returns a string which contains one sentence as POS tag sequence per line

'''
convert a sentence into this form: nmod_NN_PRON, dobj_VB_NN etc. i.e., each word is replaced by a dep. trigram of that form.
So full text will look like this instead of a series of words or POS tags:
root_X_ROOT punct_PUNCT_X case_ADP_PROPN nmod_PROPN_X flat_PROPN_PROPN
 root_PRON_ROOT nsubj_NOUN_PRON case_ADP_PROPN det_DET_PROPN nmod_PROPN_NOUN
 case_ADP_NOUN det_DET_NOUN nummod_NUM_NOUN obl_NOUN_VERB root_VERB_ROOT case_ADP_NOUN det_DET_NOUN obl_NOUN_VERB appos_PROPN_NOUN flat_PROPN_PROPN case_ADP_NOUN obl_NOUN_VERB cc_CCONJ_PART conj_PART_PROPN punct_PUNCT_VERB
 advmod_ADJ_VERB case_ADP_VERB case_ADP_VERB nmod_NOUN_ADP case_ADP_VERB nmod_NOUN_ADP case_ADP_VERB det_DET_NUM obl_NUM_VERB root_VERB_ROOT punct_PUNCT_VERB
 root_PRON_ROOT obj_NOUN_PROPN det_DET_PROPN amod_PROPN_PRON cc_CCONJ_ADV conj_ADV_PROPN cc_CCONJ_ADV punct_PUNCT_PROPN advmod_ADV_PUNCT case_ADP_ADJ advmod_ADV_PUNCT conj_ADV_PROPN amod_PROPN_PRON appos_PROPN_PROPN flat_PROPN_PROPN punct_PUNCT_PROPN
'''
def makeDepRelSentences(conllufilepath):
    fh =  open(conllufilepath)
    wanted_features = []
    deprels_sentence = []
    sent_id = 0
    head_ids_sentence = []
    pos_tags_sentence = []
    wanted_sentence_form = []
    id_dict = {} #Key: Index, Value: Word or POS depending on what dependency trigram we need. I am taking POS for now.
    id_dict['0'] = "ROOT"
    for line in fh:
        if line == "\n":
            for rel in deprels_sentence:
                wanted = rel + "_" + pos_tags_sentence[deprels_sentence.index(rel)] + "_" +id_dict[head_ids_sentence[deprels_sentence.index(rel)]]
                wanted_sentence_form.append(wanted)
                #Trigrams of the form case_ADP_PROPN, flat_PROPN_PROPN etc.
            wanted_features.append(" ".join(wanted_sentence_form) + "\n")
            deprels_sentence = []
            pos_tags_sentence = []
            head_ids_sentence = []
            wanted_sentence_form = []
            sent_id = sent_id+1
            id_dict = {}
            id_dict['0'] = "root" #LOWERCASING. Some problem with case of features in vectorizer.

        elif not line.startswith("#") and "-" not in line.split("\t")[0]:
            fields = line.split("\t")
            pos_tag = fields[3]
            deprels_sentence.append(fields[7])
            id_dict[fields[0]] = pos_tag
            pos_tags_sentence.append(pos_tag)
            head_ids_sentence.append(fields[6])
    fh.close()
    return " ".join(wanted_features)


"""
As described in Lu, 2010: http://onlinelibrary.wiley.com/doi/10.1111/j.1540-4781.2011.01232_1.x/epdf
Lexical words (N_lex: all open-class category words in UD (ADJ, ADV, INTJ, NOUN, PROPN, VERB)
All words (N)
Lex.Density = N_lex/N
Lex. Variation = Uniq_Lex/N_Lex
Type-Token Ratio = Uniq_words/N
Verb Variation = Uniq_Verb/N_verb
Noun Variation
ADJ variation
ADV variation
Modifier variation
"""
def getLexFeatures(conllufilepath,lang):
    fh =  open(conllufilepath)
    ndw = [] #To get number of distinct words
    ndn = [] #To get number of distinct nouns - includes propn
    ndv = [] #To get number of distinct verbs
    ndadj = []
    ndadv = []
    ndint = []
    numN = 0.0 #INCL PROPN
    numV = 0.0
    numI = 0.0 #INTJ
    numADJ = 0.0
    numADV = 0.0
    numIntj = 0.0
    total = 0.0
    numSent = 0.0
    for line in fh:
        if not line == "\n" and not line.startswith("#"):
            fields = line.split("\t")
            word = fields[1]
            pos_tag = fields[3]
            if word.isalpha():
                if not word in ndw:
                    ndw.append(word)
                if pos_tag == "NOUN" or pos_tag == "PROPN":
                    numN = numN +1
                    if not word in ndn:
                        ndn.append(word)
                elif pos_tag == "ADJ":
                    numADJ = numADJ+1
                    if not word in ndadj:
                        ndadj.append(word)
                elif pos_tag == "ADV":
                    numADV = numADV+1
                    if not word in ndadv:
                        ndadv.append(word)
                elif pos_tag == "VERB":
                    numV = numV+1
                    if not word in ndv:
                        ndv.append(word)
                elif pos_tag == "INTJ":
                    numI = numI +1
                    if not word in ndint:
                        ndint.append(word)
        elif line == "\n":
            numSent = numSent +1
        total = total +1
    error_features = getErrorFeatures(conllufilepath,lang)
    nlex = float(numN + numV + numADJ + numADV + numI) #Total Lexical words i.e., tokens
    dlex = float(len(ndn) + len(ndv) + len(ndadj) + len(ndadv) + len(ndint)) #Distinct Lexical words i.e., types

    #Scriptlen, Mean Sent Len, TTR, LexD, LexVar, VVar, NVar, AdjVar, AdvVar, ModVar, Total_Errors, Total Spelling errors
    result = [total, round(total/numSent,2), round(len(ndw)/total,2), round(nlex/total,2), round(dlex/nlex,2), round(len(ndv)/nlex,2), round(len(ndn)/nlex,2),
              round(len(ndadj)/nlex,2), round(len(ndadv)/nlex,2), round((len(ndadj) + len(ndadv))/nlex,2),error_features[0], error_features[1]]
    return result

"""
Num. Errors. NumSpellErrors
May be other error based features can be added later.
"""
def getErrorFeatures(conllufilepath, lang):
    checker = language_check.LanguageTool(lang)
    text = makeTextOnly(conllufilepath)
    matches = checker.check(text)
    numerr = 0
    numspellerr = 0
    for match in matches:
        if not match.locqualityissuetype == "whitespace":
            numerr = numerr +1
            if match.locqualityissuetype == "typographical" or match.locqualityissuetype == "misspelling":
                numspellerr = numspellerr +1
    return [numerr, numspellerr]


"""
get features that are typically used in scoring models using getErrorFeatures and getLexFeatures functions.
"""
def getScoringFeatures(dirpath,lang):
    files = os.listdir(dirpath)
    fileslist = []
    featureslist = []
    for file in files:
        if file.endswith(".txt"):
            features_for_this_file = getLexFeatures(os.path.join(dirpath,file),lang)
            fileslist.append(file)
            featureslist.append(features_for_this_file)
    return fileslist, featureslist


"""
Function to get n-gram like features for Word, POS, and Dependency representations
"""
def getLangData(dirpath):
    files = os.listdir(dirpath)
    fileslist = []
    posversionslist = []
    for file in files:
        if file.endswith(".txt"):
            pos_version_of_file = makePOSsentences(os.path.join(dirpath,file)) #DO THIS TO GET POS N-GRAM FEATURES later
            #pos_version_of_file = makeDepRelSentences(os.path.join(dirpath,file)) #DO THIS TO GET DEP-TRIAD N-gram features later
            #pos_version_of_file = makeTextOnly(os.path.join(dirpath,file)) #DO THIS TO GET Word N-gram features later
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
    uni_to_tri_vectorizer =  CountVectorizer(analyzer = "word", tokenizer = None, preprocessor = None, stop_words = None, ngram_range=(1,5), min_df=10)
    vectorizers = [uni_to_tri_vectorizer]
    classifiers = [RandomForestClassifier(class_weight="balanced")] #RandomForestClassifier(), , XGBClassifier()] #Add more.GradientBoostingClassifier(),
    k_fold = StratifiedKFold(10)
    for vectorizer in vectorizers:
        for classifier in classifiers:
            print("Printing results for: " + str(classifier) + str(vectorizer))
            train_vector = vectorizer.fit_transform(train_data).toarray()
            #print(vectorizer.get_feature_names()) #To see what features were selected.
            cross_val = cross_val_score(classifier, train_vector, train_labels, cv=k_fold, n_jobs=1)
            predicted = cross_val_predict(classifier, train_vector, train_labels, cv=k_fold, n_jobs=1)
            print(cross_val)
            print(sum(cross_val)/float(len(cross_val)))
            print(vectorizer.get_feature_names())
            print(confusion_matrix(train_labels, predicted, labels=["A1","A2","B1","B2", "C1", "C2"]))
    print("SAME LANG EVAL DONE")

def train_onelang_regression(train_scores,train_data):
    uni_to_tri_vectorizer =  CountVectorizer(analyzer = "word", tokenizer = None, preprocessor = None, stop_words = None, ngram_range=(1,5), min_df=10, max_features = 2000)
    vectorizers = [uni_to_tri_vectorizer]
    regressors = [LinearRegression(), RandomForestRegressor(), GradientBoostingRegressor(), XGBRegressor()]
    k_fold = StratifiedKFold(10)
    for vectorizer in vectorizers:
        for regressor in regressors:
            train_vector = vectorizer.fit_transform(train_data).toarray()
            print("Printing results for: " + str(regressor) + str(vectorizer))
            cross_val = cross_val_score(regressor, train_vector, train_scores, cv=k_fold, n_jobs=1)
            predicted = cross_val_predict(regressor, train_vector, train_scores, cv=k_fold)
            predicted[predicted < 0] = 0
            n = len(predicted)
            print("RMSLE: ", np.sqrt((1/n) * sum(np.square(np.log10(predicted +1) - (np.log10(train_scores) +1)))))
            print("MAE: ", mean_absolute_error(train_scores,predicted))
            print("Pearson: ", pearsonr(train_scores,predicted))
            print("Spearman: ", spearmanr(train_scores,predicted))
    print("SAME LANG EVAL DONE")

def cross_lang_testing_classification(train_labels,train_data, test_labels, test_data):
    uni_to_tri_vectorizer =  CountVectorizer(analyzer = "word", tokenizer = None, preprocessor = None, stop_words = None, ngram_range=(1,5), min_df=10, max_features = 2000)
    vectorizers = [uni_to_tri_vectorizer]
    classifiers = [RandomForestClassifier(class_weight="balanced")] #RandomForestClassifier(), RandomForestClassifier(class_weight="balanced"), GradientBoostingClassifier()] #Side note: gradient boosting needs a dense array. Testing fails for that. Should modifiy the pipeline later to account for this.
    #Check this discussion for handling the sparseness issue: https://stackoverflow.com/questions/28384680/scikit-learns-pipeline-a-sparse-matrix-was-passed-but-dense-data-is-required
    for vectorizer in vectorizers:
        for classifier in classifiers:
            print("Printing results for: " + str(classifier) + str(vectorizer))
            text_clf = Pipeline([('vect', vectorizer), ('clf', classifier)])
            text_clf.fit(train_data,train_labels)
            print(vectorizer.get_feature_names())
            predicted = text_clf.predict(test_data)
            #print(vectorizer.get_feature_names())
            print(np.mean(predicted == test_labels,dtype=float))
            print(confusion_matrix(test_labels, predicted, labels=["A1","A2","B1","B2", "C1", "C2"]))
            print("CROSS LANG EVAL DONE")
"""
Note: XGBoost classifier has some issue with retaining feature names between train and test data properly. This is resulting in error while doing cross language classification.
Strangely, I did not encounter this issue with POS trigrams. Only encountering with dependency features.
Seems to be a known issue: https://github.com/dmlc/xgboost/issues/2334
"""

def cross_lang_testing_regression(train_scores, train_data, test_scores, test_data):
    uni_to_tri_vectorizer =  CountVectorizer(analyzer = "char", tokenizer = None, preprocessor = None, stop_words = None, ngram_range=(1,10), min_df=10, max_features = 10000)
    vectorizers = [uni_to_tri_vectorizer]
    regressors = [RandomForestRegressor()] #linear_model.LinearRegression(),  - seems to be doing badly for cross-lang.
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
            print("Pearson: ", pearsonr(test_scores,predicted))
            print("Spearman: ", spearmanr(test_scores,predicted))

def doExperimentsWithoutVectorizer(train_vector,train_labels): #test_vector,test_labels):

    k_fold = StratifiedKFold(10)
    classifiers = [LogisticRegression(C=0.1, max_iter=500)] #Add more later
    #classifiers = [MLPClassifier(max_iter=500)]
    #RandomForestClassifer(), GradientBoostClassifier()
    #Not useful: SVC with kernels - poly, sigmoid, rbf.

    for classifier in classifiers:
        print(classifier)
        cross_val = cross_val_score(classifier, train_vector, train_labels, cv=k_fold, n_jobs=1)
        predicted = cross_val_predict(classifier, train_vector, train_labels, cv=k_fold)
        print(cross_val)
        print(sum(cross_val)/float(len(cross_val)))
        print(confusion_matrix(train_labels, predicted))

def main():

    dedirpath = "/Users/sowmya/Research/CrossLing-Scoring/CrossLingualScoring/Datasets/DE-Parsed"
    defiles,deposdata = getScoringFeatures(dedirpath, "de")
    delabels = getcatlist(defiles)
    print(collections.Counter(delabels))
    print("DE data details: ", len(delabels), len(deposdata))
    doExperimentsWithoutVectorizer(deposdata,delabels)


    """
    itdirpath = "/Users/sowmya/Research/CrossLing-Scoring/CrossLingualScoring/Datasets/IT-Parsed"
    fileslist,itposdata = getLangData(itdirpath)
    itlabels = getcatlist(fileslist)
    itscores = getnumlist(fileslist)
    print("IT data details: ", len(fileslist), len(itposdata))
    print(collections.Counter(itlabels))

    dedirpath = "/Users/sowmya/Research/CrossLing-Scoring/CrossLingualScoring/Datasets/DE-Parsed"
    defiles,deposdata = getLangData(dedirpath)
    delabels = getcatlist(defiles)
    descores = getnumlist(defiles)
    print(collections.Counter(delabels))
    print("DE data details: ", len(delabels), len(deposdata))

    czdirpath = "/Users/sowmya/Research/CrossLing-Scoring/CrossLingualScoring/Datasets/CZ-Parsed"
    czfiles,czposdata = getLangData(czdirpath)
    czlabels = getcatlist(czfiles)
    czscores = getnumlist(czfiles)
    print("CZ data details: ", len(czlabels), len(czposdata))
    print(collections.Counter(czlabels))

    print("Training and Testing with German - Classification")
    train_onelang_classification(delabels,deposdata)
    print("***********")
    train_onelang_classification(itlabels,itposdata)
    train_onelang_classification(czlabels,czposdata)


    print("Training with German, Testing on Italian  - Classification: ")
    cross_lang_testing_classification(delabels,deposdata, itlabels, itposdata)
     #get basic stats
    print("***********")

    print("Training with German, Testing on Czech  - Classification: ")
    cross_lang_testing_classification(delabels,deposdata,czlabels, czposdata)
     #get basic stats
    #print(collections.Counter(getcatlist(fileslist))) #get basic stats
    print("***********")

    print("Training and Testing with German - Regression")
    train_onelang_regression(descores,deposdata)
    print("***********")

    print("Training with German, Testing on Italian  - Regression: ")
    cross_lang_testing_regression(descores,deposdata,itscores,itposdata)
    print("***********")

    print("Training with German, Testing on Czech - Regression: ")
    cross_lang_testing_regression(descores,deposdata,czscores,czposdata)
    print("***********")
    """
#print(getLexFeatures("/Users/sowmya/Research/CrossLing-Scoring/CrossLingualScoring/Datasets/DE-Parsed/1031_0003076_DE_C1.txt.parsed.txt", "de"))
#exit(1)

if __name__ == "__main__":
    main()
