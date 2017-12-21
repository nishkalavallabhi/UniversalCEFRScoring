#Purpose: Build a scorer with POS N-grams. Use it on another language.

import pprint
import os
import collections
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import Imputer #to replace NaN with mean values.
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score,cross_val_predict,StratifiedKFold 
from sklearn.metrics import f1_score,classification_report,accuracy_score,confusion_matrix, mean_absolute_error
from xgboost import XGBClassifier, XGBRegressor
from sklearn.svm import LinearSVC

from scipy.stats import spearmanr, pearsonr

import language_check

seed=1234

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
def getLexFeatures(conllufilepath,lang, err):
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
    if err:
        error_features = getErrorFeatures(conllufilepath,lang)
    else:
        error_features = ['NA', 'NA']

    nlex = float(numN + numV + numADJ + numADV + numI) #Total Lexical words i.e., tokens
    dlex = float(len(ndn) + len(ndv) + len(ndadj) + len(ndadv) + len(ndint)) #Distinct Lexical words i.e., types

    #Scriptlen, Mean Sent Len, TTR, LexD, LexVar, VVar, NVar, AdjVar, AdvVar, ModVar, Total_Errors, Total Spelling errors
    result = [total, round(total/numSent,2), round(len(ndw)/total,2), round(nlex/total,2), round(dlex/nlex,2), round(len(ndv)/nlex,2), round(len(ndn)/nlex,2),
              round(len(ndadj)/nlex,2), round(len(ndadv)/nlex,2), round((len(ndadj) + len(ndadv))/nlex,2),error_features[0], error_features[1]]
    if not err: #remove last two features - they are error features which are NA for cz
       return result[:-2]
    else:
       return result

"""
Num. Errors. NumSpellErrors
May be other error based features can be added later.
"""
def getErrorFeatures(conllufilepath, lang):
    try:
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
    except:
        print("Ignoring this text: ", conllufilepath)
        numerr = np.nan
        numspellerr = np.nan
    return [numerr, numspellerr]


"""
get features that are typically used in scoring models using getErrorFeatures and getLexFeatures functions.
err - indicates whether or not error features should be extracted. Boolean. True/False
"""
def getScoringFeatures(dirpath,lang,err):
    files = os.listdir(dirpath)
    fileslist = []
    featureslist = []
    for filename in files:
        if filename.endswith(".txt"):
            features_for_this_file = getLexFeatures(os.path.join(dirpath,filename),lang,err)
            fileslist.append(filename)
            featureslist.append(features_for_this_file)
    return fileslist, featureslist


"""
Function to get n-gram like features for Word, POS, and Dependency representations
option takes: word, pos, dep. default is word
"""
def getLangData(dirpath, option):
    files = os.listdir(dirpath)
    fileslist = []
    posversionslist = []
    for filename in files:
        if filename.endswith(".txt"):
            if option == "pos":
            	pos_version_of_file = makePOSsentences(os.path.join(dirpath,filename)) #DO THIS TO GET POS N-GRAM FEATURES later
            elif option == "dep":
                pos_version_of_file = makeDepRelSentences(os.path.join(dirpath,filename)) #DO THIS TO GET DEP-TRIAD N-gram features later
            else:
                pos_version_of_file = makeTextOnly(os.path.join(dirpath,filename)) #DO THIS TO GET Word N-gram features later
            fileslist.append(filename)
            posversionslist.append(pos_version_of_file)
    return fileslist, posversionslist

#Get categories from filenames  -Classification
def getcatlist(filenameslist):
    result = []
    for name in filenameslist:
        #result.append(name.split("_")[3].split(".txt")[0])
        result.append(name.split(".txt")[0].split("_")[-1])
    return result

#Get langs list from filenames - to use in megadataset classification
def getlangslist(filenameslist):
    result = []
    for name in filenameslist:
        if "_DE_" in name:
           result.append("de")
        elif "_IT_" in name:
           result.append("it")
        else:
           result.append("cz")
    return result

#Get numbers from filenames - Regression
def getnumlist(filenameslist):
    result = []
    mapping = {"A1":1, "A2":2, "B1":3, "B2":4, "C1":5, "C2":6}
    for name in filenameslist:
        #result.append(mapping[name.split("_")[3].split(".txt")[0]])
        result.append(mapping[name.split(".txt")[0].split("_")[-1]])
    return result

#Regression evaluation.
def regEval(predicted,actual):
    n = len(predicted)
    MAE = mean_absolute_error(actual,predicted)
    pearson = pearsonr(actual,predicted)
    spearman = spearsssmanr(actual,predicted)
    rmsle = np.sqrt((1/n) * sum(np.square(np.log10(predicted +1) - (np.log10(actual) +1))))
    return {"MAE": MAE, "rmlse": rmsle, "pearson": pearson, "spearman":spearman}

#Training on one language data, Stratified 10 fold CV
def train_onelang_classification(train_labels,train_data,labelascat=False, langslist=None):
    uni_to_tri_vectorizer =  CountVectorizer(analyzer = "word", tokenizer = None, preprocessor = None, stop_words = None, ngram_range=(1,5), min_df=10)
    vectorizers = [uni_to_tri_vectorizer]
    classifiers = [RandomForestClassifier(class_weight="balanced"), LinearSVC(class_weight="balanced"), LogisticRegression(class_weight="balanced")] #Add more.GradientBoostingClassifier(),
    k_fold = StratifiedKFold(10,random_state=seed)
    for vectorizer in vectorizers:
        for classifier in classifiers:
            print("Printing results for: " + str(classifier) + str(vectorizer))
            train_vector = vectorizer.fit_transform(train_data).toarray()
            print(len(train_vector[0]))
            if labelascat and len(langslist) > 1:
               train_vector = enhance_features_withcat(train_vector,language=None,langslist=langslist)
            print(len(train_vector[0]))
            #print(vectorizer.get_feature_names()) #To see what features were selected.
            cross_val = cross_val_score(classifier, train_vector, train_labels, cv=k_fold, n_jobs=1)
            predicted = cross_val_predict(classifier, train_vector, train_labels, cv=k_fold, n_jobs=1)
           # print(cross_val)
            print(sum(cross_val)/float(len(cross_val)), f1_score(train_labels,predicted,average='weighted'))
            #print(vectorizer.get_feature_names())
            print(confusion_matrix(train_labels, predicted, labels=["A1","A2","B1","B2", "C1", "C2"]))
            #print(predicted)
    print("SAME LANG EVAL DONE FOR THIS LANG")
    

"""
Combine features like this: get probability distribution over categories with n-gram features. Use that distribution as a feature set concatenated with the domain features - one way to combine sparse and dense feature groups.
Just testing this approach here. 
"""
def combine_features(train_labels,train_sparse,train_dense):
    k_fold = StratifiedKFold(10,random_state=seed)
    vectorizer =  CountVectorizer(analyzer = "word", tokenizer = None, preprocessor = None, stop_words = None, ngram_range=(1,3), min_df=10, max_features = 2000)
    train_vector = vectorizer.fit_transform(train_sparse).toarray()
    classifier = RandomForestClassifier(class_weight="balanced")
  #  cross_val = cross_val_score(classifier, train_vector, train_labels, cv=k_fold, n_jobs=1)
  #  print("Old CV score with sparse features", str(sum(cross_val)/float(len(cross_val))))
  #  predicted = cross_val_predict(classifier, train_vector, train_labels, cv=k_fold)
    #print(f1_score(train_labels,predicted,average='weighted'))

    #Get probability distribution for classes.
    predicted = cross_val_predict(classifier, train_vector, train_labels, cv=k_fold, method="predict_proba")
    #Use those probabilities as the new featureset.
    new_features = []
    for i in range(0,len(predicted)):
       temp = list(predicted[i]) + list(train_dense[i])
       new_features.append(temp)
    #predict with new features
    new_predicted = cross_val_predict(classifier, new_features, train_labels, cv=k_fold)
    cross_val = cross_val_score(classifier, train_vector, train_labels, cv=k_fold, n_jobs=1)
   # print("new CV score", str(cross_val))
    print("Acc: " ,str(sum(cross_val)/float(len(cross_val))))
    print("F1: ", str(f1_score(train_labels,new_predicted,average='weighted')))
    
"""
Single language, regression with 10 fold CV
"""
def train_onelang_regression(train_scores,train_data):
    uni_to_tri_vectorizer =  CountVectorizer(analyzer = "word", tokenizer = None, preprocessor = None, stop_words = None, ngram_range=(1,5), min_df=10) #can specify max_features but dataset seems small enough
    vectorizers = [uni_to_tri_vectorizer]
    regressors = [LinearRegression(), RandomForestRegressor(), GradientBoostingRegressor(), XGBRegressor()]
    k_fold = StratifiedKFold(10,random_state=seed)
    for vectorizer in vectorizers:
        for regressor in regressors:
            train_vector = vectorizer.fit_transform(train_data).toarray()
            print("Printing results for: " + str(regressor) + str(vectorizer))
            cross_val = cross_val_score(regressor, train_vector, train_scores, cv=k_fold, n_jobs=1)
            predicted = cross_val_predict(regressor, train_vector, train_scores, cv=k_fold)
            predicted[predicted < 0] = 0
            n = len(predicted)
            print(regEval(predicted,train_scores))
    print("SAME LANG EVAL DONE")

"""
train on one language and test on another, classification
"""
def cross_lang_testing_classification(train_labels,train_data, test_labels, test_data):
    uni_to_tri_vectorizer =  CountVectorizer(analyzer = "word", tokenizer = None, preprocessor = None, stop_words = None, ngram_range=(1,5), min_df=10) #, max_features = 2000
    vectorizers = [uni_to_tri_vectorizer]
    classifiers = [RandomForestClassifier(class_weight="balanced"), LinearSVC(class_weight="balanced"), LogisticRegression(class_weight="balanced")] #, LinearSVC()RandomForestClassifier(), RandomForestClassifier(class_weight="balanced"), GradientBoostingClassifier()] #Side note: gradient boosting needs a dense array. Testing fails for that. Should modifiy the pipeline later to account for this.
    #Check this discussion for handling the sparseness issue: https://stackoverflow.com/questions/28384680/scikit-learns-pipeline-a-sparse-matrix-was-passed-but-dense-data-is-required
    for vectorizer in vectorizers:
        for classifier in classifiers:
            print("Printing results for: " + str(classifier) + str(vectorizer))
            text_clf = Pipeline([('vect', vectorizer), ('clf', classifier)])
            text_clf.fit(train_data,train_labels)
            #print(vectorizer.get_feature_names())
            predicted = text_clf.predict(test_data)
            #print(vectorizer.get_feature_names())
            print(np.mean(predicted == test_labels,dtype=float))
            print(confusion_matrix(test_labels, predicted, labels=["A1","A2","B1","B2", "C1", "C2"]))
            print("CROSS LANG EVAL DONE. F1score: ")
            print(f1_score(test_labels,predicted,average='weighted'))
"""
Note: XGBoost classifier has some issue with retaining feature names between train and test data properly. This is resulting in error while doing cross language classification.
Strangely, I did not encounter this issue with POS trigrams. Only encountering with dependency features.
Seems to be a known issue: https://github.com/dmlc/xgboost/issues/2334
"""

#train on one language and test on another, classification
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

#Single language, 10 fold cv for domain features - i.e., non n-gram features.
def singleLangClassificationWithoutVectorizer(train_vector,train_labels): #test_vector,test_labels):
    k_fold = StratifiedKFold(10,random_state=seed)
    classifiers = [RandomForestClassifier(class_weight="balanced"), LinearSVC(class_weight="balanced"), LogisticRegression(class_weight="balanced")] #Add more later
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
        print(f1_score(train_labels,predicted,average='macro'))

#cross lingual classification evaluation for non ngram features
def crossLangClassificationWithoutVectorizer(train_vector, train_labels, test_vector, test_labels):
    print("CROSS LANG EVAL")
    classifiers = [RandomForestClassifier(class_weight="balanced"), LinearSVC(class_weight="balanced"), LogisticRegression(class_weight="balanced")]
    for classifier in classifiers:
        classifier.fit(train_vector,train_labels)
        predicted = classifier.predict(test_vector)
        print(np.mean(predicted == test_labels,dtype=float))
        print(confusion_matrix(test_labels,predicted))
        print(f1_score(test_labels,predicted,average='weighted'))

#cross lingual regression evaluation for non ngram features
def crossLangRegressionWithoutVectorizer(train_vector, train_scores, test_vector, test_scores):
    print("CROSS LANG EVAL")
    regressors = [RandomForestRegressor()]
    k_fold = StratifiedKFold(10,random_state=seed)
    for regressor in regressors:
        cross_val = cross_val_score(regressor, train_vector, train_scores, cv=k_fold, n_jobs=1)
        predicted = cross_val_predict(regressor, train_vector, train_scores, cv=k_fold)
        predicted[predicted < 0] = 0
        print("Cross Val Results: ")
        print(regEval(predicted,train_scores))
        regressor.fit(train_vector,train_scores)
        predicted =regressor.predict(test_vector)
        predicted[predicted < 0] = 0
        print("Test data Results: ")
        print(regEval(predicted,test_scores))

#add label features as one hot vector. de - 1 0 0, it - 0 1 0, cz - 0 0 1 as sklearn has issues with combination of cat and num features.
def enhance_features_withcat(features,language=None,langslist=None):
   addition = {'de':[1,0,0], 'it': [0,1,0], 'cz': [0,0,1]}
   if language:
        for i in range(0,len(features)):
           features[i].extend(addition[language])
        return features
   if langslist:
        features = np.ndarray.tolist(features)
        for i in range(0,len(features)):
           features[i].extend(addition[langslist[i]])
        return features



"""
Goal: combine all languages data into one big model
setting options: pos, dep, domain
labelascat = true, false (to indicate whether to add label as a categorical feature)
"""
def do_mega_multilingual_model_all_features(lang1path,lang1,lang2path,lang2,lang3path,lang3,modelas, setting,labelascat):
   print("Doing: take all data as if it belongs to one large dataset, and do classification")   
   if not setting == "domain":
      lang1files,lang1features = getLangData(lang1path,setting)
      lang1labels = getcatlist(lang1files)
      lang2files,lang2features = getLangData(lang2path,setting)
      lang2labels = getcatlist(lang2files)
      lang3files,lang3features = getLangData(lang3path,setting)
      lang3labels = getcatlist(lang3files)

   else: #i.e., domain features only.
      lang1files,lang1features = getScoringFeatures(lang1path,lang1,False)
      lang1labels = getcatlist(lang1files)
      lang2files,lang2features = getScoringFeatures(lang2path,lang2,False)
      lang2labels = getcatlist(lang2files)
      lang3files,lang3features = getScoringFeatures(lang3path,lang3,False)
      lang3labels = getcatlist(lang3files)

   megalabels = []
   megalabels = lang1labels + lang2labels + lang3labels
   megalangs = getlangslist(lang1files) + getlangslist(lang2files) + getlangslist(lang3files)
   if labelascat and setting == "domain": 
      megadata = enhance_features_withcat(lang1features,"de") + enhance_features_withcat(lang2features,"it") + enhance_features_withcat(lang3features,"cz")
   else:
      megadata = lang1features + lang2features + lang3features
   print("Mega classification for: ", setting, " features")	
   
   print(len(megalabels), len(megadata), len(megalangs), len(megadata[0]))
  
   print("Distribution of labels: ")
   print(collections.Counter(megalabels))
   if setting == "domain":
      singleLangClassificationWithoutVectorizer(megadata,megalabels)
   else:
      train_onelang_classification(megalabels,megadata,labelascat,megalangs)

"""
this function does cross language evaluation.
takes a language data directory path, and lang code for both source and target languages. 
gets all features (no domain features for cz), and prints the results with those.
lang codes: de, it, cz (lower case)
modelas: "class" for classification, "regr" for regression
"""
def do_cross_lang_all_features(sourcelangdirpath,sourcelang,modelas, targetlangdirpath, targetlang):
   #Read source language data
   sourcelangfiles,sourcelangposngrams = getLangData(sourcelangdirpath, "pos")
   sourcelangfiles,sourcelangdepngrams = getLangData(sourcelangdirpath, "dep")
   #Read target language data
   targetlangfiles,targetlangposngrams = getLangData(targetlangdirpath, "pos")
   targetlangfiles,targetlangdepngrams = getLangData(targetlangdirpath, "dep")
   #Get label info
   sourcelanglabels = getcatlist(sourcelangfiles)
   targetlanglabels = getcatlist(targetlangfiles)

   if "cz" not in [sourcelang, targetlang]:
      sourcelangfiles,sourcelangdomain = getScoringFeatures(sourcelangdirpath,sourcelang,True)
      targetlangfiles,targetlangdomain = getScoringFeatures(targetlangdirpath,targetlang,True)
   else: 
      sourcelangfiles,sourcelangdomain = getScoringFeatures(sourcelangdirpath,sourcelang,False)
      targetlangfiles,targetlangdomain = getScoringFeatures(targetlangdirpath,targetlang,False)
      #if targetlang == "it": #Those two files where langtool throws error
      #   mean_imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
      #   mean_imputer = mean_imputer.fit(targetlangdomain)
      #   imputed_df = mean_imputer.transform(targetlangdomain)
      #   targetlangdomain = imputed_df
      #   print("Modified domain feature vector for Italian")
      #TODO: it can be sourcelang too! I am ignoring that for now.
   if modelas == "class":
      print("Printing cross-corpus classification evaluation results: ")

      print("*******", "\n", "Setting - Train with: ", sourcelang, " Test with: ", targetlang, " ******", "\n")
      print("Features: pos")
      cross_lang_testing_classification(sourcelanglabels,sourcelangposngrams, targetlanglabels, targetlangposngrams)
      print("Features: dep")
      cross_lang_testing_classification(sourcelanglabels,sourcelangdepngrams, targetlanglabels, targetlangdepngrams)
      print("Features: domain")
      crossLangClassificationWithoutVectorizer(sourcelangdomain,sourcelanglabels,targetlangdomain,targetlanglabels)
   if modelas == "regr":
          print("Did not add for regression yet")
 
"""
this function takes a language data directory path, and lang code, 
gets all features, and prints the results with those.
lang codes: de, it, cz (lower case)
modelas: "class" for classification, "regr" for regression
"""
def do_single_lang_all_features(langdirpath,lang,modelas):
    langfiles,langwordngrams = getLangData(langdirpath, "word")
    langfiles,langposngrams = getLangData(langdirpath, "pos")
    langfiles,langdepngrams = getLangData(langdirpath, "dep")
    if not lang == "cz":
       langfiles,langdomain = getScoringFeatures(langdirpath,lang,True)
    else:
       langfiles,langdomain = getScoringFeatures(langdirpath,lang,False)
    
    print("Extracted all features: ")
    langlabels = getcatlist(langfiles)
    langscores = getnumlist(langfiles)

   # if lang == "it": #Those two files where langtool throws error
   #    mean_imputer = Imputer(missing_values='NA', strategy='mean', axis=0)
   #    mean_imputer = mean_imputer.fit(langdomain)
   #    imputed_df = mean_imputer.transform(langdomain)
   #    langdomain = imputed_df
   #    print("Modified domain feature vector for Italian")

    print("Printing class statistics")
    print(collections.Counter(langlabels))

    if modelas == "class":
       print("With Word ngrams:", "\n", "******") 
       train_onelang_classification(langlabels,langwordngrams)
       print("With POS ngrams: ", "\n", "******") 
       train_onelang_classification(langlabels,langposngrams)
       print("Dep ngrams: ", "\n", "******") 
       train_onelang_classification(langlabels,langdepngrams)
       print("Domain features: ", "\n", "******")
       singleLangClassificationWithoutVectorizer(langdomain,langlabels)
          
       print("Combined feature rep: wordngrams + domain")
       combine_features(langlabels,langwordngrams,langdomain)
       print("Combined feature rep: posngrams + domain")
       combine_features(langlabels,langposngrams,langdomain)
       print("Combined feature rep: depngrams + domain")
       combine_features(langlabels,langdepngrams,langdomain)
       #TODO
       #print("ALL COMBINED")

       """
       defiles,dedense = getScoringFeatures(dedirpath, "de", True)
       defiles,desparse = getLangData(dedirpath)
       delabels = getcatlist(defiles)
       combine_features(delabels,desparse,dedense)
       """
    elif modelas == "regr":
       print("With Word ngrams:", "\n", "******") 
       train_onelang_regression(langscores,langwordngrams)
       print("With POS ngrams: ", "\n", "******") 
       train_onelang_regression(langscores,langposngrams)
       print("Dep ngrams: ", "\n", "******") 
       train_onelang_regression(langscores,langwordngrams)
       #TODO: singleLangRegressionWithoutVectorizer function.
       #print("Domain features: ", "\n", "******")
       #singleLangRegressionWithoutVectorizer(langdomain,langlabels)

def main():

    itdirpath = "/home/bangaru/CrossLingualScoring/Datasets/IT-Parsed"
    dedirpath = "/home/bangaru/CrossLingualScoring/Datasets/DE-Parsed"
    czdirpath = "/home/bangaru/CrossLingualScoring/Datasets/CZ-Parsed"
    #do_single_lang_all_features(dedirpath,"de", "class")
    #do_cross_lang_all_features(dedirpath,"de","class", itdirpath, "it")
    #do_cross_lang_all_features(dedirpath,"de","class", czdirpath, "cz")
    do_mega_multilingual_model_all_features(dedirpath,"de",itdirpath,"it",czdirpath,"cz","class", "dep", True)

if __name__ == "__main__":
    main()

"""
TODO: Refactoring, reducing redundancy

"""

#print(getLexFeatures("/Users/sowmya/Research/CrossLing-Scoring/CrossLingualScoring/Datasets/DE-Parsed/1031_0003076_DE_C1.txt.parsed.txt", "de"))
#exit(1):
