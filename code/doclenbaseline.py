#Adding a document length baseline for final version.
import os
import collections
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score,cross_val_predict,StratifiedKFold 
from sklearn.metrics import f1_score,classification_report,accuracy_score,confusion_matrix, mean_absolute_error
from sklearn.svm import LinearSVC

seed = 1234

def getdoclen(conllufilepath):
    fh =  open(conllufilepath, encoding="utf-8")
    allText = []
    sent_id = 0
    for line in fh:
        if line == "\n":
            sent_id = sent_id+1
        elif not line.startswith("#") and line.split("\t")[3] != "PUNCT":
            word = line.split("\t")[1]
            allText.append(word)
    fh.close()
    return len(allText)

def getfeatures(dirpath):
    files = os.listdir(dirpath)
    cats = []
    doclenfeaturelist = []
    for filename in files:
        if filename.endswith(".txt"):
            doclenfeaturelist.append([getdoclen(os.path.join(dirpath,filename))])
            cats.append(filename.split(".txt")[0].split("_")[-1])
    return doclenfeaturelist,cats

def singleLangClassificationWithoutVectorizer(train_vector,train_labels): #test_vector,test_labels):
    k_fold = StratifiedKFold(10,random_state=seed)
    classifiers = [RandomForestClassifier(class_weight="balanced",n_estimators=300,random_state=seed), LinearSVC(class_weight="balanced",random_state=seed), LogisticRegression(class_weight="balanced",random_state=seed)] #Add more later
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

def crossLangClassificationWithoutVectorizer(train_vector, train_labels, test_vector, test_labels):
    classifiers = [RandomForestClassifier(class_weight="balanced",n_estimators=300,random_state=seed), LinearSVC(class_weight="balanced",random_state=seed), LogisticRegression(class_weight="balanced",random_state=seed)]
    for classifier in classifiers:
        classifier.fit(train_vector,train_labels)
        predicted = classifier.predict(test_vector)
        print(np.mean(predicted == test_labels,dtype=float))
        print(confusion_matrix(test_labels,predicted))
        print(f1_score(test_labels,predicted,average='weighted'))


def main():
    itdirpath = "../Datasets/IT-Parsed"
    dedirpath = "../Datasets//DE-Parsed"
    czdirpath = "../Datasets/CZ-Parsed"
    print("************DE baseline:****************")
    defeats,delabels = getfeatures(dedirpath)
    singleLangClassificationWithoutVectorizer(defeats,delabels)
    print("************IT baseline:****************")
    itfeats,itlabels = getfeatures(itdirpath)
    singleLangClassificationWithoutVectorizer(itfeats,itlabels)
    print("************CZ baseline:****************")
    czfeats,czlabels = getfeatures(czdirpath)
    singleLangClassificationWithoutVectorizer(czfeats,czlabels)

    print("*** Train with DE, test with IT baseline******")
    crossLangClassificationWithoutVectorizer(defeats,delabels, itfeats,itlabels)
     
    print("*** Train with DE, test with CZ baseline ******")
    crossLangClassificationWithoutVectorizer(defeats,delabels, czfeats,czlabels)
    
    bigfeats = []
    bigcats = []
    bigfeats.extend(defeats)
    bigfeats.extend(itfeats)
    bigfeats.extend(czfeats)
    bigcats.extend(delabels)
    bigcats.extend(itlabels)
    bigcats.extend(czlabels)
    print("****Multilingual classification baseline*************")
    singleLangClassificationWithoutVectorizer(bigfeats,bigcats)

    
    
main()

