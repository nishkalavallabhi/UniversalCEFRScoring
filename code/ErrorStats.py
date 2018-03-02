"""
Purpose: Knowing error statistics in the data for DE and IT using LanguageTool
"""

import language_check
import os, collections, pprint

def write_featurelist(file_path,some_list):
    fh = open(file_path, "w")
    for item in some_list:
      fh.write(item)
      fh.write("\n")
    fh.close()

def error_stats(inputpath,lang,output_path):
    files = os.listdir(inputpath)
    checker = language_check.LanguageTool(lang)
    rules = {}
    locqualityissuetypes = {}
    categories = {}

    for file in files:
        if file.endswith(".txt"):
            text = open(os.path.join(inputpath,file)).read()
            matches = checker.check(text)
            for match in matches:
                rule = match.ruleId
                loc = match.locqualityissuetype
                cat = match.category
                rules[rule] = rules.get(rule,0) +1
                locqualityissuetypes[loc] = locqualityissuetypes.get(loc,0) +1
                categories[cat] = categories.get(cat,0)+1

    write_featurelist(output_path+lang+"-rules.txt", sorted(rules.keys()))
    write_featurelist(output_path+lang+"-locquality.txt", sorted(locqualityissuetypes.keys()))
    write_featurelist(output_path+lang+"-errorcats.txt", sorted(categories.keys()))

inputpath_de = "/home/bangaru/GitProjects/CrossLingualScoring/Datasets/DE/"
inputpath_it = "/home/bangaru/GitProjects/CrossLingualScoring/Datasets/IT/"

error_stats(inputpath_de, "de", "../features/")
error_stats(inputpath_it, "it","../features/")


