"""
Purpose: Knowing error statistics in the data for DE and IT using LanguageTool
"""

import language_check
import os, collections, pprint

def error_stats(inputpath,lang):
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

    print("unique rules for", lang, "  ", str(len(rules.keys())))
    #pprint.pprint(rules)
    pprint.pprint(locqualityissuetypes)
    pprint.pprint(categories)


inputpath_de = "/Users/sowmya/Research/CrossLing-Scoring/CrossLingualScoring/Datasets/DE/"
inputpath_it = "/Users/sowmya/Research/CrossLing-Scoring/CrossLingualScoring/Datasets/IT/"

error_stats(inputpath_de, "de")
error_stats(inputpath_it, "it")


"""
Results for locqualityissuetype:
unique rules for de    137
{'duplication': 19,
 'misspelling': 21,
 'style': 185,
 'typographical': 609,
 'uncategorized': 4746,
 'whitespace': 1280}
unique rules for it    31
{'duplication': 21,
 'misspelling': 3157,
 'typographical': 169,
 'uncategorized': 3956,
 'whitespace': 2239}

Results for category:
{'Briefe und E-Mails': 162,
 'Grammatik': 1703,
 'Groß-/Kleinschreibung': 2410,
 'Leicht zu verwechselnde Wörter': 25,
 'Mögliche Tippfehler': 98,
 'Redundanz': 2,
 'Semantische Unstimmigkeiten': 4,
 'Sonstiges': 1673,
 'Stil, Umgangssprache': 11,
 'Typographie': 412,
 'Zeichensetzung': 324,
 'Zusammen-/Getrenntschreibung': 36}
 
{'Altre': 2275,
 'Grammatica - Articoli': 200,
 'Grammatica - Elisioni e troncamenti': 99,
 'Grammatica - Frase': 641,
 'Grammatica - Preposizioni': 2,
 'Grammatica - Punteggiatura': 32,
 'Grammatica - Verbi': 192,
 'Possibile errore di battitura': 3157,
 'Stile - Espressioni': 5,
 'Stile - Frase': 130,
 'Stile - Leggibilità': 1396,
 'Stile - Numeri': 777,
 'Ulteriori errori comuni - ortografia': 57,
 "Ulteriori errori comuni - voci del verbo 'avere'": 425,
 'Uso delle maiuscole': 154}


There are 137 unique rules for German, and 31 for Italian
"""