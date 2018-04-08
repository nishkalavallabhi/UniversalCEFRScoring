# About this repo: 
This paper contains the code and some of the data and result files for the experiments described in the following paper:

> Experiments with Universal CEFR Classification  
> Authors: Sowmya Vajjala and Taraka Rama  
> (to appear) In Proceedings of The 13th Workshop on Innovative Use of NLP for Building Educational Applications

For enquiries, contact one of the authors, at:
sowmya@iastate.edu, tarakark@ifi.uio.no

About this github repo's folders:  
  
- **Datasets/:**  
  * This contains folders with learner corpora for three languages(German-DE, Italian-IT, Czech-CZ), from [MERLIN project](http://www.merlin-platform.eu/), and their dependency parsed versions (DE-Parsed, IT-Parsed, CZ-Parsed) using [UDPipe](http://ufal.mff.cuni.cz/udpipe).
  * The [original MERLIN corpus](http://www.merlin-platform.eu/C_data.php) is available under a [Creative Commons Attribution-ShareAlike 4.0 International License](https://creativecommons.org/licenses/by-sa/4.0/), which allows us to: "copy and redistribute the material in any medium or format".
  * The original corpus files have been renamed for our purposes, and all metadata showing information about the learner background, different dimensions of proficiency etc have been removed. All .Metadata.txt files in this folder contain information about other CEFR proficiency dimensions for the learner writings, which we did not use in this paper.
  * RemovedFiles/ folder in this folder contains files that we did not use for our analysis, either because they are unrated, or because that proficiency level has too few examples (Read the paper for details).

- **code/:**
  * CreateDataset.py: takes the original MERLIN corpus and creates a new version by removing metadata inside files, and renaming them.
  * CreateMetaDataFile.py: takes the original MERLIN corpus files, and creates a file with filenames and information about
different proficiency dimensions (resulting files for all languages are seen in Datasets/)
  * ErrorStats.py: uses [LanguageTool](https://languagetool.org/) to extract spelling/grammar error information for DE and IT corpora.
  * ExtractErrorFeatures.py: extracts information about individual error rules from LanguageTool, and stores them as numpy arrays - this is not used in this paper.
  * IdeaPOC.py: contains the bulk of the code - basically, all experiments without neural networks are seen here. 
  * bulklangparse.sh: Script to parse all files for a given language using its UDPipe model.  
  * monolingual_cv.py: mono-lingual neural network training using keras, with tensorflow backend.
  * multi_lingual.py: multi-lingual, multi-task learning (learning the language, and learning its CEFR)
  * multi_lingual_no_langfeat.py: multi-lingual, without language identification.  
  (For more details, read the paper, and see the code!)

- **features/:**
  * This directory contains some analysis of the data using [LanguageTool](https://languagetool.org/) to understand the different error types in different languages, as identified by the tool. Only the files ending with errorcats.txt have been used in this paper.

- **results/:**
  * This folder contains some the result files generated while doing the experiments, primarily to keep a record. It is useful for replication and comparison purposes in future.

- **README.md**: This file  

- **bea-naacl2018-supplementarymaterial.zip**: Submitted/pre-peer-reviewed version of this folder for BEA. The current version primarily has more documentation, and no added code.

- **notesonfeatures.txt**: Just some initial notes made about what to implement.



