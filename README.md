# CrossLingualScoring
This idea of pursuing a general-language agnostic-CEFR level classifier

# Neural Networks:
1. Monolingual FastText: A character embedding based Fast Text model with CV evaluation. Word based embedding combine with character embeddings. A single model.
`python3 code/monolingual_cv.py Datasets/IT`
2. Monolingual LSTM: A character embedding with word based embedding with LSTM model.
3. Multilingual model: A char- and word embedding model with two types of targets. One to train a language feature and the other to train a predictor of difficulty level of the texts.
