
import string
import nltk

class EnglishTextProcessor():

    def __init__(self):
        pass
    
    def process(self, texts):
        sentences = nltk.sent_tokenize(texts.lower())
        tokens = []
        for sentence in sentences:
            words = nltk.wordpunct_tokenize(sentence)
            words = [word for word in words if word not in string.punctuation]
            tokens.extend(words)
        return tokens

