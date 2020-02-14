import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


class lowo:

    def __init__(self, clf, df, lower):
        self.clf = clf
        self.tokenizer = Tokenizer(document_count=df, lower=lower)

    def preprocess_keras(self, texts, padding, maxlen):
        return pad_sequences(self.tokenizer.texts_to_sequences(texts),
                             padding=padding,
                             maxlen=maxlen)

    def ablate(self, text, sorted=False):
        x_in = self.preprocess([text])
        # hold the initial score of the whole text
        initial_score = self.clf.predict(x_in).flatten()[0]
        # ablate words: split to space, remove each word, merge the rest, and move on
        words = text.split()  # todo: this needs to respect the CLF's tokenizer
        texts, X_text = [], []
        for word_index in range(len(words)):
            words_copy = [w for w in words]
            words_copy.pop(word_index)
            text = " ".join(words_copy)
            texts.append(text)
            X_text.append(self.preprocess([text])[0])
        # score all ablated texts
        scores = self.clf.predict(np.array(X_text)).flatten()
        scores_pd = pd.DataFrame({"word": words, "score": scores})
        # measure the change in the score; decrease means guilty
        scores_pd.score = scores_pd.score.apply(lambda score: (initial_score - score) / initial_score)
        if sorted: scores_pd = scores_pd.sort_values(by=["score"])
        return initial_score, scores_pd

    def show(self, texts, figsize=(20, 9)):
        for text in texts:
            text_score, word_scores_pd = self.ablate(text)
            if text_score > 0.5:
                word_scores_pd.plot(x="word", y="score", style="--o", figsize=figsize)
            else:
                print ("WARNING: Negative class was found!")
