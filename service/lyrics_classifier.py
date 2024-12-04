import math
import json
import re

import numpy as np
import pandas as pd
import nltk

# from base.enums import Genre

# Best top K features by feature
top_K_by_feature = {
    "family": 10000,
    "subfamily": 10000,
    "gender": 4000,
}

nltk.download("stopwords")


class Classifier:
    def __init__(self, feature, multi_label=False, top_K=None):
        self.top_K = top_K if top_K != None else top_K_by_feature.get(feature, 1000)
        self.multi_label = multi_label
        with open(f"../dump/class_p_{feature}_top_{top_K}.json", "rb") as file:
            p_file = json.load(file)
            self.word_p = p_file["p"]
            self.num_classes = len(self.word_p.keys())
            self.vocabulary = set(p_file["vocabulary"])
        self.stopwords = nltk.corpus.stopwords.words("english")

    @staticmethod
    def text_split(text):
        text = re.split("[^a-zA-Z]+", text)
        text = [x for x in text if x]
        return text

    def tokenize(self, text):
        # turn into lower case
        text = text.lower()
        # tokenize
        words = Classifier.text_split(text)

        # words =  [''.join(filter(str.isalnum, word)) for word in words]
        words = [word for word in words if word != ""]

        # PorterStemmer algorithm
        words = [nltk.stem.PorterStemmer().stem(word) for word in words]

        # remove stopwords
        words = [word for word in words if not word in self.stopwords]
        return words

    @staticmethod
    def normalize(result):
        total = sum(np.exp(result["max_p_class"]))
        result["normalized"] = np.exp(result["max_p_class"]) / total
        return result

    def result_class(self, result):
        if self.multi_label:
            candidate = result[result["normalized"] > (1 / self.num_classes)]
            return candidate.value.tolist()
        else:
            return result["value"][0]

    def classification(self, document):
        word_p = self.word_p
        document = self.tokenize(document)
        document = [x for x in document if x in self.vocabulary]

        score = pd.DataFrame({"max_p_class": [], "value": []})
        for c in word_p.keys():
            p_class = math.log(word_p[c]["prior"], 10)
            for word in document:
                p_class += math.log(word_p[c]["cond_prob"][word], 10)
            score = pd.concat(
                [score, pd.DataFrame({"max_p_class": -p_class, "value": c}, index=[0])],
                ignore_index=True,
            )
        result = score.sort_values(by="max_p_class", ignore_index=True, ascending=True)
        result = self.normalize(result=result)
        result = self.result_class(result=result)
        return result
