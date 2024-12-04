import pickle
import random
import re
from collections import defaultdict, Counter


# This class is modified from https://towardsdatascience.com/text-generation-using-n-gram-model-8d12d9802aa0
class NgramModel:
    START_TOKEN = "<s>"

    def __init__(self, n, documents=None, context=None, ngram_counter=None, init=True):
        self.n = n
        self.documents = documents
        self.context = context or defaultdict(list)
        self.ngram_counter = ngram_counter or defaultdict(int)
        if init:
            self.init_model()

    def init_model(self):
        for document in self.documents:
            self.update(document)

    def load_pickle(self, feature, class_name):
        with open(f"../dump/{feature}_{class_name}_context.pkl", "rb") as file:
            self.context = pickle.load(file)
        with open(f"../dump/{feature}_{class_name}_ngram_counter.pkl", "rb") as file:
            self.ngram_counter = pickle.load(file)

    @staticmethod
    def tokenize(text: str):
        # print(text)
        for punct in "!?,.~$":
            text = text.replace(punct, f" {punct} ")
        return [item for item in re.split(r'[ /"@#%^&*()_`:;{}\[\]+-]', text) if item]

    def get_ngrams(self, tokens: list) -> list:
        tokens = (self.n - 1) * [self.START_TOKEN] + tokens  # padding
        return [
            (tuple([tokens[i - p - 1] for p in range(self.n - 2, -1, -1)]), tokens[i])
            for i in range(self.n - 1, len(tokens))
        ]

    def update(self, sentence: str):
        for ngram in self.get_ngrams(self.tokenize(sentence)):
            self.ngram_counter[ngram] += 1
            prev_words, target_word = ngram
            self.context[prev_words].append(target_word)
        return self

    def prob(self, context, token):
        return self.ngram_counter[(context, token)] / len(self.context[context])

    def random_token(self, context):
        r = random.random()
        map_to_probs = {
            token: self.prob(context, token) for token in self.context[context]
        }

        prob_sum = 0
        for token in sorted(map_to_probs):
            prob_sum += map_to_probs[token]
            if prob_sum > r:
                return token

    def generate_text(self, token_count: int):
        print("start generation")
        n = self.n
        prev_context = (n - 1) * [self.START_TOKEN]
        result = []
        for _ in range(token_count):
            obj = self.random_token(tuple(prev_context))
            result.append(obj)
            if n > 1:
                prev_context.pop(0)
                if obj == ".":  # new sentence
                    prev_context = (n - 1) * [self.START_TOKEN]
                else:
                    prev_context.append(obj)
        print("generation done")
        return " ".join(result)

    def to_pickle(self, feature, class_name: str):
        with open(f"../dump/{feature}_{class_name}_context.pkl", "wb") as file:
            pickle.dump(self.context, file, protocol=pickle.HIGHEST_PROTOCOL)
        with open(f"../dump/{feature}_{class_name}_ngram_counter.pkl", "wb") as file:
            pickle.dump(self.ngram_counter, file, protocol=pickle.HIGHEST_PROTOCOL)


class UserModel(NgramModel):

    def __init__(self, base_models={}, ratios={}, n=3):
        super().__init__(n, init=False)
        self.load_pickle(base_models, ratios)

    def load_pickle(self, base_models, ratios):
        for feature, classes in base_models.items():
            ratio = ratios[feature]
            for class_name in classes:
                with open(f"../dump/{feature}_{class_name}_context.pkl", "rb") as file:
                    data: dict = pickle.load(file)
                    for prev, now in data.items():
                        self.context[prev].extend(now * ratio)
                with open(
                    f"../dump/{feature}_{class_name}_ngram_counter.pkl", "rb"
                ) as file:
                    data = pickle.load(file)
                    self.ngram_counter = sum(
                        (
                            Counter(item)
                            for item in [
                                self.ngram_counter,
                                Counter(
                                    {key: value * ratio for key, value in data.items()}
                                ),
                            ]
                        ),
                        Counter(),
                    )

    def update(self, sentence: str, ratio: int = 10000):
        print("start updating....")
        sentences = sentence.split("\n")
        for sentence in sentences:
            for ngram in self.get_ngrams(self.tokenize(sentence)):
                self.ngram_counter[ngram] += ratio
                prev_words, target_word = ngram
                self.context[prev_words].extend([target_word] * ratio)
        print("updated.")
        return self
