import random
from PyDictionary import PyDictionary
import nltk
#nltk.download()
from nltk.corpus import wordnet
from nltk.corpus import stopwords
NLTK_STOPWORDS = set(stopwords.words("english"))

class Augmentation:

    def __init__(self, hyperparams={}):
        self.dictionary = PyDictionary()
        self.hyperparams = hyperparams

    def augment_line(self, line):
        return line

    def run_augmentation(self, reader_writer, comment_out, label_out, dataparams):
        reader_writer.apply(self.augment_line, size=dataparams.get("size"), mode=dataparams.get("mode", "append"),
                            label=dataparams.get("label", "all"), equal=dataparams.get("equal", False))

        reader_writer.flush(comment_out, label_out)


class NullAugmentation(Augmentation):

    def augment_line(self, line):
        return ""


class UniqueWordsAugmentation(Augmentation):

    def augment_line(self, line):
        new_line = []
        seen = set()
        for word in line.split():
            if word not in seen:
                seen.add(word)
                new_line.append(word)

        return " ".join(new_line)


class MaskWords(Augmentation):

    def augment_line(self, line):
        new_line = []
        for word in line.split():
            threshold = self.hyperparams.get('threshold', .2)
            if random.random() > threshold:
                new_line.append(word)

        return " ".join(new_line)

class RemoveStopWords(Augmentation):
    def augment_line(self, line):
        new_line = []
        for word in line.split():
            if word not in NLTK_STOPWORDS:
                new_line.append(word)
        return " ".join(new_line)

class SynonymWords(Augmentation):

    def augment_line(self, line):
        new_line = []
        for word in line.split():
            threshold = self.hyperparams.get('threshold', .2)
            if random.random() < threshold:
                synonyms = wordnet.synsets(word)
                if synonyms is not None and len(synonyms):
                    new_line.append(synonyms[0].lemmas()[0].name())
                else:
                    new_line.append(word)
            else:
                new_line.append(word)

        return " ".join(new_line)

