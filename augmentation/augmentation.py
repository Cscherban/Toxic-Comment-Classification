import random
from PyDictionary import PyDictionary


class Augmentation:

    def __init__(self, hyperparams={}):
        self.dictionary = PyDictionary()
        self.hyperparams = hyperparams

    def augment_line(self, line):
        return line

    def run_augmentation(self, reader_writer, outfile):
        reader_writer.apply(self.augment_line)
        reader_writer.flush(outfile)


class NullAugmentation(Augmentation):

    def augment_line(self, line):
        return ""


class UniqueWordsAugmentation(Augmentation):

    def augment_line(self, line):

        return ""


class MaskWords(Augmentation):

    def augment_line(self, line):
        new_line = []
        for word in line.split():
            threshold = self.hyperparams.get('threshold', .2)
            if random.random() > threshold:
                new_line.append(word)

        return " ".join(new_line)


class SynonymWords(Augmentation):

    def augment_line(self, line):
        new_line = []
        for word in line.split():
            threshold = self.hyperparams.get('threshold', .2)
            if random.random() < threshold:
                synonyms = self.dictionary.synonym(word)
                if synonyms is not None and len(synonyms):
                    new_line.append(synonyms[0])
                else:
                    new_line.append(word)
            else:
                new_line.append(word)

        return " ".join(new_line)

