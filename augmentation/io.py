import pandas as pd


class CSVReadWriter:

    def __init__(self, file):
        self.file = file
        self.data =  pd.read_csv(file)

        self.dataOut = self.data.copy()
        self.dataCol = "comment_text"
        # read file into csv

    def read(self):
        for line in self.data:
            yield line

    def record(self, data):
        self.dataOut.append(data)

    def apply(self, func):
        self.dataOut[self.dataCol] = self.data[self.dataCol].apply(func)

    def flush(self, file):
        self.dataOut.to_csv(file)
        self.reset()

    def reset(self):
        self.dataOut = self.data.copy()