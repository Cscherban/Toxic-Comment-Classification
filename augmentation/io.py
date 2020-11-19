import pandas as pd
import time
import numpy as np

class CSVReadWriter:

    def __init__(self, comments_loc, labels_loc):
        comments = pd.read_csv(comments_loc)
        labels = pd.read_csv(labels_loc)

        self.data = pd.DataFrame({'comment': comments['0'], 'label': labels['0'] })
        self.dataOut = self.data.copy()
        self.dataCol = 'comment'
        self.labelCol = 'label'
        # read file into csv

    def read(self):
        for line in self.data:
            yield line

    def record(self, data):
        self.dataOut.append(data)

    def _get_input_df(self, label, size=None, equal=False):
        labels = self.data[self.labelCol]
        input_df = self.data.copy()

        classes = labels.value_counts()
        maxidx = classes.idxmax()
        minidx = classes.idxmin()

        print(f"Value Counts: \t {classes.to_string()}")

        if label == "majority":
            input_df = input_df[input_df[self.labelCol] == maxidx]
        elif label == "minority":
            input_df = input_df[input_df[self.labelCol] == minidx]

        if equal and label != "all":
            size = classes[maxidx] - classes[minidx]

        if size:
            idx = np.arange(0, size) % len(input_df)
            input_df = input_df.iloc[idx]
        return input_df

    def apply(self, func, mode="append", label="all", size=None, equal=False):
        print(f"\n mode: {mode} \t label: {label} \t size: {size} \t equal: {equal}\n")
        start_time = time.time()
        print(f"Started at {start_time}")
        mod_df = self._get_input_df(label, size, equal)

        mod_df[self.dataCol] = mod_df[self.dataCol].apply(func)

        if mode == "append":
            copy_orig = self.data.copy()
            self.dataOut = copy_orig.append(mod_df, ignore_index=True)
        else:
            self.dataOut = mod_df

        end_time = time.time()
        print(f"Ended at {end_time}")
        total_time = end_time - start_time
        print(f"Elapsed time: {total_time} seconds; \t {total_time * 1000} millis")

    def flush(self, fcomment, flabel):
        print(f"Flushing to file. \n X-Train {fcomment} \n Y-train {flabel} \n Data-len: ({len(self.data)}) \n Out-len ({len(self.dataOut)})")
        comments = self.dataOut[self.dataCol]
        labels = self.dataOut[self.labelCol]
        comments.to_csv(fcomment, index=False, header=['0'])
        labels.to_csv(flabel, index=False, header=['0'])
        self.reset()

    def reset(self):
        self.dataOut = self.data.copy()