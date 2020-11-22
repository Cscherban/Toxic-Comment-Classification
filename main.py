from augmentation.augmentation import NullAugmentation, MaskWords, SynonymWords, RemoveStopWords, \
    UniqueWordsAugmentation
from augmentation.io import CSVReadWriter
import os.path
from os import path

in_dir = '/home/cscherban/Documents/School/DLTProj/8803Data/stop_words/'
comment_file = "X_train.csv"
label_file = "y_train.csv"

read_writer = CSVReadWriter(in_dir + comment_file, in_dir + label_file)

jobs = [
    {
        "title": "Synonym Augmentation",
        "obj": SynonymWords,
        "equal": True,
        "label": "minority",
        "mode": "append",
        "outdir": in_dir + "synonym_augmentation_minority_50/",
        "hyperparams": {}
    },
    {
        "title": "Mask Words",
        "obj": MaskWords,
        "equal": True,
        "label": "minority",
        "mode": "append",
        "outdir": in_dir + "mask_augmentation_minority_50/",
        "hyperparams": {}
    },
    {
        "title": "Unique Words",
        "obj": UniqueWordsAugmentation,
        "equal": True,
        "label": "minority",
        "mode": "append",
        "outdir": in_dir + "unique_augmentation_minority_50/",
        "hyperparams": {}
    }
    # {
    #     "title": "Stop Words",
    #     "obj": RemoveStopWords,
    #     #"size": .7,
    #     "label": "all",
    #     "mode": "update",
    #     "outdir": in_dir + "stop_words/",
    #     "hyperparams": {}
    # },
]


def clear(dir, files):
    if not path.exists(dir):
        os.mkdir(dir)
    for file in files:
        if path.exists(file):
            os.remove(file)

def run_job(job):
    print("===== ===== ===== ===== ===== \n" * 4)
    print(f"Starting {job['title']}")
    aug = job["obj"](job['hyperparams'])
    outdir = job["outdir"]
    comments_out = outdir + comment_file
    labels_out = outdir + label_file
    clear(outdir, [comments_out, labels_out])
    aug.run_augmentation(read_writer, comments_out, labels_out, job)
    print(f"Finished job \n")

for job in jobs:
    run_job(job)
