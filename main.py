from augmentation.augmentation import NullAugmentation, MaskWords, SynonymWords
from augmentation.io import CSVReadWriter

in_dir = '/home/cscherban/Documents/School/DLTProj/jigsaw-multilingual-toxic-comment-classification/'
out_dir = '/home/cscherban/Documents/School/DLTProj/jigsaw-multilingual-toxic-comment-classification/augmented/'

in_name = 'jigsaw-toxic-comment-train.csv'
out_name_null = 'jigsaw-toxic-null-train.csv'
out_name_syn = 'jigsaw-toxic-synonym-train.csv'
out_name_mask = 'jigsaw-toxic-mask-train.csv'

read_writer = CSVReadWriter(in_dir + in_name)
null_augmentation = NullAugmentation()
null_augmentation.run_augmentation(read_writer, outfile=out_dir + out_name_null)
MaskWords().run_augmentation(read_writer, outfile=out_dir + out_name_mask)
SynonymWords().run_augmentation(read_writer, outfile=out_dir + out_name_syn)