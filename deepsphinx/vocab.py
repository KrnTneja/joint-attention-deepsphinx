'''Vocabulary for the dataset'''
import numpy as np

VOCAB = np.asarray(['q', 'ay', 'tcl', 't', 'uh', 'kcl', 'k', 'hh', 'er', 
	'w', 'f', 'ao', 'r', 'eh', 'b', 'ih', 'dx', 'sh', 'iy', 'l', 'gcl', 'g', 
	'ow', 'ng', 'th', 'y', 'ux', 'pcl', 's', 'ax', 'bcl', 'ey', 'axr', 'ah', 
	'dh', 'ix', 'ch', 'z', 'aw', 'n', 'hv', 'ae', 'dcl', 'jh', 'd', 'aa', 
	'epi', 'v', 'pau', 'p', 'm', 'oy', 'uw', 'nx', 'en', 'el', 'ax-h', 'em', 
	'zh', 'eng', '<s>', '</s>', 'h#']
)
VOCAB_TO_INT = {}

for ch in VOCAB:
    VOCAB_TO_INT[ch] = len(VOCAB_TO_INT)

# backoff is not in vocabulary
VOCAB_SIZE = len(VOCAB)


# INFLECTION TASK VOCAB
VOCAB_INF = np.asarray(['ß', '</s>', 'é', 'è', '<s>', 'æ', 'ä', 'ü', 'ö', 'a', 'c', 'b', 'e', 'd', 'g', 'f', 'i', 'h', 'k', 'j', 'm',
		 'l', 'o', 'n', 'q', 'p', 's', 'r', 'u', 't', 'w', 'v', 'y', 'x', 'z'])
VOCAB_TO_INT_INF = {}

for ch in VOCAB_INF:
    VOCAB_TO_INT_INF[ch] = len(VOCAB_TO_INT_INF)

# backoff is not in vocabulary
VOCAB_SIZE_INF = len(VOCAB_INF)

print("VOCAB_TO_INT_INF:", VOCAB_TO_INT_INF)