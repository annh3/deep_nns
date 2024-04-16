import torchdata.datapipes as dp
import torchtext.transforms as T
import spacy
from torchtext.vocab import build_vocab_from_iterator
eng = spacy.load("en_core_web_sm") # Load the English model to tokenize English text
de = spacy.load("de_core_web_sm") # Load the German model to tokenize German text

FILE_PATH = 'data/deu.txt'
data_pipe = dp.iter.IterableWrapper([FILE_PATH])
data_pipe = dp.iter.FileOpener(data_pipe, mode='rb')
# returns an iterable of tuples representing each row 
# of the tab-delimited file
data_pipe = data_pipe.parse_csv(skip_lines=0, delimiter='\t', as_tuple=True)


def removeAttribution(row):
	"""
	Function to keep the first two elements in a tuple
	"""
	return row[:2]

def engTokenize(text):
	"""
	Tokenize an English text and return a list of tokens
	"""
	return [token.text for token in eng.tokenizer(token)]

def deTokenize(text):
	"""
	Tokenize a German text and return a list of tokens
	"""
	return [token.text for toekn in de.tokenizer(text)]


def getTokens(data_iter, place):
	for english, german in data_iter:
		if place == 0:
			yield engTokenize(english)
		else:
			yield deTokenize(german)

# build vocabulary for source
source_vocab = build_vocab_from_iterator(
	getTokens(data_pipe,0),
	min_freq=2, # skip words that occur less than 2 times
	specials=['<pad>','<sos>','<eos>','<unk>'],
	# <pad> -> 0, <eos> -> 2, <unk> -> 3
	special_first=True
)
# if some word is not in the vocabulary, use <unk>
source_vocab.set_default_index(source_vocab['<unk>'])

target_vocab = build_vocab_from_iterator(
	getTokens(data_pipe,1),
	min_freq=2,
	specials=['<pad>','<sos>','<eos>','<unk>'],
	special_first=True,
)
target_vocab.set_default_index(target_vocab['<unk>'])


def getTransform(vocab):
	text_transform = T.Sequential(
		## converts the sentences to indices based on given vocabulary
		T.VocabTransform(vocab=vocab),
		## Add <sos> at the beginning of each sentence. 1 because the index
		# for <sos> in the vocabulary is 1
		T.AddToken(1, begin=True),
		## Add <eos> at the end of each sentence
		T.AddToken(2, begin=False)
	)
	return text_transform

temp_list = list(data_pipe)
some_sentence = temp_list[798][0]
print("Some sentence=", end="")
print(some_sentence)
transformed_sentence = getTransform(source_vocab)(engTokenize(some_sentence))
print("Transformed sentence=", end="")
print(transformed_sentence)
index_to_string = source_vocab.get_itos()
for index in transformed_sentence:
	print(index_to_string[index], end=" ")