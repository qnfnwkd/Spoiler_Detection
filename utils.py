import random

def load_data(train_file_path, test_file_path):
	"""
		Load the dataset
	"""
	train = [line.strip() for line in open(train_file_path).readlines()]
	test = [line.strip() for line in open(test_file_path).readlines()]
	return train, test

def build_dic(train, test, word_list):
	"""
		We constructed the word_list, in advance, because we do not need the 
		word embedding vector of words the dataset does not include.
	"""
	id2word = [line.strip() for line in open(word_list).readlines()]
	word2id = {id2word[i]:i for i in xrange(len(id2word))}

	genre2id = {"<PAD>":0}
	id2genre = ["<PAD>"]

	"""
		Data format:
		<word_1 word_2 ...> \t <label> \t <genre_1 genre_2 ...>
	"""
	for line in train+test:
		text, spoiler, genre = line.split("\t")
		for _genre in genre.split():
			if _genre not in genre2id:
				genre2id[_genre] = len(genre2id)
				id2genre.append(_genre)

	return word2id, id2word, genre2id, id2genre

def converting(_train, _test, word2id, genre2id):

	# For zero padding, calculates the maximum lengths
	maximum_document_length = max([len(line.split("\t")[0].split()) for line in _train+_test])
	maximum_genre_length = max([len(line.split("\t")[2].split()) for line in _train+_test])

	train, test = [], []
	for line in _train:
		words, spoiler, genre = line.split("\t")
		spoiler = int(spoiler)
		words = [word2id[word] for word in words.split()]
		words = words + [0]*(maximum_document_length-len(words))
		genre = [genre2id[_genre] for _genre in genre.split()]
		genre = genre + [0]*(maximum_genre_length-len(genre))
		train.append((spoiler, words, genre))

	for line in _test:
		words, spoiler, genre = line.split("\t")
		spoiler = int(spoiler)
		words = [word2id[word] for word in words.split()]
		words = words + [0]*(maximum_document_length-len(words))
		genre = [genre2id[_genre] for _genre in genre.split()]
		genre = genre + [0]*(maximum_genre_length-len(genre))
		test.append((spoiler, words, genre))

	return train, test, maximum_document_length, maximum_genre_length

def batch_iter(data, batch_size):
	batch_num = int(len(data)/batch_size) + 1
	random.shuffle(data)
	for i in xrange(batch_num):
		left = i*batch_size
		right = min(len(data), (i+1)*batch_size)
		spoiler, content, genre = [], [], []
		for line in data[left:right]:
			spoiler.append(line[0])
			content.append(line[1])
			genre.append(line[2])
		yield content, genre, spoiler