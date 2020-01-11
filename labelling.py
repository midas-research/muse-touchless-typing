


char_to_cluster = {
	'a': 4,
	'b': 9,
	'c': 7,
	'd': 4,
	'e': 1, 
	'f': 5,
	'g': 5,
	'h': 5,
	'i': 3,
	'j': 6,
	'k': 6,
	'l': 6,
	'm': 9,
	'n': 9,
	'o': 3,
	'p': 3,
	'q': 1,
	'r': 2,
	's': 4,
	't': 2,
	'u': 2,
	'v': 7,
	'w': 1,
	'x': 7,
	'y': 2,
	'z': 7,
	' ': 8
}

samples = ['locate','single','family','would','place','large','work','take','live','box','method','listen','house','learn','come','some','ice','old','fly','leg','i never gave up','best time to live','catch the trade winds','hear a voice within you','he will forget it','hello','excuse me','i am sorry','thank you','good bye','see you','nice to meet you','you are welcome','how are you','have a good time']

print (samples)
print (len(samples))


def get_labelling():
	word_to_cluster = dict()
	for sample in samples:
		word_to_cluster[sample] = [char_to_cluster[s] for s in sample]
	return word_to_cluster


print (get_labelling())