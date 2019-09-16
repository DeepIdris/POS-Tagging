# Baseline model that computes the probability of
# of a word given a tag, that is based on its most
# frequent tag 

# import libraries
from nltk import FreqDist

# variable declarations
emission_prob = {}
word_and_tag = []
taglist = []
temp_tags = []
words_list = []
words = []
tagger = []

# read in from file and sort
in_file = open('yo_ytb-ud-train.txt', "r", encoding="utf8")
model = in_file.readlines()

for line in model:
    if line.startswith('#'):
        continue
    if line == "\n":
        words_list.append(words)
        temp_tags.append('</s>')
        taglist.append(temp_tags)
        temp_tags = []
        continue
    position, word, word_rep, tag, a, b, c, d, e, f = line.split()
    if position == "1":
        temp_tags.append('<s>')
    word_and_tag.append((word,tag))
    words.append(word)
    temp_tags.append(tag)


# Function for computing the probability of a tag given an observation (emission probability)
def emission_probabilities():
    '''
    This function computes the probability of a tag given an observation (emission probability)
    returns the tag set and word set

    '''
    t_set = set([t for (_, t) in word_and_tag])
    w_set = set([w for (w, _) in word_and_tag])
    for tg in t_set:
        words_n = [w for (w, t) in word_and_tag if t == tg]
        emission_prob[tg] = FreqDist(words_n)

    return t_set, w_set


tag_set, word_set = emission_probabilities()
emission_probabilities()


