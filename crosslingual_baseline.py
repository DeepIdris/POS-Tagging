# Baseline model that computes the probability of
# of a word given a tag, that is based on its most
# frequent tag

# import libraries
from nltk import FreqDist

# variable declarations
transition_prob = {}
word_and_tag = []
taglist = []
temp_tags = []
words_list = []
tagger = []

# read in from file and sort
in_file = open('train.txt', "r", encoding="utf8")
model = in_file.readlines()

for line in model:
    if line.startswith('#'):
        continue
    if line == "\n":
        temp_tags.append('</s>')
        taglist.append(temp_tags)
        temp_tags = []
        continue
    position, word, word_rep, tag, a, b, c, d, e, f = line.split()
    if position == "1":
        temp_tags.append('<s>')
    word_and_tag.append((word,tag))
    temp_tags.append(tag)


# Function that computes the naive transition probabilities for the tags/states
def naive_transition_probabilities():
    '''
    Function that computes naive transition probabilities for the tags/states
    using a uniform probability.

    returns tag set

    '''
    tag_s = set([t for (_, t) in word_and_tag])
    tag_s.add('<s>')
    for tg in tag_s:
        transition_prob[tg] = FreqDist(tag_s)

    return tag_s


tag_set = naive_transition_probabilities()



