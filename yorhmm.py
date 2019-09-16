# HMM file that computes tge transition and emission probabailities
# for a bigram HMM. Uses Witten Bell Smoothing

# import libraries
from nltk import ngrams, FreqDist, WittenBellProbDist

# variable declarations
smoothed_emission_prob = {}
smoothed_transition_prob = {}
word_and_tag = []
taglist = []
temp_tags = []

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


# Function that computes the emission probabilities for the tags and words
def emission_probabilities():
    tag_s = set([t for (_, t) in word_and_tag])
    for tg in tag_s:
        words = [w for (w, t) in word_and_tag if t == tg]   #convert all words for better accuracy
        smoothed_emission_prob[tg] = WittenBellProbDist(FreqDist(words), bins=1e5)


# Function that computes the transition probabilities for the tags/states
def transition_probabilities():
    '''
    Function that computes the transition probabilities for the tags/states

    returns tag set

    '''
    bigrams = []
    for tg in taglist:
        bigrams += ngrams(tg, 2)
    tag_s = set([t for (t, _) in bigrams])

    for tg in tag_s:
        current_tag = [ct for (t, ct) in bigrams if t == tg]
        smoothed_transition_prob[tg] = WittenBellProbDist(FreqDist(current_tag), bins=1e5)
    return tag_s


tag_set = transition_probabilities()
emission_probabilities()
transition_probabilities()