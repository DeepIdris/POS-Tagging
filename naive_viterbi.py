# Program that implements the viterbi Algorithm
# Using transition and emission probabilities
# from naive transitions
from yorhmm import tag_set, smoothed_emission_prob
from crosslingual_baseline import transition_prob
import sys
from nltk.metrics import ConfusionMatrix
from collections import Counter
from viterbi_algorithm import classification_metrics

# Tagset containing all the states of the corpus
tag_set -= {'<s>'}
predictions = []


# Function that implements the viterbi algorithm
def viterbi():

    # File to to write the correct/expected tags
    # and generated tags from a corpus
    model_file2 = open('expected_tags.txt', 'w')
    model_file = open('generated_tags.txt', 'w')

    in_file = open('test.txt', "r", encoding="utf8")
    test_sentences = in_file.readlines()

    # Store the words and their corresponding tags
    words_list = []
    tag_list = []
    words = []
    tags = []
    for sent in test_sentences:
        if sent.startswith('#'):
            continue
        if sent == "\n":
            words_list.append(words)
            tag_list.append(tags)
            words = []
            tags = []
            continue
        position, word, word_rep, tag, a, b, c, d, e, f = sent.split()
        words.append(word)
        tags.append(tag)

    # Print the correct/expected tags to a file
    sys.stdout = model_file2
    for tg in tag_list:
        print(tg)

    gold = [item for sublist in tag_list for item in sublist]

    for words in words_list:
        obs = [w for w in words]
        viterbi_prob = [{}]
        backpointer = {}

        # Initialize for all states
        for states in tag_set:
            viterbi_prob[0][states] = transition_prob['<s>'].freq(states) * smoothed_emission_prob[states].prob(obs[0])
            backpointer[states] = [states]

        # Loop over i from 1 to n
        for i in range(1, len(obs)):
            viterbi_prob.append({})
            newpath = {}

            # for all states
            for states in tag_set:

                # Compute maximum probability of going to each state
                # from previous states
                (prob, state) = max((viterbi_prob[i - 1][prev_state] * transition_prob['<s>'].freq(states)
                                   * smoothed_emission_prob[states].prob(obs[i]), prev_state) for prev_state in tag_set)

                viterbi_prob[i][states] = prob
                newpath[states] = backpointer[state] + [states]

            backpointer = newpath

            (prob, state) = max((viterbi_prob[i][st], st) for st in tag_set)

        #  Print the generated tags to a file
        sys.stdout = model_file
        print(backpointer[state])

        predictions.append(backpointer[state])

    predict = [item for sublist in predictions for item in sublist]

    classification_metrics(gold, predict)


viterbi()