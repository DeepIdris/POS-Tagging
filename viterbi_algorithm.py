# Program that implements the viterbi Algorithm
# Using transition and emmission probabilities
# from the hmm
from yorhmm import tag_set, smoothed_transition_prob, smoothed_emission_prob
from nltk.metrics import ConfusionMatrix
from collections import Counter
import sys

# Tagset containing all the states of the corpus
tag_set -= {'<s>'}

predictions = []


#  Function to compute and print the classification metrics
def classification_metrics(gold, predict):
    '''
    This function compute and print the classification metrics
    precision, recall and f1 score for different classes using
    the confusion matrix provided by NLTK.
    
    gold- the gold standard/actual values 
    predict- the predicted values

    ''' 
    cm = ConfusionMatrix(gold, predict)

    model_file = open('classification_metrics.txt', 'w')
    sys.stdout = model_file

    print("\t\t---------------------------------------")
    print("\t\t\t\tConfusion Matrix 1")
    print("\t\t---------------------------------------\n")
    print(cm)

    print("\t\t---------------------------------------")
    print("\t\t\t\tConfusion Matrix 2")
    print("\t\t----------------------------------------\n")
    print(cm.pretty_format(sort_by_count=True, show_percents=True, truncate=10))

    labels = tag_set

    true_positives = Counter()
    false_negatives = Counter()
    false_positives = Counter()

    for i in labels:
        for j in labels:
            if i == j:
                true_positives[i] += cm[i, j]
            else:
                false_negatives[i] += cm[i, j]
                false_positives[j] += cm[i, j]

    print("\t---------------------------------------")
    print("\tPrecision Recall F-score")
    print("\t---------------------------------------\n")

    for i in sorted(labels):
        if true_positives[i] == 0:
            fscore = 0
        else:
            precision = true_positives[i] / float(true_positives[i] + false_positives[i])
            recall = true_positives[i] / float(true_positives[i] + false_negatives[i])
            fscore = 2 * (precision * recall) / float(precision + recall)

        print(i, "\t", "%.2f" % precision, "\t", "%.2f" % recall, "\t", "%.2f" % fscore)


# Function that implements the viterbi algorithm
def viterbi():

    # File to to write the correct/expected tags
    # and generated tags from a corpus
    model_file2 = open('expected_tags.txt', 'w')
    model_file = open('generated_tags.txt', 'w')

    in_file = open('yo_ytb-ud-te.txt', "r", encoding="utf8")
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
            viterbi_prob[0][states] = smoothed_transition_prob['<s>'].prob(states) * smoothed_emission_prob[states].prob(obs[0])
            backpointer[states] = [states]

        # Loop over i from 1 to n
        for i in range(1, len(obs)):
            viterbi_prob.append({})
            newpath = {}

            # for all states
            for states in tag_set:

                # Compute maximum probability of going to each state
                # from previous states
                (prob, state) = max((viterbi_prob[i - 1][prev_state] * smoothed_transition_prob[prev_state].prob(states)
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