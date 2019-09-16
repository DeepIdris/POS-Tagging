# Computes the accuracy of tags by comparing two files going through
# tag sequences in each
import sys

infile1 = open('generated_tags.txt', 'r')
generated_tags = infile1.readlines()

infile2 = open('expected_tags.txt', 'r')
expected_tags = infile2.readlines()

correct_tags = 0
incorrect_tags = 0

for gen_sent, cor_sent in zip(generated_tags, expected_tags):
    gen_tag = gen_sent.split()
    cor_tag = cor_sent.split()

    for g, c in zip(gen_tag, cor_tag):
        if g == c:
            correct_tags += 1
        else:
            incorrect_tags += 1

accuracy = 100 * float(correct_tags) / (correct_tags + incorrect_tags)
print("correct", correct_tags)
print("incorrect", incorrect_tags)
print("accuracy = ", accuracy)