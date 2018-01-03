#!/usr/bin/python
# -*- coding: utf-8 -*-

import re
import os
from collections import Counter

base_dir = 'data/cornell_movie'
lines_dir = os.path.join(base_dir, 'movie_lines.txt')
convs_dir = os.path.join(base_dir, 'movie_conversations.txt')


def clean_text(text):
    """Clean text by removing unnecessary characters and altering the format of words."""

    text = text.lower()

    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"it's", "it is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"what's", "that is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"how's", "how is", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"n'", "ng", text)
    text = re.sub(r"'bout", "about", text)
    text = re.sub(r"'til", "until", text)
    text = re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,]", "", text)

    return text


# Create a dictionary to map each line's id with its text
# Clean the data
id2line = {}
lines = open(lines_dir, encoding='utf-8', errors='ignore').read().strip().split('\n')
for line in lines:
    _line = line.split(' +++$+++ ')
    if len(_line) == 5:
        id2line[_line[0]] = clean_text(_line[4]).split()

print("Total lines:", len(id2line))

# Create a list of all of the conversations' lines' ids.
convs = []
conv_lines = open(convs_dir, encoding='utf-8', errors='ignore').read().strip().split('\n')
for line in conv_lines:
    convs.append(line.split(' +++$+++ ')[-1][1:-1].replace("'", "").replace(" ", "").split(","))

print("Total conversations:", len(convs))

# Remove questions and answers that are shorter than 2 words and longer than 20 words.
min_length = 2
max_length = 20

clean_questions = []
clean_answers = []
total_cnt = 0
for conv in convs:
    for i in range(len(conv) - 1):
        cur_q = id2line[conv[i]]
        cur_a = id2line[conv[i + 1]]
        total_cnt += 1
        if min_length <= len(cur_q) <= max_length and min_length <= len(cur_a) <= max_length:
            clean_questions.append(cur_q)
            clean_answers.append(cur_a)

print("Cleaned Q-A pairs:", len(clean_questions))

# Create a vocabulary with most frequent words
vocabs = ['<SOS>', '<EOS>', '<UNK>']
data = []
for line in clean_questions + clean_answers:
    data.extend(line)
counter = Counter(data)
count_pairs = counter.most_common(8000 - len(vocabs))
words, _ = list(zip(*count_pairs))
vocabs += words

print("Vocabulary size:", len(vocabs))

# Save the vocabs into the vocab_dir
vocab_dir = os.path.join(base_dir, 'movie_vocabs.txt')
open(vocab_dir, 'w', encoding='utf-8').write('\n'.join(vocabs) + '\n')

print("Vocabulary saved.")

# Save the preprocessed Q-A pairs
clean_dir = os.path.join(base_dir, 'movie_clean.txt')
with open(clean_dir, 'w', encoding='utf-8') as f:
    for i in range(len(clean_questions)):
        f.write(' '.join(clean_questions[i]) + ' +++$+++ ' + ' '.join(clean_answers[i]) + '\n')

print("Clean corpus saved.")
