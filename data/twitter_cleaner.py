# coding: utf-8

import os
import numpy as np
from collections import Counter
from nltk.tokenize import TweetTokenizer

base_dir = 'twitter'
data_file = os.path.join(base_dir, 'twitter_en.txt')
clean_file = os.path.join(base_dir, 'twitter_clean.txt')
vocab_file = os.path.join(base_dir, 'twitter_vocab.txt')

print("Tokenizing data...")
data = []
tknzr = TweetTokenizer()
for line in open(data_file, 'r', encoding='utf-8', errors='ignore'):
    data.append(' '.join(tknzr.tokenize(line.strip().lower())).split())  # special characters like ''

print("Original QA pairs:", len(data) // 2)
print("  Max length:", np.max(list(map(len, data))))
print("  Min length:", np.min(list(map(len, data))))
print("  Mean length:", np.mean(list(map(len, data))))
print("  Median length:", np.median(list(map(len, data))))

print("Removing data whose length > 20...")
max_len = 20
questions, answers = [], []
for i in range(0, len(data) - 1, 2):
    q, a = data[i], data[i + 1]
    if len(q) <= max_len and len(a) <= max_len:
        questions.append(q)
        answers.append(a)

print("Cleaned QA pairs:", len(questions))

vocab_size = 10000

vocabs = ['<pad>', '<sos>', '<eos>', '<unk>']
all_data = []
for line in questions + answers:
    all_data.extend(line)
counter = Counter(all_data)
count_pairs = counter.most_common(vocab_size - len(vocabs))

words, _ = list(zip(*count_pairs))
vocabs += words
word_to_id = dict(zip(vocabs, range(len(vocabs))))


def clean_str(string):
    return [x if x in word_to_id else '<unk>' for x in string]


new_qs, new_as = [], []
unk_cnt, total_cnt = 0, 0
for i in range(len(questions)):
    q, a = clean_str(questions[i]), clean_str(answers[i])
    new_qs.append(q)
    new_as.append(a)
    unk_cnt += (q + a).count('<unk>')
    total_cnt += len(q + a)

print("Percentage of unknown words: {:.3f}%".format(unk_cnt / total_cnt * 100))

print("Writing vocabulary to", vocab_file)
open(vocab_file, 'w', encoding='utf-8').write('\n'.join(vocabs) + '\n')

print("Writing cleaned data to", clean_file)
with open(clean_file, 'w', encoding='utf-8') as f:
    for i in range(len(new_qs)):
        f.write(' '.join(new_qs[i]) + ' ==> ' + ' '.join(new_as[i]) + '\n')

print("Testing the cleaned data...")
ques, ans = [], []
with open(clean_file, 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        try:
            q, a = line.strip().split(' ==> ')
            ques.append(q)
            ans.append(a)
        except:
            pass

for i in range(10):
    print(ques[i])
    print(ans[i])
    print()
