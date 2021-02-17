#-------------------------------------------------------
# Compares text file content to Harvard Sentences list 1
#-------------------------------------------------------

file1 = input("Enter a file to be compared: ")
sentence1 = 'harvard_sentences1.txt'

with open(sentence1) as f:
    correct_lines = f.readlines()

with open(file1) as f:
    lines = f.readlines()

correct_lines = [x.strip() for x in correct_lines]
lines = [x.strip() for x in lines]


s = 0       # Number of substitutions / incorrect words
d = 0       # Number of deletions / missing words
i = 0       # Number of insertions / extra words
c = 0       # Number of correct words

for i in range(len(correct_lines)):
    words = lines[i].split()
    correct_words = correct_lines[i].split()

    # See if there are deletions or insertions
    if len(words) < len(correct_words):
        d = d + (len(correct_words) - len(words))
    elif len(words) > len(correct_words):
        i = i + (len(words) - len(correct_words))

    for word in words:
        if word in correct_words:
            c = c + 1
        else:
            s = s + 1

wer = (s+d+i)/(s+d+c) 

print('Substitutions:', s)
print('Deletions:', d)
print('Insertions:', i)
print('Correct words:', c)
print('WER:', wer)
   

