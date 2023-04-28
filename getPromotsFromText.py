import os
import random

exclude = ['العنوان', 'http', 'المحتويات'] # strings to exclude

# define function to get random words from a file
def get_random_words(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        words = []
        for line in f:
            for word in line.split():
                if word not in exclude and len(word) >= 3:
                    words.append(word)
        if len(words) >= 2:
            return random.sample(words, 2)
        else:
            return None

# define input and output directories
input_dir = '/texts'
output_file = '/Datasets/wordsTopromot.txt'

# get list of files in nested directories
file_list = []
for subdir, dirs, files in os.walk(input_dir):
    for file in files:
        file_list.append(os.path.join(subdir, file))

# get two random words from each file and write them to output file
with open(output_file, 'w', encoding='utf-8') as f_out:
    for file_path in file_list:
        words = get_random_words(file_path)
        if words:
            f_out.write(words[0] + '\n')
            f_out.write(words[1] + '\n')
          
# Print a message indicating that the program has finished running
print('Done!')
