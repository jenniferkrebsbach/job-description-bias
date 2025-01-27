# Import necessary libraries
import os
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
import nltk

# Ensure stopwords are downloaded
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))


# Add custom words to exclude
custom_stop_words = {'role', 'work', 'responsible', 'ensuring', 'join', 'ensure', 'seeking', 
                    'maintaining', 'play', 'various', 'weather', 'also'}
stop_words.update(custom_stop_words)

# Configurable directory paths
data_dir = './data'
feminine_words_path = os.path.join(data_dir, 'female_words.txt')
masculine_words_path = os.path.join(data_dir, 'male_words.txt')

# Load word lists
def load_word_list(file_path):
    with open(file_path, 'r') as file:
        words = [line.strip().lower() for line in file if line.strip()]
    return words

feminine_words = load_word_list(feminine_words_path)
masculine_words = load_word_list(masculine_words_path)

# Job description files
files_to_analyze = {
    'female_coded': 'combined_woman.txt',
    'male_coded': 'combined_man.txt',
    'gender_neutral': 'combined_other.txt'
}

# Function to read and process the content of a file (removing stopwords)
def read_file_and_clean(file_path):
    with open(file_path, 'r') as file:
        words = file.read().lower().split()
        cleaned_words = [word for word in words if word.isalpha() and word not in stop_words]
    return cleaned_words


# Initialize a dictionary to store cleaned text data for each category
text_data = {category: read_file_and_clean(os.path.join(data_dir, filename)) for category, filename in files_to_analyze.items()}

# Generate word frequencies for each category
word_frequencies = {category: Counter(text_data[category]) for category in text_data}


# Define a function to create a word cloud from word frequencies
def create_wordcloud(frequencies, title):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(frequencies)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(title, fontsize=18)
    plt.axis('off')
    plt.show()
    plt.savefig('wordcloud_analysis_frequencies_'+ title + '.png')

# Generate and display word clouds for each category
for category, frequencies in word_frequencies.items():
    create_wordcloud(frequencies, title=f"Word Cloud for {category.capitalize()} Narrative")
    
