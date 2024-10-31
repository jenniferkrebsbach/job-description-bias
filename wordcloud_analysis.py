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

# Paths to the text files for analysis
directory = '/Users/jenniferkrebsbach'
files_to_analyze = {
    'male_coded': 'combined_man.txt',
    'female_coded': 'combined_woman.txt',
    'gender_neutral': 'combined_other.txt'
}

# Function to read and process the content of a file (removing stopwords)
def read_file_and_clean(file_path):
    with open(file_path, 'r') as file:
        words = file.read().lower().split()
        cleaned_words = [word for word in words if word.isalpha() and word not in stop_words]
    return cleaned_words

# Initialize a dictionary to store cleaned text data for each category
text_data = {category: read_file_and_clean(os.path.join(directory, filename)) for category, filename in files_to_analyze.items()}

# Generate word frequencies for each category
word_frequencies = {category: Counter(text_data[category]) for category in text_data}

# Remove the top 3 words for each category
filtered_word_frequencies = {}
for category, frequencies in word_frequencies.items():
    # Get the top 3 words to exclude
    top_3_words = [word for word, _ in frequencies.most_common(3)]
    # Create a new frequency dictionary without the top 3 words
    filtered_frequencies = {word: count for word, count in frequencies.items() if word not in top_3_words}
    filtered_word_frequencies[category] = filtered_frequencies

# Define a function to create a word cloud from word frequencies
def create_wordcloud(frequencies, title):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(frequencies)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(title, fontsize=18)
    plt.axis('off')
    plt.show()

# Create word clouds for each category excluding the top 3 words
for category, frequencies in filtered_word_frequencies.items():
    title = f"Word Cloud for {category.replace('_', ' ').title()} Job Descriptions (Excluding Top 3 Words)"
    create_wordcloud(frequencies, title)
