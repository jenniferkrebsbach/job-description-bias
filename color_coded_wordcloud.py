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

# Add custom stopwords
custom_stop_words = {
    'role', 'work', 'responsible', 'ensuring', 'join', 'ensure', 'seeking',
    'maintaining', 'play', 'various', 'weather', 'also'
}
stop_words.update(custom_stop_words)

# File paths
data_dir = './data'
feminine_words_path = os.path.join(data_dir, 'female_words.txt')
masculine_words_path = os.path.join(data_dir, 'male_words.txt')


# Read and clean job description file
def read_file_and_clean(file_path):
    with open(file_path, 'r') as file:
        words = file.read().lower().split()
        cleaned_words = [word for word in words if word.isalpha() and word not in stop_words]
    return cleaned_words

#Job description files to analyze
files_to_analyze = {
    'female_coded': 'combined_woman.txt',
    'male_coded': 'combined_man.txt',
    'gender_neutral': 'combined_other.txt'
}
# Prepare data
text_data = {
    category: read_file_and_clean(os.path.join(data_dir, filename))
    for category, filename in files_to_analyze.items()
}

# Word frequencies
word_frequencies = {
    category: Counter(text_data[category])
    for category in text_data
}

# Define your custom highlight words and colors here
highlight_words = {
    'strong': 'green',
    'communication' : 'purple',
    'community' : 'purple',
    'team' : 'purple',
    'individual' : 'green'
    }

# Custom color function
def get_color_func(highlight_dict):
    def color_func(word, *args, **kwargs):
        return highlight_dict.get(word, 'black')
    return color_func

# Create word cloud
def create_wordcloud(frequencies, title, highlight_dict):
    color_func = get_color_func(highlight_dict)
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white',
        color_func=color_func
    ).generate_from_frequencies(frequencies)

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(title, fontsize=18)
    plt.axis('off')
    plt.show()
    filename = 'wordcloud_analysis_frequencies_' + title.replace(" ", "_") + '.png'
    plt.savefig(filename)

# Generate word clouds
for category, frequencies in word_frequencies.items():
    title = f"Word Cloud for {category.replace('_', ' ').capitalize()} Narrative"
    create_wordcloud(frequencies, title, highlight_dict=highlight_words)
