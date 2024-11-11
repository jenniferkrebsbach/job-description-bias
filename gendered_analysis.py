import os
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt

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

# Get total counts for the word lists
total_feminine_words = len(feminine_words)
total_masculine_words = len(masculine_words)

# Job description files to analyze
files_to_analyze = ['male_coded.txt', 'female_coded.txt', 'gender_neutral.txt']
files_to_analyze_paths = [os.path.join(data_dir, file) for file in files_to_analyze]

# Function to read the content of a file
def read_file(file_path):
    with open(file_path, 'r') as file:
        return file.read()

# Function to calculate gendered word proportions relative to word lists
def calculate_gendered_proportions(text, feminine_words, masculine_words):
    words = text.lower().split()  # Simple whitespace-based tokenization
    
    feminine_counter = Counter(word for word in words if word in feminine_words)
    masculine_counter = Counter(word for word in words if word in masculine_words)
    
    feminine_proportion = (sum(feminine_counter.values()) / total_feminine_words) * 100 if total_feminine_words > 0 else 0
    masculine_proportion = (sum(masculine_counter.values()) / total_masculine_words) * 100 if total_masculine_words > 0 else 0

    return feminine_proportion, masculine_proportion, feminine_counter, masculine_counter

# Data collection
results = []
word_data = []

for file_path in files_to_analyze_paths:
    text = read_file(file_path)
    file_label = os.path.basename(file_path).replace('.txt', '').replace('_', ' ').title()
    
    # Calculate proportions and word counts
    feminine_proportion, masculine_proportion, feminine_counter, masculine_counter = calculate_gendered_proportions(
        text, feminine_words, masculine_words
    )
    
    # Store results for proportion plot
    results.append({
        'file': file_label,
        'feminine_proportion': feminine_proportion,
        'masculine_proportion': masculine_proportion
    })
    
    # Collect data for top word plots and grouped comparison plot
    for word, count in feminine_counter.items():
        word_data.append({
            'word': word,
            'file': file_label,
            'count': count,
            'gender': 'feminine'
        })
    for word, count in masculine_counter.items():
        word_data.append({
            'word': word,
            'file': file_label,
            'count': count,
            'gender': 'masculine'
        })

# Create a DataFrame for proportion plot
proportions_df = pd.DataFrame(results)

# Plot 1: Proportion Comparison of Gendered Words
ax = proportions_df.plot(x='file', kind='bar', stacked=False, color=['purple', 'lavender'], width=0.8, figsize=(10, 6))
ax.set_ylabel('Proportion of Gendered Words (Relative to Word List)')
ax.set_title('Feminine vs. Masculine Word Proportion by File')
ax.set_xticklabels(proportions_df['file'], rotation=0)
ax.legend(['Feminine words', 'Masculine words'], title='Gender')
ax.set_xlabel('Job Title Type')

# Format the y-axis as percentages
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}%'))

plt.show()
plt.savefig('gendered_analysis_similarity.png')

# Create a DataFrame for Plot 2 (Grouped Bar Chart)
word_data_df = pd.DataFrame(word_data)
pivot_df = word_data_df.pivot_table(
    index='word',
    columns='file',
    values='count',
    fill_value=0
).reset_index()

# Define custom colors for the bars
colors = ['lavender', 'purple', 'mediumorchid']

# Plot 3: Grouped Bar Chart for Word Comparison Across Files
fig, ax = plt.subplots(figsize=(14, 8))
pivot_df.plot(
    x='word', 
    kind='bar', 
    ax=ax, 
    width=0.8, 
    color=colors
)

# Customize legend
ax.legend(proportions_df['file'], title='Job Title Type')
plt.title('Gendered Word Frequency Comparison Across Files')
plt.xlabel('Gendered Words')
plt.ylabel('Frequency')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
plt.savefig('gendered_analysis_clustering.png')
