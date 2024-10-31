import os
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt

# Define the paths to the feminine and masculine word lists
feminine_words_path = '/Users/jenniferkrebsbach/Female_words.txt'
masculine_words_path = '/Users/jenniferkrebsbach/Male_words.txt'

# Function to load words from a file into a list
def load_word_list(file_path):
    with open(file_path, 'r') as file:
        words = [line.strip().lower() for line in file if line.strip()]
    return words

# Load the feminine and masculine words
feminine_words = load_word_list(feminine_words_path)
masculine_words = load_word_list(masculine_words_path)

# Get total counts for the word lists
total_feminine_words = len(feminine_words)
total_masculine_words = len(masculine_words)

# Paths to the text files for analysis
directory = '/Users/jenniferkrebsbach'
files_to_analyze = ['combined_man.txt', 'combined_woman.txt', 'combined_other.txt']

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

for file in files_to_analyze:
    file_path = os.path.join(directory, file)
    text = read_file(file_path)
    
    # Calculate proportions and word counts
    feminine_proportion, masculine_proportion, feminine_counter, masculine_counter = calculate_gendered_proportions(
        text, feminine_words, masculine_words
    )
    
    # Store results for proportion plot
    results.append({
        'file': file,
        'feminine_proportion': feminine_proportion,  # Proportion based on word lists
        'masculine_proportion': masculine_proportion  # Proportion based on word lists
    })
    
    # Collect data for top word plots and grouped comparison plot
    for word, count in feminine_counter.items():
        word_data.append({
            'word': word,
            'file': file,
            'count': count,
            'gender': 'feminine'
        })
    for word, count in masculine_counter.items():
        word_data.append({
            'word': word,
            'file': file,
            'count': count,
            'gender': 'masculine'
        })

# Create a DataFrame for proportion plot
proportions_df = pd.DataFrame(results)

# Plot 1: Proportion Comparison of Gendered Words
proportions_df['file'] = proportions_df['file'].replace({
    'combined_man.txt': 'Male-coded',
    'combined_woman.txt': 'Female-coded',
    'combined_other.txt': 'Gender-neutral'
})

ax = proportions_df.plot(x='file', kind='bar', stacked=False, color=['purple', 'lavender'], width=0.8, figsize=(10, 6))
ax.set_ylabel('Proportion of Gendered Words (Relative to Word List)')
ax.set_title('Feminine vs. Masculine Word Proportion by File')
ax.set_xticklabels(proportions_df['file'], rotation=0)
ax.legend(['Feminine words', 'Masculine words'], title='Gender')
ax.set_xlabel('Job Title Type')

# Format the y-axis as percentages
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}%'))

plt.show()

# Plot 2: Top 5 Feminine and Masculine Words for Each File
for file in files_to_analyze:
    subset = [d for d in word_data if d['file'] == file]
    word_counts_df = pd.DataFrame(subset)
    
    # Separate by gender and get top 5 words for each
    feminine_df = word_counts_df[word_counts_df['gender'] == 'feminine'].nlargest(5, 'count')
    masculine_df = word_counts_df[word_counts_df['gender'] == 'masculine'].nlargest(5, 'count')
    
    # Determine custom title based on file type
    if file == 'combined_man.txt':
        title = 'Top Gendered Words For Male-Coded Job Descriptions'
    elif file == 'combined_woman.txt':
        title = 'Top Gendered Words For Female-Coded Job Descriptions'
    else:
        title = 'Top Gendered Words For Gender-Neutral Job Descriptions'
    
    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(title, fontsize=16)
    
    # Feminine words plot with integer x-axis ticks
    feminine_df.plot(kind='barh', x='word', y='count', ax=axes[0], color='purple', legend=False)
    axes[0].set_title('Feminine Words')
    axes[0].invert_yaxis()
    axes[0].set_xlabel('Frequency of word occurrence')
    axes[0].set_xticks(range(0, int(feminine_df['count'].max()) + 1))  # Set integer x-ticks only

    # Masculine words plot
    masculine_df.plot(kind='barh', x='word', y='count', ax=axes[1], color='lavender', legend=False)
    axes[1].set_title('Masculine Words')
    axes[1].invert_yaxis()
    axes[1].set_xlabel('Frequency of word occurrence')
    axes[1].set_xticks(range(0, int(masculine_df['count'].max()) + 1))  # Set integer x-ticks only
    
    plt.show()

# Create a DataFrame for Plot 3 (Grouped Bar Chart)
word_data_df = pd.DataFrame(word_data)
pivot_df = word_data_df.pivot_table(
    index='word',
    columns='file',
    values='count',
    fill_value=0
).reset_index()

# Replace legend labels with the requested names
file_labels = {
    'combined_man.txt': 'Male-coded',
    'combined_woman.txt': 'Female-coded',
    'combined_other.txt': 'Gender-neutral'
}
pivot_df.rename(columns=file_labels, inplace=True)

# Define custom colors for the bars
colors = ['lavender', 'purple', 'mediumorchid']  # Distinct colors for each file

# Plot 3: Grouped Bar Chart for Word Comparison Across Files with Varying Colors
fig, ax = plt.subplots(figsize=(14, 8))
pivot_df.plot(
    x='word', 
    kind='bar', 
    ax=ax, 
    width=0.8, 
    color=colors  # Use distinct colors for each file
)

# Customize legend with updated labels
ax.legend(file_labels.values(), title='Job Title Type')

# Update axis labels and title
plt.title('Gendered Word Frequency Comparison Across Files')
plt.xlabel('Gendered Words')
plt.ylabel('Frequency')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
