import os
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt

# Define the paths to the feminine and masculine word lists
data_dir = './data'
feminine_words_path = os.path.join(data_dir, 'female_words.txt')
masculine_words_path = os.path.join(data_dir, 'male_words.txt')

# Function to load words from a file into a list
def load_word_list(file_path):
    with open(file_path, 'r') as file:
        words = [line.strip().lower() for line in file if line.strip()]
    return set(words)  # Use a set for faster lookup

# Load the feminine and masculine words
feminine_words = load_word_list(feminine_words_path)
masculine_words = load_word_list(masculine_words_path)

# Job description files to analyze
files_to_analyze = ['male_coded.txt', 'female_coded.txt', 'gender_neutral.txt']
files_to_analyze_paths = [os.path.join(data_dir, file) for file in files_to_analyze]


# Function to read the content of a file
def read_file(file_path):
    with open(file_path, 'r') as file:
        return file.read()

# Function to calculate gendered word proportions relative to word lists
def calculate_gendered_proportions(text, feminine_words, masculine_words):
    words = text.lower().split()
    
    # Count matches for feminine and masculine words
    feminine_matches = len(set(words).intersection(feminine_words))
    masculine_matches = len(set(words).intersection(masculine_words))
    
    # Calculate proportions based on the size of the respective word lists
    feminine_proportion = (feminine_matches / len(feminine_words)) * 100 if feminine_words else 0
    masculine_proportion = (masculine_matches / len(masculine_words)) * 100 if masculine_words else 0

    return feminine_proportion, masculine_proportion, feminine_matches, masculine_matches

# Data collection
results = []
word_data = []

for file in files_to_analyze_paths:
    text = read_file(file_path)
    file_label = os.path.basename(file_path).replace('.txt', '').replace('_', ' ').title()
    
    # Calculate proportions and word counts
    feminine_proportion, masculine_proportion, feminine_matches, masculine_matches = calculate_gendered_proportions(
        text, feminine_words, masculine_words
    )
    
    # Store results for proportion plot
    results.append({
        'file': file_label,
        'feminine_proportion': feminine_proportion,
        'masculine_proportion': masculine_proportion,
        'feminine_matches': feminine_matches,
        'masculine_matches': masculine_matches
    })
    
    # Collect data for top word plots
    words = text.lower().split()
    feminine_counter = Counter(word for word in words if word in feminine_words)
    masculine_counter = Counter(word for word in words if word in masculine_words)
    
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

# Create a DataFrame for proportions
proportions_df = pd.DataFrame(results)

# Replace file names with more descriptive labels
proportions_df['file'] = proportions_df['file'].replace({
    'male_coded.txt': 'Male-coded',
    'female_coded.txt': 'Female-coded',
    'gender_neutral.txt': 'Gender-neutral'
})

# Plot 1: Proportion Comparison of Gendered Words
ax = proportions_df.plot(
    x='file', 
    y=['feminine_proportion', 'masculine_proportion'], 
    kind='bar', 
    stacked=False, 
    color=['#696969', '#D3D3D3'], 
    width=0.8, 
    figsize=(10, 6)
)
ax.set_ylabel('Proportion of Gendered Words (Relative to Word List)')
ax.set_title('Feminine vs. Masculine Word Proportion by File')
ax.set_xticklabels(proportions_df['file'], rotation=0)
ax.legend(['Feminine words', 'Masculine words'], title='Gender')
ax.set_xlabel('Job Title Type')

# Format the y-axis as percentages
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}%'))

plt.show()
plt.savefig('gendered_analysis_similarity.png')

# Create a DataFrame for word data
word_data_df = pd.DataFrame(word_data)

# Plot 2: Top 5 Feminine and Masculine Words for Each File
for file in files_to_analyze:
    subset = word_data_df[word_data_df['file'] == file]
    
    # Separate by gender and get top 5 words for each
    feminine_df = subset[subset['gender'] == 'feminine'].nlargest(5, 'count')
    masculine_df = subset[subset['gender'] == 'masculine'].nlargest(5, 'count')
    
    # Determine custom title based on file type
    if file == 'male_coded.txt':
        title = 'Top Gendered Words For Male-Coded Job Descriptions'
    elif file == 'female_coded.txt':
        title = 'Top Gendered Words For Female-Coded Job Descriptions'
    else:
        title = 'Top Gendered Words For Gender-Neutral Job Descriptions'
    
    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(title, fontsize=16)
    
    # Feminine words plot
    feminine_df.plot(kind='barh', x='word', y='count', ax=axes[0], color='#696969', legend=False)
    axes[0].set_title('Feminine Words')
    axes[0].invert_yaxis()
    axes[0].set_xlabel('Frequency of word occurrence')

    # Masculine words plot
    masculine_df.plot(kind='barh', x='word', y='count', ax=axes[1], color='#D3D3D3', legend=False)
    axes[1].set_title('Masculine Words')
    axes[1].invert_yaxis()
    axes[1].set_xlabel('Frequency of word occurrence')
    
    plt.tight_layout()
    plt.show()
    plt.savefig('gendered_analysis_top_five.png')
    

# Plot 3: All Words with Frequency > 4 Across Files
all_words_data = word_data_df[word_data_df['count'] > 4]

# Assign descriptive labels for files
all_words_data['file'] = all_words_data['file'].replace({
    'male_coded.txt': 'Male-coded',
    'female_coded.txt': 'Female-coded',
    'gender_neutral.txt': 'Gender-neutral'
})

# Prepare data for the bar chart
word_frequencies = all_words_data.groupby(['word', 'file'])['count'].sum().reset_index()

# Pivot to format data for the bar chart
pivoted_data = word_frequencies.pivot(
    index='word', 
    columns='file', 
    values='count'
).fillna(0)

# Sort words by total frequency across all files
pivoted_data['total'] = pivoted_data.sum(axis=1)
pivoted_data = pivoted_data[pivoted_data['total'] > 4].drop(columns='total')  # Filter words > 4 occurrences
pivoted_data = pivoted_data.sort_values(by=pivoted_data.columns.tolist(), ascending=False)

# Define colors and hatch patterns
colors = ['#696969', '#A9A9A9', '#D3D3D3']
hatches = ['', '///', '']  # Hatching for medium gray bars (female-coded)

# Plot non-stacked bar chart for all words with frequency > 4
ax = pivoted_data.plot(
    kind='bar',
    figsize=(16, 8),
    width=0.8,
    color=colors  # Set the colors
)

# Add hatch patterns for specific bars
for bar_group, hatch in zip(ax.containers, hatches):
    for bar in bar_group:
        bar.set_hatch(hatch)

# Add titles and labels
plt.title("Words with Total Frequency > 4 Across Files", fontsize=16)
plt.ylabel("Frequency", fontsize=12)
plt.xlabel("Words", fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.legend(title="File Type", fontsize=10)
plt.tight_layout()
plt.show()
plt.savefig('gendered_analysis_clustering.png')
