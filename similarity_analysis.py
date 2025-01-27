import os
import re
import numpy as np
import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import nltk
from nltk.corpus import stopwords


# Ensure stopwords are downloaded
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Add custom words to exclude
custom_stop_words = {'role', 'work', 'responsible', 'ensuring', 'join', 'ensure', 'seeking', 
                    'maintaining', 'play', 'various', 'weather', 'also'}
stop_words.update(custom_stop_words)

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
print("BERT model and tokenizer loaded successfully.")

# Function to load a word list
def load_word_list(file_path):
    with open(file_path, 'r') as file:
        words = [line.strip().lower() for line in file if line.strip()]
    return set(words)  # Use a set for faster lookup

# Configurable directory paths
data_dir = './data'
feminine_words_path = os.path.join(data_dir, 'female_words.txt')
masculine_words_path = os.path.join(data_dir, 'male_words.txt')

feminine_words = load_word_list(feminine_words_path)
masculine_words = load_word_list(masculine_words_path)

# Job description files
job_descriptions = {
    'female_coded': 'combined_woman.txt',
    'male_coded': 'combined_man.txt',
    'gender_neutral': 'combined_other.txt'
}

for category in job_descriptions.keys():
    job_descriptions[category] = os.path.join(data_dir, job_descriptions[category]) 

# Function to preprocess text
def preprocess_text(text):
    # Lowercase and remove punctuation
    text = re.sub(r'[^\w\s]', '', text.lower())
    return text

# Function to get BERT embeddings for a list of words
def get_embeddings(text, words_to_embed):
    preprocessed_text = preprocess_text(text)
    tokens = tokenizer(preprocessed_text, return_tensors='pt', truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**tokens)
    embeddings = outputs.last_hidden_state.squeeze(0)  # Shape: [sequence_length, hidden_size]

    # Extract embeddings for words in the list
    word_embeddings = []
    for word in words_to_embed:
        if word in preprocessed_text.split():
            word_tokens = tokenizer.tokenize(word)
            word_ids = tokenizer.convert_tokens_to_ids(word_tokens)
            for idx, input_id in enumerate(tokens['input_ids'].squeeze()):
                if input_id in word_ids:
                    word_embeddings.append(embeddings[idx].numpy())
    return np.mean(word_embeddings, axis=0) if word_embeddings else None

# Dictionary to store embeddings
job_embeddings = {}

# Process each job description
for job_type, file_path in job_descriptions.items():
    with open(file_path, 'r') as f:
        text = f.read()
    
    feminine_embedding = get_embeddings(text, feminine_words)
    masculine_embedding = get_embeddings(text, masculine_words)
    
    if feminine_embedding is not None and masculine_embedding is not None:
        job_embeddings[job_type] = {
            "feminine": feminine_embedding,
            "masculine": masculine_embedding
        }

# Ensure embeddings were successfully created
if not job_embeddings:
    print("No embeddings were created. Check your input files and word lists.")
    exit()

# Calculate cosine similarities
cosine_similarity_results = {"Feminine": {}, "Masculine": {}}

for word_type in ["feminine", "masculine"]:
    for job1 in job_embeddings:
        for job2 in job_embeddings:
            if job1 != job2:
                sim = cosine_similarity(
                    job_embeddings[job1][word_type].reshape(1, -1),
                    job_embeddings[job2][word_type].reshape(1, -1)
                )[0][0]
                cosine_similarity_results[word_type.capitalize()][(job1, job2)] = sim

# Create DataFrames for heatmaps
heatmap_data_feminine = pd.DataFrame({
    "Male Coded": [1.0, cosine_similarity_results["Feminine"].get(("male_coded", "female_coded"), 0), cosine_similarity_results["Feminine"].get(("male_coded", "gender_neutral"), 0)],
    "Female Coded": [cosine_similarity_results["Feminine"].get(("male_coded", "female_coded"), 0), 1.0, cosine_similarity_results["Feminine"].get(("female_coded", "gender_neutral"), 0)],
    "Gender Neutral": [cosine_similarity_results["Feminine"].get(("male_coded", "gender_neutral"), 0), cosine_similarity_results["Feminine"].get(("female_coded", "gender_neutral"), 0), 1.0]
}, index=["Male Coded", "Female Coded", "Gender Neutral"])

heatmap_data_masculine = pd.DataFrame({
    "Male Coded": [1.0, cosine_similarity_results["Masculine"].get(("male_coded", "female_coded"), 0), cosine_similarity_results["Masculine"].get(("male_coded", "gender_neutral"), 0)],
    "Female Coded": [cosine_similarity_results["Masculine"].get(("male_coded", "female_coded"), 0), 1.0, cosine_similarity_results["Masculine"].get(("female_coded", "gender_neutral"), 0)],
    "Gender Neutral": [cosine_similarity_results["Masculine"].get(("male_coded", "gender_neutral"), 0), cosine_similarity_results["Masculine"].get(("female_coded", "gender_neutral"), 0), 1.0]
}, index=["Male Coded", "Female Coded", "Gender Neutral"])

# Plot heatmaps
plt.figure(figsize=(12, 6))
plt.suptitle('Cosine Similarity of Gendered Words Across Job Types')


# Feminine words heatmap
plt.subplot(1, 2, 1)
sns.heatmap(
    heatmap_data_feminine, 
    annot=True, 
    cmap="coolwarm",  
    vmin=0, 
    vmax=1, 
    square=True,
    annot_kws={"size": 12},  # Adjust font size for annotations
    cbar_kws={'label': 'Cosine Similarity'}  # Add label to color bar
)
plt.title("Feminine Words Similarity", fontsize=12)
plt.xlabel("Job Type")
plt.ylabel("Job Type")

# Masculine words heatmap
plt.subplot(1, 2, 2)
sns.heatmap(
    heatmap_data_masculine, 
    annot=True, 
    cmap="coolwarm",  
    vmin=0, 
    vmax=1, 
    square=True,
    annot_kws={"size": 12},  # Adjust font size for annotations
    cbar_kws={'label': 'Cosine Similarity'}  # Add label to color bar
)
plt.title("Masculine Words Similarity", fontsize=12)
plt.xlabel("Job Type")
plt.ylabel("Job Type")
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
plt.savefig('cosine_similarity_heatmap.png')
