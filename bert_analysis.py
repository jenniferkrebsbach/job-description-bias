import os
import numpy as np
import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity

# Load the pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

print("BERT model and tokenizer loaded successfully.")

# Load the lemmatized gendered word lists
def load_word_list(file_path):
    with open(file_path, 'r') as file:
        words = [line.strip().lower() for line in file if line.strip()]
    return words

feminine_words = load_word_list('/Users/jenniferkrebsbach/Female_words.txt')
masculine_words = load_word_list('/Users/jenniferkrebsbach/Male_words.txt')

# Define paths to job description files
job_descriptions = {
    "male_coded": "/Users/jenniferkrebsbach/combined_man.txt",
    "female_coded": "/Users/jenniferkrebsbach/combined_woman.txt",
    "gender_neutral": "/Users/jenniferkrebsbach/combined_other.txt"
}

# Debugging-enabled function to get embeddings for gendered words
def get_embeddings(text, words_to_embed):
    # Tokenize and convert text to tensors
    tokens = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**tokens)
    embeddings = outputs.last_hidden_state.squeeze(0)  # Shape: [sequence_length, hidden_size]
    
    # Extract embeddings for words of interest
    word_embeddings = []
    for word in words_to_embed:
        word_tokens = tokenizer.tokenize(word)
        word_ids = tokenizer.convert_tokens_to_ids(word_tokens)
        
        found = False  # Track if the word was found
        for idx, input_id in enumerate(tokens['input_ids'].squeeze()):
            if input_id in word_ids:
                word_embeddings.append(embeddings[idx].numpy())
                found = True
        if found:
            print(f"Embedding found for word: {word}")
        else:
            print(f"Word not found in text: {word}")
                
    # Return average embedding for words of interest
    return np.mean(word_embeddings, axis=0) if word_embeddings else None

print("Function for embedding extraction defined successfully.")

# Dictionary to store the embeddings for each job type
job_embeddings = {}

# Process each job description type
for job_type, file_path in job_descriptions.items():
    # Load the text content of each job description
    with open(file_path, 'r') as f:
        text = f.read()
    
    # Get embeddings for feminine and masculine words
    feminine_embedding = get_embeddings(text, feminine_words)
    masculine_embedding = get_embeddings(text, masculine_words)
    
    # Store embeddings if they exist
    if feminine_embedding is not None and masculine_embedding is not None:
        job_embeddings[job_type] = {
            "feminine": feminine_embedding,
            "masculine": masculine_embedding
        }

print("Gendered word embeddings extracted for each job description type.")

# Calculate and display cosine similarities for feminine and masculine embeddings between job types
for word_type in ["feminine", "masculine"]:
    print(f"\nCosine Similarity for {word_type.capitalize()} words:")
    for job1 in job_embeddings:
        for job2 in job_embeddings:
            if job1 != job2:
                if job_embeddings[job1][word_type] is not None and job_embeddings[job2][word_type] is not None:
                    # Compute cosine similarity only if embeddings are valid
                    sim = cosine_similarity(
                        job_embeddings[job1][word_type].reshape(1, -1),
                        job_embeddings[job2][word_type].reshape(1, -1)
                    )[0][0]
                    print(f"{job1} vs {job2}: {sim:.4f}")
                else:
                    print(f"Missing embeddings for comparison between {job1} and {job2} for {word_type} words.")

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Cosine similarity values
similarity_data = {
    "Feminine": {
        ("male_coded", "female_coded"): 0.5691,
        ("male_coded", "gender_neutral"): 0.4301,
        ("female_coded", "gender_neutral"): 0.3762
    },
    "Masculine": {
        ("male_coded", "female_coded"): 0.7400,
        ("male_coded", "gender_neutral"): 0.8840,
        ("female_coded", "gender_neutral"): 0.7920
    }
}

# Convert data to DataFrame format for easier plotting
heatmap_data_feminine = pd.DataFrame({
    "male_coded": [1.0, similarity_data["Feminine"][("male_coded", "female_coded")], similarity_data["Feminine"][("male_coded", "gender_neutral")]],
    "female_coded": [similarity_data["Feminine"][("male_coded", "female_coded")], 1.0, similarity_data["Feminine"][("female_coded", "gender_neutral")]],
    "gender_neutral": [similarity_data["Feminine"][("male_coded", "gender_neutral")], similarity_data["Feminine"][("female_coded", "gender_neutral")], 1.0]
}, index=["male_coded", "female_coded", "gender_neutral"])

heatmap_data_masculine = pd.DataFrame({
    "male_coded": [1.0, similarity_data["Masculine"][("male_coded", "female_coded")], similarity_data["Masculine"][("male_coded", "gender_neutral")]],
    "female_coded": [similarity_data["Masculine"][("male_coded", "female_coded")], 1.0, similarity_data["Masculine"][("female_coded", "gender_neutral")]],
    "gender_neutral": [similarity_data["Masculine"][("male_coded", "gender_neutral")], similarity_data["Masculine"][("female_coded", "gender_neutral")], 1.0]
}, index=["male_coded", "female_coded", "gender_neutral"])

# Plotting the Heatmap for Feminine Words
plt.figure(figsize=(10, 5))
plt.suptitle('Cosine Similarity of Gendered Words Across Job Types')

plt.subplot(1, 2, 1)
sns.heatmap(heatmap_data_feminine, annot=True, cmap="Purples", vmin=0, vmax=1, square=True)
plt.title("Feminine Words Similarity")
plt.xlabel("Job Type")
plt.ylabel("Job Type")

# Plotting the Heatmap for Masculine Words
plt.subplot(1, 2, 2)
sns.heatmap(heatmap_data_masculine, annot=True, cmap="Blues", vmin=0, vmax=1, square=True)
plt.title("Masculine Words Similarity")
plt.xlabel("Job Type")
plt.ylabel("Job Type")

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# Plotting Clustered Bar Chart
similarity_df = pd.DataFrame({
    "Job Type Comparison": [
        "male vs female", "male vs gender-neutral", "female vs gender-neutral"
    ],
    "Feminine Similarity": [
        similarity_data["Feminine"][("male_coded", "female_coded")],
        similarity_data["Feminine"][("male_coded", "gender_neutral")],
        similarity_data["Feminine"][("female_coded", "gender_neutral")]
    ],
    "Masculine Similarity": [
        similarity_data["Masculine"][("male_coded", "female_coded")],
        similarity_data["Masculine"][("male_coded", "gender_neutral")],
        similarity_data["Masculine"][("female_coded", "gender_neutral")]
    ]
})

# Melt data for grouped bar chart
similarity_melted = similarity_df.melt(id_vars="Job Type Comparison", var_name="Gendered Word Type", value_name="Similarity")

plt.figure(figsize=(8, 6))
sns.barplot(data=similarity_melted, x="Job Type Comparison", y="Similarity", hue="Gendered Word Type", palette=["purple", "blue"])
plt.title("Cosine Similarity of Gendered Words Across Job Type Comparisons")
plt.ylim(0, 1)
plt.legend(title="Word Type")
plt.show()
