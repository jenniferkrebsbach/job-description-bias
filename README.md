# job-description-bias
# Gendered Language Analysis with BERT and Word Clouds
This project analyzes gendered language in job descriptions to identify patterns in word usage associated with male-coded, female-coded, and gender-neutral job descriptions. It includes scripts for:

* Calculating gendered word frequencies
* Generating visualizations for feminine and masculine word distribution
* Running BERT-based similarity analysis
* Clustering and word cloud generation


# Table of Contents
1. Project Structure

2. Installation

3. Data Preparation

4. Running the Analysis

5. Interpreting the Results

6. Project Structure

# This repository includes the following key files and folders:

* gendered_analysis.py – Main analysis script for gendered word frequencies and visualizations
* bert_analysis.py – BERT similarity and clustering analysis script
* wordcloud_analysis.py – Script for generating word clouds
* data/ – Folder containing gendered word lists and job description text files
* requirements.txt – Dependencies for running the analysis

# Installation
Clone the Repository

git clone https://github.com/yourusername/your-repo-name.git

cd your-repo-name

# Install Required Python Packages Make sure you have Python 3.6+ installed, then install dependencies using:

pip install -r requirements.txt

# Verify PyTorch Compatibility 
Ensure that PyTorch is installed correctly, as it is required by the BERT model. You may need to install it separately depending on your hardware (CPU vs. GPU). Check the PyTorch installation page for details.

# Data Preparation
**Gendered Word Lists:**
- The folder data/ should contain two text files, Female_words.txt and Male_words.txt, with gendered words for analysis. These files come from well-known gender bias studies.
**Job Description Files:**
- Place your job description files in the data/ folder. The sample file names used in the script are:
  - male_coded_jobs.txt
  - female_coded_jobs.txt
  - gender_neutral_jobs.txt
- You can add or rename files, but be sure to update the files_to_analyze variable in each script accordingly.

# Running the Analysis
1. Run Gendered Word Frequency and Visualization Analysis

python gendered_analysis.py

This script will:

* Calculate and display the percentage of feminine and masculine words in each job description file.
* Generate bar charts for feminine vs. masculine word usage and the top words in each category.

2. Run BERT Similarity and Clustering Analysis

python bert_analysis.py

This script will:

* Calculate cosine similarity between job description types for gendered words using BERT embeddings.
* Create a clustering plot to visualize similarities.

3. Generate Word Clouds

python wordcloud_analysis.py

This script will:

* Generate word clouds for each job description type, highlighting frequent words after removing common stopwords and top words.

# Interpreting the Results
* Frequency Visualizations:
  - The bar charts provide an overview of the distribution of gendered words across job description types.
* BERT Analysis:
  - Cosine similarity values indicate how similar or different the job descriptions are in terms of gendered language.
  - The clustering plot visualizes the groupings of sentences based on gendered word usage.
* Word Clouds:
  - Word clouds highlight prominent words, offering insights into the themes and language patterns within each job description category.
