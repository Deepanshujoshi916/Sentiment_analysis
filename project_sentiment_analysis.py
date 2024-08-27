from bs4 import BeautifulSoup
from newspaper import Article
import pandas as pd
import nltk
import os
import re
import string

nltk.download('vader_lexicon')
nltk.download('punkt')
nltk.download('words')

from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import words

sia = SentimentIntensityAnalyzer()

sentiment_data = []

# Complexity threshold to classify words as complex
COMPLEXITY_THRESHOLD = 7  # You can adjust this threshold as needed
# Define your lists of positive and negative words
positive_words = ["good", "happy", "excellent", "positive"]  
negative_words = ["bad", "sad", "terrible", "negative"] 
# Function to calculate Fog Index
def fog_index(text):
    words = text.split()
    sentences = sent_tokenize(text)
    num_complex_words = sum(1 for word in words if len(word) > COMPLEXITY_THRESHOLD)
    avg_words_per_sentence = len(words) / len(sentences)
    fog_index = 0.4 * (avg_words_per_sentence + num_complex_words)
    return fog_index

def avg_word_length(text):
    words = text.split()
    total_word_length = sum(len(word) for word in words)
    avg_length = total_word_length / len(words)
    return avg_length

def syllable_count(word):
    vowels = "AEIOUaeiou"
    count = 0
    prev_char = ''
    for char in word:
        if char in vowels and prev_char not in vowels:
            count += 1
        prev_char = char
    if word.endswith('e'):
        count -= 1
    if count == 0:
        count = 1
    return count

def personal_pronouns_count(text):
    pronoun_list = ['I', 'we', 'my', 'ours', 'us']
    pronoun_count = sum(1 for word in text.split() if word.lower() in pronoun_list)
    return pronoun_count

def sanitize_filename(filename):
    valid_chars = "-_.() %s%s" % (string.ascii_letters, string.digits)
    sanitized = ''.join(c for c in filename if c in valid_chars)
    return sanitized

#read list of URLs from the file
with open('urls.txt', "r", encoding='utf-8') as f:
    urls = f.readlines()

#create a directory to store description files
if not os.path.exists('descriptions'):
    os.makedirs('descriptions')

#process each URL
for url in urls:
    try:
        # Create an Article object and download, parse, and apply natural language processing
        article = Article(url)
        article.download()
        article.parse()
        article.nlp()

        # Get the full content of the article
        full_content = article.text

        # Tokenize the article text
        tokens = word_tokenize(full_content)

        # Calculate sentiment score using NLTK's SentimentIntensityAnalyzer
        sentiment_scores = sia.polarity_scores(full_content)

        # Calculate additional features
        word_count = len(tokens)
        avg_sentence_length = len(full_content) / len(sent_tokenize(full_content))
        complex_word_count = sum(1 for token in tokens if len(token) > COMPLEXITY_THRESHOLD)
        percentage_complex_words = (complex_word_count / len(tokens)) * 100
        fog_index_value = fog_index(full_content)
        avg_words_per_sentence = len(tokens) / len(sent_tokenize(full_content))
       
        avg_word_length_value = avg_word_length(full_content)
        avg_syllables_per_word = sum(syllable_count(word) for word in tokens) / len(tokens)
        personal_pronouns = personal_pronouns_count(full_content)

        positive_score = sum(1 for word in tokens if word.lower() in positive_words)
        negative_score = sum(1 for word in tokens if word.lower() in negative_words)
        polarity_score = (positive_score - negative_score) / ((positive_score + negative_score) + 0.000001)
        subjectivity_score = (positive_score + negative_score) / (word_count + 0.000001)
        # Sanitize the article title for use as a filename
        article_title = article.title if article.title else 'untitled'
        sanitized_title = sanitize_filename(article_title)

        #ad the data to sentiment_data list
        sentiment_data.append({
            'URL': url.strip(),
            'Positive_Score': sentiment_scores['pos'],
            'Negative_Score': sentiment_scores['neg'],
            'Polarity_Score': polarity_score,
            'Subjectivity_Score': subjectivity_score,
            'Word_Count': word_count,
            'Complex_Words_Count': complex_word_count,
            'avg sentence length': avg_sentence_length,
            'Percentage of complex words': percentage_complex_words,
            'fog index': fog_index_value,
            'avg number of words per sentence': avg_words_per_sentence,
            'avg word length': avg_word_length_value,
            'Syllable_per_word': avg_syllables_per_word,
            'Personal_Pronoun': personal_pronouns
        })

        # Save data into a seperate file
        description_filename = f'descriptions/{sanitized_title}.txt'
        with open(description_filename, 'w', encoding='utf-8') as desc_file:
            desc_file.write(full_content)
        
    except Exception as e:
        print("An error occurred for URL", url, ":", e)

#create a Pandas DataFrame from sentiment_data
df = pd.DataFrame(sentiment_data)

#save the DataFrame to a CSV file
df.to_csv("sentiment_data.csv", index=False)
