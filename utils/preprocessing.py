
# Standard library imports
import os
import re
from multiprocessing import Pool, cpu_count

# Third-party imports
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize

# Local imports
from utils.general import N_PROCESSES
from utils.data_load_extract import collect_filepaths

def clean_text(text, lemmatizer=None, stopwords_list=list()):
    """
    Cleans a given text by applying various transformations.

    The function tokenizes the text into sentences, and then for each sentence:
    1. Converts to lowercase.
    2. Removes URLs, emails, numbers, and some special characters.
    3. Tokenizes into words.
    4. Removes stopwords.
    5. Lemmatizes words.
    6. Joins the words back into a sentence.

    Args:
    - text (str): Text to be cleaned.
    - lemmatizer (function, optional): Lemmatizing function. Defaults to None.
    - stopwords_list (list, optional): List of stopwords to remove. Defaults to an empty list.

    Returns:
    - list: A list of lists, where each sub-list contains information about each cleaned sentence.
    """
    # Tokenize the document into sentences
    sentences = sent_tokenize(text)

    # Initialize the results list
    results = []

    offset = 0

    for sentence in sentences:
        # Calculate the length and offset of each sentence
        length = len(sentence)
        offset_beggining_of_sentence = offset
        offset += length + 1  # Account for spaces

        # Filter out sentences with length shorter than 50 characters
        if length > 50:
            # Apply the transformations to clean the sentences
            cleaned = sentence.lower()
            cleaned = re.sub(
                r"http\S+|www\S+|email\S+", "", cleaned
            )  # Remove URLs and emails
            cleaned = re.sub(r"\d+", "", cleaned)  # Remove numbers
            cleaned = re.sub(
                r"[^a-zA-Zñéüäöß. ]", "", cleaned
            )  # Remove unwanted special characters

            tokens = word_tokenize(cleaned)

            if stopwords_list:
                tokens = [token for token in tokens if token not in stopwords_list]

            if lemmatizer:
                tokens = [lemmatizer.lemmatize(token) for token in tokens]

            cleaned = " ".join(tokens)
            cleaned = re.sub(" +", " ", cleaned).strip()  # Final cleanup

            # Check if the cleaned sentence is not empty, avoid NaNs
            if not cleaned:
                cleaned = " "

            results.append([sentence, length, offset_beggining_of_sentence, cleaned])

    return results


def preprocess_document(filepath, cleaned_path, lemmatizer=None, stopwords_list=list()):
    filename = os.path.basename(filepath)
    cleaned_filepath = os.path.join(cleaned_path, filename)[:-3] + "parquet"

    with open(filepath, "r") as file:
        doc = file.read()
        results = clean_text(doc, lemmatizer, stopwords_list)

    cols = ["sentence", "length", "offset", "cleaned_sentence"]
    df = pd.DataFrame(data=results, columns=cols)
    df.to_parquet(cleaned_filepath)

def preprocessing_data(original_dir, new_dir, limit):
    # Initialize lemmatizer for word normalization
    lemmatizer = WordNetLemmatizer()
    # Initialize list of stopwords to be removed during preprocessing
    stopwords_list = set(stopwords.words("english"))
    # Collect filepaths
    filepaths = collect_filepaths(original_dir)[:limit]
    # Use multi-processing to parallelize data preprocessing tasks for better performance
    with Pool(processes=N_PROCESSES) as pool:
        pool.starmap(
            preprocess_document,
            [(filepath, new_dir, lemmatizer, stopwords_list) for filepath in filepaths],
        )