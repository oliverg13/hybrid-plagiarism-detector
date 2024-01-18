# Standard library imports
from multiprocessing import Pool

# Third-party library imports
import numpy as np
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from scipy.sparse import vstack

# Warnings Filtering
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

def transform_chunk(docs_chunk, vectorizer_params):
    """Worker function to transform a chunk of documents."""
    try:
        # Reconstruct the vectorizer in each subprocess
        local_vectorizer = CountVectorizer(**vectorizer_params)
        transformed_chunk = local_vectorizer.fit_transform(docs_chunk)
        return transformed_chunk
    except Exception as e:
        # Handle exceptions and possibly log them
        print(f"Error processing chunk: {e}")
        return None

def transform_in_parallel(docs, vectorizer, N_PROCESSES):
    """Transforms a list of documents in parallel using a vectorizer."""
    # Serialize vectorizer's parameters
    vectorizer_params = vectorizer.get_params()
    # This is necessary to make sure we are ussing the proper vocab
    vectorizer_params["vocabulary"] = list(vectorizer.get_feature_names_out())

    # Chunk the documents list for parallel processing
    chunk_size = int(np.ceil(len(docs) / N_PROCESSES))
    chunks = [docs[i : i + chunk_size] for i in range(0, len(docs), chunk_size)]

    with Pool(N_PROCESSES) as pool:
        # Use pool.map() to apply transform_chunk() on each chunk
        transformed_chunks = pool.starmap(
            transform_chunk, [(chunk, vectorizer_params) for chunk in chunks]
        )

    # Merge the results (vertically stack the matrices)
    result_matrix = vstack([chunk for chunk in transformed_chunks if chunk is not None])
    return result_matrix

def train_tfidf_vectorizer(suspicious_docs, source_docs):
    tfidf_vectorizer = TfidfVectorizer(norm=None, tokenizer=word_tokenize)
    tfidf_vectorizer.fit(suspicious_docs + source_docs)
    common_vocabulary = tfidf_vectorizer.get_feature_names_out()
    return tfidf_vectorizer, common_vocabulary

def get_word_weights_and_vectors(ft, tfidf_vectorizer):
    # Extract the vocabulary from the TF-IDF vectorizer
    tfidf_vocab = tfidf_vectorizer.get_feature_names_out()

    # Get the diagonal TF-IDF weights for the terms
    tfidf_weights = tfidf_vectorizer.transform(tfidf_vocab).diagonal()

    # Fetch the FastText word vectors for each term in the vocabulary
    word_vectors = [ft.get_word_vector(token) for token in tfidf_vocab]

    # Sum the word vectors across words
    sum_word_vectors = np.sum(np.array(word_vectors), axis=1)

    return tfidf_weights, sum_word_vectors