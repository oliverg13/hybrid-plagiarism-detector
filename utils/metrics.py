# Standard library imports
import os

# Third-party library imports
import numpy as np
import pandas as pd
from tqdm import tqdm
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import fasttext

# Local imports
from utils.general import printt

# Warnings Filtering
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

def overlap(a1, a2, b1, b2):
    """Check if two intervals (a1, a2) and (b1, b2) overlap."""
    return max(0, min(a2, b2) - max(a1, b1)) > 0

def evaluate_detections(sent_matches_dir, THRESHOLD, df_references):
    sent_matches_filenames = os.listdir(sent_matches_dir)
    TP, FP = 0, 0 # Initiliaze counting for this threshold value
    S_R, R_S = set(), set() # initialize empty sets to calculate granularity and plagdet
    for sent_matches_filename in sent_matches_filenames:
        sent_matches_filepath = os.path.join(sent_matches_dir, sent_matches_filename)
        df = pd.read_parquet(sent_matches_filepath)
        df = df[df["hybrid_similarity"]>=THRESHOLD]

        printt(f"Evaluate detections of {sent_matches_filename}")
        for idx in tqdm(range(len(df)), unit="row"):
            row = df.iloc[idx] # select row from df detections
            suspicious_filename = row["suspicious_filename"]
            suspicious_offset = row["suspicious_offset"]
            suspicious_length = row["suspicious_length"]
            source_filename = row["source_filename"]
            source_offset = row["source_offset"]
            source_length = row["source_length"]

            # Filter by filenames
            matching_rows = df_references[
                (df_references["suspicious_filename"] == suspicious_filename)
                & (df_references["source_filename"] == source_filename)
            ]
            
            # Search using overlap function
            overlap_found = False
            for _, reference in matching_rows.iterrows():
                if overlap(
                    suspicious_offset,
                    suspicious_offset + suspicious_length,
                    reference["suspicious_offset"],
                    reference["suspicious_offset"] + reference["suspicious_length"],
                ) and overlap(
                    source_offset,
                    source_offset + source_length,
                    reference["source_offset"],
                    reference["source_offset"] + reference["source_length"],
                ):
                    overlap_found = True
                    break
            
            if overlap_found:
                TP += 1
                R_S.add(row)
                S_R.add(reference)
            else:
                FP += 1

    FN = len(df_references) - TP
    granularity = len(R_S) / len(S_R) if len(S_R) != 0 else 1
    # Calculate precision, recall, f1-score, plagdet
    precision = TP / (TP + FP) if TP + FP != 0 else 0
    recall = TP / (TP + FN) if TP + FN != 0 else 0
    f1_score = (
        2 * (precision * recall) / (precision + recall)
        if precision + recall != 0
        else 0
    )
    plagdet = f1_score / np.log2(1 + granularity)

    return precision, recall, f1_score, plagdet


def create_syn_sem_matrices(token_to_tfidf_weight, token_to_word_vector, join_matrix_vocab, sent_tokenized):
    sent_semantic_matrix = []
    sent_syntactic_matrix = []
    sent_matrix_embeddings = np.array([token_to_word_vector[sent_token] for sent_token in sent_tokenized])
    for jm_token in join_matrix_vocab:
        if jm_token in sent_tokenized:
            sent_semantic_matrix.append(1)
            sent_syntactic_matrix.append(token_to_tfidf_weight[jm_token])
        else:
            jm_token_vector = token_to_word_vector[jm_token]
            cosine_similarity_matrix = sent_matrix_embeddings@jm_token_vector.T
            idx_largest_similarity_token = np.argmax(cosine_similarity_matrix)
            largest_similarity_value = cosine_similarity_matrix[idx_largest_similarity_token]
            sent_token = sent_tokenized[idx_largest_similarity_token]
            sent_semantic_matrix.append(largest_similarity_value)
            sent_syntactic_matrix.append(token_to_tfidf_weight[sent_token])
    sent_semantic_matrix = np.array(sent_semantic_matrix)
    sent_syntactic_matrix = np.array(sent_syntactic_matrix)
    return sent_semantic_matrix, sent_syntactic_matrix

def similarity_semantic(vec1,vec2):
    numerator = 2*vec1.dot(vec2)
    denominator = vec1.dot(vec1) + vec2.dot(vec2)
    return numerator / denominator

def similarity_syntactic(vec1, vec2):
    diff = vec1 - vec2
    summ = vec1 + vec2
    numerator = np.linalg.norm(diff)
    denominator = np.linalg.norm(summ)
    return 1 - numerator / denominator

def similarity_computation(cleaned_suspicious_path, cleaned_source_path, sent_matches_dir, beta):
    # Load fasttext
    printt("Loading FastText")
    ft = fasttext.load_model("cc.en.300.bin")

    # Load TF-IDF vectorizer and train it on all the suspicious and source documents available
    tfidf_vectorizer = TfidfVectorizer(norm=None, tokenizer=word_tokenize)

    cleaned_suspicious_filenames = os.listdir(cleaned_suspicious_path)
    cleaned_suspicious_filepaths = [os.path.join(cleaned_suspicious_path, suspicious_filename) for suspicious_filename in cleaned_suspicious_filenames]

    cleaned_source_filenames = os.listdir(cleaned_source_path)
    cleaned_source_filepaths = [os.path.join(cleaned_source_path, source_filename) for source_filename in cleaned_source_filenames]

    all_cleaned_documents_filepaths = cleaned_suspicious_filepaths + cleaned_source_filepaths

    all_documents = []
    for document_filepath in all_cleaned_documents_filepaths:
        df_temp = pd.read_parquet(document_filepath)
        cleaned_sentences = df_temp["cleaned_sentence"]
        cleaned_document = " ".join(cleaned_sentences)
        all_documents.append(cleaned_document)

    tfidf_vectorizer.fit(all_documents)
    vocab = tfidf_vectorizer.get_feature_names_out()
    tfidf_weights = tfidf_vectorizer.transform(vocab).diagonal()
    token_to_tfidf_weight = {token: tfidf_weight for token, tfidf_weight in zip(vocab, tfidf_weights)}

    del tfidf_vectorizer, all_documents

    def normalize_vec(x): return x/np.linalg.norm(x)
    token_to_word_vector = {token: normalize_vec(ft.get_word_vector(token)) for token in vocab} 

    del ft

    # Now is time to go with the dataframes of sentences matches

    sent_matches_filenames = os.listdir(sent_matches_dir)
    counter = 0
    total_elements = len(sent_matches_filenames)
    for sent_matches_filename in sent_matches_filenames: 
        counter += 1
        printt(f"Calculating similarities for sentences of {sent_matches_filename}, [{counter}/{total_elements}]")
        sent_matches_filepath = os.path.join(sent_matches_dir, sent_matches_filename)
        df_sent_matches = pd.read_parquet(sent_matches_filepath)
        all_hybrid_similarities =[]
        for idx in tqdm(range(len(df_sent_matches))):
            row = df_sent_matches.iloc[idx]
            sus_sent_tokenized = word_tokenize(row["suspicious_cleaned_sentence"])
            src_sent_tokenized = word_tokenize(row["detected_source_cleaned_sentence"])
            join_matrix_vocab = list(set(sus_sent_tokenized + src_sent_tokenized))

            sus_semantic_matrix, sus_syntactic_matrix = create_syn_sem_matrices(token_to_tfidf_weight, token_to_word_vector, join_matrix_vocab, sus_sent_tokenized)
            src_semantic_matrix, src_syntactic_matrix = create_syn_sem_matrices(token_to_tfidf_weight, token_to_word_vector, join_matrix_vocab, src_sent_tokenized)

            similarity_semantic_value = similarity_semantic(sus_semantic_matrix, src_semantic_matrix)
            simimilarity_syntactic_value = similarity_syntactic(sus_syntactic_matrix, src_syntactic_matrix)

            hybrid_similarity = beta * similarity_semantic_value + (1 - beta) * simimilarity_syntactic_value

            all_hybrid_similarities.append(hybrid_similarity)

        df_sent_matches["hybrid_similarity"] = pd.Series(all_hybrid_similarities)

        df_sent_matches.to_parquet(sent_matches_filepath)