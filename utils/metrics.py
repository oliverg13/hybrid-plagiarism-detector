# Standard library imports
import os
from multiprocessing import Pool

# Third-party library imports
import numpy as np
import pandas as pd
from tqdm import tqdm
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Local imports
from utils.general import printt, N_PROCESSES
from utils.data_load_extract import load_data_and_models
from utils.machine_learning import transform_in_parallel, train_tfidf_vectorizer, get_word_weights_and_vectors

def similarity_semantic(S1, S2):
    numerator = 2 * np.sum(S1.multiply(S2), axis=1)
    denominator = np.sum(S1.multiply(S1), axis=1) + np.sum(S2.multiply(S2), axis=1)
    return numerator / denominator

def similarity_syntactic(S1, S2):
    diff = S1 - S2
    summ = S1 + S2
    numerator = np.sqrt(np.sum(diff.multiply(diff), axis=1))
    denominator = np.sqrt(np.sum(summ.multiply(summ), axis=1))
    return 1 - numerator / denominator

def overlap(a1, a2, b1, b2):
    """Check if two intervals (a1, a2) and (b1, b2) overlap."""
    return max(0, min(a2, b2) - max(a1, b1)) > 0

def evaluate_detection(df_references, df_detections):
    TP = 0
    FP = 0
    FN = 0

    detected_cases_count = (
        {}
    )  # dictionary to store count of detections for each reference
    for idx, detection in tqdm(
        df_detections.iterrows(), total=len(df_detections), unit="row"
    ):
        suspicious_filename = detection["suspicious_filename"]
        source_filename = detection["source_filename"]
        suspicious_offset = detection["suspicious_offset"]
        suspicious_length = detection["suspicious_length"]
        source_offset = detection["source_offset"]
        source_length = detection["source_length"]

        matching_rows = df_references[
            (df_references["suspicious_filename"] == suspicious_filename)
            & (df_references["source_filename"] == source_filename)
        ]

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
            ref_key = (
                suspicious_filename,
                source_filename,
            )  # Use file pair as a unique key
            detected_cases_count[ref_key] = detected_cases_count.get(ref_key, 0) + 1
        else:
            FP += 1

    FN = len(df_references) - TP

    sum_Rs = sum(detected_cases_count.values())
    abs_SR = len(detected_cases_count)
    granularity = sum_Rs / abs_SR if abs_SR != 0 else 0

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


def process_row(detection, df_references_dict):
    # Convert dictionary back to DataFrame
    df_references = pd.DataFrame.from_dict(df_references_dict)

    suspicious_filename = detection["suspicious_filename"]
    source_filename = detection["source_filename"]
    suspicious_offset = detection["suspicious_offset"]
    suspicious_length = detection["suspicious_length"]
    source_offset = detection["source_offset"]
    source_length = detection["source_length"]

    matching_rows = df_references[
        (df_references["suspicious_filename"] == suspicious_filename)
        & (df_references["source_filename"] == source_filename)
    ]

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

    return overlap_found, (suspicious_filename, source_filename)


def evaluate_detection_multiprocessing(df_references, df_detections):
    TP = 0
    FP = 0
    detected_cases_count = {}

    # Convert DataFrame to dictionary for serialization
    df_references_dict = df_references.to_dict("list")

    detections_list = df_detections.to_dict("records")

    with Pool(N_PROCESSES) as pool:
        results = pool.starmap(
            process_row,
            [(detection, df_references_dict) for detection in detections_list],
        )

    for overlap_found, ref_key in results:
        if overlap_found:
            TP += 1
            detected_cases_count[ref_key] = detected_cases_count.get(ref_key, 0) + 1
        else:
            FP += 1

    FN = len(df_references) - TP

    sum_Rs = sum(detected_cases_count.values())
    abs_SR = len(detected_cases_count)
    granularity = sum_Rs / abs_SR if abs_SR != 0 else 0

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

def similarity_computation(cleaned_suspicious_path, cleaned_source_path, output_dir, BETA):
    # Load data and pre-trained models
    printt("Loading FastText and data")
    ft, suspicious_docs, source_docs = load_data_and_models(
        cleaned_suspicious_path, cleaned_source_path
    )

    # Train the TF-IDF vectorizer and retrieve the common vocabulary
    printt("Train the TF-IDF vectorizer and retrieve the common vocabulary")
    tfidf_vectorizer, common_vocabulary = train_tfidf_vectorizer(
        suspicious_docs, source_docs
    )

    # Get TF-IDF weights and word vectors for the common vocabulary
    printt("Get TF-IDF weights and word vectors for the common vocabulary")
    tfidf_weights, sum_word_vectors = get_word_weights_and_vectors(ft, tfidf_vectorizer)

    # Initialize the binary count vectorizer with the shared vocabulary
    count_vectorizer = CountVectorizer(
        binary=True, vocabulary=common_vocabulary, tokenizer=word_tokenize
    )

    del ft, suspicious_docs, source_docs, tfidf_vectorizer, common_vocabulary

    df_filenames = os.listdir(output_dir)
    df_filepaths = [
        os.path.join(output_dir, df_filename) for df_filename in df_filenames
    ]

    for df_filepath in tqdm(df_filepaths, unit="file"):
        df = pd.read_parquet(df_filepath)

        suspicious_sentences = df["suspicious_cleaned_sentence"]
        source_sentences = df["detected_source_cleaned_sentence"]

        suspicious_count = transform_in_parallel(
            suspicious_sentences, count_vectorizer, N_PROCESSES
        )
        source_count = transform_in_parallel(
            source_sentences, count_vectorizer, N_PROCESSES
        )

        S1_syntactic = suspicious_count.multiply(tfidf_weights)
        S2_syntactic = source_count.multiply(tfidf_weights)

        S1_semantic = suspicious_count.multiply(sum_word_vectors)
        S2_semantic = source_count.multiply(sum_word_vectors)

        sim_syntactic = similarity_syntactic(S1_syntactic, S2_syntactic)
        sim_semantic = similarity_semantic(S1_semantic, S2_semantic)

        hybrid_similarity = BETA * sim_semantic + (1 - BETA) * sim_syntactic

        df["hybrid_similarity"] = hybrid_similarity

        df.to_parquet(df_filepath)

    df_list = []
    for df_filepath in df_filepaths:
        df = pd.read_parquet(df_filepath)
        df_list.append(df)

    df_similarity = pd.concat(df_list)

    return df_similarity

