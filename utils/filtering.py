# Standard library imports
import os
from multiprocessing import Pool

# Third-party library imports
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import word_tokenize
from tqdm import tqdm

# Local imports
from utils.data_load_extract import (read_dataframe,read_files_from_directory)
from utils.machine_learning import transform_in_parallel
from utils.general import printt, N_PROCESSES

def filter_documents(
    i, suspicious_filename, suspicious_matrix, common_tokens_matrix, source_filenames
):
    """
    Filters potential source documents based on common tokens.

    Parameters:
    - i (int): Index of the suspicious document.
    - suspicious_filename (str): Filename of the suspicious document.
    - suspicious_matrix (ndarray): Matrix representation of the suspicious document.
    - common_tokens_matrix (ndarray): Matrix of common tokens between suspicious and source documents.
    - source_filenames (list): List of source filenames.

    Returns:
    - str: Suspicious filename.
    - list: Potential source filenames.
    """
    suspicious_len = suspicious_matrix[i].sum()
    threshold = suspicious_len / 3  # At least 1/3 common tokens

    # Find source documents that meet the threshold
    potential_sources = []
    for j in range(len(source_filenames)):
        if common_tokens_matrix[i, j] >= threshold:
            potential_sources.append(source_filenames[j])

    return suspicious_filename, potential_sources

def document_filtering(cleaned_suspicious_path, cleaned_source_path):
    # Initialize a binary CountVectorizer for tokenizing documents
    vectorizer = CountVectorizer(binary=True, tokenizer=word_tokenize)

    # Load preprocessed suspicious and source documents
    suspicious_docs, suspicious_filenames = read_files_from_directory(
        cleaned_suspicious_path, filetype=".parquet"
    )
    source_docs, source_filenames = read_files_from_directory(
        cleaned_source_path, filetype=".parquet"
    )

    # Convert documents to binary matrix representations
    suspicious_matrix = vectorizer.fit_transform(suspicious_docs)
    source_matrix = transform_in_parallel(source_docs, vectorizer, N_PROCESSES)

    # Calculate common tokens between suspicious and source documents
    printt("Starting common_tokens_matrix multiplication ")
    common_tokens_matrix = np.dot(suspicious_matrix, source_matrix.T)

    # Use multi-processing to parallelize document filtering tasks for better performance
    printt("Start filter index location of documents")
    with Pool(processes=N_PROCESSES) as pool:
        results = pool.starmap(
            filter_documents,
            [
                (
                    i,
                    suspicious_filename,
                    suspicious_matrix,
                    common_tokens_matrix,
                    source_filenames,
                )
                for i, suspicious_filename in enumerate(suspicious_filenames)
            ],
        )

    # Create a dictionary mapping suspicious documents to their potential source documents
    suspicious_to_sources = {
        suspicious_filename: potential_sources
        for suspicious_filename, potential_sources in results
    }

    return suspicious_to_sources


def filter_sentences(
    cleaned_source_path,
    source_filename,
    source_sparse_matrices,
    suspicious_sentences_matrix,
    df_suspicious,
    suspicious_filename,
):
    detections = []
    source_filepath = os.path.join(cleaned_source_path, source_filename)
    df_source = read_dataframe(source_filepath)
    source_sentences_matrix = source_sparse_matrices[source_filename]
    common_tokens_matrix = suspicious_sentences_matrix.dot(
        source_sentences_matrix.T
    ).tocoo()
    # Using coo_matrix to find pairs of sentences sharing at least 5 tokens
    # The index i is for the suspicious_sentences indexing
    # The index j is for teh source_sentences indexing

    for i, j, v in zip(
        common_tokens_matrix.row, common_tokens_matrix.col, common_tokens_matrix.data
    ):
        if v >= 5:
            detected_suspicious_offset = df_suspicious["offset"][i]
            detected_suspicious_length = df_suspicious["length"][i]
            detected_suspicious_cleaned_sentence = df_suspicious["cleaned_sentence"][i]
            detected_source_offset = df_source["offset"][j]
            detected_source_length = df_source["length"][j]
            detected_source_cleaned_sentence = df_source["cleaned_sentence"][j]
            new_register = [
                suspicious_filename,
                source_filename,
                detected_suspicious_offset,
                detected_suspicious_length,
                detected_source_offset,
                detected_source_length,
                detected_suspicious_cleaned_sentence,
                detected_source_cleaned_sentence,
            ]
            detections.append(new_register)

    return detections

def sentence_filtering(
    cleaned_suspicious_path, cleaned_source_path, suspicious_to_sources
):
    vectorizer = CountVectorizer(binary=True, tokenizer=word_tokenize)

    printt("Fitting vectorizer for sentences filtering")
    all_suspicious_sentences = []
    for suspicious_filename in suspicious_to_sources.keys():
        suspicious_filepath = os.path.join(cleaned_suspicious_path, suspicious_filename)
        df_suspicious = read_dataframe(suspicious_filepath)
        suspicious_sentences = df_suspicious["cleaned_sentence"]
        all_suspicious_sentences.extend(suspicious_sentences)

    vectorizer.fit(all_suspicious_sentences)

    printt("Calculating sparse matrices of sentences from source documents")
    reduced_source_filenames = []
    for potential_sources in suspicious_to_sources.values():
        reduced_source_filenames.extend(potential_sources)
    reduced_source_filenames = list(set(reduced_source_filenames))

    source_sparse_matrices = {}
    for source_filename in tqdm(
        reduced_source_filenames, total=len(reduced_source_filenames), unit="file"
    ):
        source_filepath = os.path.join(cleaned_source_path, source_filename)
        df_source = read_dataframe(source_filepath)
        source_sentences = df_source["cleaned_sentence"]
        source_sentences_matrix = transform_in_parallel(
            source_sentences, vectorizer, N_PROCESSES
        )
        source_sparse_matrices[source_filename] = source_sentences_matrix

    printt("Matrix multiplication and matching value index location")
    suspicious_filenames = list(suspicious_to_sources.keys())
    all_detections = []
    for suspicious_filename in tqdm(
        suspicious_filenames, total=len(suspicious_filenames), unit="file"
    ):
        suspicious_filepath = os.path.join(cleaned_suspicious_path, suspicious_filename)
        df_suspicious = read_dataframe(suspicious_filepath)
        suspicious_sentences = df_suspicious["cleaned_sentence"]
        suspicious_sentences_matrix = transform_in_parallel(
            suspicious_sentences, vectorizer, N_PROCESSES
        )

        potential_sources = suspicious_to_sources[suspicious_filename]
        arguments = [
            (
                cleaned_source_path,
                source_filename,
                source_sparse_matrices,
                suspicious_sentences_matrix,
                df_suspicious,
                suspicious_filename,
            )
            for source_filename in potential_sources
        ]
        # Use multi-processing to parallelize sentence filtering tasks
        with Pool(processes=N_PROCESSES) as pool:
            detections_for_this_suspicious_filename = pool.starmap(
                filter_sentences, arguments
            )

        flattened_detections = [
            subsublist
            for sublist in detections_for_this_suspicious_filename
            for subsublist in sublist
        ]
        all_detections.extend(flattened_detections)

    # Convert detections into a DataFrame
    cols = [
        "suspicious_filename",
        "source_filename",
        "suspicious_offset",
        "suspicious_length",
        "source_offset",
        "source_length",
        "suspicious_cleaned_sentence",
        "detected_source_cleaned_sentence",
    ]
    df_sentences_detections = pd.DataFrame(data=all_detections, columns=cols)

    df_sentences_detections["suspicious_filename"] = df_sentences_detections[
        "suspicious_filename"
    ].str.replace(".parquet", ".txt")
    df_sentences_detections["source_filename"] = df_sentences_detections[
        "source_filename"
    ].str.replace(".parquet", ".txt")

    return df_sentences_detections