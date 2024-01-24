# Standard library imports
import os
import time

# Local imports
from utils.general import printt, print_sucess, N_PROCESSES
from utils.data_load_extract import (
    ensure_directory_exists_and_is_empty,
    extract_sources_from_suspicious_xml
    )
from utils.preprocessing import preprocessing_data
from utils.filtering import document_filtering, sentence_filtering
from utils.metrics import similarity_computation, evaluate_detections

def main():
    # 1. Initial setup (paths and number of processes for parallelization)
    printt("Starting the algorithm...")
    # Getting the current directory path and setting up the base path for data.
    current_dir = os.getcwd()
    base_data_path = os.path.join(current_dir, "pan-plagiarism-corpus-2011/external-detection-corpus")
    sent_matches_dir = os.path.join(current_dir, "sent-matches-dir")
    # Print the number of processes that will be used for parallel operations.
    printt(f"N_PROCESSES: {N_PROCESSES}")

    # 2. Directory setup
    printt("Setting up directories...")
    # Initialize the directories for raw and cleaned versions of both source and suspicious documents.
    # Set up paths for raw and cleaned source documents.
    cleaned_source_path = os.path.join(current_dir, "cleaned-source-documents")
    source_path = os.path.join(base_data_path, "source-document")
    # Set up paths for raw and cleaned suspicious documents.
    cleaned_suspicious_path = os.path.join(current_dir, "cleaned-suspicious-documents")
    suspicious_path = os.path.join(base_data_path, "suspicious-document")
    if reload_sources:
        ensure_directory_exists_and_is_empty(cleaned_source_path)
    if reload_suspicious:
        ensure_directory_exists_and_is_empty(cleaned_suspicious_path)
    if reload_sent_matches:
        ensure_directory_exists_and_is_empty(sent_matches_dir)

    # Step 3: Preprocess the raw data to prepare it for the subsequent analysis
    printt("Preprocessing data...")
    printt("Source documents preprocessing...")
    if reload_sources:
        preprocessing_data(source_path, cleaned_source_path, SOURCE_LIMIT)
    printt("Suspicious documents preprocessing...")
    if reload_suspicious:
        preprocessing_data(suspicious_path, cleaned_suspicious_path, SUSPICIOUS_LIMIT)

    # 4. Extract real results from XML files to understand the ground truth of plagiarism references
    printt("Extracting reference results...")
    # Extract and store the plagiarism references from the suspicious XML files
    df_references = extract_sources_from_suspicious_xml(suspicious_path)
    suspicious_filenames = [filename.replace("parquet", "txt") for filename in os.listdir(cleaned_suspicious_path)]
    source_filenames = [filename.replace("parquet", "txt") for filename in os.listdir(cleaned_source_path)]
    df_references = df_references[df_references["suspicious_filename"].isin(suspicious_filenames)].reset_index(drop=True)

    # Step 5: Identify potential source documents for each suspicious document
    printt("Filtering documents...")
    suspicious_to_sources = document_filtering(cleaned_suspicious_path, cleaned_source_path)

    # Step 6: Filter sentences based on potential plagiarisms
    printt("Filtering sentences...")
    if reload_sent_matches:
        sentence_filtering(cleaned_suspicious_path, cleaned_source_path, suspicious_to_sources, sent_matches_dir)

    # Step 7: Compute and save similarity scores between suspicious and source sentences
    printt("Processing similarity calculations...")
    output_dir = os.path.join(current_dir, "sentences-detections")
    similarity_computation(cleaned_suspicious_path, cleaned_source_path, sent_matches_dir, BETA)

    # Step 8. Evaluate detections
    printt("Evaluate detections against ground truth")
    printt(f"{SOURCE_LIMIT=}, {SUSPICIOUS_LIMIT=}, {THRESHOLD=}, {BETA=}")
    precision, recall, f1_score, plagdet = evaluate_detections(sent_matches_dir, THRESHOLD, df_references)

    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1_score:.4f}")
    print(f"Plagdet-Score: {plagdet:.4f}")

    # 9. Save results
    printt(f"Saving results to {results_filename}")
    with open(results_filename, "a") as f:
        f.write(f"{SUSPICIOUS_LIMIT}, {THRESHOLD}, {BETA}, {precision}, {recall}, {f1_score}, {plagdet}\n")

if __name__ == "__main__":
    reload_sources = True
    reload_suspicious = True
    reload_sent_matches = True
    results_filename = "data_for_figure_20.txt"
    SOURCE_LIMIT = 64
    SUSPICIOUS_LIMIT = 8
    THRESHOLD_LIST = [0.65] # [0.4, 0.45, 0.5, 0.55, 0.60, 0.65, 0.70]
    BETA = 0.5

    start_time = time.time()
    for THRESHOLD in THRESHOLD_LIST:
        main()

    end_time = time.time()
    diff_time = end_time - start_time
    print(f"Time taken for Figure 20 replication: {diff_time:.2f} seconds")

    print_sucess()