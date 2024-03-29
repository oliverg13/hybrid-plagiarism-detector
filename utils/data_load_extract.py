# Standard library imports
import os
import shutil
import xml.etree.ElementTree as ET

# Third-party imports
import pandas as pd
import fasttext

# Warnings Filtering
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
fasttext.FastText.eprint = lambda x: None  # Suppress fasttext's warning outputs

def ensure_directory_exists_and_is_empty(path):
    # If directory exists, delete it and all its contents
    if os.path.exists(path):
        shutil.rmtree(path)

    # Create the directory
    os.makedirs(path)

def collect_filepaths(directory, filetype=".txt"):
    filepaths = []  # Store the paths of files

    # Filter and sort subfolders
    subfolders = sorted(
        item
        for item in os.listdir(directory)
        if os.path.isdir(os.path.join(directory, item))
    )

    # Iterate through subfolders
    for folder_name in subfolders:
        folder_path = os.path.join(directory, folder_name)

        # Collect file paths of the specified filetype
        for filename in os.listdir(folder_path):
            filepath = os.path.join(folder_path, filename)
            if os.path.isfile(filepath) and filename.endswith(filetype):
                filepaths.append(filepath)
                # filepaths.append(filename)

    return filepaths

def extract_source_references(filepath):
    tree = ET.parse(filepath)
    root = tree.getroot()
    filename = os.path.basename(filepath)[:-3] + "txt"

    plagiarism_information = []

    features = root.findall(".//feature[@name='plagiarism']")
    for plagiarism in features:
        source_reference = plagiarism.attrib["source_reference"]
        suspicious_offset = int(plagiarism.attrib["this_offset"])
        suspicious_length = int(plagiarism.attrib["this_length"])
        source_offset = int(plagiarism.attrib["source_offset"])
        source_length = int(plagiarism.attrib["source_length"])

        plagiarism_information.append(
            [
                filename,
                source_reference,
                suspicious_offset,
                suspicious_length,
                source_offset,
                source_length,
            ]
        )

    return plagiarism_information

def read_files_from_directory(directory_path, filetype=".parquet"):
    """
    Reads files of a specified type from the provided directory.

    Parameters:
    - directory_path (str): Path to the directory.
    - filetype (str, optional): Type of files to read. Defaults to ".parquet".

    Returns:
    - list: List of concatenated `cleaned_sentence` from each file.
    - list: List of filenames.
    """
    docs = []
    filenames = [
        filename
        for filename in os.listdir(directory_path)
        if filename.endswith(filetype)
    ]
    for filename in filenames:
        filepath = os.path.join(directory_path, filename)
        
        df = read_dataframe(filepath)

        cleaned_sentences = df["cleaned_sentence"]
        doc = " ".join(cleaned_sentences)  # + " "
        docs.append(doc)
    return docs, filenames

def read_dataframe(path: str):
    if path.endswith(".csv"):
        df = pd.read_csv(path)
    elif path.endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported filetype")
    return df

def extract_sources_from_suspicious_xml(suspicious_path):
    # Collect all XML file paths from the given directory
    filepaths = collect_filepaths(suspicious_path, filetype=".xml")

    # Extract plagiarism information from each XML file
    plagiarism_information_data = []
    for filepath in filepaths:
        plagiarism_information = extract_source_references(filepath)
        if plagiarism_information:
            plagiarism_information_data.extend(plagiarism_information)

    # Convert the extracted information to a DataFrame
    cols = [
        "suspicious_filename",
        "source_filename",
        "suspicious_offset",
        "suspicious_length",
        "source_offset",
        "source_length",
    ]
    df_plagiarism_references = pd.DataFrame(plagiarism_information_data, columns=cols)

    return df_plagiarism_references
