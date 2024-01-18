# Standard library imports
import os
import time

# Third-party imports
import numpy as np
import pandas as pd
from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import fasttext

# Local application imports
from utils.shortcuts import printt

# Warnings Filtering
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
fasttext.FastText.eprint = lambda x: None  # Suppress fasttext's warning outputs