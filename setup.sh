#!/bin/bash

# Update and Upgrade Packages
sudo apt update && sudo apt upgrade -y

# Install python3-pip
sudo apt install -y python3-pip

# Install speedtest-cli
sudo apt install -y speedtest-cli

# Install aria2
sudo apt-get install -y aria2

# Install unrar
sudo apt-get install -y unrar

# Run Speedtest
speedtest-cli --simple

# Download Files
aria2c -x 16 https://zenodo.org/record/3250095/files/pan-plagiarism-corpus-2011.part1.rar
aria2c -x 16 https://zenodo.org/record/3250095/files/pan-plagiarism-corpus-2011.part2.rar

# Unrar Files
unrar x "pan-plagiarism-corpus-2011.part1.rar"

# Install Python Requirements
pip install -r "requirements.txt"

# Run Python Scripts
python3 downloads_nltk.py
python3 download_fasttext.py
python3 --version
