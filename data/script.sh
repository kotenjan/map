#!/bin/bash

# Directory containing the images
source_dir="val"

# Directories to distribute the images
target_dir1="val_0"
target_dir2="val_1"
target_dir3="val_2"
target_dir4="val_3"

# Create target directories if they don't exist
mkdir -p "$target_dir1" "$target_dir2" "$target_dir3" "$target_dir4"

# Get all image files from source directory
files=($(find "$source_dir" -type f -name "*.jpg" -o -name "*.png" -o -name "*.jpeg"))

# Shuffle the file list
shuffled_files=($(shuf -e "${files[@]}"))

# Calculate the number of files to distribute to each directory
num_files=${#shuffled_files[@]}
files_per_dir=$((num_files / 4))
remainder=$((num_files % 4))

# Split and move files
for (( i = 0; i < num_files; i++ )); do
    if (( i < files_per_dir + (i < remainder ? 1 : 0) )); then
        cp "${shuffled_files[$i]}" "$target_dir1"
    elif (( i < (files_per_dir * 2) + (i < remainder * 2 ? 1 : 0) )); then
        cp "${shuffled_files[$i]}" "$target_dir2"
    elif (( i < (files_per_dir * 3) + (i < remainder * 3 ? 1 : 0) )); then
        cp "${shuffled_files[$i]}" "$target_dir3"
    else
        cp "${shuffled_files[$i]}" "$target_dir4"
    fi
done
