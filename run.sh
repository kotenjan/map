#!/bin/bash

# Initial values
current_epoch=0
num_epochs=$((current_epoch + 2))
dir_index=$((current_epoch % 4))
resize=96

# Array of directories
dirs=("0" "1" "2" "3")

# Loop through the epochs
while [ $current_epoch -lt 2000 ]; do
    # Run the Python script with the current directory and epoch values
    echo "Running main.py with arguments: ${dirs[$dir_index]} $current_epoch $num_epochs $resize"

    python main.py ${dirs[$dir_index]} $current_epoch $num_epochs $resize

    # Update epochs
    let current_epoch+=2
    let num_epochs+=2

    if [ $((current_epoch % 4)) -eq 0 ]; then
        let dir_index+=1
        let dir_index%=4
    fi
    if [ $((current_epoch % 16)) -eq 0 ]; then
        let resize+=32
        if [ $resize -gt 256 ]; then
            resize=256
        fi
    fi
done
