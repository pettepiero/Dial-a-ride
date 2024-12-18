#!/bin/bash

# Directory to check (default is the current directory)
DIR=${1:-.}

# Iterate over all files matching the pattern
for FILE in "$DIR"/pr*; do
    echo "Reading file $FILE"
    if [[ -f "$FILE" ]]; then
        # Read the last number from the first line of the file
        LAST_NUMBER=$(head -n 1 "$FILE" | awk '{print $NF}' | tr -d '[:space:]')
        echo "Last number for file $FILE is $LAST_NUMBER"

        # Validate that LAST_NUMBER is a positive integer
        if ! [[ "$LAST_NUMBER" =~ ^[0-9]+$ ]]; then
            echo "File $FILE has an invalid last number in the first line: $LAST_NUMBER"
            exit 1
        fi

        # Extract the first $LAST_NUMBER rows (excluding the first line)
        ROWS=$(sed -n "2,$((LAST_NUMBER+1))p" "$FILE")

        # Check if all rows in the range are identical
        FIRST_ROW=$(echo "$ROWS" | head -n 1)
        DIFFERENT_ROW=$(echo "$ROWS" | grep -vF "$FIRST_ROW")

        if [[ -n "$DIFFERENT_ROW" ]]; then
            echo "File $FILE has non-identical rows within the first $LAST_NUMBER rows."
            exit 1
        fi
    fi
done

echo "All files have identical rows in the first $LAST_NUMBER rows within each file."

