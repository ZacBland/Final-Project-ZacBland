#!/bin/bash
# Save all training/test artifacts from an experiment to models/<name>/

# Ask for experiment name
read -p "Experiment name: " name

if [ -z "$name" ]; then
    echo "Error: no name provided."
    exit 1
fi

dest="models/$name"

if [ -d "$dest" ]; then
    echo "Error: '$dest' already exists."
    exit 1
fi

mkdir -p "$dest"

# Copy checkpoints
if [ -d "checkpoints" ]; then
    cp -r checkpoints/ "$dest/checkpoints/"
    echo "  Saved checkpoints/"
fi

# Copy plots
if [ -d "plots" ]; then
    cp -r plots/ "$dest/plots/"
    echo "  Saved plots/"
fi

# Copy training log
if [ -f "train.log" ]; then
    cp train.log "$dest/train.log"
    echo "  Saved train.log"
fi

# Copy test log if it exists
if [ -f "test.log" ]; then
    cp test.log "$dest/test.log"
    echo "  Saved test.log"
fi

# Snapshot current training config for reference
if [ -f "Code/train.py" ]; then
    head -60 Code/train.py > "$dest/train_config_snapshot.txt"
    echo "  Saved train config snapshot"
fi

# Copy predictions CSV if it exists
for csv in predictions*.csv; do
    if [ -f "$csv" ]; then
        cp "$csv" "$dest/$csv"
        echo "  Saved $csv"
    fi
done

echo ""
echo "Experiment saved to: $dest/"
