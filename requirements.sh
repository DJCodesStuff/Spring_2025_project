#!/bin/bash

# List of packages to install
packages=(
    "numpy==1.24.2"
    "pandas"
    "torch"
    "wandb"
    "transformers"
    "peft"
    "datasets"
    "scikit-learn"
    "ollama"
    "textacy"
    "kagglehub"
    "cython"
    "murmurhash"
    "cymem"
    "preshed"
    "thinc"
    "blis"
    "spacy"
    "scikit-learn"
)

# Loop through each package and install it
for package in "${packages[@]}"; do
    echo "Installing $package..."
    python3 -m pip install "$package"
done

echo "All packages installed!"
