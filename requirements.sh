#!/bin/bash

# List of packages to install
packages=(
    "numpy>=2.0.0,<3.0.0"
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
    "sklearn"
)

# Loop through each package and install it
for package in "${packages[@]}"; do
    echo "Installing $package..."
    python3 -m pip install "$package"
done

echo "All packages installed!"
