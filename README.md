## Dream Generator with Markov Chains

A program for generating dreams based on a corpus of existing texts using Markov chains. Implemented in pure Python, without the use of library components.

## 📋 Description

The generator analyzes the structure of uploaded dream descriptions and creates new ones by combining words and phrases based on statistical phenomena. The more source data, the more coherent the results.

## 🚀 Features

- Dream generation based on a text corpus
- Customizable Markov chain order (analysis depth)
- Seed word generation support
- Interactive experimentation mode
- Model statistics
- Adjustable minimum and maximum dream description lengths

## 🔧 Requirements

- Python 3.6 or higher
- Only standard Python libraries (random, re, collections, argparse)

## 📥 Installation

1. Clone the repository or copy the `dreams_generator.py` file
2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate # For Linux/Mac
venv\Scripts\activate # For Windows
