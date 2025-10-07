# Getting Started Guide

Welcome to the Data Science and Learning Analytics repository! This guide will help you get started with the examples and tutorials.

## Quick Start

### 1. Installation

First, make sure you have Python 3.7 or higher installed. Then, install the required packages:

```bash
pip install -r requirements.txt
```

### 2. Running Examples

Each module has its own directory with example scripts. Navigate to any module and run the scripts:

```bash
# Example: Running Pandas basics
cd 01_pandas_basics
python pandas_basics.py

# Example: Running data visualization
cd 06_data_visualization
python matplotlib_basics.py
```

## Module Overview

### Module 1: Pandas Basics
**Location:** `01_pandas_basics/`

Learn fundamental Pandas operations:
- `pandas_basics.py` - Introduction to Series and DataFrames
- `data_manipulation.py` - Merging, reshaping, and manipulating data
- `grouping_aggregation.py` - GroupBy operations and aggregations

**Run Time:** ~5 seconds per script

### Module 2: Parsing Data
**Location:** `02_parsing_data/`

Learn to work with different data formats:
- `csv_operations.py` - Reading and writing CSV files
- `json_operations.py` - Parsing and manipulating JSON data

**Run Time:** ~5 seconds per script
**Output:** Sample data files in `datasets/` directory

### Module 3: Data Cleansing
**Location:** `03_data_cleansing/`

Learn essential data cleaning techniques:
- `missing_values.py` - Handling missing data with various strategies
- `outliers.py` - Detecting and treating outliers

**Run Time:** ~10 seconds per script
**Output:** Visualization files in `outputs/` directory

### Module 4: Text Classification
**Location:** `04_text_classification/`

Learn text processing and classification:
- `text_preprocessing.py` - Text cleaning, tokenization, stemming, lemmatization
- `text_classification.py` - Building text classifiers with ML algorithms

**Run Time:** ~30 seconds per script (first run downloads NLTK data)

### Module 5: Sentiment Classification
**Location:** `05_sentiment_classification/`

Advanced sentiment analysis with feature engineering:
- `feature_engineering.py` - Creating features from text data
- `sentiment_classification.py` - Logistic regression for sentiment analysis

**Run Time:** ~30 seconds per script
**Output:** Model evaluation plots in `outputs/` directory

### Module 6: Data Visualization
**Location:** `06_data_visualization/`

Create professional visualizations:
- `matplotlib_basics.py` - Basic plotting with Matplotlib
- `seaborn_visualizations.py` - Statistical plots with Seaborn
- `advanced_visualizations.py` - Complex multi-plot figures and dashboards

**Run Time:** ~15 seconds per script
**Output:** Multiple visualization files in `outputs/` directory

## Directory Structure

```
Data-science-and-learning-analytics/
├── 01_pandas_basics/          # Pandas fundamentals
│   ├── README.md
│   ├── pandas_basics.py
│   ├── data_manipulation.py
│   └── grouping_aggregation.py
│
├── 02_parsing_data/            # CSV and JSON parsing
│   ├── README.md
│   ├── csv_operations.py
│   └── json_operations.py
│
├── 03_data_cleansing/          # Data cleaning techniques
│   ├── README.md
│   ├── missing_values.py
│   └── outliers.py
│
├── 04_text_classification/     # Text classification
│   ├── README.md
│   ├── text_preprocessing.py
│   └── text_classification.py
│
├── 05_sentiment_classification/ # Sentiment analysis
│   ├── README.md
│   ├── feature_engineering.py
│   └── sentiment_classification.py
│
├── 06_data_visualization/      # Visualization examples
│   ├── README.md
│   ├── matplotlib_basics.py
│   ├── seaborn_visualizations.py
│   └── advanced_visualizations.py
│
├── datasets/                   # Generated sample data
├── outputs/                    # Generated visualizations
├── requirements.txt            # Python dependencies
└── README.md                   # Main documentation
```

## Learning Path

For best results, follow this learning path:

1. **Week 1: Pandas Fundamentals**
   - Complete all scripts in `01_pandas_basics/`
   - Practice with your own datasets

2. **Week 2: Data I/O and Cleaning**
   - Work through `02_parsing_data/`
   - Learn data cleaning in `03_data_cleansing/`

3. **Week 3: Text Processing**
   - Master text preprocessing in `04_text_classification/`
   - Build your first text classifier

4. **Week 4: Advanced Text Analysis**
   - Feature engineering in `05_sentiment_classification/`
   - Build sentiment analysis models

5. **Week 5: Visualization**
   - Create visualizations in `06_data_visualization/`
   - Build comprehensive dashboards

## Tips for Success

1. **Run Examples First:** Always run the example scripts before modifying them
2. **Read the Output:** Pay attention to what the scripts print to understand the concepts
3. **Experiment:** Modify the code and see what happens
4. **Check Outputs:** Look at generated files in `datasets/` and `outputs/` directories
5. **Read Module READMEs:** Each module has a README with detailed explanations

## Troubleshooting

### NLTK Download Issues
If you encounter NLTK-related errors, run:
```python
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
```

### Missing Packages
If you get import errors, install missing packages:
```bash
pip install package_name
```

### Permission Errors
If you can't write to the `outputs/` or `datasets/` directories, they will be created automatically when you run the scripts.

## Next Steps

After completing all modules:
1. Apply concepts to your own datasets
2. Combine techniques from different modules
3. Build end-to-end data science projects
4. Explore advanced topics in each area

## Getting Help

- Read the module README files for detailed explanations
- Check the inline comments in the code
- Review the example outputs
- Experiment with the code to understand behavior

Happy Learning! 🚀
