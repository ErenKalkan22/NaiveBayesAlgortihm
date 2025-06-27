# Naive Bayes Sentiment Analysis Project

A machine learning web application that performs sentiment analysis on movie reviews using the Naive Bayes algorithm. The application classifies user input text as either positive or negative sentiment.

## Project Overview

This project implements a sentiment analysis system using:
- **Naive Bayes** classification algorithm
- **TF-IDF** vectorization for text processing
- **Flask** web framework for the user interface
- **IMDB Movie Reviews** dataset for training

## Project Structure

```
NaiveBayes/
├── main.py              # Flask web application
├── train.py             # Model training script
├── cv.pkl               # Trained TF-IDF vectorizer
├── review.pkl           # Trained Naive Bayes model
├── IMDB Dataset.csv     # Training dataset
├── templates/
│   └── index.html       # Web interface template
└── input/               # Input directory
```

## Features

- **Real-time sentiment analysis** of text input
- **Web-based interface** for easy interaction
- **Pre-trained model** ready for immediate use
- **Visualization** of training progress (accuracy over epochs)
- **High accuracy** sentiment classification

## Installation

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Required Dependencies

```bash
pip install flask pandas scikit-learn numpy matplotlib
```
### Required Dependencies
dataset:IMDB Dataset of 50K Movie Reviews

### Alternative Installation

If you have a `requirements.txt` file:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Web Application

1. Navigate to the NaiveBayes directory:
```bash
cd NaiveBayes/
```

2. Start the Flask application:
```bash
python main.py
```

3. Open your web browser and go to:
```
http://localhost:5000
```

4. Enter a movie review or any text in the input field and click "Predict" to get the sentiment analysis result.

### Training the Model (Optional)

If you want to retrain the model with your own data:

1. Ensure you have the IMDB dataset in the correct location
2. Run the training script:
```bash
python train.py
```

This will:
- Load and preprocess the IMDB dataset
- Train the Naive Bayes model
- Generate accuracy visualizations
- Save the trained model and vectorizer as pickle files

## How It Works

### 1. Data Preprocessing
- Loads the IMDB movie reviews dataset
- Maps sentiment labels: 'positive' → 1, 'negative' → 0
- Splits data into training (80%) and testing (20%) sets

### 2. Text Vectorization
- Uses TF-IDF (Term Frequency-Inverse Document Frequency) vectorization
- Removes English stop words
- Limits features to 5000 most important terms

### 3. Model Training
- Implements Multinomial Naive Bayes classifier
- Uses incremental learning with `partial_fit()` for epoch-based training
- Optimizes with alpha parameter (α = 0.1) for better generalization

### 4. Web Interface
- Flask-based web application
- Clean, responsive HTML interface
- Real-time prediction display

## Model Performance

The model is trained with the following configuration:
- **Algorithm**: Multinomial Naive Bayes
- **Vectorization**: TF-IDF with 5000 features
- **Training Epochs**: 10,000
- **Alpha Parameter**: 0.1 (for smoothing)
- **Dataset**: IMDB Movie Reviews (50,000 reviews)

## API Endpoints

### GET `/`
Returns the main web interface.

### POST `/predict`
Accepts form data with a 'message' field and returns sentiment prediction.

**Request Format**:
```
Content-Type: application/x-www-form-urlencoded
message=Your review text here
```

**Response**: HTML page with prediction result

## Example Predictions

**Positive Examples**:
- "This movie is absolutely fantastic! Great acting and storyline."
- "I loved every minute of it. Highly recommended!"

**Negative Examples**:
- "Terrible movie, waste of time and money."
- "Poor acting and boring plot. Very disappointed."

## File Descriptions

- **`main.py`**: Flask web application with prediction endpoints
- **`train.py`**: Complete training pipeline with visualization
- **`cv.pkl`**: Serialized TF-IDF vectorizer
- **`review.pkl`**: Serialized Naive Bayes model
- **`index.html`**: Web interface template with modern styling

## Troubleshooting

### Common Issues

1. **FileNotFoundError for pickle files**:
   - Ensure you're running the script from the NaiveBayes directory
   - The code now uses absolute paths to prevent this issue

2. **scikit-learn version warnings**:
   - These are compatibility warnings and don't affect functionality
   - Consider updating scikit-learn: `pip install --upgrade scikit-learn`

3. **Dataset path issues**:
   - Update the file path in `train.py` to match your dataset location

## Future Improvements

- Add support for batch predictions
- Implement confidence scores for predictions
- Add more preprocessing options
- Support for multiple languages
- Model comparison with other algorithms
- REST API endpoints for integration

## Authors

- Efe Eren Yağcı
- Eren Kalkan

## License

This project is for educational purposes as part of Algorithm Analysis coursework. 
