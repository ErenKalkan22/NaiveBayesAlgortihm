import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Loading the dataset
file_path = '/Users/erenkalkan/PycharmProjects/NaiveBayes/IMDB Dataset.csv'
data = pd.read_csv(file_path)

# Cleaning and preparing the data
data['sentiment'] = data['sentiment'].map({'positive': 1, 'negative': 0})
reviews = data['review']
labels = data['sentiment']

# Splitting data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(reviews, labels, test_size=0.2, random_state=42)

# Text vectorization (TF-IDF)
tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_vect = tfidf.fit_transform(X_train)
X_test_vect = tfidf.transform(X_test)

# Training the model and epoch-based accuracy graph
epochs = 10000 # Try more epochs
train_accuracies = []
test_accuracies = []

model = MultinomialNB(alpha=0.1)  # Playing with alpha (smaller alpha, better generalization)

for epoch in range(1, epochs + 1):
    model.partial_fit(X_train_vect, y_train, classes=[0, 1])
    train_pred = model.predict(X_train_vect)
    test_pred = model.predict(X_test_vect)

    train_accuracy = accuracy_score(y_train, train_pred)
    test_accuracy = accuracy_score(y_test, test_pred)

    train_accuracies.append(train_accuracy)
    test_accuracies.append(test_accuracy)

    print(f"Epoch {epoch}: Train Accuracy: {train_accuracy:.2f}, Test Accuracy: {test_accuracy:.2f}")

# Creating the graph
plt.figure(figsize=(10, 6))
plt.plot(range(1, epochs + 1), train_accuracies, label='Train Accuracy', marker='o')
plt.plot(range(1, epochs + 1), test_accuracies, label='Test Accuracy', marker='o')
plt.title("Naive Bayes Model Training Progress")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.grid()
plt.show()