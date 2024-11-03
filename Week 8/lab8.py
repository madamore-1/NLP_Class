# Import necessary libraries
import nltk
from nltk.corpus import sentence_polarity
import random

# Download the necessary NLTK data if you haven't already
nltk.download('sentence_polarity')
nltk.download('stopwords')

# Load the movie review sentences from the NLTK corpus
sentences = sentence_polarity.sents()
documents = [(sent, category) for category in sentence_polarity.categories() 
             for sent in sentence_polarity.sents(categories=category)]
random.shuffle(documents)  # Shuffle the documents

# Check the number of sentences and categories
print(f"Total number of sentences: {len(sentences)}")
print(f"Categories: {sentence_polarity.categories()}")
print("First four sentences:")
for sent in sentences[:4]:
    print(sent)



# Extract all words and get the 2000 most common words for BOW features
all_words_list = [word.lower() for (sent, _) in documents for word in sent]
all_words = nltk.FreqDist(all_words_list)
word_features = [word for (word, _) in all_words.most_common(2000)]

# Define a function for BOW features
def document_features(document, word_features):
    document_words = set(document)
    features = {f'V_{word}': (word in document_words) for word in word_features}
    return features

# Create feature sets using BOW
featuresets = [(document_features(d, word_features), c) for (d, c) in documents]

# Split into training and testing sets (90/10 split)
train_set, test_set = featuresets[1000:], featuresets[:1000]

# Train a Naive Bayes classifier and evaluate it
classifier = nltk.NaiveBayesClassifier.train(train_set)
accuracy = nltk.classify.accuracy(classifier, test_set)
print(f'Accuracy with BOW features: {accuracy * 100:.2f}%')
classifier.show_most_informative_features(10)



# Import the readSubjectivity function from Subjectivity.py (ensure the file is in the same directory)
from subjectivity import readSubjectivity  # Adjust the path if necessary

# Load the Subjectivity Lexicon
SLpath = "subjclueslen1-HLTEMNLP05.tff"  # Path to the subjectivity lexicon file
SL = readSubjectivity(SLpath)
print(f"Loaded {len(SL)} words from the subjectivity lexicon.")

# Define features using the Subjectivity Lexicon
def SL_features(document, word_features, SL):
    document_words = set(document)
    features = {f'V_{word}': (word in document_words) for word in word_features}
    weakPos, strongPos, weakNeg, strongNeg = 0, 0, 0, 0
    for word in document_words:
        if word in SL:
            strength, _, _, polarity = SL[word]
            if strength == 'weaksubj' and polarity == 'positive':
                weakPos += 1
            elif strength == 'strongsubj' and polarity == 'positive':
                strongPos += 1
            elif strength == 'weaksubj' and polarity == 'negative':
                weakNeg += 1
            elif strength == 'strongsubj' and polarity == 'negative':
                strongNeg += 1
    features['positivecount'] = weakPos + (2 * strongPos)
    features['negativecount'] = weakNeg + (2 * strongNeg)
    return features

# Create feature sets using the Subjectivity Lexicon features
SL_featuresets = [(SL_features(d, word_features, SL), c) for (d, c) in documents]
train_set, test_set = SL_featuresets[1000:], SL_featuresets[:1000]

# Train and evaluate the classifier
classifier = nltk.NaiveBayesClassifier.train(train_set)
accuracy = nltk.classify.accuracy(classifier, test_set)
print(f'Accuracy with Subjectivity Lexicon features: {accuracy * 100:.2f}%')



negationwords = ['no', 'not', 'never', 'none', 'rather', 'hardly', 'scarcely', 
                 'rarely', 'seldom', 'neither', 'nor']

# Define features with negation handling
def NOT_features(document, word_features, negationwords):
    features = {f'V_{word}': False for word in word_features}
    features.update({f'V_NOT{word}': False for word in word_features})
    for i in range(len(document)):
        word = document[i]
        if ((i + 1) < len(document)) and ((word in negationwords) or (word.endswith("n't"))):
            i += 1
            if document[i] in word_features:
                features[f'V_NOT{document[i]}'] = True
        elif word in word_features:
            features[f'V_{word}'] = True
    return features

# Create feature sets using negation handling
NOT_featuresets = [(NOT_features(d, word_features, negationwords), c) for (d, c) in documents]
train_set, test_set = NOT_featuresets[1000:], NOT_featuresets[:1000]

# Train and evaluate the classifier
classifier = nltk.NaiveBayesClassifier.train(train_set)
accuracy = nltk.classify.accuracy(classifier, test_set)
print(f'Accuracy with Negation features: {accuracy * 100:.2f}%')
classifier.show_most_informative_features(10)
