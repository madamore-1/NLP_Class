import nltk
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
import random

# Download  NLTK data
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('universal_tagset')
nltk.download('sentence_polarity')

# Load and shuffle data
from nltk.corpus import sentence_polarity
documents = [(sent, cat) for cat in sentence_polarity.categories()
             for sent in sentence_polarity.sents(categories=cat)]
random.shuffle(documents)

# Extract 1500 unigrams
all_words_list = [word for (sent, cat) in documents for word in sent]
all_words = nltk.FreqDist(all_words_list)
word_items = all_words.most_common(1500)
word_features = [word for (word, count) in word_items]

# unigram feature extraction
def unigram_features(document, word_features):
    document_words = set(document)
    features = {'contains({})'.format(word): (word in document_words) for word in word_features}
    return features

#  bigram feature extraction
def bigram_features_func(document, word_features, bigram_features):
    document_words = set(document)
    document_bigrams = list(nltk.bigrams(document))
    features = {'contains({})'.format(word): (word in document_words) for word in word_features}
    features.update({'bigram({}_{})'.format(b[0], b[1]): (b in document_bigrams) for b in bigram_features})
    return features

# Define simplified suffix-based POS tagging
def simplified_POS_features(document):
    suffix_tags = Counter()
    for word in document:
        if word.endswith('ing'):
            suffix_tags['verbs'] += 1
        elif word.endswith('ly'):
            suffix_tags['adverbs'] += 1
        elif word.endswith('ed'):
            suffix_tags['verbs'] += 1
        elif word.endswith('ous') or word.endswith('able') or word.endswith('ive'):
            suffix_tags['adjectives'] += 1
        elif word.isalpha() and len(word) > 5:
            suffix_tags['nouns'] += 1

    features = {
        'nouns': suffix_tags['nouns'],
        'verbs': suffix_tags['verbs'],
        'adjectives': suffix_tags['adjectives'],
        'adverbs': suffix_tags['adverbs'],
    }
    return features

# improved POS tagging 
def improved_POS_features(document):
    tagged_words = nltk.pos_tag(document, tagset='universal')
    pos_counts = Counter(tag for _, tag in tagged_words)

    features = {
        'nouns': pos_counts.get('NOUN', 0),
        'verbs': pos_counts.get('VERB', 0),
        'adjectives': pos_counts.get('ADJ', 0),
        'adverbs': pos_counts.get('ADV', 0),
    }
    return features

# feature sets for each method
unigram_featuresets = [(unigram_features(d, word_features), c) for (d, c) in documents]
bigram_featuresets = [(bigram_features_func(d, word_features, []), c) for (d, c) in documents]
simplified_POS_featuresets = [(simplified_POS_features(d), c) for (d, c) in documents]
improved_POS_featuresets = [(improved_POS_features(d), c) for (d, c) in documents]

# Train and evaluate Naive Bayes classifier
def train_and_evaluate_naive_bayes(featuresets, label):
    train_set, test_set = featuresets[1000:], featuresets[:1000]
    classifier = nltk.NaiveBayesClassifier.train(train_set)
    accuracy = nltk.classify.accuracy(classifier, test_set)
    print(f"{label} Accuracy with Naive Bayes: {accuracy:.3f}")

# Convert features to sklearn-compatible format
def prepare_sklearn_data(featuresets):
    vectorizer = DictVectorizer()
    X = vectorizer.fit_transform([features for features, _ in featuresets])
    y = [label for _, label in featuresets]
    return X, y

# Train and evaluate SVM classifier
def train_and_evaluate_svm(featuresets, label):
    X, y = prepare_sklearn_data(featuresets)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    classifier = LinearSVC()
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{label} Accuracy with SVM: {accuracy:.3f}")

# Train and evaluate Random Forest classifier
def train_and_evaluate_random_forest(featuresets, label):
    X, y = prepare_sklearn_data(featuresets)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{label} Accuracy with Random Forest: {accuracy:.3f}")

print("\n--- Model Evaluation with Naive Bayes ---")
train_and_evaluate_naive_bayes(unigram_featuresets, "Unigram Features")
train_and_evaluate_naive_bayes(bigram_featuresets, "Bigram Features")
train_and_evaluate_naive_bayes(simplified_POS_featuresets, "Simplified POS Features")
train_and_evaluate_naive_bayes(improved_POS_featuresets, "Improved POS Features")

print("\n--- Model Evaluation with SVM ---")
train_and_evaluate_svm(improved_POS_featuresets, "Improved POS Features")

print("\n--- Model Evaluation with Random Forest ---")
train_and_evaluate_random_forest(improved_POS_featuresets, "Improved POS Features")
