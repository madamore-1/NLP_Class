'''
This small development program can be run to parse the sentences for Homework 3.
You may either run this as a stand-alone python program or copy it to jupyter notebook.
In either case, it must be in the same directory as the sentences and the grammar.
'''

import nltk
from nltk.parse import RecursiveDescentParser


# Load the grammar from the CFG file
with open('camelot_grammar.cfg', 'r') as grammar_file:
    grammar_text = grammar_file.read()

grammar = nltk.CFG.fromstring(grammar_text)
parser = RecursiveDescentParser(grammar)

# Load sentences from sentences.txt
with open('sentences.txt', 'r') as sentences_file:
    sentences = sentences_file.readlines()

# Function to parse and display each sentence's parse tree
def parse_sentences(sentences):
    for sentence in sentences:
        sentence = sentence.strip()  # Clean up any leading/trailing whitespace
        if sentence:
            print(f"Parsing sentence: '{sentence}'")
            words = sentence.split()
            try:
                for tree in parser.parse(words):
                    print(tree)
                    tree.pretty_print()  # Print the tree structure
            except ValueError as e:
                print(f"Error parsing sentence '{sentence}': {e}")

# Parse all sentences
parse_sentences(sentences)
