'''
  This program shell reads tweet data for the twitter sentiment classification problem.
  The input to the program is the path to the Semeval directory "corpus" and a limit number.
  The program reads the first limit number of tweets
  It creates a "tweetdocs" variable with a list of tweets consisting of a pair
    with the list of tokenized words from the tweet and the label pos, neg or neu.
  It prints a few example tweets, as text and as tokens.
  Your task is to generate features sets and train and test a classifier.

  Usage:  python classifySemevalTweets.py  <corpus directory path> <limit number>
'''
# open python and nltk packages needed for processing
# while the semeval tweet task b data has tags for "positive", "negative", 
#  "objective", "neutral", "objective-OR-neutral", we will combine the last 3 into "neutral"
import os
import sys
import nltk
from nltk.tokenize import TweetTokenizer

# read stop words from file if used
# stopwords = [line.strip() for line in open('stopwords_twitter.txt')]

# define a feature definition function here


# function to read tweet training file, train and test a classifier 
def processtweets(dirPath,limitStr):
  # convert the limit argument from a string to an int
  limit = int(limitStr)
  # initialize NLTK built-in tweet tokenizer
  twtokenizer = TweetTokenizer()
  
  os.chdir(dirPath)
  
  f = open('./downloaded-tweeti-b-dist.tsv', 'r')
  # loop over lines in the file and use the first limit of them
  #    assuming that the tweets are sufficiently randomized
  tweetdata = []
  for line in f:
    if (len(tweetdata) < limit):
      # remove final end of line character
      line = line.strip()
      # each line has 4 items separated by tabs
      # ignore the tweet and user ids, and keep the sentiment and tweet text
      tweetdata.append(line.split('\t')[2:4])
  
  for tweet in tweetdata[:10]:
    print (tweet)
  
  # create list of tweet documents as (list of words, label)
  # where the labels are condensed to just 3:  'pos', 'neg', 'neu'
  tweetdocs = []
  # add all the tweets except the ones whose text is Not Available
  for tweet in tweetdata:
    if (tweet[1] != 'Not Available'):
      # run the tweet tokenizer on the text string - returns unicode tokens, so convert to utf8
      tokens = twtokenizer.tokenize(tweet[1])

      if tweet[0] == '"positive"':
        label = 'pos'
      else:
        if tweet[0] == '"negative"':
          label = 'neg'
        else:
          if (tweet[0] == '"neutral"') or (tweet[0] == '"objective"') or (tweet[0] == '"objective-OR-neutral"'):
            label = 'neu'
          else:
            label = ''
      tweetdocs.append((tokens, label))
  
  # print a few
  for tweet in tweetdocs[:10]:
    print (tweet)

  # possibly filter tokens

  # continue as usual to get all words and create word features
  
  # feature sets from a feature definition function

  # train and test a classifier

  # show most informative features


"""
commandline interface takes a directory name with semeval task b training subdirectory 
       for downloaded-tweeti-b-dist.tsv
   and a limit to the number of tweets to use
It then processes the files and trains a tweet sentiment classifier.

"""
if __name__ == '__main__':
    if (len(sys.argv) != 3):
        print ('usage: classifytweets.py <corpus-dir> <limit>')
        sys.exit(0)
    processtweets(sys.argv[1], sys.argv[2])