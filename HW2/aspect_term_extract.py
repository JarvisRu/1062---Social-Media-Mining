import xml.etree.ElementTree as ET
import nltk
import subprocess
from baselines import *

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_treebank_pos_tagger')
nltk.download('vader_lexicon')

# a set to store all aspect terms occur in Train data
aspectTerms_set = set([])
task1_in = Corpus(ET.parse('Restaurants_Test_Data_PhaseA.xml').getroot().findall('sentence'))

# create baseline_task1.xml from BaselineAspectExtractor to compare
def runForBase():
    corpus = Corpus(ET.parse('Restaurants_Train.xml').getroot().findall('sentence'))
    unseen = task1_in
    b1 = BaselineAspectExtractor(corpus)
    predicted = b1.tag(unseen.corpus)
    corpus.write_out('baseline_task1.xml', predicted, False)

# get all aspectTerms from train data
def getAspectTermsFromTrain():
    root = ET.parse('Restaurants_Train.xml').getroot()
    
    for sentence in root.findall('sentence'):
        aspeTerms = sentence.find('aspectTerms')
        if aspeTerms is not None:
            for aspeTerm in aspeTerms.findall('aspectTerm'):
                aspectTerms_set.add(Aspect('', '', []).create(aspeTerm).term.lower().strip('s'))

# check if this word is a Noun 
def isNoun(pos):
    if pos == 'NN' or pos == 'NNS':
        return True
    else:
        return False

# extract noun or Compound noun from test data
def extractFromTestData(text):
    count_noun = 0
    tokenized = nltk.word_tokenize(text)
    nouns = nltk.pos_tag(tokenized)
    for i, pos in enumerate(nouns):
        if nouns[i][0] not in stopwords:
            if isNoun(nouns[i][1]):
                if count_noun >= 1:
                    # connect nouns as Compound Noun
                    nouns[i] = (nouns[i-1][0] + ' ' + nouns[i][0], nouns[i][1])
                    del nouns[i-1]
                count_noun += 1
            else:
                count_noun = 0
        else:
            count_noun = 0
    return [word for (word, pos) in nouns if isNoun(pos) ]

def extract():
    getAspectTermsFromTrain()
    # extract noun from sentence and check if it's exist in the set which we create from train data 
    for cor in task1_in.corpus:
        words = extractFromTestData(cor.text)
        for word in words:
            if word.strip('s') in aspectTerms_set:
                word_from = cor.text.index(word)
                word_to = word_from + len(word)
                offset = {'from': str(word_from), 'to': str(word_to)}
                cor.aspect_terms = set(cor.aspect_terms)
                cor.aspect_terms.add(Aspect(word, '', offset))
    # write to task1_out
    task1_in.write_out('task1_out.xml', task1_in.corpus, False)

def doEval(test, ref):
    command = 'java -cp ./eval.jar Main.Aspects {} {}'.format(test, ref)
    command_list = command.split()
    subprocess.call(command_list)

if __name__ == "__main__": 
    runForBase()
    extract()
    print("======================================")
    print("Do eval for Baseline:")
    doEval('Restaurants_Test_Gold.xml', 'baseline_task1.xml')
    print("======================================")
    print("Do eval for my task:")
    doEval('Restaurants_Test_Gold.xml', 'task1_out.xml')