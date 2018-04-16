# Aspect Based Sentiment Analysis

## Requirement
- Install
    - xml : for parse .xml
    - nltk : for NLP

## Task1 : Aspect term extraction

### Steps
1. Run baselines.py to get **baseline_task1.xml**
2. Extract
    - Get aspectTerms_set : all aspect terms from **Restaurants_Train.xml**
    - Using nltk to extract noun or compound noun from **Restaurants_Test_Data_PhaseA.xml**
    - Check if the word is in aspectTerms_set too
        - Yes : extract this word
        - No : do nothing
    - write it out to **task1_out.xml**
3. Evaluate
    - Do evaluation of 
        - test = baseline_task1.xml , ref = Restaurants_Test_Gold.xml
        - test = task1_out.xml , ref = Restaurants_Test_Gold.xml
