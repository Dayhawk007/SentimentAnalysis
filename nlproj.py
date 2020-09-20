from nltk.tag import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer
import random,re,string
from nltk.corpus import stopwords
from nltk import FreqDist
from nltk import classify
from nltk import NaiveBayesClassifier
from nltk.tokenize import word_tokenize
import twitterscrape
stop_words=stopwords.words('english')
train=open("final.csv",'r',encoding="UTF-8")
list=train.readlines()
new_list=[]
def tokenize(arr):
    new_arr=[]
    for line in arr:
        sentence=line.split(',')
        tokenized=sentence[0].split(" ")
        new_arr.append([tokenized,sentence[1][:-1]])
    return new_arr
def remove_noise(tweet_tokens, stop_words = ()):

    cleaned_tokens = []

    for token, tag in pos_tag(tweet_tokens):
        token = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|'\
                       '(?:%[0-9a-fA-F][0-9a-fA-F]))+','', token)
        token = re.sub("(@[A-Za-z0-9_]+)","", token)

        if tag.startswith("NN"):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'

        lemmatizer = WordNetLemmatizer()
        token = lemmatizer.lemmatize(token, pos)

        if len(token) > 0 and token not in string.punctuation and token.lower() not in stop_words:
            cleaned_tokens.append(token.lower())
    return cleaned_tokens
def get_all_words(cleaned_tokens_list):
    for tokens in cleaned_tokens_list:
        for token in tokens:
            yield token

new_list=tokenize(list)
random.shuffle(new_list)
lem_sent=[]
cleaned=[]
positive=[]
negative=[]
fear=[]
anger=[]
joy=[]
sad=[]
for line in new_list:
    cleaned.append(remove_noise([i for i in line[0] if i],stop_words))
    lem_sent.append([remove_noise([i for i in line[0] if i],stop_words),line[1]])
    if(line[1]=='Positive'):
        positive.append(remove_noise([i for i in line[0] if i],stop_words))
    elif (line[1] == 'Negative'):
        negative.append(remove_noise([i for i in line[0] if i], stop_words))
    elif (line[1] == 'fear'):
        fear.append(remove_noise([i for i in line[0] if i], stop_words))
    elif (line[1] == 'sad'):
        sad.append(remove_noise([i for i in line[0] if i], stop_words))
    elif (line[1] == 'joy'):
        joy.append(remove_noise([i for i in line[0] if i], stop_words))
    elif (line[1] == 'anger'):
        anger.append(remove_noise([i for i in line[0] if i], stop_words))
all_positive=get_all_words(positive)
all_negative=get_all_words(negative)
all_sad=get_all_words(sad)
all_fear=get_all_words(fear)
all_joy=get_all_words(joy)
all_anger=get_all_words(anger)
freq_pos=FreqDist(all_positive)
freq_neg=FreqDist(all_negative)
freq_sad=FreqDist(all_sad)
freq_anger=FreqDist(all_anger)
freq_fear=FreqDist(all_fear)
freq_joy=FreqDist(all_joy)
def get_tweets_for_model(cleaned_tokens_list):
    for tweet_tokens in cleaned_tokens_list:
        yield dict([token, True] for token in tweet_tokens)

pos_model=get_tweets_for_model(positive)
neg_model=get_tweets_for_model(negative)
sad_model=get_all_words(sad)
anger_model=get_tweets_for_model(anger)
joy_model=get_tweets_for_model(joy)
fear_model=get_tweets_for_model(fear)

positive_dataset = [(tweet_dict, "Positive")
                    for tweet_dict in pos_model]

negative_dataset = [(tweet_dict, "Negative")
                     for tweet_dict in neg_model]

sad_dataset=[(tweet_dict, "sad")
                     for tweet_dict in sad_model]
joy_dataset=[(tweet_dict, "joy")
                     for tweet_dict in joy_model]
anger_dataset=[(tweet_dict, "anger")
                     for tweet_dict in anger_model]
fear_dataset=[(tweet_dict, "fear")
                     for tweet_dict in fear_model]
dataset=positive_dataset+negative_dataset+joy_dataset+sad_dataset+anger_dataset+fear_dataset
new_dataset=[]

for tweet_dict,emo in dataset:
    count=0
    if(type(tweet_dict)==dict):

        new_dataset.append((tweet_dict,emo))
    count+=1
random.shuffle(new_dataset)
train_data=new_dataset[:20000]
test_data=new_dataset[20000:]

Classifier=NaiveBayesClassifier.train(train_data)

print("Accuracy is: "+str(classify.accuracy(Classifier,test_data)))

for tweet_data in twitterscrape.tweet_list:
    tweet_id,tweet_text=tweet_data
    custom_tokens = remove_noise(word_tokenize(tweet_text))
    print(tweet_text+"\n")
    print(Classifier.classify(dict([token, True] for token in custom_tokens)))





