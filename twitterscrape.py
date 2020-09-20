
import tweepy as tw
import pandas as pd
keys=open('keys','r')
cs_key,cs_secret,acc_tok,acc_secret=keys.read().split(",")
auth=tw.OAuthHandler(cs_key,cs_secret)
auth.set_access_token(acc_tok,acc_secret)
api=tw.API(auth,wait_on_rate_limit=True)
option=int(input("Select what to scrape from twitter\n1.User's Data\n2.Search Query\n"))

if(option==1):
    user=input("Enter UserName\n")
    tweets=tw.Cursor(api.user_timeline,id=user).items(100)
    tweet_list=[[tweet.id,tweet.text] for tweet in tweets]

    df=pd.DataFrame(tweet_list)
elif(option==2):
    search=input("Enter Search Query\n")
    tweets=tw.Cursor(api.search,q=search).items(100)
    tweet_list = [[tweet.id, tweet.text] for tweet in tweets]

    df = pd.DataFrame(tweet_list)
print(df)

