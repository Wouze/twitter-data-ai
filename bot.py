import time
import tweepy
import json
from threading import Thread
from mega import Mega
from pyarabic.araby import strip_tashkeel, strip_diacritics, strip_tatweel
import pyarabic.trans

consumer_key = 'MP6nI2pGFDy0z84ExG7Q0Gs4y'
consumer_secret = '9dAiXDzPtjE45u1bKZBzs7vMneHBf9Gg6EHsClEL8OcovAS9UO'

access_token = '1470828550441820163-KL82G5hNNP0VyNU70M6QrUQ105KttW'
access_secret = 'qn88AmDL06bUGoINPdlL9umnpgS8DZfcIkRKPZjZY9fAd'

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_secret)
api = tweepy.API(auth, wait_on_rate_limit=True)

# xxxxxx = pyarabic.trans.normalize_digits(strip_tashkeel(strip_diacritics(strip_tatweel(text))), source='all', out='west')

print(time.ctime().replace(' ', '_').replace(':', "-"))
xxx = '( محمد OR من OR الى OR عن OR على OR في OR بل OR انت OR وقطفت OR هل OR رفاقي OR التواجد OR رحبوا OR يقال OR أعلم OR ثاقبة OR ذائقة OR نخبة OR تجنب OR لكن OR حتى OR ما OR ليه OR لماذا OR لم OR لما OR أن OR بلى OR اجل OR اسم OR ضحل OR بوت OR روبوت OR كيف OR السلام OR عليكم OR مرحبا OR كيفك OR شخبارك OR ايه OR طيب OR قبل OR اقول OR قلت OR عشان OR لأن ) (lang:ar -is:retweet is:reply -is:quote)'
print(len(xxx))

mega = Mega()
m = mega.login('m-ohm1@hotmail.com', '4250104m')

Folder = mega.find('twitter_data')


twets = []
data = {'data':[]}

def save():
    global data
    if data != {'data':[]}:
        name = f"{time.ctime().replace(' ', '_').replace(':', '-')}[{len(data['data'])}].json"

        with open("data/" + name, 'w', encoding="utf-8") as outfile:
            json.dump(data, outfile, ensure_ascii=False, indent=4)
        
        data = {'data':[]}
        time.sleep(1)
        m.upload("data/" +name, Folder[0])

def main_lop():
    global stop_bkup
    while True:
        time.sleep((60*60) - 20)
        try:
            stop_bkup = True
            time.sleep(19)
            save()
            stop_bkup = False
        except BaseException as a:
            print(a)


stop = False
stop_bkup = False
def main_loops():
    global twets, stop
    while True:
        if len(twets) > 110:
            stop=True
            time.sleep(0.1)
            chuks = twets[0:99]
            twets = twets[100:]
            stop=False
            chuk = [x[0] for x in chuks]
            txt = {chuks[i][0]:chuks[i][1] for i in range(len(chuks))}
            rplys = {}

            statuses = api.lookup_statuses(chuk, include_entities=False, map=True)
            
            # for i, t in zip(statuses, txt):
            for i in range(len(statuses)):
                try:
                    
                    rplys[statuses[i].in_reply_to_status_id] = [txt[statuses[i].id], statuses[i].id, statuses[i].user.id, int(statuses[i].in_reply_to_status_id)]
                except:
                    pass

            chuk = [rplys[x][3] for x in rplys.keys()]
            statuses = api.lookup_statuses(chuk, include_entities=False, tweet_mode='extended', map=True)
            # print(f'{len(rplys) = }')
            # print(f'{len(statuses) = }')
            # print(f'{len(twets) = }')
            for stat in statuses:
                try:
                    t = stat.full_text

                    for i in t.split():
                        if i.startswith('@'):
                            t = t.replace(i, "")
                        elif i.startswith('http'):
                            t = t.replace(i, '')
                        elif i.startswith('ههههههههههه'):
                            t = t.replace(i, '')
                        elif i.startswith('#'):
                            t = t.replace(i, '')
                    
                    t = pyarabic.trans.normalize_digits(strip_tashkeel(strip_diacritics(strip_tatweel(str(t)))), source='all', out='west')
                    t = t.strip()
                    data['data'].append([[t, stat.id, stat.user.id], [rplys[stat.id][0], rplys[stat.id][1], rplys[stat.id][2]]])
                except:
                    pass

        time.sleep(10)
        # print(len(twets))
        if stop_bkup:
            time.sleep(30)


BAR_TOKEN = 'AAAAAAAAAAAAAAAAAAAAAKKMkgEAAAAAuQ1%2BU26mJLOKR2DC%2BXTA%2B7%2FBcdM%3DNCC1krzyqTCw8syngxLisyIWDv4ZE6en9LEZTGflHJ7yhVRNbH'


class MyStreamListener(tweepy.StreamingClient):
    def on_tweet(self, tweet):
        try:
            if not stop:
                if tweet.referenced_tweets[0].type == "replied_to":
                    t = str(tweet.text)
                    # print(tweet.id)                    

                    for i in t.split():
                        if i.startswith('@'):
                            t = t.replace(i, "")
                        elif i.startswith('http'):
                            t = t.replace(i, '')
                        elif i.startswith('ههههههههههه'):
                            t = t.replace(i, '')
                        elif i.startswith('#'):
                            t = t.replace(i, '')

                    t = pyarabic.trans.normalize_digits(strip_tashkeel(strip_diacritics(strip_tatweel(str(t)))), source='all', out='west')
                    t = t.strip()

                    if len(t) > 200:
                        twets.append([tweet.id, t])

        except:
            pass

    def on_error(self, status_code):
        print(status_code)
        if status_code == 420:
            print("Bot has been failed")
            save()
            return False


streaming_client = MyStreamListener(BAR_TOKEN, wait_on_rate_limit=True)


# print(streaming_client.add_rules(tweepy.StreamRule("PixelThis_bot")))
# streaming_client.delete_rules(1605279071348424709)
# print(streaming_client.get_rules())


# print(streaming_client.add_rules(tweepy.StreamRule(xxx)))
# 1605316521676472321

# (محمد ما ليه اي ايه احس لا طيب قبل اقول قلت عشان لأن لان ) lang:ar -is:retweet is:reply

Thread(target=main_loops).start()
time.sleep(0.5)
Thread(target=main_lop).start()
time.sleep(0.5)
print("Starting the stream")
streaming_client.filter(tweet_fields=['referenced_tweets'])
