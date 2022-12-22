import os
import json
from numba import njit
from numba.typed import List
from time import time, sleep
data = []
# from tex import data
# from text import data_after_str as data

@njit
def proces(data) -> list[int]:
    testing = []
    testing.append(data[0][1][2])

def load():
    global data
    with open('texts.json', 'r', encoding='utf-8') as f:
        data = json.load(f)


def fetch_all_twitter_data():
    dataa = {'data': []}

    files = os.listdir('data/twitter_data/')

    for i in files:
        with open('data/twitter_data/'+i, 'r', encoding='utf-8') as f:
            js = json.load(f)
            dataa['data'].extend(js['data'])

        print('don')

    with open('texts.json', 'w', encoding='utf-8') as f:
        json.dump(dataa, f, indent=4, ensure_ascii=False)


def make_sure_of_count():
    from tex import data

    print(len(data))


def check_for_similar():
    load()
    s = time()

    # replys = sorted(([x[1][0], x[1][2]] for x in data))
    data_ = tuple(data['data'])
    replys_ = tuple(sorted(((str(x[1][0]), x[1][2]) for x in data_)))
    data_replys_ = tuple(str(x[1][0]) for x in data_)
    
    # print(replys_[0])
    # print(data_replys_[0])
    print('checkging')

    i = len(set(replys_))
    le = len(replys_)
    print(le)
    print(i)

    new_data = tuple(
            (
                (str(x[0][0]), str(x[1][0])) for i, x in enumerate(data_) if x[1][0] not in data_replys_[:i] and x[0][0] != ''
            )
        )
    

    print(len(new_data))
    print(len(set(new_data)))
    

    print(new_data[0])
    
    s = time() - s
    s = str(s) + ' seconds'

    print(s)

    with open('final_for_train.json', 'w', encoding='utf-8') as f:
        json.dump(new_data, f, ensure_ascii=False, indent=4)

    # x = proces(data)
    # print(x)
    
[
    ['وزحمة ', 1605568513611120641, 1397166580841304066],
    ['نكدتو علبنا ', 1605569079468732417, 1397166580841304066]
]


# make_sure_of_count()
check_for_similar()
# fetch_all_twitter_data()
