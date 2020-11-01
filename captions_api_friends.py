from convokit import Corpus, download
from collections import namedtuple
import pickle


corpus = Corpus(filename=download("friends-corpus"))

Caption = namedtuple('Caption', ['character', 'message', 'startTime', 'endTime', 'comments'])

captions = []

i = 1
while True:
    convoNumber = '{:0>2}'.format(i)
    try:
        convo = corpus.get_conversation(f"s08_e14_c{convoNumber}_u001")
        for utterance_id in convo.get_utterance_ids():
            utterance = corpus.get_utterance(utterance_id)
            if utterance.retrieve_meta("caption") is None:
                continue
            startTime, endTime, _ = utterance.retrieve_meta("caption")
            captions.append(Caption(utterance.speaker.id, utterance.text, startTime // 1000, endTime // 1000, None))
        i += 1
    except KeyError:
        break  # there are no more conversations

captionsPath = "./data/friends/captions.pkl"

with open(captionsPath, 'wb') as captionsFile:
    pickle.dump(captions, captionsFile)
