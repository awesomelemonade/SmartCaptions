from convokit import Corpus, download
corpus = Corpus(filename=download("friends-corpus"))
# have to do it like this because there's no easy way of getting all conversations for a particular episode
i = 1

captions = []

while True:
    convoNumber = '{:0>2}'.format(i)
    try:
        convo = corpus.get_conversation(f"s08_e14_c{convoNumber}_u001")
        for utterance_id in convo.get_utterance_ids():
            utterance = corpus.get_utterance(utterance_id)
            nextCaption = {}
            nextCaption['speaker'] = utterance.speaker.id
            nextCaption['text'] = utterance.text
            nextCaption['startTime'] = utterance.retrieve_meta("caption")[0]
            nextCaption['endTime'] = utterance.retrieve_meta("caption")[1]
            captions.append(nextCaption)
            # print(utterance_id)
            # print(utterance.retrieve_meta("caption"))
            # print(utterance.text)
            # print(utterance.speaker)
        print(convo)
        i += 1
    except KeyError:
        break  # there are no more conversations

