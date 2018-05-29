from webcam import webcam
import json
import numpy as np
import cPickle

def set_default(obj):
    if isinstance(obj, set):
        return list(obj)
    raise TypeError

def main():
    data = {}

    with open('data.json', 'r') as outfile:
        data = cPickle.load(outfile)

    wc = webcam(data)
    wc.video_capture('Dulan')
    # wc.video_capture('Dulan')
    wc.video_capture('Sharada')

    print wc.data['Dulan']['feature']

    with open('data.json', 'w') as outfile:
        cPickle.dump(wc.data, outfile)
    # print type(wc.data)
    # print wc.data

if __name__=='__main__':
    main()