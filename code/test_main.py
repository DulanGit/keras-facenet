from webcam import webcam
from facenet import facenet
import cv2
import json
import numpy as np
import cPickle
import os

def set_default(obj):
    if isinstance(obj, set):
        return list(obj)
    raise TypeError

def main():

    data = {}

    if os.path.exists('data.json'):
        with open('data.json', 'r') as outfile:
            data = cPickle.load(outfile)

    fn = facenet()

    wc = webcam(data, fn_o= fn)
    wc.video_capture('Dulan')

    print wc.data['Dulan']['image'][0]
    print "FEA"
    print wc.data['Dulan']['feature'][0]

    with open('data.json', 'w') as outfile:
        cPickle.dump(wc.data, outfile)


if __name__=='__main__':
    main()