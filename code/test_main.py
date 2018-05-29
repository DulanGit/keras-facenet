from webcam import webcam
from facenet import facenet
import cv2
import json
import numpy as np
import cPickle

def set_default(obj):
    if isinstance(obj, set):
        return list(obj)
    raise TypeError

def main():
    fn = facenet()
    data = {}

    with open('data.json', 'r') as outfile:
        data = cPickle.load(outfile)

    wc = webcam(data)
    # wc.video_capture('Dulan')

    # image = cv2.imread('/home/dulan/PycharmProjects/keras-facenet/data/test/gallary/q2.jpg')
    # print wc.data['Dulan'].shape
    # cv2.imshow('Disp', np.array(wc.data['Dulan'][0], dtype=np.uint8))
    # cv2.waitKey(0)
    print fn.proc_image(np.array(wc.data['Dulan'][0]))
    #
    # wc.video_capture('Sharada')
    #
    with open('data.json', 'w') as outfile:
        cPickle.dump(wc.data, outfile)


if __name__=='__main__':
    main()