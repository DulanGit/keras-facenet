from webcam import webcam
from facenet import facenet
import cPickle
import os

def set_default(obj):
    if isinstance(obj, set):
        return list(obj)
    raise TypeError

def main():
    save = True
    data = {}

    if os.path.exists('data.json'):
        with open('data.json', 'r') as outfile:
            data = cPickle.load(outfile)

    fn = facenet()

    wc = webcam(data, fn_o= fn)
    # wc.video_capture('Murali')
    wc.detect_face()

    # wc.video_capture('Dilina')
    #
    # print wc.data['Dulan']['image'][0]
    # print "FEA"
    # print wc.data['Dulan']['feature'][0]
    if save:
        with open('data.json', 'w') as outfile:
            cPickle.dump(wc.data, outfile)


if __name__=='__main__':
    main()