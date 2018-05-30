from params import params

import cv2 as cv
import sys
import numpy as np
import time

class webcam():
    def __init__(self, data={}, fn_o = None):
        self.vc =None
        self.margin = params.MARGIN
        cascade_path = '../model/cv2/haarcascade_frontalface_alt2.xml'
        self.cascade = cv.CascadeClassifier(cascade_path)
        self.n_img_per_person = 1
        self.data = data
        self.fn_o = fn_o

    def detect_face(self):
        stime = time.time()
        name = "unknown"
        vc = cv.VideoCapture(0)
        self.vc = cv.VideoCapture(0)
        if vc.isOpened():
            is_capturing, _ = vc.read()
            print "Cam opened"
        else:
            is_capturing = False
        key_input = ''
        imgs = []
        while is_capturing:

            is_capturing, frame = vc.read()
            frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            faces = self.cascade.detectMultiScale(frame,
                                                  scaleFactor=1.1,
                                                  minNeighbors=3,
                                                  minSize=(100, 100))
            if len(faces) != 0:
                face = faces[0]
                (x, y, w, h) = face
                left = x - self.margin // 2
                right = x + w + self.margin // 2
                bottom = y - self.margin // 2
                top = y + h + self.margin // 2
                # print time.time() - stime
                if time.time() - stime > 5:
                    stime = time.time()
                    img = cv.resize(frame[bottom:top, left:right, :],
                                    (160, 160))
                    p_feature = self.fn_o.proc_image(img)
                    name_list = self.data.keys()
                    dist_list = []
                    for p_name in name_list:
                        dist_list.append(self.fn_o.calc_emb_dist(self.data[p_name]['feature'][0], p_feature))
                    print name_list
                    p_name = name_list[dist_list.index(min(dist_list))]
                    if min(dist_list) < 10:
                        name = p_name
                    else:
                        name = "unknown"
                    print "Matched with Gallary image: ", p_name, "Dist", min(dist_list)

                cv.rectangle(frame,
                             (left - 1, bottom - 1),
                             (right + 1, top + 1),
                             (255, 0, 0), thickness=2)

                txt_loc = (left, bottom + 20)
                cv.putText(frame, name, txt_loc, cv.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 0), 2)
            frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
            cv.imshow('display', frame)
            key_input = cv.waitKey(1)
            if key_input == ord('q'):
                sys.exit(0)

    def video_capture(self, name='Unknown'):
        vc = cv.VideoCapture(0)
        self.vc = cv.VideoCapture(0)
        if vc.isOpened():
            is_capturing, _ = vc.read()
            print "Cam opened"
        else:
            is_capturing = False
        key_input = ''
        imgs = []
        while is_capturing:
            is_capturing, frame = vc.read()
            frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            faces = self.cascade.detectMultiScale(frame,
                                                  scaleFactor=1.1,
                                                  minNeighbors=3,
                                                  minSize=(100, 100))
            if len(faces) != 0:
                face = faces[0]
                (x, y, w, h) = face
                left = x - self.margin // 2
                right = x + w + self.margin // 2
                bottom = y - self.margin // 2
                top = y + h + self.margin // 2
                if key_input == ord('c'):
                    img = cv.resize(frame[bottom:top, left:right, :],
                                 (160, 160))
                    imgs.append(img)
                elif key_input == ord('d'):
                    img = cv.resize(frame[bottom:top, left:right, :],
                                    (160, 160))
                    p_feature = self.fn_o.proc_image(img)
                    name_list = self.data.keys()
                    dist_list = []
                    for p_name in name_list:
                        dist_list.append(self.fn_o.calc_emb_dist(self.data[p_name]['feature'][0], p_feature))
                    print name_list
                    print "Matched with Gallary image: ", name_list[dist_list.index(min(dist_list))]

                cv.rectangle(frame,
                              (left - 1, bottom - 1),
                              (right + 1, top + 1),
                              (255, 0, 0), thickness=2)

                txt_loc = (left, bottom + 20)
                cv.putText(frame, name, txt_loc, cv.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 0), 2)

            if len(imgs) == self.n_img_per_person:
                capture_text = 'captured'
                txt_loc = (left, bottom - 20)
                cv.putText(frame, capture_text, txt_loc, cv.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 0), 2)
                if name in self.data:
                    capture_text = 'Duplicated'
                    txt_loc = (left+120, bottom - 20)
                    cv.putText(frame, capture_text, txt_loc, cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
                else:
                    self.data[name] = {}
                    if self.fn_o is not None:
                        fea = self.fn_o.proc_images(imgs)
                        print fea
                        self.data[name]['feature'] = fea
                    self.data[name]['image'] = np.array(imgs)
                cv.imshow('display', frame)
                cv.waitKey(0)
                vc.release()
                return 0
            frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
            cv.imshow('display', frame)
            key_input = cv.waitKey(1)
            if key_input == ord('q'):
                return 0