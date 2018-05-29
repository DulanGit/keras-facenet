from params import params

import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
from imageio import imread
from skimage.transform import resize
from scipy.spatial import distance

import inception_resnet_v1

class facenet:
    def __init__(self):
        self.cascade_path = params.CASCADE_PATH
        self.image_dir_basepath = params.IMAGE_DIR
        self.image_size = 160

        print("Loading model...")
        self.model = inception_resnet_v1.InceptionResNetV1(weights_path=params.WEIGHTS_PATH)

        self.FACE_CROP = False


    def prewhiten(self ,x):
        if x.ndim == 4:
            axis = (1, 2, 3)
            size = x[0].size
        elif x.ndim == 3:
            axis = (0, 1, 2)
            size = x.size
        else:
            raise ValueError('Dimension should be 3 or 4')

        mean = np.mean(x, axis=axis, keepdims=True)
        std = np.std(x, axis=axis, keepdims=True)
        std_adj = np.maximum(std, 1.0/np.sqrt(size))
        y = (x - mean) / std_adj
        return y

    def l2_normalize(self ,x, axis=-1, epsilon=1e-10):
        output = x / np.sqrt(np.maximum(np.sum(np.square(x), axis=axis, keepdims=True), epsilon))
        return output


    def load_and_align_images(self ,filepaths, margin):
        cascade = cv2.CascadeClassifier(self.cascade_path)

        aligned_images = []
        for filepath in filepaths:
            img = imread(filepath)
            if self.FACE_CROP:
                faces = cascade.detectMultiScale(img,
                                                 scaleFactor=1.1,
                                                 minNeighbors=3)
                (x, y, w, h) = faces[0]
                cropped = img[y - margin // 2:y + h + margin // 2,
                          x - margin // 2:x + w + margin // 2, :]

                aligned = resize(cropped, (self.image_size, self.image_size), mode='reflect')
            else:
                aligned = resize(img, (self.image_size, self.image_size), mode='reflect')
            aligned_images.append(aligned)

        return np.array(aligned_images)

    def calc_embs(self ,filepaths, margin=10, batch_size=1):
        aligned_images = self.prewhiten(self.load_and_align_images(filepaths, margin))
        pd = []
        for start in range(0, len(aligned_images), batch_size):
            pd.append(self.model.predict_on_batch(aligned_images[start:start+batch_size]))
        embs = self.l2_normalize(np.concatenate(pd))

        return embs

    def calc_dist(self ,img_name0, img_name1):
        return distance.euclidean(self.data_g[img_name0]['emb'], self.data_g[img_name1]['emb'])

    def calc_emb_dist(self ,emb1, emb2):
        return distance.euclidean(emb1, emb2)

    def proc_image(self, image):
        print "Obtaining feature vector..."
        return self.model.predict(self.prewhiten(np.array([resize(image, (self.image_size, self.image_size), mode='reflect')])))

    def proc_images(self, images):
        print "Obtaining feature vector..."
        return self.model.predict(self.prewhiten(np.array([resize(image, (self.image_size, self.image_size), mode='reflect') for image in images])))

    def main(self):
        print("Processing gallary...")
        data_g = {}
        gallary_folder = os.path.join(self.image_dir_basepath, 'gallary')
        gallary_images = [os.path.join(gallary_folder, f) for f in os.listdir(gallary_folder)]
        embs = self.calc_embs(gallary_images)
        for i in range(len(gallary_images)):
            data_g['{}'.format(i)] = {'image_filepath' : gallary_images[i], 'emb' : embs[i]}

        print("Processing quary...")
        data_q = {}
        quary_folder = os.path.join(self.image_dir_basepath, 'quary')
        quary_images = [os.path.join(quary_folder, f) for f in os.listdir(quary_folder)]
        embs = self.calc_embs(quary_images)
        for i in range(len(quary_images)):
            data_q['{}'.format(i)] = {'image_filepath' : quary_images[i], 'emb' : embs[i]}

        for j in range(len(quary_images)):
            print('')
            print("Quary image: ", os.path.basename(data_q[str(j)]['image_filepath']))
            basename_list = []
            distance_list = []
            for i in range(len(gallary_images)):
                basename_list.append(os.path.basename(data_g[str(i)]['image_filepath']))
                distance_list.append(self.calc_emb_dist(data_q[str(j)]['emb'], data_g[str(i)]['emb']))

            print("Matched with Gallary image: ", basename_list[distance_list.index(min(distance_list))])


        # print(calc_dist('BillGates0', 'LarryPage0'))
        # print(calc_dist('MarkZuckerberg0', 'MarkZuckerberg1'))



        print("Done...")

if __name__=='__main__':
    fn = facenet()
    fn.main()