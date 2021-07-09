import tensorflow as tf
import pretrained_networks
import os
from align_images import start_img
import project_images as pimg
import glob

import numpy as np
from PIL import Image
import dnnlib
import dnnlib.tflib as tflib
from pathlib import Path

from mtcnn import MTCNN
import cv2


def toon():

    blended_url = "https://drive.google.com/uc?id=1H73TfV5gQ9ot7slSed_l-lim9X7pMRiU" 
    ffhq_url = "http://d36zk2xti64re0.cloudfront.net/stylegan2/networks/stylegan2-ffhq-config-f.pkl"
    _, _, Gs_blended = pretrained_networks.load_networks(blended_url)
    _, _, Gs = pretrained_networks.load_networks(ffhq_url)


    start_img('./raw','./aligned')
    pimg.main()

    latent_dir = Path("input")
    latents = latent_dir.glob("*.npy")
    for latent_file in latents:
        latent = np.load(latent_file)
        latent = np.expand_dims(latent,axis=0)
        synthesis_kwargs = dict(output_transform=dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=False), minibatch_size=8)
        images = Gs_blended.components.synthesis.run(latent, randomize_noise=False, **synthesis_kwargs)
        Image.fromarray(images.transpose((0,2,3,1))[0], 'RGB').save(latent_file.parent / (f"{latent_file.stem}-toon.jpg"))

    images = glob.glob('./input/*.jpg')

    for img_path in images:
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        detector = MTCNN()
        detections = detector.detect_faces(img)
        if detections != []:
            f = open(img_path[:-3]+'txt','w')
            for key in detections[0]['keypoints'].keys():
                f.write("%f %f\n" %(detections[0]['keypoints'][key][0].round(2),detections[0]['keypoints'][key][1].round(2)))
            f.close()
        else:
            os.remove(img_path)



if __name__ == '__main__':

    toon()
