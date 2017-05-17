#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function, division
import argparse
import caffe
import cv2
import json
import numpy as np
from os.path import dirname, exists, join, splitext
import sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', nargs='?',
                        choices=['pascal_voc', 'camvid', 'kitti', 'cityscapes'])
    parser.add_argument('input_path', nargs='?', default='',
                        help='Required path to input image')
    parser.add_argument('-p', '--prototxt', default=None)
    parser.add_argument('-w', '--weights', default=None)
    parser.add_argument('-i', '--info', default=None)
    parser.add_argument('--gpu', type=int, default=-1,
                        help='GPU ID to run CAFFE. '
                             'If -1 (default), CPU is used')
    args = parser.parse_args()

    if args.gpu >= 0:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu)
        print('Using GPU ', args.gpu)
    else:
        caffe.set_mode_cpu()
        print('Using CPU')

    print( 'Load weights from %s' % (args.weights) )
    net = caffe.Net(args.prototxt, args.weights, caffe.TRAIN)
    
    

    with open(args.info, 'r') as fp:
            info = json.load(fp)
    palette = np.array(info['palette'], dtype=np.uint8)

    import matplotlib.pyplot as plt

    for ii in range(10):
        out = net.forward()
        data = net.blobs['data'].data.copy()
        data = data[0].transpose((1,2,0))

        mean = np.array( [[[104.008, 116.669, 122.675]]] )

        data += mean
        data = data[:,:,(2,1,0)]
        data = data.astype(np.uint8)

        plt.imshow(data)
        plt.savefig('data.jpg')


        label = net.blobs['label'].data.copy()
        label = label[0][0]
        label = label.astype(np.uint8)
        
        label[label == 255] = 21
                
        color_image = palette[label.ravel()].reshape(data.shape)
        color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)

        plt.imshow(color_image)
        plt.savefig('label.jpg')

        pred = net.blobs['conv6_interp'].data.copy()
        pred = pred[0]
        pred = pred.argmax(axis=0)

        pred_image = palette[pred.ravel()].reshape(data.shape)
        pred_image = cv2.cvtColor(pred_image, cv2.COLOR_RGB2BGR)

        plt.imshow(pred_image)
        plt.savefig('pred.jpg')

        import ipdb
        ipdb.set_trace()


    ##### Data layer
    # for ii in range(10):
    #     out = net.forward()
    #     data = out['data'].copy()
    #     data = data[0].transpose((1,2,0))

    #     mean = np.array( [[[104.008, 116.669, 122.675]]] )

    #     data += mean
    #     data = data[:,:,(2,1,0)]
    #     data = data.astype(np.uint8)

    #     plt.imshow(data)
    #     plt.savefig('data.jpg')


    #     label = out['label'].copy()
    #     label = label[0][0]
    #     label = label.astype(np.uint8)
        
    #     label[label == 255] = 21
                
    #     color_image = palette[label.ravel()].reshape(data.shape)
    #     color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)

    #     plt.imshow(color_image)
    #     plt.savefig('label.jpg')

    #     import ipdb
    #     ipdb.set_trace()

if __name__ == '__main__':
    main()
