#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function, division
import argparse
import caffe
import cv2
import numpy as np
import os
from os.path import exists, join, split, splitext
import shutil
import json

import matplotlib.pyplot as plt
# import network
# import util

__author__ = 'Soonmin Hwang'
__email__ = 'smhwang@rcv.kaist.ac.kr'
__description__ = 'This code is a modified version of F.Yus implementation. \
                    (https://github.com/fyu/dilated.git) '


# def read_array(filename):
#     with open(filename, 'rb') as fp:
#         type_code = np.fromstring(fp.read(4), dtype=np.int32)
#         shape_size = np.fromstring(fp.read(4), dtype=np.int32)
#         shape = np.fromstring(fp.read(4 * shape_size), dtype=np.int32)
#         if type_code == cv2.CV_32F:
#             dtype = np.float32
#         if type_code == cv2.CV_64F:
#             dtype = np.float64
#         return np.fromstring(fp.read(), dtype=dtype).reshape(shape)


# def write_array(filename, array):
#     with open(filename, 'wb') as fp:
#         if array.dtype == np.float32:
#             typecode = cv2.CV_32F
#         elif array.dtype == np.float64:
#             typecode = cv2.CV_64F
#         else:
#             raise ValueError("type is not supported")
#         fp.write(np.array(typecode, dtype=np.int32).tostring())
#         fp.write(np.array(len(array.shape), dtype=np.int32).tostring())
#         fp.write(np.array(array.shape, dtype=np.int32).tostring())
#         fp.write(array.tostring())


def test_image(options):

    # label_margin = 186

    if options.gpu >= 0:
        caffe.set_mode_gpu()
        caffe.set_device(options.gpu)
        print('Using GPU ', options.gpu)
    else:
        caffe.set_mode_cpu()
        print('Using CPU')

    mean_pixel = np.array(options.mean, dtype=np.float32)
    net = caffe.Net(options.deploy_net, options.weights, caffe.TEST)

    image_paths = [line.strip() for line in open(options.image_list, 'r')]
    image_names = [split(p)[1] for p in image_paths]
    input_dims = list(net.blobs['data'].shape)

    assert input_dims[0] == 1
    batch_size, num_channels, input_height, input_width = input_dims
    print('Input size:', input_dims)
    caffe_in = np.zeros(input_dims, dtype=np.float32)

    output_height = input_height
    output_width = input_width

    result_list = []
    feat_list = []


    with open(options.info, 'r') as fp:
            info = json.load(fp)
    palette = np.array(info['palette'], dtype=np.uint8)


    for i in range(len(image_names)):
        print('Predicting', image_names[i])
        image_ori = cv2.imread(image_paths[i]).astype(np.float32) - mean_pixel        
        image_size = image_ori.shape    
        print('Image size:', image_size)

        image = cv2.resize(image_ori, (input_dims[2],input_dims[3]), interpolation = cv2.INTER_CUBIC)

        caffe_in[0] = image.transpose([2, 0, 1])
        out = net.forward_all(blobs=[], **{net.inputs[0]: caffe_in})
        prob = out['pred'][0]

        # image = cv2.copyMakeBorder(image, label_margin, label_margin,
        #                            label_margin, label_margin,
        #                            cv2.BORDER_REFLECT_101)
        # num_tiles_h = input_height // output_height + \
        #               (1 if image_size[0] % output_height else 0)
        # num_tiles_w = input_width // output_width + \
        #               (1 if image_size[1] % output_width else 0)
        # prediction = []
        # feat = []
        # for h in range(num_tiles_h):
        #     col_prediction = []
        #     col_feat = []
        #     for w in range(num_tiles_w):
        #         offset = [output_height * h,
        #                   output_width * w]
        #         tile = image[offset[0]:offset[0] + input_height,
        #                      offset[1]:offset[1] + input_width, :]
        #         margin = [0, input_height - tile.shape[0],
        #                   0, input_width - tile.shape[1]]
        #         # tile = cv2.copyMakeBorder(tile, margin[0], margin[1],
        #         #                           margin[2], margin[3],
        #         #                           cv2.BORDER_REFLECT_101)

        #         caffe_in[0] = tile.transpose([2, 0, 1])
        #         blobs = []
        #         out = net.forward_all(blobs=blobs, **{net.inputs[0]: caffe_in})
        #         prob = out['pred'][0]
        #         col_prediction.append(prob)
        #     col_prediction = np.concatenate(col_prediction, axis=2)
        #     prediction.append(col_prediction)
        # prob = np.concatenate(prediction, axis=1)

        # zoom_prob = prob[:, :image_size[0], :image_size[1]]
        # prediction = np.argmax(zoom_prob.transpose([1, 2, 0]), axis=2)
        prediction = np.argmax(prob.transpose([1, 2, 0]), axis=2)
        prediction = cv2.resize(prediction, (image_size[1], image_size[0]), interpolation=cv2.INTER_NEAREST)
        
        from PIL import PngImagePlugin, Image
        


        out_path = join(options.result_dir,
                        splitext(image_names[i])[0] + '.png')
        print('Writing', out_path)
        # cv2.imwrite(out_path, prediction)        

        im = Image.fromarray(prediction.astype(np.uint8), mode='P')
        im.putpalette(palette.flatten())
        # im.info['palette'] = palette
        im.save(out_path)

        # meta = PngImagePlugin.PngInfo()
        # reserved = ('interlace', 'gamma', 'dpi', 'transparency', 'aspect')
        # for k, v, in im.info.iteritems():
        #     if k in reserved: continue
        #     meta.add_text(k, v, 0)                    
        # im.save(out_path, "PNG", pnginfo=meta)


        # out_path = join(options.result_dir,
        #                 'seg_' + splitext(image_names[i])[0] + '.png')

        # color_image = palette[prediction.ravel()].reshape(image_size)        
        # color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)

        # color_image = color_image * 0.3 + image_ori * 0.7
        # cv2.imwrite(out_path, color_image)

        # import ipdb
        # ipdb.set_trace()

    print('================================')
    print('All results are generated.')
    print('================================')

    # result_list_path = join(options.result_dir, 'results.txt')
    # print('Writing', result_list_path)
    # with open(result_list_path, 'w') as fp:
    #     fp.write('\n'.join(result_list))
    # if options.bin:
    #     feat_list_path = join(options.feat_dir, 'feats.txt')
    #     print('Writing', feat_list_path)
    #     with open(feat_list_path, 'w') as fp:
    #         fp.write('\n'.join(feat_list))


def process_options(options):
    assert exists(options.image_list), options.image_list + ' does not exist'
    assert exists(options.weights), options.weights + ' does not exist'
    
    work_dir = options.work_dir
    model = options.model

    assert exists(options.deploy_net), options.deploy_net + 'does not exist'
    shutil.copy(options.deploy_net, join(work_dir, 'deploy.prototxt'))

    options.result_dir = join(work_dir, 'results', options.sub_dir)

    if not exists(work_dir):
        print('Creating working directory', work_dir)
        os.makedirs(work_dir)
    if not exists(options.result_dir):
        print('Creating', options.result_dir)
        os.makedirs(options.result_dir)
    
    return options


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='pspnet101', choices=['pspnet101'])
    parser.add_argument('--work_dir', default='jobs/pascal_voc/pspnet101_init',
                        help='Working directory')
    parser.add_argument('--sub_dir', default='',
                        help='Subdirectory to store the model testing results. '
                             'For example, if it is set to "val", the testing '
                             'results will be saved in <work_dir>/results/val/ '
                             'folder. By default, the results are saved in '
                             '<work_dir>/results/ directly.')
    parser.add_argument('--image_list', required=True,
                        help='List of images to test on. This is required '
                             'for context module to deal with variable image '
                             'size.')
    parser.add_argument('--weights', required=True)
    parser.add_argument('--deploy_net', required=True)
    parser.add_argument('--info', required=True)
    parser.add_argument('--mean', nargs='*', default=[104.008, 116.669, 122.675], type=float,
                        help='Mean pixel value (BGR) for the dataset.\n'
                             'Default is the mean pixel of PASCAL dataset.')
    parser.add_argument('--classes', type=int, required=True,
                        help='Number of categories in the data')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU for testing. If it is less than 0, '
                             'CPU is used instead.')
    
    options = process_options(parser.parse_args())    
    test_image(options)


if __name__ == '__main__':
    main()
