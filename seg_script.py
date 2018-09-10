#!/home/mbajaj01/anaconda3/envs/py27/bin/python

# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################

"""Perform inference on a single image or all images with a certain extension
(e.g., .jpg) in a folder.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import defaultdict
import argparse
import cv2  # NOQA (Must import before importing caffe2 due to bug in cv2)
import glob
import logging
import os
import sys
sys.path.append('lib/')




import time
import pickle
from caffe2.python import workspace

from core.config import assert_and_infer_cfg
from core.config import cfg
from core.config import merge_cfg_from_file
from utils.io import cache_url
from utils.timer import Timer
import core.test_engine as infer_engine
import datasets.dummy_datasets as dummy_datasets
import utils.c2 as c2_utils
import utils.logging
import utils.vis as vis_utils

c2_utils.import_detectron_ops()
# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)


def parse_args():
    parser = argparse.ArgumentParser(description='End-to-end inference')
    parser.add_argument(
        '--cfg',
        dest='cfg',
        help='cfg model file (/path/to/model_config.yaml)',
        #default='configs/bbox2mask_vg/eval_sw_R101/runtest_clsbox_2_layer_mlp_nograd_R101.yaml',
        default='configs/bbox2mask_vg/eval_sw/runtest_clsbox_2_layer_mlp_nograd.yaml',
        type=str
    )
    parser.add_argument(
        '--wts',
        dest='weights',
        help='weights model file (/path/to/model_weights.pkl)',
        default='lib/datasets/data/trained_models/33241332_model_final_coco2vg3k_seg.pkl',
        #default='lib/datasets/data/trained_models/33219850_model_final_coco2vg3k_seg.pkl',
        type=str
    )
    parser.add_argument(
        '--output-dir',
        dest='output_dir',
        help='directory for visualization pdfs (default: /tmp/infer_simple)',
        #default='bboxes_100_res50',
        default='/media/mynewdrive/activityNet/bboxes_50_res50',
        type=str
    )
    parser.add_argument(
        '--image-ext',
        dest='image_ext',
        help='image file name extension (default: jpg)',
        default='jpg',
        type=str
    )
    parser.add_argument(
        '--im_or_folder',
        dest='im_or_folder',
        help='image or folder of images',
        #default='jpg_videos/',
        default='/media/mynewdrive/activityNet/fps_videos/',
        #default='demo_vg3k/',
        type=str
    )
    parser.add_argument(
        '--use-vg3k',
        dest='use_vg3k',
        default=True,
        help='use Visual Genome 3k classes (instead of COCO 80 classes)',
        action='store_true'
    )
    parser.add_argument(
        '--thresh',
        default=0.3,
        #default=0.7,
        type=float,
        help='score threshold for predictions',
    )
#     if len(sys.argv) == 1:
#         parser.print_help()
#         sys.exit(1)
    import sys
    sys.argv=['']
    #del sys
    return parser.parse_args()



if __name__ == '__main__':
    log_dir = 'logs'
    for i in range(100):
        log_file = 'log_' + str(i) + '.txt'
        log_path = os.path.join(log_dir, log_file)
        if(os.path.exists(log_path)):
            continue
        else:
            sys.stdout = open(log_path, "w")
            break
    
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
    utils.logging.setup_logging(__name__)
    args = parse_args()
    #main(args)
    logger = logging.getLogger(__name__)
    first_try = True
    if first_try:
        merge_cfg_from_file(args.cfg)
        cfg.NUM_GPUS = 1
        args.weights = cache_url(args.weights, cfg.DOWNLOAD_CACHE)
        assert_and_infer_cfg(cache_urls=False)
        model = infer_engine.initialize_model_from_cfg(args.weights)
        #print(model)
        #return
        dummy_coco_dataset = (
            dummy_datasets.get_vg3k_dataset()
            if args.use_vg3k else dummy_datasets.get_coco_dataset())
    
    seg_vids = []
    skip_vids = []
    pickle_path = "./seg_vids.p"
    skip_path = "./skip_vids.p"
    count = 0
    processed_count = 0
    average_time = 0.
    skipVid = False
    if(os.path.exists(pickle_path)):   
        seg_vids = pickle.load( open( pickle_path , "rb" ) )
        processed_count = len(seg_vids)
        print("\n**************************************************\n")
        print("\n***********Already processed videos: ", len(seg_vids))
        print("\n**************************************************\n")
        
    if(os.path.exists(skip_path)):   
        skip_vids = pickle.load( open( skip_path , "rb" ) )
        print("\n**************************************************\n")
        print("\n***********Already skipped videos: ", len(skip_vids))
        print("\n**************************************************\n")
    
    dirs = os.listdir(args.im_or_folder)
    for dir_name in dirs:
        vid_timers = defaultdict(Timer)
        vid_timers['video_bbox'].tic()
        #print(dir_name)
        if dir_name in seg_vids:
            continue
        if dir_name in skip_vids:
            continue   
        video_path = os.path.join(args.im_or_folder, dir_name)
        images = os.listdir(video_path)    
        im_dict = {}
    

    
#     if os.path.isdir(args.im_or_folder):
#         im_list = glob.iglob(args.im_or_folder + '/*.' + args.image_ext)
#     else:
#         im_list = [args.im_or_folder]

        out_name = os.path.join(
                args.output_dir, '{}'.format(os.path.basename(dir_name) + '.p')
            )
        logger.info('Processing {} -> {}'.format(dir_name, out_name))


        for i, image in enumerate(images):
            im_name, ext = os.path.splitext(image)
            
            #print(image)
            im_path = os.path.join(video_path, image)
            im = cv2.imread(im_path)
            if(im is None):
                print("**Problem detected with: ",dir_name, " Skipping ...")
                skip_vids.append(dir_name)
                print("Total videos skipped: ",len(skip_vids))
                skipVid = True
                break
            timers = defaultdict(Timer)
            t = time.time()
            with c2_utils.NamedCudaScope(0):
        
                ans_scores, ans_boxes, cls_boxes, im_scales, fc7_feats, im_info = infer_engine.im_detect_all(model, im, None, timers=timers)


   

            im_dict[im_name] = {"scores":ans_scores, "boxes":ans_boxes, "cls_boxes":cls_boxes,
                                "im_scales": im_scales, "fc7_feats": fc7_feats, "im_info":im_info}


        if skipVid:
            skipVid = False
            pickle.dump( skip_vids, open( skip_path, "wb" ) )
            continue
        pickle.dump( im_dict, open( out_name, "wb" ) ) 
        seg_vids.append(dir_name)
        pickle.dump( seg_vids, open( pickle_path, "wb" ) ) 
        count +=1
        processed_count += 1
        print("\n**************************************************\n")
        print("Done: ", dir_name)
        print("Processed in this batch: ", count)
        print("Processed in total: ", processed_count)
        print("\n**************************************************\n")
        vid_timers['video_bbox'].toc()
        k='video_bbox'
        v=vid_timers[k]
        logger.info(' | {}: {:.3f}s'.format(k, v.average_time))
        average_time += v.average_time
        print("Average time per video: ",(average_time/count))



#         vis_utils.vis_one_image(
#             im[:, :, ::-1],  # BGR -> RGB for visualization
#             im_name,
#             args.output_dir,
#             cls_boxes,
#             cls_segms,
#             cls_keyps,
#             dataset=dummy_coco_dataset,
#             box_alpha=0.3,
#             show_class=True,
#             thresh=args.thresh,
#             kp_thresh=2
#         )