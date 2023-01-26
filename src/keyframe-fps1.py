import sys
import os
sys.path.append(os.path.dirname(__file__).replace('src', '', 1))
from   src.utils.inputVideo import InputVideo
import src.utils.localFeatures as local
import src.utils.imageHistogram as hist
import shutil
import time
import cv2 as cv
import time
import pdb

config = {
'width':299,
'height':299,
'sampling_rate':1,
'use_cached_cnn':False,
'cnn_params':{'model':'keras'},
'use_multiprocessing':True,
'scene_cut_features':'color',
'min_scene_length':2,
'scene_cut_thresh':0.65,
'scene_cut_features_params':{'difference_metric':'correlation'},
'clustering':'kmeans',
'scene_based_removal':True,
'global_removal':False,
'scene_processing_features':'cnn',
'scene_processing_features_params':{'model':'keras'},
'global_removal_thresh':0.82,
'global_hsv_thresh':0.85,
'scene_based_removal_thresh':0.8,
'cnn_vects_path':'cached_cnn_vects_2'}

def summarize(input_video, out_dir):
    t1 = time.time()
    video = InputVideo(input_video,out_dir,config,resize = False)
    print("Processing Video: {}".format(video.getVideoName()))
    print("######################################################")
    # sampled_video = video.getSampledInputVideo(config['sampling_rate'])
    video.getAdjacentDifferenceList('cnn', config['cnn_params'], loadCNNfromCache=config['use_cached_cnn'])
    if(config['use_multiprocessing']):
        kfs = video.generateKeyframes_multiprocessing()
    else:
        kfs = video.generateKeyframes_sequential()
    # os.remove(sampled_video.path)  # remove sampled video
    kf_path = 'kfs/' + video.getVideoName()[:video.getVideoName().find('.')]
    if(os.path.isdir(kf_path)):
        shutil.rmtree(kf_path)  # remove old results if any
    os.makedirs(kf_path)
    for i, kf in enumerate(kfs):
        cv.imwrite('{}/{}.jpg'.format(kf_path, i), kf.image)
    # pdb.set_trace()
    out_file=open(out_dir+f"/frame_indices_keyframe_extraction_{config['sampling_rate']}.txt",'w')
    for kf in kfs:
        out_file.write(str(kf.index)+'\n')
    print("Formatted length in seconds: {}".format(video.getFormattedVideoLenghtInSeconds()))
    print("Time: {}".format(time.time() - t1))
    kfs.sort(key = lambda k: k.index)
    return [kf.image for kf in kfs], video.getFormattedVideoLenghtInSeconds()


if __name__ == '__main__':
    input_video = sys.argv[1] if len(sys.argv) > 1 else None
    out_dir = sys.argv[2]
    if(input_video != None):
        summarize(input_video, out_dir)
    else:
        print("Missing Video Path Argument")
