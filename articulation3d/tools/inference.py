import argparse
import json
import numpy as np
import os
from regex import P
import torch
import torch.nn.functional as F
from collections import defaultdict
import cv2
from tqdm import tqdm
import pickle
import imageio
import random
import math
import pdb
from glob import glob

import pycocotools.mask as mask_util
from fvcore.common.file_io import PathManager
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures.boxes import pairwise_iou, pairwise_ioa
from detectron2.structures import Boxes, BoxMode, Instances
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import GenericMask, ColorMode
from detectron2.config import get_cfg

import pytorch3d
from pytorch3d.structures.meshes import Meshes
from pytorch3d.renderer import Textures
from pytorch3d.utils import ico_sphere

import articulation3d.modeling  # noqa
from articulation3d.data import PlaneRCNNMapper
from articulation3d.data.planercnn_transforms import axis_to_angle_offset, angle_offset_to_axis
from articulation3d.visualization.unit_vector_plot import get_normal_figure
from articulation3d.evaluation import ArtiEvaluator
from articulation3d.config import get_planercnn_cfg_defaults
from articulation3d.utils.vis import get_pcd, project2D, random_colors, get_single_image_mesh_arti
from articulation3d.utils.mesh_utils import save_obj, get_camera_meshes, transform_meshes, rotate_mesh_for_webview
from articulation3d.utils.opt_utils import track_planes, optimize_planes
from articulation3d.utils.arti_vis import create_instances, PlaneRCNN_Branch, draw_pred, draw_gt, get_normal_map, draw_mask
from articulation3d.utils.visualizer import ArtiVisualizer

def transform_image(im):
    height = im.shape[0]
    width = im.shape[1]
    center = np.array(im.shape[:2]) / 2
    if height > 480:
        y = int(center[0] - 480/2)
        im = im[y:y+480, :, :]
    elif height < 480:
        pad = int((480 - height)/2)
        if (480 - height) % 2 == 0:
            im = cv2.copyMakeBorder(
                im.copy(), pad, pad, 0, 0, cv2.BORDER_CONSTANT, value=0)
        else:
            im = cv2.copyMakeBorder(
                im.copy(), pad, pad+1, 0, 0, cv2.BORDER_CONSTANT, value=0)
    if width > 640:
        x = int(center[1] - 640/2)
        im = im[:, x:x+640, :]
    elif width < 640:
        pad = int((640-width)/2)
        if (640-width) % 2 == 0:
            im = cv2.copyMakeBorder(
                im.copy(), 0, 0, pad, pad, cv2.BORDER_CONSTANT, value=0)
        else:
            im = cv2.copyMakeBorder(
                im.copy(), 0, 0, pad, pad+1, cv2.BORDER_CONSTANT, value=0)

    return im

def save_metric_output(args, preds, frames, frame_id, motion_type, axis_dir='l'):
    p_instance = preds[frame_id]
    if p_instance.scores.shape[0] == 0:
        print("no prediction!")
        return

    box_id = p_instance.scores.argmax()
    vis = ArtiVisualizer(frames[frame_id])
    im = frames[frame_id]

    # computing the rotation axis
    pred_mask = p_instance.pred_masks[box_id]
    pred_plane = p_instance.pred_planes[box_id:(box_id + 1)].clone()
    pred_plane[:, [1, 2]] = pred_plane[:, [2, 1]]
    pred_plane[:, 1] = - pred_plane[:, 1]
    pred_box_centers = p_instance.pred_boxes.get_centers()

    pts = angle_offset_to_axis(p_instance.pred_rot_axis, pred_box_centers)
    verts = pred_mask.nonzero().flip(1)
    normal = F.normalize(pred_plane, p=2)[0]
    offset = torch.norm(pred_plane, p=2)
    verts_axis = pts[box_id].reshape(-1, 2)
    verts_axis_3d = get_pcd(verts_axis, normal, offset)

    if args.webvis:
        # 3d transformation for model-viewer
        verts_axis_3d = torch.tensor((np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])@np.array(
            [[-1, 0, 0], [0, -1, 0], [0, 0, 1]])@verts_axis_3d.numpy().T).T)
    dir_vec = verts_axis_3d[1] - verts_axis_3d[0]
    dir_vec = dir_vec / np.linalg.norm(dir_vec)

    # computing pcd
    plane_params = p_instance.pred_planes[box_id:(box_id + 1)]
    segmentations = p_instance.pred_masks[box_id:(box_id + 1)]
    reduce_size = False
    height = 480*2
    width = 640*2

    mesh, uv_maps = get_single_image_mesh_arti(
        plane_params, segmentations, img=im, height=height, width=width, webvis=args.webvis, reduce_size=reduce_size)

    output_dir = os.path.join(args.output, 'frame_{:0>4}'.format(frame_id))
    # output_dir = os.path.join(args.output, 'frame_reference'.format(frame_id))
    os.mkdir(output_dir)
    basename = 'arti'
    save_obj(output_dir, basename+'_pred', mesh, uv_maps=uv_maps)
    
    motion_para = {
        'motion_type': motion_type,
        'motion_axis': dir_vec.cpu().numpy().tolist(),
        'motion_origin': verts_axis_3d[0].cpu().numpy().tolist(),
        # 'motion_state': p_instance.angle
        'motion_state': []
    }
    with open(f'{output_dir}/motion_para.json', 'w') as fp:
        json.dump(motion_para, fp)


def get_single_video_results(args, path, model, metadata, cls_name_map):
    video_path = path
    is_video = True
    if video_path.endswith('mp4'):
        reader = imageio.get_reader(video_path)
        fps = reader.get_meta_data()['fps']
    elif video_path.endswith('png'):  # accept images as input
        is_video = False
        reader = [imageio.imread(video_path)]

    frames = []
    preds = []
    org_vis_list = []
    seg_list = []
    for i, im in enumerate(tqdm(reader)):
        height = im.shape[0]
        width = im.shape[1]
        im = cv2.resize(im, (int(width*(517.97/983)),
                        int(height*(517.97/983))))
        im = transform_image(im)
        # pdb.set_trace()

        frames.append(im)
        im = im[:, :, ::-1]
        pred = model.inference(im)
        pred_dict = model.process(pred)
        p_instance = create_instances(
            pred_dict['instances'], im.shape[:2],
            pred_planes=pred_dict['pred_plane'].numpy(),
            pred_rot_axis=pred_dict['pred_rot_axis'],
            pred_tran_axis=pred_dict['pred_tran_axis'],
            conf_threshold=args.conf_threshold,
        )

        preds.append(p_instance)

        # visualization without optmization
        if args.output is not None:
            vis = ArtiVisualizer(im[:, :, ::-1])
            seg_pred = draw_pred(vis, p_instance, metadata,
                                 cls_name_map, conf_threshold=args.conf_threshold)

            # import pdb
            # pdb.set_trace()
            # surface normal
            if len(p_instance.pred_boxes) == 0:
                normal_vis = get_normal_map(torch.tensor(
                    [[1., 0, 0]]), torch.zeros(1, 480, 640))
                seg = seg_pred
            else:
                normal_vis = get_normal_map(
                    p_instance.pred_planes, p_instance.pred_masks.cpu())
            # get the frame with pred mask, bbox, axis
            # mask = p_instance.pred_masks.cpu().permute(1,2,0)[:,:,:1].numpy()
            # zero_mask = np.zeros((seg_pred.shape[0], seg_pred.shape[1], seg_pred.shape[2]))
            # zero_mask[:,:,0] = zero_mask[:,:,0] + mask[:,:,0]*255
            # zero_mask = zero_mask.astype(np.uint8)
            # seg = cv2.addWeighted(seg_pred, 1, zero_mask, 0.5, 0)
                seg = draw_mask(p_instance.pred_masks, seg_pred)

            # combine visualization and generate output
            combined_vis = np.concatenate((seg_pred, normal_vis), axis=1)
            org_vis_list.append(combined_vis)
            seg_list.append(seg)
            
            # save 2D mask
            # mask_num = str(i).zfill(4)
            # np.save(f'{args.output}/Mask_{mask_num}', p_instance.pred_masks.cpu().numpy())
    if is_video:
        reader.close()

    # temporal optimization
    planes = track_planes(preds)
    opt_preds, cluster, rsq, ref_idx = optimize_planes(
        preds, planes, '3dc', frames=frames)

    is_video = True
    # video visualization in 2D
    if is_video:
        writer = imageio.get_writer(os.path.join(
            args.output, '{}.mp4'.format('output')), fps=fps)
    else:
        write_path = os.path.join(args.output, '{}.png'.format('output'))

    for i, im in (enumerate(frames)):
        p_instance = opt_preds[i]
        org_vis = org_vis_list[i]

        vis = ArtiVisualizer(im)

        seg_pred = draw_pred(vis, p_instance, metadata, cls_name_map)

        # surface normal
        if len(p_instance.pred_boxes) == 0:
            normal_vis = get_normal_map(torch.tensor(
                [[1., 0, 0]]), torch.zeros(1, 480, 640))
            seg = seg_pred
        else:
            normal_vis = get_normal_map(
                p_instance.pred_planes, p_instance.pred_masks.cpu())

            seg = draw_mask(p_instance.pred_masks, seg_pred)
        # combine visualization and generate output

        # combined_vis = np.concatenate((seg_pred, normal_vis, org_vis), axis=1)

        seg = seg_list[i]

        if is_video:
            writer.append_data(seg)
            write_path = f"{args.output}/output_{i}.png"
        else:
            # imageio.imwrite(write_path, combined_vis)
            write_path = f"{args.output}/output_{i}.png"
            imageio.imwrite(write_path, seg)
            
    if args.save_obj:
        # select frame_ids you want to visualize
        # frame_ids = [0, 30, 60, 90]

        if ref_idx['trans'] != []:
            frame_ids = ref_idx['trans']
            motion_type = 'translation'
        else:
            frame_ids = ref_idx['rot']
            motion_type = 'rotation'

        frame_ids = np.arange(len(frames))

        print("<================Cluster Info====================>")
        print(cluster)
        print("<================RSQ Value====================>")
        print(rsq)
        print("<================Reference ID====================>")
        print(ref_idx)


        pdb.set_trace()
        for frame_id in frame_ids:
            save_metric_output(args, opt_preds, frames, frame_id, motion_type)
            # pdb.set_trace()
    return



def main():
    random.seed(2020)
    np.random.seed(2020)

    # command line arguments
    parser = argparse.ArgumentParser(
        description="A script that generates results of articulation prediction."
    )
    parser.add_argument("--config", required=True, help="config/config.yaml")
    parser.add_argument("--input", required=True, help="input video series name")
    parser.add_argument("--output", required=True, help="output directory")
    parser.add_argument('--save-obj', action='store_true')
    parser.add_argument('--webvis', action='store_true')
    parser.add_argument("--conf-threshold", default=0.7,
                        type=float, help="confidence threshold")
    args = parser.parse_args()

    # create output directory
    # os.makedirs(args.output, exist_ok=True)

    # setup logger
    logger = setup_logger()

    # load model
    cfg = get_cfg()
    get_planercnn_cfg_defaults(cfg)
    cfg.merge_from_file(args.config)
    model = PlaneRCNN_Branch(cfg)
    shortened_class_names = {'arti_rot': 'R', 'arti_tran': 'T'}
    metadata = MetadataCatalog.get('arti_train')
    cls_name_map = [shortened_class_names[cls]
                    for cls in metadata.thing_classes]   

    videos = open(args.input,'r').read().splitlines()

    output_dir = args.output
    for video in videos:
        path = f"../dataset/{video.replace('_b', '/b', 1)}/frames/video.mp4"
        args.output = f"{output_dir}/{video.replace('_b', '/b', 1)}"
        os.makedirs(args.output, exist_ok=True)
        get_single_video_results(args, path, model, metadata, cls_name_map)




if __name__ == "__main__":
    main()