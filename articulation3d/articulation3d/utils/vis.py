import numpy as np
import mapbox_earcut as earcut
import torch
from collections import defaultdict
from torch.nn.utils.rnn import pad_sequence
from PIL import Image, ImageDraw
import imageio
import cv2
import colorsys
import pdb

# pytorch3d has removed pytorch3d.structures.Textures
import pytorch3d
from pytorch3d.structures import Meshes#, Textures
from pytorch3d.renderer.mesh import TexturesVertex, TexturesUV, Textures

from pycocotools import mask
from .mesh_utils import save_obj
from .pycococreatortools import binary_mask_to_polygon
from detectron2.structures.masks import polygons_to_bitmask



def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    np.random.shuffle(colors)
    return colors


def precompute_K_inv_dot_xy_1(h=480, w=640):
        focal_length = 517.97
        offset_x = 320
        offset_y = 240

        K = [[focal_length, 0, offset_x],
             [0, focal_length, offset_y],
             [0, 0, 1]]

        K_inv = np.linalg.inv(np.array(K))

        K_inv_dot_xy_1 = np.zeros((3, h, w))
        for y in range(h):
            for x in range(w):
                yy = float(y) / h * 480
                xx = float(x) / w * 640
                
                ray = np.dot(K_inv,
                             np.array([xx, yy, 1]).reshape(3, 1))
                K_inv_dot_xy_1[:, y, x] = ray[:, 0]

        # precompute to speed up processing
        return K_inv_dot_xy_1


def project2D(pcd,h=480,w=640,focal_length=368.635):
    #pcd is Nx3
    offset_x = w/2
    offset_y = h/2
    K = [[focal_length, 0, offset_x],
        [0, focal_length, offset_y],
        [0, 0, 1]]

    #pdb.set_trace()
    if torch.is_tensor(pcd) and pcd.is_cuda:
        K = torch.FloatTensor(K).cuda()
        proj = (K@(pcd.T)).T
        proj = proj[:,:2] / proj[:,2][:,None]
        return proj 

    #proj is Nx2
    proj = (np.array(K)@(pcd.T)).T
    proj = proj[:,:2] / proj[:,2][:,None]
    return proj 


    
def get_pcd(verts, normal, offset, h=480, w=640, focal_length=368.635):
    """
    convert 2d verts to 3d point cloud based on plane normal and offset
    depth = offset / n \dot K^{-1}q
    """
    offset_x = w/2
    offset_y = h/2
    K = [[focal_length, 0, offset_x],
        [0, focal_length, offset_y],
        [0, 0, 1]]
    K_inv = np.linalg.inv(np.array(K))
    homogeneous = np.hstack((verts, np.ones(len(verts)).reshape(-1,1)))

    ray = K_inv@homogeneous.T
    depth = offset / np.dot(normal, ray)
    pcd = depth.reshape(-1,1) * ray.T
    return pcd


def get_pcd_depth(verts, depth, h=480, w=640, focal_length = 517.97):
    """
    convert 2d verts to 3d point cloud based on depth map
    depth = offset / n \dot K^{-1}q
    """
    offset_x = w/2
    offset_y = h/2
    K = [[focal_length, 0, offset_x],
        [0, focal_length, offset_y],
        [0, 0, 1]]
    K_inv = np.linalg.inv(np.array(K))
    homogeneous = np.hstack((verts, np.ones(len(verts)).reshape(-1,1)))
    ray = K_inv@homogeneous.T
    pcd = depth[tuple(np.transpose(verts))].reshape(-1,1) * ray.T
    return pcd


def rle2polygon(segmentations):
    """
    convert rle format segmentation to polygon
    """
    assert isinstance(segmentations[0], dict)
    # decode
    binary_masks = mask.decode(segmentations).transpose(2,0,1)
    # binary masks 2 polygon masks
    poly_masks = [binary_mask_to_polygon(bm) for bm in binary_masks]
    return poly_masks


def get_single_image_mesh_plane(plane_params, segmentations, img_file, height=480, width=640, focal_length=571.623718, webvis=False, reduce_size=True):
    plane_params = np.array(plane_params)
    plane_params[:, [1,2]] = plane_params[:, [2, 1]]
    plane_params[:, 1] = -plane_params[:, 1]
    offsets = np.linalg.norm(plane_params, ord=2, axis=1)
    norms = plane_params / offsets.reshape(-1,1)

    if type(segmentations[0]) == dict:
        poly_segmentations = rle2polygon(segmentations)
    else:
        poly_segmentations = segmentations
    verts_list = []
    faces_list = []
    verts_uvs = []
    uv_maps = []
    imgs = []

    for planeI,(segm, normal, offset) in enumerate(zip(poly_segmentations, norms, offsets)):
        if len(segm) == 0:
            continue
        
        #####DF#####
        I = np.array(imageio.imread(img_file))
        #####DF#####

        warped_img = None
        HUse = None

        # save uv_map
        tmp_verts = []
        for s in segm:
            tmp_verts.extend(s)
        tmp_verts = np.array(tmp_verts).reshape(-1,2)
        #####DF#####
        #pick an arbitrary point
        # get 3d pointcloud
        tmp_pcd = get_pcd(tmp_verts, normal, offset, focal_length)  
        point0 = tmp_pcd[0,:]
        #pick the furthest point from here
        dPoint0 = np.sum((tmp_pcd-point0[np.newaxis,:])**2,axis=1)
        point1 = tmp_pcd[np.argmax(dPoint0),:]

        #dir1 and dir2 are orthogonal to the normal
        dir1 = (point1 - point0)
        dir1 = dir1 / np.linalg.norm(dir1)
        dir2 = np.cross(dir1,normal)

        #control points in 3D 
        control3D = [point0, point0+dir1, point0+dir2, point0+dir1+dir2]
        control3D = np.vstack([p[None,:] for p in control3D])
        control3DProject = project2D(control3D, focal_length)

        #pick an arbitrary square
        targetSize = 300
        fakePoints = np.array([[0,0],[0,targetSize],[targetSize,0],[targetSize,targetSize]]).astype(np.float32)

        #fit, then adjust
        H = cv2.getPerspectiveTransform(control3DProject.astype(np.float32),fakePoints)
        #this maps the control points to the square; now make sure the full mask warps in
        P = cv2.perspectiveTransform(tmp_verts.reshape(1,-1,2),H)[0,:,:] 
        xTrans, yTrans = P[:,0].min(), P[:,1].min()
        maxScale = max(P[:,0].max() - P[:,0].min(), P[:,1].max() - P[:,1].min())
        HShuffle = np.array([[targetSize/maxScale, 0, -xTrans * targetSize / maxScale],[0, targetSize/maxScale, -yTrans * targetSize / maxScale],[0,0,1]])
        HUse = HShuffle@H

        #warped_image is now the rectified image; warped_image2 has it with a 100px fudge factor
        warped_image = cv2.warpPerspective(I, HUse, (targetSize,targetSize))
        uv_maps.append(warped_image)

        verts_3d = []
        faces = []
        uvs = []

        for ring_idx, ring in enumerate(segm):
            verts = np.array(ring).reshape(-1,2)
            # get 3d pointcloud
            pcd = get_pcd(verts, normal, offset, focal_length)            

            if webvis:
                # Rotate by 11 degree around x axis to push things on the ground.
                # pcd = (np.array([[-1,0,0], [0,1,0], [0,0,-1]])@np.array([[1,0,0],[0,0.9816272,-0.1908090],[0,0.1908090,0.9816272]])@np.array([[-1,0,0],[0,-1,0],[0,0,1]])@pcd.T).T
                pcd = (np.array([[-1,0,0], [0,1,0], [0,0,-1]])@np.array([[-1,0,0],[0,-1,0],[0,0,1]])@pcd.T).T
            #####DF#####
            uvsPoly = np.array([0,1]) + np.array([1,-1])*verts/np.array([width, height])
            uvsRectified = cv2.perspectiveTransform(verts.astype(np.float32).reshape(1,-1,2),HUse)[0,:,:]
            uvsRectified = np.array([0,1]) + np.array([1,-1])*uvsRectified/np.array([targetSize, targetSize])
            uvs.extend(uvsRectified)
            #####DF#####

            # triangulate polygon using earcut algorithm
            triangles = earcut.triangulate_float32(verts, [len(verts)])
            #pdb.set_trace()
            # triangles = earcut.triangulate_float32(verts_2d, rings)
            # add base index of vertice
            triangles += len(verts_3d)
            triangles = triangles.reshape(-1,3)
            # convert to counter-clockwise
            triangles[:,[0, 2]] = triangles[:,[2, 0]]
            
            if triangles.shape[0] == 0:
                continue

            verts_3d.extend(pcd)
            faces.extend(triangles)

        verts_list.append(torch.tensor(verts_3d, dtype=torch.float32))
        faces_list.append(torch.tensor(faces, dtype=torch.int32))
        verts_uvs.append(torch.tensor(uvs, dtype=torch.float32))
        imgs.append(torch.FloatTensor(imageio.imread(img_file)))


    #pdb.set_trace()

    # pytorch3d mesh
    verts_uvs = pad_sequence(verts_uvs, batch_first=True)
    faces_uvs = pad_sequence(faces_list, batch_first=True, padding_value=-1)
    tex = Textures(verts_uvs=verts_uvs, faces_uvs=faces_uvs, maps=imgs)
    meshes = Meshes(verts=verts_list, faces=faces_list, textures=tex)

    return meshes, uv_maps


def get_single_image_mesh_arti(plane_params, segmentations, img, height=480, width=640, focal_length=571.623718, webvis=False, reduce_size=True):
    plane_params = np.array(plane_params)
    plane_params[:, [1,2]] = plane_params[:, [2, 1]]
    plane_params[:, 1] = -plane_params[:, 1]
    offsets = np.linalg.norm(plane_params, ord=2, axis=1)
    norms = plane_params / offsets.reshape(-1,1)

    #pdb.set_trace()

    poly_segmentations = []
    for i in range(segmentations.shape[0]):
        poly_segmentations.append(binary_mask_to_polygon(segmentations[i]))
    
    #if type(segmentations[0]) == dict:
    #    poly_segmentations = rle2polygon(segmentations)
    #else:
    #    poly_segmentations = segmentations
    
    verts_list = []
    faces_list = []
    verts_uvs = []
    uv_maps = []
    imgs = []

    for planeI,(segm, normal, offset) in enumerate(zip(poly_segmentations, norms, offsets)):
        if len(segm) == 0:
            continue

        #####DF#####
        #I = np.array(imageio.imread(img_file))
        I = img
        #pdb.set_trace()
        #####DF#####

        warped_img = None
        HUse = None

        # save uv_map
        tmp_verts = []
        for s in segm:
            tmp_verts.extend(s)
        tmp_verts = np.array(tmp_verts).reshape(-1,2)
        #####DF#####
        #pick an arbitrary point
        # get 3d pointcloud
        tmp_pcd = get_pcd(tmp_verts, normal, offset, focal_length = focal_length)  
        point0 = tmp_pcd[0,:]
        #pick the furthest point from here
        dPoint0 = np.sum((tmp_pcd-point0[np.newaxis,:])**2,axis=1)
        point1 = tmp_pcd[np.argmax(dPoint0),:]

        #dir1 and dir2 are orthogonal to the normal
        dir1 = (point1 - point0)
        dir1 = dir1 / np.linalg.norm(dir1)
        dir2 = np.cross(dir1,normal)

        #control points in 3D 
        control3D = [point0, point0+dir1, point0+dir2, point0+dir1+dir2]
        control3D = np.vstack([p[None,:] for p in control3D])
        control3DProject = project2D(control3D, focal_length = focal_length)

        #pick an arbitrary square
        targetSize = 300
        fakePoints = np.array([[0,0],[0,targetSize],[targetSize,0],[targetSize,targetSize]]).astype(np.float32)

        #fit, then adjust
        H = cv2.getPerspectiveTransform(control3DProject.astype(np.float32),fakePoints)
        #this maps the control points to the square; now make sure the full mask warps in
        P = cv2.perspectiveTransform(tmp_verts.reshape(1,-1,2),H)[0,:,:] 
        xTrans, yTrans = P[:,0].min(), P[:,1].min()
        maxScale = max(P[:,0].max() - P[:,0].min(), P[:,1].max() - P[:,1].min())
        HShuffle = np.array([[targetSize/maxScale, 0, -xTrans * targetSize / maxScale],[0, targetSize/maxScale, -yTrans * targetSize / maxScale],[0,0,1]])
        HUse = HShuffle@H

        #warped_image is now the rectified image; warped_image2 has it with a 100px fudge factor
        warped_image = cv2.warpPerspective(I, HUse, (targetSize,targetSize))
        uv_maps.append(warped_image)

        verts_3d = []
        faces = []
        uvs = []

        for ring_idx, ring in enumerate(segm):
            verts = np.array(ring).reshape(-1,2)
            # get 3d pointcloud
            pcd = get_pcd(verts, normal, offset, focal_length)            

            if webvis:
                # Rotate by 11 degree around x axis to push things on the ground.
                # pcd = (np.array([[-1,0,0], [0,1,0], [0,0,-1]])@np.array([[1,0,0],[0,0.9816272,-0.1908090],[0,0.1908090,0.9816272]])@np.array([[-1,0,0],[0,-1,0],[0,0,1]])@pcd.T).T
                pcd = (np.array([[-1,0,0], [0,1,0], [0,0,-1]])@np.array([[-1,0,0],[0,-1,0],[0,0,1]])@pcd.T).T
            #####DF#####
            uvsPoly = np.array([0,1]) + np.array([1,-1])*verts/np.array([width, height])
            uvsRectified = cv2.perspectiveTransform(verts.astype(np.float32).reshape(1,-1,2),HUse)[0,:,:]
            uvsRectified = np.array([0,1]) + np.array([1,-1])*uvsRectified/np.array([targetSize, targetSize])
            uvs.extend(uvsRectified)
            #####DF#####

            # triangulate polygon using earcut algorithm
            triangles = earcut.triangulate_float32(verts, [len(verts)])
            #pdb.set_trace()
            # triangles = earcut.triangulate_float32(verts_2d, rings)
            # add base index of vertice
            triangles += len(verts_3d)
            triangles = triangles.reshape(-1,3)
            # convert to counter-clockwise
            triangles[:,[0, 2]] = triangles[:,[2, 0]]
            
            if triangles.shape[0] == 0:
                continue

            verts_3d.extend(pcd)
            faces.extend(triangles)

        verts_list.append(torch.tensor(verts_3d, dtype=torch.float32))
        #faces_list.append(torch.tensor(faces, dtype=torch.int32))
        faces = torch.tensor(faces, dtype=torch.int32)
        faces = faces.long()
        faces_list.append(faces)

        #verts_uvs.append(torch.tensor(uvs, dtype=torch.float32))
        uvs = torch.tensor(uvs, dtype=torch.float32)
        #uvs = uvs.long()
        verts_uvs.append(uvs)

        #imgs.append(torch.FloatTensor(imageio.imread(img_file)))
        imgs.append(torch.FloatTensor(img))


    #pdb.set_trace()

    # pytorch3d mesh
    verts_uvs = pad_sequence(verts_uvs, batch_first=True)
    faces_uvs = pad_sequence(faces_list, batch_first=True, padding_value=-1)
    tex = Textures(verts_uvs=verts_uvs, faces_uvs=faces_uvs, maps=imgs)
    meshes = Meshes(verts=verts_list, faces=faces_list, textures=tex)

    return meshes, uv_maps


"""
def get_single_image_mesh_plane(plane_params, segmentations, img_file, height=480, width=640, focal_length=368.635, webvis=False, reduce_size=True):
    plane_params = np.array(plane_params)
    offsets = np.linalg.norm(plane_params, ord=2, axis=1)
    norms = plane_params / offsets.reshape(-1,1)

    if type(segmentations[0]) == dict:
        poly_segmentations = rle2polygon(segmentations)
    else:
        poly_segmentations = segmentations
    verts_list = []
    faces_list = []
    verts_uvs = []
    uv_maps = []
    imgs = []

    for planeI,(segm, normal, offset) in enumerate(zip(poly_segmentations, norms, offsets)):
        if len(segm) == 0:
            continue
        verts_3d = []
        faces = []
        uvs = []

        #####DF#####
        I = np.array(imageio.imread(img_file))
        #####DF#####

        warped_img = None
        HUse = None

        rings = []
        verts_2d = []
        verts_base_id = 0

        for ring_idx, ring in enumerate(segm):
            #if ring_idx != 0:
            #    continue

            verts = np.array(ring).reshape(-1,2)
            verts_2d.extend(verts)
            # get 3d pointcloud
            pcd = get_pcd(verts, normal, offset, focal_length)


            if ring_idx == 0:
                tmp_verts = []
                for s in segm:
                    tmp_verts.extend(s)
                tmp_verts = np.array(tmp_verts).reshape(-1,2)
                #####DF#####
                #pick an arbitrary point
                point0 = pcd[0,:]
                #pick the furthest point from here
                dPoint0 = np.sum((pcd-point0[np.newaxis,:])**2,axis=1)
                point1 = pcd[np.argmax(dPoint0),:]

                #dir1 and dir2 are orthogonal to the normal
                dir1 = (point1 - point0)
                dir1 = dir1 / np.linalg.norm(dir1)
                dir2 = np.cross(dir1,normal)

                #control points in 3D 
                control3D = [point0, point0+dir1, point0+dir2, point0+dir1+dir2]
                control3D = np.vstack([p[None,:] for p in control3D])
                control3DProject = project2D(control3D, focal_length)

                #pick an arbitrary square
                targetSize = 300
                fakePoints = np.array([[0,0],[0,targetSize],[targetSize,0],[targetSize,targetSize]]).astype(np.float32)

                #fit, then adjust
                H = cv2.getPerspectiveTransform(control3DProject.astype(np.float32),fakePoints)
                #this maps the control points to the square; now make sure the full mask warps in
                P = cv2.perspectiveTransform(tmp_verts.reshape(1,-1,2),H)[0,:,:] 
                xTrans, yTrans = P[:,0].min(), P[:,1].min()
                maxScale = max(P[:,0].max() - P[:,0].min(), P[:,1].max() - P[:,1].min())
                HShuffle = np.array([[targetSize/maxScale, 0, -xTrans * targetSize / maxScale],[0, targetSize/maxScale, -yTrans * targetSize / maxScale],[0,0,1]])
                HUse = HShuffle@H

                #warped_image is now the rectified image; warped_image2 has it with a 100px fudge factor
                warped_image = cv2.warpPerspective(I, HUse, (targetSize,targetSize))
                
                #HFudgeFactor = np.array([[1,0,100],[0,1,100],[0,0,1]])
                #warped_image2 = cv2.warpPerspective(I, HFudgeFactor@HUse, (targetSize+200,targetSize+200))

                #imageio.imwrite("output/rectified_%d.png" % planeI,warped_image)
                #imageio.imwrite("output/rectified2_%d.png" % planeI,warped_image2)
                uv_maps.append(warped_image)
                #####DF#####

            

            if webvis:
                # Rotate by 11 degree around x axis to push things on the ground.
                pcd = (np.array([[-1,0,0], [0,1,0], [0,0,-1]])@np.array([[1,0,0],[0,0.9816272,-0.1908090],[0,0.1908090,0.9816272]])@np.array([[-1,0,0],[0,-1,0],[0,0,1]])@pcd.T).T
            
            verts_3d.extend(pcd)
            ring = len(verts) + verts_base_id
            rings.append(ring)
            verts_base_id += len(verts)

            #####DF#####
            uvsPoly = np.array([0,1]) + np.array([1,-1])*verts/np.array([width, height])
            uvsRectified = cv2.perspectiveTransform(verts.astype(np.float32).reshape(1,-1,2),HUse)[0,:,:]
            uvsRectified = np.array([0,1]) + np.array([1,-1])*uvsRectified/np.array([targetSize, targetSize])
            #pdb.set_trace()
            #uvs.extend(uvsPoly)
            uvs.extend(uvsRectified)
            #####DF#####

        # triangulate polygon using earcut algorithm
        #triangles = earcut.triangulate_float32(verts, [len(verts)])
        #pdb.set_trace()
        triangles = earcut.triangulate_float32(verts_2d, rings)
        # add base index of vertice
        #triangles += len(verts_3d)
        triangles = triangles.reshape(-1,3)
        # convert to counter-clockwise
        triangles[:,[0, 2]] = triangles[:,[2, 0]]
        
        if triangles.shape[0] == 0:
            continue

        faces.extend(triangles)

        verts_list.append(torch.tensor(verts_3d, dtype=torch.float32))
        faces_list.append(torch.tensor(faces, dtype=torch.int32))
        verts_uvs.append(torch.tensor(uvs, dtype=torch.float32))
        imgs.append(torch.FloatTensor(imageio.imread(img_file)))


    #pdb.set_trace()

    # pytorch3d mesh
    verts_uvs = pad_sequence(verts_uvs, batch_first=True)
    faces_uvs = pad_sequence(faces_list, batch_first=True, padding_value=-1)
    tex = Textures(verts_uvs=verts_uvs, faces_uvs=faces_uvs, maps=imgs)
    meshes = Meshes(verts=verts_list, faces=faces_list, textures=tex)

    return meshes, uv_maps
"""


def get_single_image_mesh(plane_params, segmentations, img_file, height=480, width=640, focal_length=368.635, webvis=False, reduce_size=True):
    plane_params = np.array(plane_params)
    offsets = np.linalg.norm(plane_params, ord=2, axis=1)
    norms = plane_params/offsets.reshape(-1,1)

    if type(segmentations[0]) == dict:
        poly_segmentations = rle2polygon(segmentations)
    else:
        poly_segmentations = segmentations
    verts_list = []
    faces_list = []
    verts_uvs = []
    img_files = []
    imgs = []

    for segm, normal, offset in zip(poly_segmentations, norms, offsets):
        if len(segm) == 0:
            continue
        verts_3d = []
        faces = []
        uvs = []
        if reduce_size:
            for ring in segm:
                verts = np.array(ring).reshape(-1,2)
                # get 3d pointcloud
                pcd = get_pcd(verts, normal, offset, focal_length)
                if webvis:
                    # Rotate by 11 degree around x axis to push things on the ground.
                    pcd = (np.array([[-1,0,0], [0,1,0], [0,0,-1]])@np.array([[1,0,0],[0,0.9816272,-0.1908090],[0,0.1908090,0.9816272]])@np.array([[-1,0,0],[0,-1,0],[0,0,1]])@pcd.T).T
                # triangulate polygon using earcut algorithm
                triangles = earcut.triangulate_float32(verts, [len(verts)])
                # add base index of vertice
                triangles += len(verts_3d)
                triangles = triangles.reshape(-1,3)
                # convert to counter-clockwise
                triangles[:,[0, 2]] = triangles[:,[2, 0]]
                verts_3d.extend(pcd)
                faces.extend(triangles)
                uvs.extend(np.array([0,1]) + np.array([1,-1])*verts/np.array([width, height]))
            
        else:
            bitmask = polygons_to_bitmask(segm, height=height, width=width)
            verts = np.transpose(bitmask.nonzero())
            vert_id_map = defaultdict(dict)
            for idx, vert in enumerate(verts):
                vert_id_map[vert[0]][vert[1]] = idx + len(verts_3d)

            verts_3d = get_pcd(verts[:,::-1], normal, offset)
            if webvis:
                # Rotate by 11 degree around x axis to push things on the ground.
                verts_3d = (np.array([[-1,0,0], [0,1,0], [0,0,-1]])@np.array([[1,0,0],[0,0.9816272,-0.1908090],[0,0.1908090,0.9816272]])@np.array([[-1,0,0],[0,-1,0],[0,0,1]])@pcd.T).T
            triangles = []
            for vert in verts:
                # upper right triangle
                if vert[0] < height-1 and vert[1] < width - 1 and bitmask[vert[0]][vert[1]+1] and bitmask[vert[0]+1][vert[1]+1]:
                    triangles.append([vert_id_map[vert[0]][vert[1]], vert_id_map[vert[0]+1][vert[1]+1], vert_id_map[vert[0]][vert[1]+1]])
                # bottom left triangle
                if vert[0] < height-1 and vert[1] < width - 1 and bitmask[vert[0]+1][vert[1]] and bitmask[vert[0]+1][vert[1]+1]:
                    triangles.append([vert_id_map[vert[0]][vert[1]], vert_id_map[vert[0]+1][vert[1]], vert_id_map[vert[0]+1][vert[1]+1]])
            triangles = np.array(triangles)
            faces.extend(triangles)
            uvs.extend(np.array([0,1]) + np.array([1,-1])*verts[:,::-1]/np.array([width, height]))
        verts_list.append(torch.tensor(verts_3d, dtype=torch.float32))
        faces_list.append(torch.tensor(faces, dtype=torch.int32))
        verts_uvs.append(torch.tensor(uvs, dtype=torch.float32))
        img_files.append(img_file)
        imgs.append(torch.FloatTensor(imageio.imread(img_file)))
    verts_uvs = pad_sequence(verts_uvs, batch_first=True)
    faces_uvs = pad_sequence(faces_list, batch_first=True, padding_value=-1)
    
    # Create a textures object
    #tex = Textures(verts_uvs=verts_uvs, faces_uvs=faces_uvs, map_files=img_files)
    #tex = TexturesVertex(verts_features=verts_uvs)
    #tex = TexturesUV(verts_uvs=verts_uvs, faces_uvs=faces_uvs, maps=imgs)
    #pdb.set_trace()
    tex = Textures(verts_uvs=verts_uvs, faces_uvs=faces_uvs, maps=imgs)

    # Initialise the mesh with textures
    meshes = Meshes(verts=verts_list, faces=faces_list, textures=tex)
    return meshes, img_files


def get_single_image_pcd(plane_params, segmentations, height=480, width=640):
    plane_params = np.array(plane_params)
    offsets = np.maximum(np.linalg.norm(plane_params, ord=2, axis=1), 1e-5)
    norms = plane_params/offsets.reshape(-1,1)

    if type(segmentations[0]) == dict:
        poly_segmentations = rle2polygon(segmentations)
    else:
        poly_segmentations = segmentations
    verts_list = []

    for segm, normal, offset in zip(poly_segmentations, norms, offsets):
        if len(segm) == 0:
            verts_list.append(torch.tensor([[0,0,0]], dtype=torch.float32))
            continue
        verts_3d = []
        bitmask = polygons_to_bitmask(segm, height=height, width=width)
        verts = np.transpose(bitmask.nonzero())
        verts_3d = get_pcd(verts[:,::-1], normal, offset)
        verts_list.append(torch.tensor(verts_3d, dtype=torch.float32))
    return verts_list

            


def get_single_image_mesh_depth(depth, segmentations, img_file, height=480, width=640, webvis=True):
    if type(segmentations[0]) == dict:
        poly_segmentations = rle2polygon(segmentations)
    else:
        poly_segmentations = segmentations
    verts_list = []
    faces_list = []
    verts_uvs = []
    img_files = []
    imgs = []

    for segm in poly_segmentations:
        if len(segm) == 0:
            continue
        verts_3d = []
        faces = []
        uvs = []
        bitmask = polygons_to_bitmask(segm, height=height, width=width)
        verts = np.transpose(bitmask.nonzero())
        vert_id_map = defaultdict(dict)
        for idx, vert in enumerate(verts):
            vert_id_map[vert[0]][vert[1]] = idx + len(verts_3d)
        pcd = get_pcd_depth(verts[:,::-1], depth.T)
        if webvis:
            # Rotate by 11 degree around x axis to push things on the ground.
            pcd = (np.array([[-1,0,0], [0,1,0], [0,0,-1]])@np.array([[1,0,0],[0,0.9816272,-0.1908090],[0,0.1908090,0.9816272]])@np.array([[-1,0,0],[0,-1,0],[0,0,1]])@pcd.T).T
        triangles = []
        for vert in verts:
            # upper right triangle
            if vert[0] < height-1 and vert[1] < width - 1 and bitmask[vert[0]][vert[1]+1] and bitmask[vert[0]+1][vert[1]+1]:
                triangles.append([vert_id_map[vert[0]][vert[1]], vert_id_map[vert[0]+1][vert[1]+1], vert_id_map[vert[0]][vert[1]+1]])
            # bottom left triangle
            if vert[0] < height-1 and vert[1] < width - 1 and bitmask[vert[0]+1][vert[1]] and bitmask[vert[0]+1][vert[1]+1]:
                triangles.append([vert_id_map[vert[0]][vert[1]], vert_id_map[vert[0]+1][vert[1]], vert_id_map[vert[0]+1][vert[1]+1]])
        triangles = np.array(triangles)
        verts_3d.extend(pcd)
        faces.extend(triangles)
        uvs.extend(np.array([0,1]) + np.array([1,-1])*verts[:,::-1]/np.array([width, height]))
        verts_list.append(torch.tensor(verts_3d, dtype=torch.float32))
        faces_list.append(torch.tensor(faces, dtype=torch.int32))
        verts_uvs.append(torch.tensor(uvs, dtype=torch.float32))
        img_files.append(img_file)
        imgs.append(torch.FloatTensor(imageio.imread(img_file)))
    verts_uvs = pad_sequence(verts_uvs, batch_first=True)
    faces_uvs = pad_sequence(faces_list, batch_first=True, padding_value=-1)
    
    # Create a textures object
    #tex = Textures(verts_uvs=verts_uvs, faces_uvs=faces_uvs, map_files=img_files)
    #tex = TexturesVertex(verts_features=verts_uvs)
    #tex = TexturesUV(verts_uvs=verts_uvs, faces_uvs=faces_uvs, maps=imgs)
    #pdb.set_trace()
    tex = Textures(verts_uvs=verts_uvs, faces_uvs=faces_uvs, maps=imgs)

    # Initialise the mesh with textures
    meshes = Meshes(verts=verts_list, faces=faces_list, textures=tex)
    return meshes, img_files