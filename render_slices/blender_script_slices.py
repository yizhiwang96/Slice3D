"""Blender script to render images of 3D models. Adapted from https://github.com/liuyuan-pal/SyncDreamer/blob/main/blender_script.py
"""

import argparse
import json
import math
import os
import random
import sys
import time
import urllib.request
from pathlib import Path

from mathutils import Vector, Matrix
import numpy as np

import bpy
from mathutils import Vector
import pickle

def read_pickle(pkl_path):
    with open(pkl_path, 'rb') as f:
        return pickle.load(f)

def save_pickle(data, pkl_path):
    # os.system('mkdir -p {}'.format(os.path.dirname(pkl_path)))
    with open(pkl_path, 'wb') as f:
        pickle.dump(data, f)

parser = argparse.ArgumentParser()
parser.add_argument("--object_path", type=str, required=True)
parser.add_argument("--output_dir", type=str, required=True)
parser.add_argument("--engine", type=str, default="CYCLES", choices=["CYCLES", "BLENDER_EEVEE"])
parser.add_argument("--slice_direction", type=str, default="camera", choices=["camera", "axis"])
parser.add_argument("--num_slices_per_axis", type=int, default=4)
parser.add_argument("--camera_type", type=str, default='random')
parser.add_argument("--num_images", type=int, default=16)
parser.add_argument("--elevation", type=float, default=30)
parser.add_argument("--elevation_start", type=float, default=-10)
parser.add_argument("--elevation_end", type=float, default=40)
parser.add_argument("--device", type=str, default='CUDA')
parser.add_argument("--camera_dist", type=float, default=1.2)
parser.add_argument("--allow_rewrite", type=bool, default=True)
parser.add_argument("--view_path", type=str, required=True)

argv = sys.argv[sys.argv.index("--") + 1 :]
args = parser.parse_args(argv)

print('===================', args.engine, '===================')

context = bpy.context
scene = context.scene
render = scene.render

cam = scene.objects["Camera"]
cam.location = (0, 1.2, 0)
cam.data.lens = 35
cam.data.sensor_width = 32

cam_constraint = cam.constraints.new(type="TRACK_TO")
cam_constraint.track_axis = "TRACK_NEGATIVE_Z"
cam_constraint.up_axis = "UP_Y"

render.engine = args.engine
render.image_settings.file_format = "PNG"
render.image_settings.color_mode = "RGBA"
render.resolution_x = 256
render.resolution_y = 256
render.resolution_percentage = 100
scene.render.film_transparent = True


def az_el_to_points(azimuths, elevations):
    x = np.cos(azimuths)*np.cos(elevations)
    y = np.sin(azimuths)*np.cos(elevations)
    z = np.sin(elevations)
    return np.stack([x,y,z],-1) #

def set_camera_location(cam_pt):
    # from https://blender.stackexchange.com/questions/18530/
    x, y, z = cam_pt # sample_spherical(radius_min=1.5, radius_max=2.2, maxz=2.2, minz=-2.2)
    # x, y, z = 0.4869694025878821, -1.0836020046573227, 0.2988185008426229 # 1.75 * 0.7
    # print((x ** 2 + y ** 2 + z ** 2) ** (0.5))
    camera = bpy.data.objects["Camera"]
    camera.location = x, y, z

    cam_constraint = cam.constraints.new(type="TRACK_TO")

    # create an empty object to track
    empty = bpy.data.objects.new("Empty", None)
    scene.collection.objects.link(empty)
    cam_constraint.target = empty
    bpy.ops.constraint.apply(constraint="Track To", owner='OBJECT')

    return camera

def get_calibration_matrix_K_from_blender(camera):
    f_in_mm = camera.data.lens
    scene = bpy.context.scene
    resolution_x_in_px = scene.render.resolution_x
    resolution_y_in_px = scene.render.resolution_y
    scale = scene.render.resolution_percentage / 100
    sensor_width_in_mm = camera.data.sensor_width
    sensor_height_in_mm = camera.data.sensor_height
    pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y

    if camera.data.sensor_fit == 'VERTICAL':
        # the sensor height is fixed (sensor fit is horizontal),
        # the sensor width is effectively changed with the pixel aspect ratio
        s_u = resolution_x_in_px * scale / sensor_width_in_mm / pixel_aspect_ratio
        s_v = resolution_y_in_px * scale / sensor_height_in_mm
    else:  # 'HORIZONTAL' and 'AUTO'
        # the sensor width is fixed (sensor fit is horizontal),
        # the sensor height is effectively changed with the pixel aspect ratio
        s_u = resolution_x_in_px * scale / sensor_width_in_mm
        s_v = resolution_y_in_px * scale * pixel_aspect_ratio / sensor_height_in_mm

    # Parameters of intrinsic calibration matrix K
    alpha_u = f_in_mm * s_u
    alpha_v = f_in_mm * s_u
    u_0 = resolution_x_in_px * scale / 2
    v_0 = resolution_y_in_px * scale / 2
    skew = 0  # only use rectangular pixels

    K = np.asarray(((alpha_u, skew, u_0),
                    (0, alpha_v, v_0),
                    (0, 0, 1)),np.float32)
    return K

def reset_scene() -> None:
    """Resets the scene to a clean state."""
    # delete everything that isn't part of a camera or a light
    for obj in bpy.data.objects:
        if obj.type not in {"CAMERA", "LIGHT"}:
            bpy.data.objects.remove(obj, do_unlink=True)
    # delete all the materials
    for material in bpy.data.materials:
        bpy.data.materials.remove(material, do_unlink=True)
    # delete all the textures
    for texture in bpy.data.textures:
        bpy.data.textures.remove(texture, do_unlink=True)
    # delete all the images
    for image in bpy.data.images:
        bpy.data.images.remove(image, do_unlink=True)

def remove_textures():
    for material in bpy.data.materials:
        for slot in material.texture_slots:
            # Clear texture slots
            if slot is not None and slot.texture is not None:
                bpy.data.textures.remove(slot.texture, do_unlink=True)
    
# load the glb model
def load_object(object_path: str, merge_meshes: bool) -> None:
    """Loads a glb model into the scene."""
    if object_path.endswith(".glb"):
        bpy.ops.import_scene.gltf(filepath=object_path, merge_vertices=True)
    elif object_path.endswith(".fbx"):
        bpy.ops.import_scene.fbx(filepath=object_path)
    else:
        raise ValueError(f"Unsupported file type: {object_path}")

    if merge_meshes:
        msh_objs = [m for m in bpy.context.selected_objects if m.type == 'MESH'] # 
        # Mesh objects
        for obj in msh_objs:
            bpy.context.view_layer.objects.active = obj
            bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
        bpy.ops.object.select_all(action='DESELECT')
        msh_merged_name = msh_objs[-1].name
        for objs in msh_objs:
            objs.select_set(True)
        bpy.context.view_layer.objects.active = msh_objs[-1]
        bpy.ops.object.join()

    return msh_merged_name

def scene_bbox(single_obj=None, ignore_matrix=False):
    bbox_min = (math.inf,) * 3
    bbox_max = (-math.inf,) * 3
    found = False
    for obj in scene_meshes() if single_obj is None else [single_obj]:
        found = True
        for coord in obj.bound_box:
            coord = Vector(coord)
            if not ignore_matrix:
                coord = obj.matrix_world @ coord
            bbox_min = tuple(min(x, y) for x, y in zip(bbox_min, coord))
            bbox_max = tuple(max(x, y) for x, y in zip(bbox_max, coord))
    if not found:
        raise RuntimeError("no objects in scene to compute bounding box for")
    return Vector(bbox_min), Vector(bbox_max)

def scene_root_objects():
    for obj in bpy.context.scene.objects.values():
        if not obj.parent:
            yield obj

def scene_meshes():
    for obj in bpy.context.scene.objects.values():
        if isinstance(obj.data, (bpy.types.Mesh)):
            yield obj

# function from https://github.com/panmari/stanford-shapenet-renderer/blob/master/render_blender.py
def get_3x4_RT_matrix_from_blender(cam):
    bpy.context.view_layer.update()
    location, rotation = cam.matrix_world.decompose()[0:2]
    R = np.asarray(rotation.to_matrix())
    t = np.asarray(location)

    cam_rec = np.asarray([[1, 0, 0], [0, -1, 0], [0, 0, -1]], np.float32)
    R = R.T
    t = -R @ t
    R_world2cv = cam_rec @ R
    t_world2cv = cam_rec @ t

    RT = np.concatenate([R_world2cv,t_world2cv[:,None]],1)
    return RT

def duplicate_obj(obj_src, obj_dup_name):

    bpy.context.view_layer.objects.active = obj_src
    obj_src.select_set(True)
    bpy.ops.object.duplicate(linked=0, mode='TRANSLATION')
    obj_dup = bpy.context.active_object
    obj_dup.name = obj_dup_name

def delete_objs():
    for o in scene.objects:
        bpy.ops.object.select_all(action='DESELECT')
        bpy.data.objects[o.name].select_set(True)
        bpy.ops.object.delete()

def delete_objs_w_name(camera_id):
    for o in scene.objects:
        if f'cmr_{str(camera_id)}' in o.name:
            bpy.ops.object.select_all(action='DESELECT')
            bpy.data.objects[o.name].select_set(True)
            bpy.ops.object.delete()

def export_obj(obj, file_export_name):
    blend_file_path = bpy.data.filepath
    directory = os.path.dirname(blend_file_path)
    target_file = os.path.join(directory, file_export_name)

    bpy.ops.object.select_all(action='DESELECT')
    bpy.data.objects[file_export_name.replace(".glb", "")].select_set(True)

    bpy.ops.export_scene.gltf(filepath=target_file, use_selection=True)

def slice_object(slice_direction, num_slices, camera, camera_id, object_uid, msh_merged_name):
    # select the input mesh
    obj_src_ = bpy.data.objects[msh_merged_name]

    try:
        obj_src_.data.materials.clear()
    except:
        print('clear material failed')
        return None, None
    
    # camera-aligned slicing
    if slice_direction == 'camera':
        obj_name = f"obj_cmr_{str(camera_id)}"
        duplicate_obj(obj_src_, obj_name)
        obj_src = bpy.data.objects[obj_name]

        inverse_world_matrix = camera.matrix_world.inverted()
        for vert in obj_src.data.vertices:
            # obj_src.matrix_world @ v.co is the world coordinate of vertice
            world_vert = obj_src.matrix_world @ vert.co # object space to world space
            camera_vert = inverse_world_matrix @ world_vert # transformed to camera world coordinate
            vert.co = obj_src.matrix_world.inverted() @ camera_vert # world space to object space
    else:
        obj_src = obj_src_

    # export_obj(bpy.data.objects[msh_merged_name], msh_merged_name + ".glb")
    slice_names = []

    for axis in ['X', 'Y', 'Z']:
        # camera.matrix_world.inverted() is changed after this loop but I haven't found the reason
        if axis == 'X':
            min_pt = min([(obj_src.matrix_world @ v.co).x for v in obj_src.data.vertices])
            max_pt = max([(obj_src.matrix_world @ v.co).x for v in obj_src.data.vertices])
        elif axis == 'Y':
            min_pt = min([(obj_src.matrix_world @ v.co).y for v in obj_src.data.vertices])
            max_pt = max([(obj_src.matrix_world @ v.co).y for v in obj_src.data.vertices])
        else:
            min_pt = min([(obj_src.matrix_world @ v.co).z for v in obj_src.data.vertices])
            max_pt = max([(obj_src.matrix_world @ v.co).z for v in obj_src.data.vertices])
        
        step = (max_pt - min_pt) / (num_slices)
        slice_coord = [min_pt + step * i for i in range(num_slices + 1)]
        if axis != 'X':
            slice_coord = slice_coord[::-1]
        
        for slice_idx in range(1, num_slices + 1):
            slice_name = f"slice_{axis}_{slice_idx}_{object_uid}_cmr_{str(camera_id)}"

            duplicate_obj(obj_src, slice_name)
            bpy.ops.object.select_all(action='DESELECT')

            bpy.context.view_layer.objects.active = bpy.data.objects[slice_name]
            bpy.ops.object.mode_set(mode='EDIT')
            bpy.ops.mesh.select_all(action='SELECT')

            if axis == 'X':
                plane_co_prev = (slice_coord[slice_idx - 1], 0, 0)
                plane_co_next = (slice_coord[slice_idx], 0, 0)
                plane_no_prev = (-1, 0, 0)
                plane_no_next = (1, 0, 0)
            elif axis == 'Y':
                plane_co_prev = (0, slice_coord[slice_idx - 1], 0)
                plane_co_next = (0, slice_coord[slice_idx], 0)
                plane_no_prev = (0, 1, 0)
                plane_no_next = (0, -1, 0)
            else:
                plane_co_prev = (0, 0, slice_coord[slice_idx - 1])
                plane_co_next = (0, 0, slice_coord[slice_idx]) 
                plane_no_prev = (0, 0, 1)
                plane_no_next = (0, 0, -1)

            if slice_idx == 1:
                bpy.ops.mesh.bisect(plane_co=plane_co_next, plane_no=plane_no_next, clear_outer=True, clear_inner=False, use_fill=False)
            elif slice_idx > 1 and slice_idx < num_slices:
                bpy.ops.mesh.bisect(plane_co=plane_co_prev, plane_no=plane_no_prev, clear_outer=True, clear_inner=False, use_fill=False)
                bpy.ops.object.mode_set(mode='OBJECT')
                duplicate_obj(obj_src, slice_name)
                bpy.ops.object.select_all(action='DESELECT')
                bpy.context.view_layer.objects.active = bpy.data.objects[slice_name]
                bpy.ops.object.mode_set(mode='EDIT')
                bpy.ops.mesh.select_all(action='SELECT')
                bpy.ops.mesh.bisect(plane_co=plane_co_next, plane_no=plane_no_next, clear_outer=True, clear_inner=False, use_fill=False)
            elif slice_idx == num_slices:
                bpy.ops.mesh.bisect(plane_co=plane_co_prev, plane_no=plane_no_prev, clear_outer=True, clear_inner=False, use_fill=False)

            bpy.ops.object.mode_set(mode='OBJECT')
            # export_obj(bpy.data.objects[slice_name], slice_name + ".glb")
            slice_names.append(slice_name)

    if slice_direction == 'camera':
        camera = reset_camera(camera, obj_src, inverse_world_matrix)

    return slice_names, camera

def reset_camera(cam, obj, inv_wrd_mat):
    '''
    set the camera as the center of the world
    '''
    obj_location_old = Vector((0, 0, 0))
    obj_location_new = inv_wrd_mat @ obj_location_old

    cam.location = (0, 0, 0)

    cam_empty_ = bpy.data.objects.new("Empty_", None)
    cam_empty_.location = (0, 0, obj_location_new[2])

    cam_constraint = cam.constraints.new(type='TRACK_TO')
    cam_constraint.target = cam_empty_
    bpy.ops.constraint.apply(constraint="Track To", owner='OBJECT')

    return cam

def normalize_scene(object_uid, msh_merged, view_path):
    '''
    re-scale and translate the object, using the same scaling factor and offset during rendering input views
    '''
    cam_meta = read_pickle(f'{view_path}/{object_uid}/meta.pkl')
    azimuths, elevations, scale_rand, offset_rand = cam_meta[1], cam_meta[2], cam_meta[5], cam_meta[6]
    obj = msh_merged

    bbox_min, bbox_max = scene_bbox(obj)
    dimensions = bbox_max - bbox_min
    body_diagnoal = math.sqrt(dimensions.x**2 + dimensions.y**2 + dimensions.z**2)

    # scale = 1 / max(bbox_max - bbox_min)
    scale = 1 / body_diagnoal
    scale = scale * scale_rand

    obj.scale = obj.scale * scale
    
    # Apply scale to matrix_world.
    bpy.context.view_layer.update()

    bbox_min, bbox_max = scene_bbox(obj)
    offset = -(bbox_min + bbox_max) / 2
    offset.x += offset_rand[0]
    offset.y += offset_rand[1]
    offset.z += offset_rand[2]

    obj.matrix_world.translation += offset

    bpy.ops.object.select_all(action="DESELECT")

    return azimuths, elevations, scale_rand, offset_rand

def save_images(object_file: str, slice_direction: str, num_slices: int) -> None:
    object_uid = os.path.basename(object_file).split(".")[0]
    os.makedirs(args.output_dir, exist_ok=True)

    reset_scene()
    # load the object
    msh_merged_name = load_object(object_file, merge_meshes=True)

    azimuths, elevations, scale_rand, offset_rand = normalize_scene(object_uid, bpy.data.objects[msh_merged_name], args.view_path)

    world_tree = bpy.context.scene.world.node_tree
    back_node = world_tree.nodes['Background']
    env_light = 0.5
    back_node.inputs['Color'].default_value = Vector([env_light, env_light, env_light, 1.0])
    back_node.inputs['Strength'].default_value = 1.0

    distances = np.asarray([args.camera_dist for _ in range(args.num_images)])

    cam_pts = az_el_to_points(azimuths, elevations) * distances[:,None]
    cam_poses = []
    (Path(args.output_dir) / object_uid).mkdir(exist_ok=True, parents=True)

    for i in range(args.num_images):
        # if i !=1: continue
        # set camera
        bpy.context.view_layer.objects.active = bpy.data.objects[msh_merged_name]

        camera = set_camera_location(cam_pts[i])

        # perform slicing
        slice_names, camera = slice_object(slice_direction, num_slices, camera, i, object_uid, msh_merged_name)
        if slice_names is None: break

        RT = get_3x4_RT_matrix_from_blender(camera)
        cam_poses.append(RT)

        for slice_name in slice_names:
            # select slice
            for o in scene.objects:
                if o.name in ['Light', 'Camera', slice_name]:
                    continue
                else:
                    o.hide_render = True
                    # o.hide_viewport = True

            axis = slice_name.split('_')[1]
            slice_idx = slice_name.split('_')[2]

            render_path = os.path.join(args.output_dir, object_uid, f"{i:03d}", f"{axis}_{slice_idx}.png")
            if (not args.allow_rewrite) and os.path.exists(render_path): continue
            scene.render.filepath = os.path.abspath(render_path)
            bpy.ops.render.render(write_still=True)

            # uncomment the following to debug
            # bpy.ops.wm.save_as_mainfile(filepath='/localhome/ywa439/Documents/23_slice3d/cmr_scaled66.blend')
            # gyu=ft6g

            # reset hide_render
            for o in scene.objects:
                o.hide_render = False
        

        # delete_objs_w_name(i)
    
    delete_objs()

if __name__ == "__main__":
    save_images(args.object_path, args.slice_direction, args.num_slices_per_axis)