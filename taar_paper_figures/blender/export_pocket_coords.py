"""Export pocket surface vertices projected through the current camera, plus the camera
pose, so the 2D matplotlib panel matches the 3D render exactly.

Run inside Blender:  Scripting tab -> Open -> Run Script
or headless:         blender scene.blend --background --python export_pocket_coords.py

Writes next to the .blend (or to OUT_DIR):
    pocket_camera_coords.npz   {object_name: (N, 2) pixel coords} + __res__
    camera_pose.json           location, rotation, lens, resolution

Re-run this after ANY camera move. Material changes do not affect the projection.

IMPORTANT - evaluated geometry
Molecular Nodes builds the surface in a geometry node tree, so the object's base mesh is
just atom positions. The vertices we want only exist after the depsgraph evaluates the
modifier stack, which is why this uses obj.evaluated_get(depsgraph) rather than obj.data.
Reading obj.data directly would export atom centres, not the surface, and the silhouettes
would come out too small.
"""
import json
import os

import bpy
import numpy as np
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector

# ---------------------------------------------------------------- constants
# pocket objects carry "ortho" in their name. EXCLUDE keeps the backbone out - it is
# called 8JLQ_3_backbone_largest_holo, so matching on state alone would sweep it in.
POCKET_TOKEN = "ortho"
EXCLUDE = ("backbone", "ribbon", "protein", "cartoon")
# the ligand is exported too, so the 2D panel can show where it sits
LIGAND_TOKENS = ("ligand", "vrk", "a77636")

# ---------------------------------------------------------------- the slice
# The camera view is kept exactly as it is: x and y always come from the camera
# projection. The only free parameter is DEPTH along the viewing direction, and there is
# an infinite family of planes to choose from (the tip of the pockets, the middle,
# three-quarters through, ...).
#
# "cross_section" takes the plane through the WORLD ORIGIN, perpendicular to the viewing
# direction, and keeps only the geometry lying IN that plane (a thin slab about it). The
# traced outline is therefore the CUT FACE - the shape you would see having sawn the
# receptor in half at the origin in its current orientation - not the outline of the
# remaining half, and not an outer silhouette.
#
#   "cross_section"  slab about the plane          -> the cut face   (default)
#   "half"           everything beyond the plane   -> silhouette of the far half
#   "none"           everything                    -> outer silhouette
SLICE_MODE = "cross_section"
SLICE_THROUGH = "origin"      # "origin" (the receptor centre in blender) or "centre"
SLICE_OFFSET = 0.0            # move the plane along the viewing direction
SLAB = "auto"                 # slab half-thickness in blender units, or "auto"
MIN_SLAB_POINTS = 200         # "auto" widens the slab until every pocket keeps this many

OUT_DIR = None                    # None -> next to the .blend
NPZ_NAME = "pocket_camera_coords.npz"
POSE_NAME = "camera_pose.json"
CAMERA_NAME = "Camera"

MAX_VERTS = 60000                 # per object; evenly subsampled above this
BEHIND_CAMERA_DROP = True         # drop vertices behind the camera (z <= 0)


# ---------------------------------------------------------------- helpers
def is_pocket(name):
    lowered = name.lower()
    if any(token in lowered for token in EXCLUDE):
        return False
    return POCKET_TOKEN in lowered


def is_ligand(name):
    lowered = name.lower()
    if any(token in lowered for token in EXCLUDE):
        return False
    return any(token in lowered for token in LIGAND_TOKENS)


def output_dir():
    if OUT_DIR:
        return OUT_DIR
    if bpy.data.filepath:
        return os.path.dirname(bpy.data.filepath)
    raise RuntimeError("save the .blend first, or set OUT_DIR")


def render_resolution(scene):
    """pixel resolution actually rendered, including the % scale and pixel aspect"""
    scale = scene.render.resolution_percentage / 100.0
    width = int(scene.render.resolution_x * scale)
    height = int(scene.render.resolution_y * scale)
    return width, height


def project_object(obj, scene, camera, depsgraph):
    """world_to_camera_view gives normalised (0..1, 0..1, depth) with the origin at the
    bottom-left of the frame. Convert to top-left pixel coordinates so the result lines
    up with the rendered PNG when it is displayed with imshow."""
    evaluated = obj.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh()
    if mesh is None or len(mesh.vertices) == 0:
        evaluated.to_mesh_clear()
        return None

    matrix = evaluated.matrix_world
    width, height = render_resolution(scene)

    coords = np.empty((len(mesh.vertices), 3), dtype=np.float64)
    for i, vertex in enumerate(mesh.vertices):
        ndc = world_to_camera_view(scene, camera, matrix @ vertex.co)
        coords[i] = (ndc.x, ndc.y, ndc.z)
    evaluated.to_mesh_clear()

    if BEHIND_CAMERA_DROP:
        coords = coords[coords[:, 2] > 0]
    return coords if coords.size else None


def keep_slice(coords, plane, slab):
    """camera-space rows surviving the chosen slice.

    cross_section keeps a slab straddling the plane, so what is left is the material the
    plane actually passes through - the cut face. half keeps everything beyond it."""
    if plane is None or SLICE_MODE == "none":
        return coords
    if SLICE_MODE == "half":
        return coords[coords[:, 2] > plane]
    return coords[np.abs(coords[:, 2] - plane) <= slab]


def to_pixels(coords, scene):
    """camera-space (x, y, depth) -> top-left pixel coordinates"""
    if coords.size == 0:
        return None
    width, height = render_resolution(scene)
    pixels = np.column_stack([coords[:, 0] * width,
                              (1.0 - coords[:, 1]) * height])   # flip Y to image space
    if len(pixels) > MAX_VERTS:
        pixels = pixels[::int(np.ceil(len(pixels) / MAX_VERTS))]
    return pixels


def save_camera_pose(camera, scene, path):
    pose = {
        "location": list(camera.location),
        "rotation_euler": list(camera.rotation_euler),
        "rotation_mode": camera.rotation_mode,
        "lens": camera.data.lens,
        "sensor_width": camera.data.sensor_width,
        "type": camera.data.type,
        "ortho_scale": getattr(camera.data, "ortho_scale", None),
        "shift_x": camera.data.shift_x,
        "shift_y": camera.data.shift_y,
        "clip_start": camera.data.clip_start,
        "clip_end": camera.data.clip_end,
        "resolution": list(render_resolution(scene)),
    }
    with open(path, "w") as handle:
        json.dump(pose, handle, indent=2)
    return pose


def restore_camera_pose(path=None):
    """put the camera back where a previous export recorded it"""
    path = path or os.path.join(output_dir(), POSE_NAME)
    with open(path) as handle:
        pose = json.load(handle)
    camera = bpy.data.objects[CAMERA_NAME]
    camera.rotation_mode = pose.get("rotation_mode", "XYZ")
    camera.location = pose["location"]
    camera.rotation_euler = pose["rotation_euler"]
    camera.data.lens = pose["lens"]
    camera.data.shift_x = pose.get("shift_x", 0.0)
    camera.data.shift_y = pose.get("shift_y", 0.0)
    print(f"camera restored from {path}")


# ---------------------------------------------------------------- entry point
def main():
    scene = bpy.context.scene
    camera = scene.camera or bpy.data.objects.get(CAMERA_NAME)
    if camera is None:
        raise RuntimeError("no active camera in the scene")

    depsgraph = bpy.context.evaluated_depsgraph_get()
    width, height = render_resolution(scene)
    print(f"Exporting through '{camera.name}' at {width}x{height}")

    # pass 1: project everything into camera space and keep the depths
    camera_space, ligands = {}, {}
    for obj in scene.objects:
        if obj.type != "MESH":
            continue
        if is_pocket(obj.name):
            target = camera_space
        elif is_ligand(obj.name):
            target = ligands
        else:
            continue
        coords = project_object(obj, scene, camera, depsgraph)
        if coords is None:
            print(f"  {obj.name}: no visible geometry, skipped")
            continue
        target[obj.name] = coords

    if not camera_space:
        raise RuntimeError(f"no objects matched POCKET_TOKEN='{POCKET_TOKEN}'")

    # pass 2: where the plane sits, in camera depth
    plane, slab = None, 0.0
    if SLICE_MODE != "none":
        depths = np.concatenate([c[:, 2] for c in camera_space.values()])
        if SLICE_THROUGH == "origin":
            plane = world_to_camera_view(scene, camera, Vector((0.0, 0.0, 0.0))).z
        else:
            plane = (depths.min() + depths.max()) / 2
        plane += SLICE_OFFSET
        print(f"  depth {depths.min():.3f} .. {depths.max():.3f}  "
              f"plane through the {SLICE_THROUGH} at {plane:.3f}")

        if SLICE_MODE == "cross_section":
            slab = SLAB
            if slab == "auto":
                span = depths.max() - depths.min()
                slab = span * 0.01
                # widen until the thinnest pocket still has enough points to trace
                while slab < span / 2:
                    counts = [len(keep_slice(c, plane, slab)) for c in camera_space.values()]
                    if min(counts) >= MIN_SLAB_POINTS:
                        break
                    slab *= 1.5
            print(f"  slab half-thickness {slab:.3f} "
                  f"({'auto' if SLAB == 'auto' else 'fixed'})")

    arrays = {"__res__": np.array([width, height], dtype=np.int64)}
    for name, coords in camera_space.items():
        pixels = to_pixels(keep_slice(coords, plane, slab), scene)
        if pixels is None:
            print(f"  {name}: nothing survives the slice, skipped")
            continue
        arrays[name] = pixels
        print(f"  {name}: {len(pixels):,} of {len(coords):,} points  "
              f"x[{pixels[:, 0].min():.0f}, {pixels[:, 0].max():.0f}]  "
              f"y[{pixels[:, 1].min():.0f}, {pixels[:, 1].max():.0f}]")

    # the ligand is never sliced - it is the thing the cut is meant to reveal.
    # both its projected cloud and its centre of mass are stored, so the 2D panel can
    # draw it either as an outline or as a single marker.
    for name, coords in ligands.items():
        pixels = to_pixels(coords, scene)
        if pixels is None:
            continue
        arrays[f"ligand__{name}"] = pixels
        arrays[f"ligand_com__{name}"] = pixels.mean(axis=0)
        print(f"  ligand {name}: {len(pixels):,} points, "
              f"centre of mass ({pixels[:, 0].mean():.0f}, {pixels[:, 1].mean():.0f})")

    directory = output_dir()
    npz_path = os.path.join(directory, NPZ_NAME)
    pose_path = os.path.join(directory, POSE_NAME)
    np.savez_compressed(npz_path, **arrays)
    save_camera_pose(camera, scene, pose_path)

    print(f"\nSaved {npz_path}")
    print(f"Saved {pose_path}")
    print("Now render from this same camera, then run fig2_panel_d_pockets.py")


if __name__ == "__main__":
    main()
