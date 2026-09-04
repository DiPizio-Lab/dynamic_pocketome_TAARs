"""Restore the pocket surface material (transparent body + absorptive rim) on a WHITE
background, and optionally set the camera.

Run inside Blender:  Scripting tab -> Open -> Run Script
or headless:         blender scene.blend --background --python fix_pocket_materials.py

WHY THE SURFACES WENT SOLID
The original material mixed a Transparent BSDF against an Emission node, driven by
Layer Weight [Facing]. That is a rim-GLOW: it adds light at grazing angles, which reads
beautifully on a dark slide. On white there is no headroom left to add light to, so the
rim disappears. Swapping Emission for an opaque shader fixes visibility but, if the mix
factor is left at its old range, the whole surface turns solid - which is what happened.

The material built here keeps the same topology but makes the rim ABSORB instead of emit,
and it drives the mix over a wider, offset range so the body stays see-through:

    Layer Weight [Facing] -> Map Range (FROM_MIN..1 -> BODY_OPACITY..1) -> Mix Shader Fac
    Mix Shader: Transparent BSDF (Fac 0) <-> Principled BSDF (Fac 1)

  BODY_OPACITY  faint tint on faces pointing at the camera, so a pocket reads as a volume
                rather than an empty outline. 0 reproduces the old fully-clear body.
  FROM_MIN      where the rim starts. Lower = thicker rim. The old 0.7 is a hairline;
                0.45 gives a band you can actually see against white.
"""
import json
import math
import os

import bpy
from mathutils import Vector

# ---------------------------------------------------------------- constants
# EXACT paper palette - the renders use the same colours as the plots, no substitutes.
APO = (0.663, 0.541, 0.416, 1.0)      # #A98A6A sand
HOLO = (0.608, 0.333, 0.376, 1.0)     # #9B5560 dusty burgundy
LIGAND = (0.871, 0.482, 0.118, 1.0)   # #DE7B1E orange

# object matching: pockets carry "ortho" in their name. EXCLUDE matters because the
# backbone is called 8JLQ_3_backbone_largest_holo - matching on state alone would hand
# the backbone a pocket material.
POCKET_TOKEN = "ortho"
EXCLUDE = ("backbone", "ribbon", "protein", "cartoon")
STATE_COLORS = {"apo": APO, "holo": HOLO}

# look presets - switch with LOOK below.
#   body      opacity of faces pointing at the camera. THIS is what stops the pockets
#             washing out: composited over white paper, a low-alpha sand or slate turns
#             pale. 0.8 keeps the palette colour at close to full strength.
#   from_min  where the rim starts. lower = thicker rim.
#   emission  deliberately 0 for the paper: emission ADDS light, which lightens and
#             desaturates exactly the colours we are trying to keep true.
PRESETS = {
    "paper": dict(body=0.80, from_min=0.20, emission=0.00, roughness=0.45),
    "dense": dict(body=0.92, from_min=0.12, emission=0.00, roughness=0.40),
    "ghost": dict(body=0.30, from_min=0.40, emission=0.00, roughness=0.50),
}
LOOK = "paper"

# --- SCOPE ------------------------------------------------------------------
# Default: this script ONLY changes render/background settings. The materials in the
# .blend are left exactly as they are. Set FIX_MATERIALS = True only if you actually
# want the pocket materials rebuilt from scratch.
FIX_MATERIALS = False

BLEND = 0.40              # Layer Weight blend; lower = softer rim falloff

# --- nesting: large pockets ENCLOSE small ones ------------------------------
# With all four visible, a small pocket sits inside a large one, so at equal opacity the
# large surface simply paints over it. The fix is to give the two sizes different jobs:
#   large  -> almost pure outline (very low body), acting as a container you see through
#   small  -> dense, so it reads as a solid object suspended inside that container
# Size therefore stays legible even before you compare the geometry.
SIZE_OVERRIDE = {
    "large": dict(body=0.06, from_min=0.62),   # cage: rim only
    "small": dict(body=0.95, from_min=0.10),   # solid body
}
USE_SIZE_OVERRIDE = False  # only consulted when FIX_MATERIALS is True

# --- background -------------------------------------------------------------
# Render with NO background (straight alpha) and composite onto white in matplotlib.
# Two settings cause the "white renders grey-ish" problem, and both are fixed here:
#   film_transparent  otherwise the world colour is baked into the image
#   view_transform    Blender 4.x defaults to AgX, a filmic tone map that pulls pure
#                     white down to grey and desaturates saturated colour. Scientific
#                     figures want 'Standard', which is a straight, untonemapped output.
FILM_TRANSPARENT = True
VIEW_TRANSFORM = "Standard"
WORLD_STRENGTH = 1.0      # world still lights the scene even when the film is transparent
TRANSPARENT_BOUNCES = 64  # stacked transparent surfaces need far more than the default 8

SET_CAMERA = False        # camera is left alone; coords get re-exported anyway
CAMERA_JSON = "camera_pose.json"
CAMERA_NAME = "Camera"


# ---------------------------------------------------------------- material
def build_pocket_material(name, base_color, body=None, from_min=None, emission=None,
                          roughness=None):
    """transparent body with an absorptive, softly glowing rim.

    Fac 0 -> Transparent BSDF, Fac 1 -> Principled. Map Range lifts the floor so that
    faces pointing at the camera keep `body` opacity instead of vanishing, and starts the
    rim at `from_min` so the edge is a band rather than a hairline. The emission term is
    what rescues the pocket where it overlaps the dark backbone: an absorptive rim alone
    cannot separate two dark things."""
    preset = PRESETS[LOOK]
    body = preset["body"] if body is None else body
    from_min = preset["from_min"] if from_min is None else from_min
    emission = preset["emission"] if emission is None else emission
    roughness = preset["roughness"] if roughness is None else roughness

    mat = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    mat.use_nodes = True
    tree = mat.node_tree
    tree.nodes.clear()

    out = tree.nodes.new("ShaderNodeOutputMaterial")
    mix = tree.nodes.new("ShaderNodeMixShader")
    transparent = tree.nodes.new("ShaderNodeBsdfTransparent")
    principled = tree.nodes.new("ShaderNodeBsdfPrincipled")
    map_range = tree.nodes.new("ShaderNodeMapRange")
    layer_weight = tree.nodes.new("ShaderNodeLayerWeight")

    for node, (x, y) in zip((layer_weight, map_range, transparent, principled, mix, out),
                            ((-800, 200), (-600, 200), (-400, 40), (-400, -140),
                             (-150, 60), (60, 60))):
        node.location = (x, y)

    layer_weight.inputs["Blend"].default_value = BLEND
    map_range.inputs["From Min"].default_value = from_min
    map_range.inputs["From Max"].default_value = 1.0
    map_range.inputs["To Min"].default_value = min(body, 1.0)
    map_range.inputs["To Max"].default_value = 1.0
    map_range.clamp = True

    principled.inputs["Base Color"].default_value = base_color
    principled.inputs["Roughness"].default_value = roughness
    if "Metallic" in principled.inputs:
        principled.inputs["Metallic"].default_value = 0.0
    # socket names differ across versions: 4.x uses "Emission Color", 3.x "Emission"
    for key in ("Emission Color", "Emission"):
        if key in principled.inputs:
            principled.inputs[key].default_value = base_color
            break
    if "Emission Strength" in principled.inputs:
        principled.inputs["Emission Strength"].default_value = emission

    tree.links.new(layer_weight.outputs["Facing"], map_range.inputs["Value"])
    tree.links.new(map_range.outputs["Result"], mix.inputs["Fac"])
    tree.links.new(transparent.outputs["BSDF"], mix.inputs[1])
    tree.links.new(principled.outputs["BSDF"], mix.inputs[2])
    tree.links.new(mix.outputs["Shader"], out.inputs["Surface"])

    if hasattr(mat, "surface_render_method"):        # Blender 4.2+
        mat.surface_render_method = "BLENDED"
    elif hasattr(mat, "blend_method"):               # 4.1 and earlier
        mat.blend_method = "BLEND"
    if hasattr(mat, "use_transparent_shadow"):
        mat.use_transparent_shadow = True
    return mat


def build_solid_material(name, base_color, roughness=0.45):
    """plain opaque material for the ligand and the backbone"""
    mat = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    mat.use_nodes = True
    tree = mat.node_tree
    tree.nodes.clear()
    out = tree.nodes.new("ShaderNodeOutputMaterial")
    bsdf = tree.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.location = (-300, 0)
    bsdf.inputs["Base Color"].default_value = base_color
    bsdf.inputs["Roughness"].default_value = roughness
    tree.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    return mat


# ---------------------------------------------------------------- assignment
def assign_material(obj, mat):
    """set the material on the object AND on any Molecular Nodes style node.

    MN drives the material through the geometry node tree, so writing only to the object
    material slot is silently overridden on the next evaluation."""
    if obj.data is not None and hasattr(obj.data, "materials"):
        if obj.data.materials:
            obj.data.materials[0] = mat
        else:
            obj.data.materials.append(mat)

    for modifier in obj.modifiers:
        if modifier.type != "NODES" or modifier.node_group is None:
            continue
        for node in modifier.node_group.nodes:
            for socket in node.inputs:
                if socket.type == "MATERIAL":
                    socket.default_value = mat


def color_for(name):
    """(state, colour) for a pocket object, or (None, None) if this is not a pocket"""
    lowered = name.lower()
    if any(token in lowered for token in EXCLUDE):
        return None, None
    if POCKET_TOKEN not in lowered:
        return None, None
    for state, color in STATE_COLORS.items():
        if state in lowered:
            return state, color
    return None, None


def fix_pocket_objects(objects=None):
    """apply the rim material to every pocket object, keyed by apo/holo in its name"""
    objects = objects or [o for o in bpy.context.scene.objects if o.type == "MESH"]
    touched = []
    for obj in objects:
        state, color = color_for(obj.name)
        if state is None:
            continue
        size = "large" if "large" in obj.name.lower() else \
               "small" if "small" in obj.name.lower() else None
        override = SIZE_OVERRIDE.get(size, {}) if USE_SIZE_OVERRIDE else {}
        mat = build_pocket_material(f"pocket_{obj.name}", color, **override)
        # the enclosing cage should not hide what is inside it
        mat.use_backface_culling = (size == "large" and USE_SIZE_OVERRIDE)
        assign_material(obj, mat)
        touched.append(f"{obj.name} [{state}/{size or 'unsized'}]")
    return touched


# ---------------------------------------------------------------- scene
def set_world(strength=WORLD_STRENGTH):
    """white world for even lighting. With FILM_TRANSPARENT the colour is not rendered
    into the image, it only lights the objects."""
    world = bpy.context.scene.world or bpy.data.worlds.new("World")
    bpy.context.scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    if bg is None:
        bg = world.node_tree.nodes.new("ShaderNodeBackground")
        out = world.node_tree.nodes.get("World Output") or \
            world.node_tree.nodes.new("ShaderNodeOutputWorld")
        world.node_tree.links.new(bg.outputs["Background"], out.inputs["Surface"])
    bg.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    bg.inputs["Strength"].default_value = strength


def set_render_settings():
    """transparent film, straight colour output, RGBA png"""
    scene = bpy.context.scene
    if scene.render.engine == "CYCLES":
        scene.cycles.transparent_max_bounces = TRANSPARENT_BOUNCES
        scene.cycles.max_bounces = max(scene.cycles.max_bounces, 16)

    scene.render.film_transparent = FILM_TRANSPARENT
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"      # without this the alpha is discarded
    scene.render.image_settings.color_depth = "8"
    scene.render.resolution_x = 1920
    scene.render.resolution_y = 1080

    # AgX / Filmic grey out whites and desaturate; 'Standard' is what a figure needs
    try:
        scene.view_settings.view_transform = VIEW_TRANSFORM
        scene.view_settings.look = "None"
        scene.view_settings.exposure = 0.0
        scene.view_settings.gamma = 1.0
    except TypeError:
        print(f"  view transform '{VIEW_TRANSFORM}' unavailable in this build")


# ---------------------------------------------------------------- camera
def save_camera(path=None):
    """write the current camera pose next to the .blend, so it is never lost again.

    Do this once now that the framing is good - then restore_camera() reproduces it
    exactly, and the npz export and the renders stay in the same projection."""
    cam = bpy.data.objects.get(CAMERA_NAME)
    if cam is None:
        raise RuntimeError(f"no object named {CAMERA_NAME}")
    path = path or os.path.join(os.path.dirname(bpy.data.filepath), CAMERA_JSON)
    pose = {
        "location": list(cam.location),
        "rotation_euler": list(cam.rotation_euler),
        "lens": cam.data.lens,
        "type": cam.data.type,
        "ortho_scale": getattr(cam.data, "ortho_scale", None),
        "resolution": [bpy.context.scene.render.resolution_x,
                       bpy.context.scene.render.resolution_y],
    }
    with open(path, "w") as handle:
        json.dump(pose, handle, indent=2)
    print(f"  camera pose saved -> {path}")
    return path


def restore_camera(path=None):
    """put the camera back exactly where save_camera() recorded it"""
    path = path or os.path.join(os.path.dirname(bpy.data.filepath), CAMERA_JSON)
    if not os.path.exists(path):
        print(f"  no camera pose file at {path}; skipping")
        return False
    with open(path) as handle:
        pose = json.load(handle)
    cam = bpy.data.objects.get(CAMERA_NAME)
    if cam is None:
        cam_data = bpy.data.cameras.new(CAMERA_NAME)
        cam = bpy.data.objects.new(CAMERA_NAME, cam_data)
        bpy.context.scene.collection.objects.link(cam)
    cam.location = pose["location"]
    cam.rotation_euler = pose["rotation_euler"]
    cam.data.lens = pose["lens"]
    bpy.context.scene.camera = cam
    print(f"  camera restored from {path}")
    return True


def frame_objects(objects=None, azimuth_deg=35.0, elevation_deg=20.0, margin=1.35):
    """fallback framing: point the camera at the pocket cloud from a given direction.

    Only use this if the original pose was never saved - it reproduces a similar view,
    not the identical one. Call save_camera() afterwards so it becomes reproducible."""
    objects = objects or [o for o in bpy.context.scene.objects
                          if o.type == "MESH" and color_for(o.name)[0]]
    if not objects:
        raise RuntimeError("no pocket objects found to frame")

    corners = []
    for obj in objects:
        corners += [obj.matrix_world @ Vector(c) for c in obj.bound_box]
    center = sum(corners, Vector()) / len(corners)
    radius = max((c - center).length for c in corners)

    cam = bpy.data.objects.get(CAMERA_NAME)
    if cam is None:
        cam_data = bpy.data.cameras.new(CAMERA_NAME)
        cam = bpy.data.objects.new(CAMERA_NAME, cam_data)
        bpy.context.scene.collection.objects.link(cam)
    bpy.context.scene.camera = cam

    az, el = math.radians(azimuth_deg), math.radians(elevation_deg)
    fov = cam.data.angle if cam.data.type == "PERSP" else math.radians(39.6)
    distance = radius * margin / math.tan(fov / 2)
    direction = Vector((math.cos(el) * math.cos(az),
                        math.cos(el) * math.sin(az),
                        math.sin(el)))
    cam.location = center + direction * distance
    cam.rotation_euler = (direction * -1).to_track_quat("-Z", "Y").to_euler()
    print(f"  camera framed on {len(objects)} objects, distance {distance:.1f}")
    return cam


# ---------------------------------------------------------------- entry point
def main():
    """default scope: background and render settings only"""
    if FIX_MATERIALS:
        print("Rebuilding pocket materials...")
        touched = fix_pocket_objects()
        print(f"  applied to {len(touched)} objects: {', '.join(touched) or '(none)'}")
        for obj in bpy.context.scene.objects:
            if "ligand" in obj.name.lower():
                assign_material(obj, build_solid_material("ligand_orange", LIGAND))
    else:
        print("Materials left untouched (FIX_MATERIALS = False)")

    set_world()
    set_render_settings()
    if SET_CAMERA and not restore_camera():
        frame_objects()
        save_camera()
    print(f"Background: film_transparent={FILM_TRANSPARENT}, "
          f"view_transform={VIEW_TRANSFORM}, RGBA output.")
    print("Render to PNG; matplotlib composites the alpha onto white.")


if __name__ == "__main__":
    main()
