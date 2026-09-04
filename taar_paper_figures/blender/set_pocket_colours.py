"""Colour the pocket surfaces so apo reads as SAND and holo as DUSTY MAUVE, and so a
small pocket nested inside a large one of the same colour stays visible.

Run inside Blender:  Scripting tab -> Open -> Run Script

Colours only. No slicing, no geometry-node edits - TM7 is handled manually.

TWO PROBLEMS THIS SOLVES

1. Cherry-red holo.
   The earlier version drove an Emission node with a heavily "compensated" colour
   (#8C002F for holo) on the theory that the translucent body would wash it back to the
   target over white. That only holds for a SINGLE isolated surface. Emission emits the
   raw colour, so at back faces and self-overlaps the uncomposited #8C002F leaked through
   as cherry, and every extra overlapping layer pushed it further toward pure red.
   Fix: emit the TRUE target colour and raise the body opacity enough that the true colour
   is actually reachable (holo needs >= 0.91 over white; DIFFUSE has no such limit). No
   compensation, so nothing can drift toward cherry no matter how the surfaces stack.

2. Small pockets vanishing.
   apo_ortho_small sits INSIDE apo_ortho_large, same colour, so at equal opacity the large
   pocket simply paints over it. Giving the two sizes DIFFERENT jobs fixes it without a
   second hue: the large pocket is a light, mostly-transparent shell, the small one is a
   dense, saturated core. Size then reads as shell-vs-core before you compare anything.
"""
import bpy
import numpy as np

# ---------------------------------------------------------------- targets
TARGET = {"apo": "#D9B98B",     # sand
          "holo": "#9B5560"}    # dusty mauve

STATE_TOKENS = ("apo", "holo")
POCKET_TOKEN = "ortho"
EXCLUDE = ("backbone", "ribbon", "protein", "cartoon", "ligand")

# ---------------------------------------------------------------- look
# per-size settings: a large pocket is a see-through shell, a small pocket a solid core,
# so a small pocket nested in a large one of the same colour still reads.
SIZE = {
    "large": dict(body_alpha=0.30, rim_boost=1.0),   # light shell, strong rim
    "small": dict(body_alpha=0.85, rim_boost=1.0),   # dense core
    None:    dict(body_alpha=0.55, rim_boost=1.0),   # unsized fallback
}
RIM_START = 0.35         # Layer Weight facing value where the rim begins (lower = wider)
USE_EMISSION = False     # False = Diffuse (recommended): the true colour is reachable at
# any opacity, so holo cannot go cherry. True = Emission: exact colour but only faithful
# at body_alpha >= 0.91 for holo; below that it needs compensation and can leak red.
TRANSPARENT_BOUNCES = 64


# ---------------------------------------------------------------- colour helpers
def hex_to_srgb(value):
    value = value.lstrip("#")
    return tuple(int(value[i:i + 2], 16) / 255 for i in (0, 2, 4))


# ---------------------------------------------------------------- material
def _new(tree, kind, location):
    node = tree.nodes.new(kind)
    node.location = location
    return node


def build_material(name, target_srgb, body_alpha):
    """translucent body + true-colour rim. The body and the rim are the SAME hue, so the
    pocket reads as its colour at every facing angle; only the opacity varies."""
    material = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    material.use_nodes = True
    tree = material.node_tree
    tree.nodes.clear()

    colour = (*target_srgb, 1.0)

    facing = _new(tree, "ShaderNodeLayerWeight", (-820, 60))
    facing.inputs["Blend"].default_value = 0.4

    # opacity ramp: body_alpha head-on, 1.0 at the grazing rim
    ramp = _new(tree, "ShaderNodeMapRange", (-620, 160))
    ramp.inputs["From Min"].default_value = RIM_START
    ramp.inputs["From Max"].default_value = 1.0
    ramp.inputs["To Min"].default_value = body_alpha
    ramp.inputs["To Max"].default_value = 1.0
    ramp.clamp = True

    clear = _new(tree, "ShaderNodeBsdfTransparent", (-320, 150))
    clear.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)   # WHITE: a tinted
    # transparent shader multiplies, so overlaps would darken and over-saturate

    if USE_EMISSION:
        body = _new(tree, "ShaderNodeEmission", (-320, -120))
        body.inputs["Color"].default_value = colour
        body.inputs["Strength"].default_value = 1.0    # exactly the colour, no lift
        shader_out = body.outputs["Emission"]
    else:
        body = _new(tree, "ShaderNodeBsdfPrincipled", (-320, -120))
        body.inputs["Base Color"].default_value = colour
        body.inputs["Roughness"].default_value = 0.45
        if "Emission Strength" in body.inputs:
            body.inputs["Emission Strength"].default_value = 0.0
        shader_out = body.outputs["BSDF"]

    mix = _new(tree, "ShaderNodeMixShader", (-60, 0))
    output = _new(tree, "ShaderNodeOutputMaterial", (160, 0))

    links = tree.links
    links.new(facing.outputs["Facing"], ramp.inputs["Value"])
    links.new(ramp.outputs["Result"], mix.inputs["Fac"])
    links.new(clear.outputs["BSDF"], mix.inputs[1])
    links.new(shader_out, mix.inputs[2])
    links.new(mix.outputs["Shader"], output.inputs["Surface"])

    if hasattr(material, "surface_render_method"):        # Blender 4.2+
        material.surface_render_method = "BLENDED"
    elif hasattr(material, "blend_method"):
        material.blend_method = "BLEND"
    material.use_backface_culling = False
    if hasattr(material, "use_transparent_shadow"):
        material.use_transparent_shadow = True
    return material


def assign(obj, material):
    if obj.data is not None and hasattr(obj.data, "materials"):
        if obj.data.materials:
            obj.data.materials[0] = material
        else:
            obj.data.materials.append(material)
    for modifier in obj.modifiers:                 # MN drives the style's Material socket
        if modifier.type != "NODES" or modifier.node_group is None:
            continue
        for node in modifier.node_group.nodes:
            for socket in node.inputs:
                if socket.type == "MATERIAL":
                    socket.default_value = material


def classify(name):
    """(state, size) for a pocket object, or (None, None)"""
    lowered = name.lower()
    if any(token in lowered for token in EXCLUDE):
        return None, None
    if POCKET_TOKEN not in lowered:
        return None, None
    state = next((s for s in STATE_TOKENS if s in lowered), None)
    size = "large" if "large" in lowered else "small" if "small" in lowered else None
    return state, size


# ---------------------------------------------------------------- entry point
def main():
    scene = bpy.context.scene
    print(f"emission={USE_EMISSION}\n")

    touched = 0
    for obj in scene.objects:
        if obj.type != "MESH":
            continue
        state, size = classify(obj.name)
        if state is None:
            continue
        body_alpha = SIZE[size]["body_alpha"]
        material = build_material(f"pocket_{obj.name}", hex_to_srgb(TARGET[state]),
                                  body_alpha)
        assign(obj, material)
        touched += 1
        print(f"  {obj.name}: {state}/{size or 'unsized'}  body alpha {body_alpha:.2f}")

    if scene.render.engine == "CYCLES":
        scene.cycles.transparent_max_bounces = TRANSPARENT_BOUNCES
    print(f"\n{touched} pocket object(s) recoloured. Large = light shell, small = dense "
          f"core, so a nested small pocket stays visible.")


if __name__ == "__main__":
    main()
