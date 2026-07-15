#!/usr/bin/env bash
# fetch_gear_assets.sh — download the EXACT FORGE (arXiv 2408.04587) gear assets
# and author the body-root wrapper USDAs the task1 env spawns from.
#
# Idempotent. Run once per pod:  bash scripts/fetch_gear_assets.sh
#
# Why wrappers: the factory USDs nest the rigid body one level down
# (/factory_gear_medium/factory_gear_medium) and gear_base carries a
# root_joint + ArticulationRootAPI. Referencing the BODY prim directly puts
# the rigid body AT the spawn root (like the LIBERO assets task3 used), so
# contact-sensor paths resolve and the root_joint composes away.
set -euo pipefail

DEST=/workspace/assets/factory
S3=https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/IsaacLab/Factory
mkdir -p "$DEST"

for f in factory_gear_base factory_gear_small factory_gear_medium factory_gear_large; do
  if [ ! -s "$DEST/$f.usd" ]; then
    echo "[fetch] $f.usd"
    curl -sf -o "$DEST/$f.usd" "$S3/$f.usd"
  fi
  head -c 8 "$DEST/$f.usd" | grep -q "PXR-USDC" || { echo "BAD USD: $f"; exit 1; }
done

cat > "$DEST/gear_medium_body.usda" <<'EOF'
#usda 1.0
(
    defaultPrim = "gear_medium"
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "gear_medium" (
    prepend references = @./factory_gear_medium.usd@</factory_gear_medium/factory_gear_medium>
)
{
    # FULL CONVEX functional collision (factory SDF cooks WITHOUT the bore —
    # probe_bore_physics.py). RAMP chamfer: 5 x 0.6 mm bands, radii
    # 7.0->5.6 mm (0.35 mm ledges slide under press; the discrete staircase
    # ledges were hard catches — jam traces: re-approaches at ~2 mm caught
    # the 5.8->5.0 step forever). Fit band r 5.15 (0.4 mm clearance) —
    # relaxed from exact-FORGE 0.25 mm to absorb 16-gon vertex artifacts.
    # body ring: plate rest; hub ring: pad grip.
    over "collisions" (
        active = false
    )
    {
    }
    def Cube "funnel0_0" (
        prepend apiSchemas = ["PhysicsCollisionAPI"]
    )
    {
        double size = 1
        uniform token purpose = "guide"
        float3 xformOp:translate = (0.000000, 0.008100, 0.00530)
        float3 xformOp:rotateXYZ = (0, 0, -0.0)
        float3 xformOp:scale = (0.002840, 0.002200, 0.00060)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ", "xformOp:scale"]
    }
    def Cube "funnel0_1" (
        prepend apiSchemas = ["PhysicsCollisionAPI"]
    )
    {
        double size = 1
        uniform token purpose = "guide"
        float3 xformOp:translate = (0.003100, 0.007483, 0.00530)
        float3 xformOp:rotateXYZ = (0, 0, -22.5)
        float3 xformOp:scale = (0.002840, 0.002200, 0.00060)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ", "xformOp:scale"]
    }
    def Cube "funnel0_2" (
        prepend apiSchemas = ["PhysicsCollisionAPI"]
    )
    {
        double size = 1
        uniform token purpose = "guide"
        float3 xformOp:translate = (0.005728, 0.005728, 0.00530)
        float3 xformOp:rotateXYZ = (0, 0, -45.0)
        float3 xformOp:scale = (0.002840, 0.002200, 0.00060)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ", "xformOp:scale"]
    }
    def Cube "funnel0_3" (
        prepend apiSchemas = ["PhysicsCollisionAPI"]
    )
    {
        double size = 1
        uniform token purpose = "guide"
        float3 xformOp:translate = (0.007483, 0.003100, 0.00530)
        float3 xformOp:rotateXYZ = (0, 0, -67.5)
        float3 xformOp:scale = (0.002840, 0.002200, 0.00060)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ", "xformOp:scale"]
    }
    def Cube "funnel0_4" (
        prepend apiSchemas = ["PhysicsCollisionAPI"]
    )
    {
        double size = 1
        uniform token purpose = "guide"
        float3 xformOp:translate = (0.008100, 0.000000, 0.00530)
        float3 xformOp:rotateXYZ = (0, 0, -90.0)
        float3 xformOp:scale = (0.002840, 0.002200, 0.00060)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ", "xformOp:scale"]
    }
    def Cube "funnel0_5" (
        prepend apiSchemas = ["PhysicsCollisionAPI"]
    )
    {
        double size = 1
        uniform token purpose = "guide"
        float3 xformOp:translate = (0.007483, -0.003100, 0.00530)
        float3 xformOp:rotateXYZ = (0, 0, -112.5)
        float3 xformOp:scale = (0.002840, 0.002200, 0.00060)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ", "xformOp:scale"]
    }
    def Cube "funnel0_6" (
        prepend apiSchemas = ["PhysicsCollisionAPI"]
    )
    {
        double size = 1
        uniform token purpose = "guide"
        float3 xformOp:translate = (0.005728, -0.005728, 0.00530)
        float3 xformOp:rotateXYZ = (0, 0, -135.0)
        float3 xformOp:scale = (0.002840, 0.002200, 0.00060)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ", "xformOp:scale"]
    }
    def Cube "funnel0_7" (
        prepend apiSchemas = ["PhysicsCollisionAPI"]
    )
    {
        double size = 1
        uniform token purpose = "guide"
        float3 xformOp:translate = (0.003100, -0.007483, 0.00530)
        float3 xformOp:rotateXYZ = (0, 0, -157.5)
        float3 xformOp:scale = (0.002840, 0.002200, 0.00060)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ", "xformOp:scale"]
    }
    def Cube "funnel0_8" (
        prepend apiSchemas = ["PhysicsCollisionAPI"]
    )
    {
        double size = 1
        uniform token purpose = "guide"
        float3 xformOp:translate = (0.000000, -0.008100, 0.00530)
        float3 xformOp:rotateXYZ = (0, 0, -180.0)
        float3 xformOp:scale = (0.002840, 0.002200, 0.00060)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ", "xformOp:scale"]
    }
    def Cube "funnel0_9" (
        prepend apiSchemas = ["PhysicsCollisionAPI"]
    )
    {
        double size = 1
        uniform token purpose = "guide"
        float3 xformOp:translate = (-0.003100, -0.007483, 0.00530)
        float3 xformOp:rotateXYZ = (0, 0, -202.5)
        float3 xformOp:scale = (0.002840, 0.002200, 0.00060)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ", "xformOp:scale"]
    }
    def Cube "funnel0_10" (
        prepend apiSchemas = ["PhysicsCollisionAPI"]
    )
    {
        double size = 1
        uniform token purpose = "guide"
        float3 xformOp:translate = (-0.005728, -0.005728, 0.00530)
        float3 xformOp:rotateXYZ = (0, 0, -225.0)
        float3 xformOp:scale = (0.002840, 0.002200, 0.00060)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ", "xformOp:scale"]
    }
    def Cube "funnel0_11" (
        prepend apiSchemas = ["PhysicsCollisionAPI"]
    )
    {
        double size = 1
        uniform token purpose = "guide"
        float3 xformOp:translate = (-0.007483, -0.003100, 0.00530)
        float3 xformOp:rotateXYZ = (0, 0, -247.5)
        float3 xformOp:scale = (0.002840, 0.002200, 0.00060)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ", "xformOp:scale"]
    }
    def Cube "funnel0_12" (
        prepend apiSchemas = ["PhysicsCollisionAPI"]
    )
    {
        double size = 1
        uniform token purpose = "guide"
        float3 xformOp:translate = (-0.008100, -0.000000, 0.00530)
        float3 xformOp:rotateXYZ = (0, 0, -270.0)
        float3 xformOp:scale = (0.002840, 0.002200, 0.00060)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ", "xformOp:scale"]
    }
    def Cube "funnel0_13" (
        prepend apiSchemas = ["PhysicsCollisionAPI"]
    )
    {
        double size = 1
        uniform token purpose = "guide"
        float3 xformOp:translate = (-0.007483, 0.003100, 0.00530)
        float3 xformOp:rotateXYZ = (0, 0, -292.5)
        float3 xformOp:scale = (0.002840, 0.002200, 0.00060)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ", "xformOp:scale"]
    }
    def Cube "funnel0_14" (
        prepend apiSchemas = ["PhysicsCollisionAPI"]
    )
    {
        double size = 1
        uniform token purpose = "guide"
        float3 xformOp:translate = (-0.005728, 0.005728, 0.00530)
        float3 xformOp:rotateXYZ = (0, 0, -315.0)
        float3 xformOp:scale = (0.002840, 0.002200, 0.00060)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ", "xformOp:scale"]
    }
    def Cube "funnel0_15" (
        prepend apiSchemas = ["PhysicsCollisionAPI"]
    )
    {
        double size = 1
        uniform token purpose = "guide"
        float3 xformOp:translate = (-0.003100, 0.007483, 0.00530)
        float3 xformOp:rotateXYZ = (0, 0, -337.5)
        float3 xformOp:scale = (0.002840, 0.002200, 0.00060)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ", "xformOp:scale"]
    }
    def Cube "funnel1_0" (
        prepend apiSchemas = ["PhysicsCollisionAPI"]
    )
    {
        double size = 1
        uniform token purpose = "guide"
        float3 xformOp:translate = (0.000000, 0.007750, 0.00590)
        float3 xformOp:rotateXYZ = (0, 0, -0.0)
        float3 xformOp:scale = (0.002698, 0.002200, 0.00060)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ", "xformOp:scale"]
    }
    def Cube "funnel1_1" (
        prepend apiSchemas = ["PhysicsCollisionAPI"]
    )
    {
        double size = 1
        uniform token purpose = "guide"
        float3 xformOp:translate = (0.002966, 0.007160, 0.00590)
        float3 xformOp:rotateXYZ = (0, 0, -22.5)
        float3 xformOp:scale = (0.002698, 0.002200, 0.00060)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ", "xformOp:scale"]
    }
    def Cube "funnel1_2" (
        prepend apiSchemas = ["PhysicsCollisionAPI"]
    )
    {
        double size = 1
        uniform token purpose = "guide"
        float3 xformOp:translate = (0.005480, 0.005480, 0.00590)
        float3 xformOp:rotateXYZ = (0, 0, -45.0)
        float3 xformOp:scale = (0.002698, 0.002200, 0.00060)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ", "xformOp:scale"]
    }
    def Cube "funnel1_3" (
        prepend apiSchemas = ["PhysicsCollisionAPI"]
    )
    {
        double size = 1
        uniform token purpose = "guide"
        float3 xformOp:translate = (0.007160, 0.002966, 0.00590)
        float3 xformOp:rotateXYZ = (0, 0, -67.5)
        float3 xformOp:scale = (0.002698, 0.002200, 0.00060)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ", "xformOp:scale"]
    }
    def Cube "funnel1_4" (
        prepend apiSchemas = ["PhysicsCollisionAPI"]
    )
    {
        double size = 1
        uniform token purpose = "guide"
        float3 xformOp:translate = (0.007750, 0.000000, 0.00590)
        float3 xformOp:rotateXYZ = (0, 0, -90.0)
        float3 xformOp:scale = (0.002698, 0.002200, 0.00060)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ", "xformOp:scale"]
    }
    def Cube "funnel1_5" (
        prepend apiSchemas = ["PhysicsCollisionAPI"]
    )
    {
        double size = 1
        uniform token purpose = "guide"
        float3 xformOp:translate = (0.007160, -0.002966, 0.00590)
        float3 xformOp:rotateXYZ = (0, 0, -112.5)
        float3 xformOp:scale = (0.002698, 0.002200, 0.00060)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ", "xformOp:scale"]
    }
    def Cube "funnel1_6" (
        prepend apiSchemas = ["PhysicsCollisionAPI"]
    )
    {
        double size = 1
        uniform token purpose = "guide"
        float3 xformOp:translate = (0.005480, -0.005480, 0.00590)
        float3 xformOp:rotateXYZ = (0, 0, -135.0)
        float3 xformOp:scale = (0.002698, 0.002200, 0.00060)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ", "xformOp:scale"]
    }
    def Cube "funnel1_7" (
        prepend apiSchemas = ["PhysicsCollisionAPI"]
    )
    {
        double size = 1
        uniform token purpose = "guide"
        float3 xformOp:translate = (0.002966, -0.007160, 0.00590)
        float3 xformOp:rotateXYZ = (0, 0, -157.5)
        float3 xformOp:scale = (0.002698, 0.002200, 0.00060)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ", "xformOp:scale"]
    }
    def Cube "funnel1_8" (
        prepend apiSchemas = ["PhysicsCollisionAPI"]
    )
    {
        double size = 1
        uniform token purpose = "guide"
        float3 xformOp:translate = (0.000000, -0.007750, 0.00590)
        float3 xformOp:rotateXYZ = (0, 0, -180.0)
        float3 xformOp:scale = (0.002698, 0.002200, 0.00060)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ", "xformOp:scale"]
    }
    def Cube "funnel1_9" (
        prepend apiSchemas = ["PhysicsCollisionAPI"]
    )
    {
        double size = 1
        uniform token purpose = "guide"
        float3 xformOp:translate = (-0.002966, -0.007160, 0.00590)
        float3 xformOp:rotateXYZ = (0, 0, -202.5)
        float3 xformOp:scale = (0.002698, 0.002200, 0.00060)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ", "xformOp:scale"]
    }
    def Cube "funnel1_10" (
        prepend apiSchemas = ["PhysicsCollisionAPI"]
    )
    {
        double size = 1
        uniform token purpose = "guide"
        float3 xformOp:translate = (-0.005480, -0.005480, 0.00590)
        float3 xformOp:rotateXYZ = (0, 0, -225.0)
        float3 xformOp:scale = (0.002698, 0.002200, 0.00060)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ", "xformOp:scale"]
    }
    def Cube "funnel1_11" (
        prepend apiSchemas = ["PhysicsCollisionAPI"]
    )
    {
        double size = 1
        uniform token purpose = "guide"
        float3 xformOp:translate = (-0.007160, -0.002966, 0.00590)
        float3 xformOp:rotateXYZ = (0, 0, -247.5)
        float3 xformOp:scale = (0.002698, 0.002200, 0.00060)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ", "xformOp:scale"]
    }
    def Cube "funnel1_12" (
        prepend apiSchemas = ["PhysicsCollisionAPI"]
    )
    {
        double size = 1
        uniform token purpose = "guide"
        float3 xformOp:translate = (-0.007750, -0.000000, 0.00590)
        float3 xformOp:rotateXYZ = (0, 0, -270.0)
        float3 xformOp:scale = (0.002698, 0.002200, 0.00060)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ", "xformOp:scale"]
    }
    def Cube "funnel1_13" (
        prepend apiSchemas = ["PhysicsCollisionAPI"]
    )
    {
        double size = 1
        uniform token purpose = "guide"
        float3 xformOp:translate = (-0.007160, 0.002966, 0.00590)
        float3 xformOp:rotateXYZ = (0, 0, -292.5)
        float3 xformOp:scale = (0.002698, 0.002200, 0.00060)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ", "xformOp:scale"]
    }
    def Cube "funnel1_14" (
        prepend apiSchemas = ["PhysicsCollisionAPI"]
    )
    {
        double size = 1
        uniform token purpose = "guide"
        float3 xformOp:translate = (-0.005480, 0.005480, 0.00590)
        float3 xformOp:rotateXYZ = (0, 0, -315.0)
        float3 xformOp:scale = (0.002698, 0.002200, 0.00060)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ", "xformOp:scale"]
    }
    def Cube "funnel1_15" (
        prepend apiSchemas = ["PhysicsCollisionAPI"]
    )
    {
        double size = 1
        uniform token purpose = "guide"
        float3 xformOp:translate = (-0.002966, 0.007160, 0.00590)
        float3 xformOp:rotateXYZ = (0, 0, -337.5)
        float3 xformOp:scale = (0.002698, 0.002200, 0.00060)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ", "xformOp:scale"]
    }
    def Cube "funnel2_0" (
        prepend apiSchemas = ["PhysicsCollisionAPI"]
    )
    {
        double size = 1
        uniform token purpose = "guide"
        float3 xformOp:translate = (0.000000, 0.007400, 0.00650)
        float3 xformOp:rotateXYZ = (0, 0, -0.0)
        float3 xformOp:scale = (0.002556, 0.002200, 0.00060)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ", "xformOp:scale"]
    }
    def Cube "funnel2_1" (
        prepend apiSchemas = ["PhysicsCollisionAPI"]
    )
    {
        double size = 1
        uniform token purpose = "guide"
        float3 xformOp:translate = (0.002832, 0.006837, 0.00650)
        float3 xformOp:rotateXYZ = (0, 0, -22.5)
        float3 xformOp:scale = (0.002556, 0.002200, 0.00060)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ", "xformOp:scale"]
    }
    def Cube "funnel2_2" (
        prepend apiSchemas = ["PhysicsCollisionAPI"]
    )
    {
        double size = 1
        uniform token purpose = "guide"
        float3 xformOp:translate = (0.005233, 0.005233, 0.00650)
        float3 xformOp:rotateXYZ = (0, 0, -45.0)
        float3 xformOp:scale = (0.002556, 0.002200, 0.00060)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ", "xformOp:scale"]
    }
    def Cube "funnel2_3" (
        prepend apiSchemas = ["PhysicsCollisionAPI"]
    )
    {
        double size = 1
        uniform token purpose = "guide"
        float3 xformOp:translate = (0.006837, 0.002832, 0.00650)
        float3 xformOp:rotateXYZ = (0, 0, -67.5)
        float3 xformOp:scale = (0.002556, 0.002200, 0.00060)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ", "xformOp:scale"]
    }
    def Cube "funnel2_4" (
        prepend apiSchemas = ["PhysicsCollisionAPI"]
    )
    {
        double size = 1
        uniform token purpose = "guide"
        float3 xformOp:translate = (0.007400, 0.000000, 0.00650)
        float3 xformOp:rotateXYZ = (0, 0, -90.0)
        float3 xformOp:scale = (0.002556, 0.002200, 0.00060)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ", "xformOp:scale"]
    }
    def Cube "funnel2_5" (
        prepend apiSchemas = ["PhysicsCollisionAPI"]
    )
    {
        double size = 1
        uniform token purpose = "guide"
        float3 xformOp:translate = (0.006837, -0.002832, 0.00650)
        float3 xformOp:rotateXYZ = (0, 0, -112.5)
        float3 xformOp:scale = (0.002556, 0.002200, 0.00060)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ", "xformOp:scale"]
    }
    def Cube "funnel2_6" (
        prepend apiSchemas = ["PhysicsCollisionAPI"]
    )
    {
        double size = 1
        uniform token purpose = "guide"
        float3 xformOp:translate = (0.005233, -0.005233, 0.00650)
        float3 xformOp:rotateXYZ = (0, 0, -135.0)
        float3 xformOp:scale = (0.002556, 0.002200, 0.00060)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ", "xformOp:scale"]
    }
    def Cube "funnel2_7" (
        prepend apiSchemas = ["PhysicsCollisionAPI"]
    )
    {
        double size = 1
        uniform token purpose = "guide"
        float3 xformOp:translate = (0.002832, -0.006837, 0.00650)
        float3 xformOp:rotateXYZ = (0, 0, -157.5)
        float3 xformOp:scale = (0.002556, 0.002200, 0.00060)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ", "xformOp:scale"]
    }
    def Cube "funnel2_8" (
        prepend apiSchemas = ["PhysicsCollisionAPI"]
    )
    {
        double size = 1
        uniform token purpose = "guide"
        float3 xformOp:translate = (0.000000, -0.007400, 0.00650)
        float3 xformOp:rotateXYZ = (0, 0, -180.0)
        float3 xformOp:scale = (0.002556, 0.002200, 0.00060)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ", "xformOp:scale"]
    }
    def Cube "funnel2_9" (
        prepend apiSchemas = ["PhysicsCollisionAPI"]
    )
    {
        double size = 1
        uniform token purpose = "guide"
        float3 xformOp:translate = (-0.002832, -0.006837, 0.00650)
        float3 xformOp:rotateXYZ = (0, 0, -202.5)
        float3 xformOp:scale = (0.002556, 0.002200, 0.00060)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ", "xformOp:scale"]
    }
    def Cube "funnel2_10" (
        prepend apiSchemas = ["PhysicsCollisionAPI"]
    )
    {
        double size = 1
        uniform token purpose = "guide"
        float3 xformOp:translate = (-0.005233, -0.005233, 0.00650)
        float3 xformOp:rotateXYZ = (0, 0, -225.0)
        float3 xformOp:scale = (0.002556, 0.002200, 0.00060)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ", "xformOp:scale"]
    }
    def Cube "funnel2_11" (
        prepend apiSchemas = ["PhysicsCollisionAPI"]
    )
    {
        double size = 1
        uniform token purpose = "guide"
        float3 xformOp:translate = (-0.006837, -0.002832, 0.00650)
        float3 xformOp:rotateXYZ = (0, 0, -247.5)
        float3 xformOp:scale = (0.002556, 0.002200, 0.00060)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ", "xformOp:scale"]
    }
    def Cube "funnel2_12" (
        prepend apiSchemas = ["PhysicsCollisionAPI"]
    )
    {
        double size = 1
        uniform token purpose = "guide"
        float3 xformOp:translate = (-0.007400, -0.000000, 0.00650)
        float3 xformOp:rotateXYZ = (0, 0, -270.0)
        float3 xformOp:scale = (0.002556, 0.002200, 0.00060)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ", "xformOp:scale"]
    }
    def Cube "funnel2_13" (
        prepend apiSchemas = ["PhysicsCollisionAPI"]
    )
    {
        double size = 1
        uniform token purpose = "guide"
        float3 xformOp:translate = (-0.006837, 0.002832, 0.00650)
        float3 xformOp:rotateXYZ = (0, 0, -292.5)
        float3 xformOp:scale = (0.002556, 0.002200, 0.00060)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ", "xformOp:scale"]
    }
    def Cube "funnel2_14" (
        prepend apiSchemas = ["PhysicsCollisionAPI"]
    )
    {
        double size = 1
        uniform token purpose = "guide"
        float3 xformOp:translate = (-0.005233, 0.005233, 0.00650)
        float3 xformOp:rotateXYZ = (0, 0, -315.0)
        float3 xformOp:scale = (0.002556, 0.002200, 0.00060)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ", "xformOp:scale"]
    }
    def Cube "funnel2_15" (
        prepend apiSchemas = ["PhysicsCollisionAPI"]
    )
    {
        double size = 1
        uniform token purpose = "guide"
        float3 xformOp:translate = (-0.002832, 0.006837, 0.00650)
        float3 xformOp:rotateXYZ = (0, 0, -337.5)
        float3 xformOp:scale = (0.002556, 0.002200, 0.00060)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ", "xformOp:scale"]
    }
    def Cube "funnel3_0" (
        prepend apiSchemas = ["PhysicsCollisionAPI"]
    )
    {
        double size = 1
        uniform token purpose = "guide"
        float3 xformOp:translate = (0.000000, 0.007050, 0.00710)
        float3 xformOp:rotateXYZ = (0, 0, -0.0)
        float3 xformOp:scale = (0.002414, 0.002200, 0.00060)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ", "xformOp:scale"]
    }
    def Cube "funnel3_1" (
        prepend apiSchemas = ["PhysicsCollisionAPI"]
    )
    {
        double size = 1
        uniform token purpose = "guide"
        float3 xformOp:translate = (0.002698, 0.006513, 0.00710)
        float3 xformOp:rotateXYZ = (0, 0, -22.5)
        float3 xformOp:scale = (0.002414, 0.002200, 0.00060)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ", "xformOp:scale"]
    }
    def Cube "funnel3_2" (
        prepend apiSchemas = ["PhysicsCollisionAPI"]
    )
    {
        double size = 1
        uniform token purpose = "guide"
        float3 xformOp:translate = (0.004985, 0.004985, 0.00710)
        float3 xformOp:rotateXYZ = (0, 0, -45.0)
        float3 xformOp:scale = (0.002414, 0.002200, 0.00060)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ", "xformOp:scale"]
    }
    def Cube "funnel3_3" (
        prepend apiSchemas = ["PhysicsCollisionAPI"]
    )
    {
        double size = 1
        uniform token purpose = "guide"
        float3 xformOp:translate = (0.006513, 0.002698, 0.00710)
        float3 xformOp:rotateXYZ = (0, 0, -67.5)
        float3 xformOp:scale = (0.002414, 0.002200, 0.00060)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ", "xformOp:scale"]
    }
    def Cube "funnel3_4" (
        prepend apiSchemas = ["PhysicsCollisionAPI"]
    )
    {
        double size = 1
        uniform token purpose = "guide"
        float3 xformOp:translate = (0.007050, 0.000000, 0.00710)
        float3 xformOp:rotateXYZ = (0, 0, -90.0)
        float3 xformOp:scale = (0.002414, 0.002200, 0.00060)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ", "xformOp:scale"]
    }
    def Cube "funnel3_5" (
        prepend apiSchemas = ["PhysicsCollisionAPI"]
    )
    {
        double size = 1
        uniform token purpose = "guide"
        float3 xformOp:translate = (0.006513, -0.002698, 0.00710)
        float3 xformOp:rotateXYZ = (0, 0, -112.5)
        float3 xformOp:scale = (0.002414, 0.002200, 0.00060)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ", "xformOp:scale"]
    }
    def Cube "funnel3_6" (
        prepend apiSchemas = ["PhysicsCollisionAPI"]
    )
    {
        double size = 1
        uniform token purpose = "guide"
        float3 xformOp:translate = (0.004985, -0.004985, 0.00710)
        float3 xformOp:rotateXYZ = (0, 0, -135.0)
        float3 xformOp:scale = (0.002414, 0.002200, 0.00060)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ", "xformOp:scale"]
    }
    def Cube "funnel3_7" (
        prepend apiSchemas = ["PhysicsCollisionAPI"]
    )
    {
        double size = 1
        uniform token purpose = "guide"
        float3 xformOp:translate = (0.002698, -0.006513, 0.00710)
        float3 xformOp:rotateXYZ = (0, 0, -157.5)
        float3 xformOp:scale = (0.002414, 0.002200, 0.00060)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ", "xformOp:scale"]
    }
    def Cube "funnel3_8" (
        prepend apiSchemas = ["PhysicsCollisionAPI"]
    )
    {
        double size = 1
        uniform token purpose = "guide"
        float3 xformOp:translate = (0.000000, -0.007050, 0.00710)
        float3 xformOp:rotateXYZ = (0, 0, -180.0)
        float3 xformOp:scale = (0.002414, 0.002200, 0.00060)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ", "xformOp:scale"]
    }
    def Cube "funnel3_9" (
        prepend apiSchemas = ["PhysicsCollisionAPI"]
    )
    {
        double size = 1
        uniform token purpose = "guide"
        float3 xformOp:translate = (-0.002698, -0.006513, 0.00710)
        float3 xformOp:rotateXYZ = (0, 0, -202.5)
        float3 xformOp:scale = (0.002414, 0.002200, 0.00060)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ", "xformOp:scale"]
    }
    def Cube "funnel3_10" (
        prepend apiSchemas = ["PhysicsCollisionAPI"]
    )
    {
        double size = 1
        uniform token purpose = "guide"
        float3 xformOp:translate = (-0.004985, -0.004985, 0.00710)
        float3 xformOp:rotateXYZ = (0, 0, -225.0)
        float3 xformOp:scale = (0.002414, 0.002200, 0.00060)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ", "xformOp:scale"]
    }
    def Cube "funnel3_11" (
        prepend apiSchemas = ["PhysicsCollisionAPI"]
    )
    {
        double size = 1
        uniform token purpose = "guide"
        float3 xformOp:translate = (-0.006513, -0.002698, 0.00710)
        float3 xformOp:rotateXYZ = (0, 0, -247.5)
        float3 xformOp:scale = (0.002414, 0.002200, 0.00060)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ", "xformOp:scale"]
    }
    def Cube "funnel3_12" (
        prepend apiSchemas = ["PhysicsCollisionAPI"]
    )
    {
        double size = 1
        uniform token purpose = "guide"
        float3 xformOp:translate = (-0.007050, -0.000000, 0.00710)
        float3 xformOp:rotateXYZ = (0, 0, -270.0)
        float3 xformOp:scale = (0.002414, 0.002200, 0.00060)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ", "xformOp:scale"]
    }
    def Cube "funnel3_13" (
        prepend apiSchemas = ["PhysicsCollisionAPI"]
    )
    {
        double size = 1
        uniform token purpose = "guide"
        float3 xformOp:translate = (-0.006513, 0.002698, 0.00710)
        float3 xformOp:rotateXYZ = (0, 0, -292.5)
        float3 xformOp:scale = (0.002414, 0.002200, 0.00060)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ", "xformOp:scale"]
    }
    def Cube "funnel3_14" (
        prepend apiSchemas = ["PhysicsCollisionAPI"]
    )
    {
        double size = 1
        uniform token purpose = "guide"
        float3 xformOp:translate = (-0.004985, 0.004985, 0.00710)
        float3 xformOp:rotateXYZ = (0, 0, -315.0)
        float3 xformOp:scale = (0.002414, 0.002200, 0.00060)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ", "xformOp:scale"]
    }
    def Cube "funnel3_15" (
        prepend apiSchemas = ["PhysicsCollisionAPI"]
    )
    {
        double size = 1
        uniform token purpose = "guide"
        float3 xformOp:translate = (-0.002698, 0.006513, 0.00710)
        float3 xformOp:rotateXYZ = (0, 0, -337.5)
        float3 xformOp:scale = (0.002414, 0.002200, 0.00060)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ", "xformOp:scale"]
    }
    def Cube "funnel4_0" (
        prepend apiSchemas = ["PhysicsCollisionAPI"]
    )
    {
        double size = 1
        uniform token purpose = "guide"
        float3 xformOp:translate = (0.000000, 0.006700, 0.00770)
        float3 xformOp:rotateXYZ = (0, 0, -0.0)
        float3 xformOp:scale = (0.002272, 0.002200, 0.00060)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ", "xformOp:scale"]
    }
    def Cube "funnel4_1" (
        prepend apiSchemas = ["PhysicsCollisionAPI"]
    )
    {
        double size = 1
        uniform token purpose = "guide"
        float3 xformOp:translate = (0.002564, 0.006190, 0.00770)
        float3 xformOp:rotateXYZ = (0, 0, -22.5)
        float3 xformOp:scale = (0.002272, 0.002200, 0.00060)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ", "xformOp:scale"]
    }
    def Cube "funnel4_2" (
        prepend apiSchemas = ["PhysicsCollisionAPI"]
    )
    {
        double size = 1
        uniform token purpose = "guide"
        float3 xformOp:translate = (0.004738, 0.004738, 0.00770)
        float3 xformOp:rotateXYZ = (0, 0, -45.0)
        float3 xformOp:scale = (0.002272, 0.002200, 0.00060)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ", "xformOp:scale"]
    }
    def Cube "funnel4_3" (
        prepend apiSchemas = ["PhysicsCollisionAPI"]
    )
    {
        double size = 1
        uniform token purpose = "guide"
        float3 xformOp:translate = (0.006190, 0.002564, 0.00770)
        float3 xformOp:rotateXYZ = (0, 0, -67.5)
        float3 xformOp:scale = (0.002272, 0.002200, 0.00060)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ", "xformOp:scale"]
    }
    def Cube "funnel4_4" (
        prepend apiSchemas = ["PhysicsCollisionAPI"]
    )
    {
        double size = 1
        uniform token purpose = "guide"
        float3 xformOp:translate = (0.006700, 0.000000, 0.00770)
        float3 xformOp:rotateXYZ = (0, 0, -90.0)
        float3 xformOp:scale = (0.002272, 0.002200, 0.00060)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ", "xformOp:scale"]
    }
    def Cube "funnel4_5" (
        prepend apiSchemas = ["PhysicsCollisionAPI"]
    )
    {
        double size = 1
        uniform token purpose = "guide"
        float3 xformOp:translate = (0.006190, -0.002564, 0.00770)
        float3 xformOp:rotateXYZ = (0, 0, -112.5)
        float3 xformOp:scale = (0.002272, 0.002200, 0.00060)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ", "xformOp:scale"]
    }
    def Cube "funnel4_6" (
        prepend apiSchemas = ["PhysicsCollisionAPI"]
    )
    {
        double size = 1
        uniform token purpose = "guide"
        float3 xformOp:translate = (0.004738, -0.004738, 0.00770)
        float3 xformOp:rotateXYZ = (0, 0, -135.0)
        float3 xformOp:scale = (0.002272, 0.002200, 0.00060)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ", "xformOp:scale"]
    }
    def Cube "funnel4_7" (
        prepend apiSchemas = ["PhysicsCollisionAPI"]
    )
    {
        double size = 1
        uniform token purpose = "guide"
        float3 xformOp:translate = (0.002564, -0.006190, 0.00770)
        float3 xformOp:rotateXYZ = (0, 0, -157.5)
        float3 xformOp:scale = (0.002272, 0.002200, 0.00060)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ", "xformOp:scale"]
    }
    def Cube "funnel4_8" (
        prepend apiSchemas = ["PhysicsCollisionAPI"]
    )
    {
        double size = 1
        uniform token purpose = "guide"
        float3 xformOp:translate = (0.000000, -0.006700, 0.00770)
        float3 xformOp:rotateXYZ = (0, 0, -180.0)
        float3 xformOp:scale = (0.002272, 0.002200, 0.00060)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ", "xformOp:scale"]
    }
    def Cube "funnel4_9" (
        prepend apiSchemas = ["PhysicsCollisionAPI"]
    )
    {
        double size = 1
        uniform token purpose = "guide"
        float3 xformOp:translate = (-0.002564, -0.006190, 0.00770)
        float3 xformOp:rotateXYZ = (0, 0, -202.5)
        float3 xformOp:scale = (0.002272, 0.002200, 0.00060)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ", "xformOp:scale"]
    }
    def Cube "funnel4_10" (
        prepend apiSchemas = ["PhysicsCollisionAPI"]
    )
    {
        double size = 1
        uniform token purpose = "guide"
        float3 xformOp:translate = (-0.004738, -0.004738, 0.00770)
        float3 xformOp:rotateXYZ = (0, 0, -225.0)
        float3 xformOp:scale = (0.002272, 0.002200, 0.00060)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ", "xformOp:scale"]
    }
    def Cube "funnel4_11" (
        prepend apiSchemas = ["PhysicsCollisionAPI"]
    )
    {
        double size = 1
        uniform token purpose = "guide"
        float3 xformOp:translate = (-0.006190, -0.002564, 0.00770)
        float3 xformOp:rotateXYZ = (0, 0, -247.5)
        float3 xformOp:scale = (0.002272, 0.002200, 0.00060)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ", "xformOp:scale"]
    }
    def Cube "funnel4_12" (
        prepend apiSchemas = ["PhysicsCollisionAPI"]
    )
    {
        double size = 1
        uniform token purpose = "guide"
        float3 xformOp:translate = (-0.006700, -0.000000, 0.00770)
        float3 xformOp:rotateXYZ = (0, 0, -270.0)
        float3 xformOp:scale = (0.002272, 0.002200, 0.00060)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ", "xformOp:scale"]
    }
    def Cube "funnel4_13" (
        prepend apiSchemas = ["PhysicsCollisionAPI"]
    )
    {
        double size = 1
        uniform token purpose = "guide"
        float3 xformOp:translate = (-0.006190, 0.002564, 0.00770)
        float3 xformOp:rotateXYZ = (0, 0, -292.5)
        float3 xformOp:scale = (0.002272, 0.002200, 0.00060)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ", "xformOp:scale"]
    }
    def Cube "funnel4_14" (
        prepend apiSchemas = ["PhysicsCollisionAPI"]
    )
    {
        double size = 1
        uniform token purpose = "guide"
        float3 xformOp:translate = (-0.004738, 0.004738, 0.00770)
        float3 xformOp:rotateXYZ = (0, 0, -315.0)
        float3 xformOp:scale = (0.002272, 0.002200, 0.00060)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ", "xformOp:scale"]
    }
    def Cube "funnel4_15" (
        prepend apiSchemas = ["PhysicsCollisionAPI"]
    )
    {
        double size = 1
        uniform token purpose = "guide"
        float3 xformOp:translate = (-0.002564, 0.006190, 0.00770)
        float3 xformOp:rotateXYZ = (0, 0, -337.5)
        float3 xformOp:scale = (0.002272, 0.002200, 0.00060)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ", "xformOp:scale"]
    }
    def Cube "bore_col_0" (
        prepend apiSchemas = ["PhysicsCollisionAPI"]
    )
    {
        double size = 1
        uniform token purpose = "guide"
        float3 xformOp:translate = (0.000000, 0.006900, 0.01900)
        float3 xformOp:rotateXYZ = (0, 0, -0.0)
        float3 xformOp:scale = (0.002090, 0.003500, 0.02200)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ", "xformOp:scale"]
    }
    def Cube "bore_col_1" (
        prepend apiSchemas = ["PhysicsCollisionAPI"]
    )
    {
        double size = 1
        uniform token purpose = "guide"
        float3 xformOp:translate = (0.002641, 0.006375, 0.01900)
        float3 xformOp:rotateXYZ = (0, 0, -22.5)
        float3 xformOp:scale = (0.002090, 0.003500, 0.02200)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ", "xformOp:scale"]
    }
    def Cube "bore_col_2" (
        prepend apiSchemas = ["PhysicsCollisionAPI"]
    )
    {
        double size = 1
        uniform token purpose = "guide"
        float3 xformOp:translate = (0.004879, 0.004879, 0.01900)
        float3 xformOp:rotateXYZ = (0, 0, -45.0)
        float3 xformOp:scale = (0.002090, 0.003500, 0.02200)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ", "xformOp:scale"]
    }
    def Cube "bore_col_3" (
        prepend apiSchemas = ["PhysicsCollisionAPI"]
    )
    {
        double size = 1
        uniform token purpose = "guide"
        float3 xformOp:translate = (0.006375, 0.002641, 0.01900)
        float3 xformOp:rotateXYZ = (0, 0, -67.5)
        float3 xformOp:scale = (0.002090, 0.003500, 0.02200)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ", "xformOp:scale"]
    }
    def Cube "bore_col_4" (
        prepend apiSchemas = ["PhysicsCollisionAPI"]
    )
    {
        double size = 1
        uniform token purpose = "guide"
        float3 xformOp:translate = (0.006900, 0.000000, 0.01900)
        float3 xformOp:rotateXYZ = (0, 0, -90.0)
        float3 xformOp:scale = (0.002090, 0.003500, 0.02200)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ", "xformOp:scale"]
    }
    def Cube "bore_col_5" (
        prepend apiSchemas = ["PhysicsCollisionAPI"]
    )
    {
        double size = 1
        uniform token purpose = "guide"
        float3 xformOp:translate = (0.006375, -0.002641, 0.01900)
        float3 xformOp:rotateXYZ = (0, 0, -112.5)
        float3 xformOp:scale = (0.002090, 0.003500, 0.02200)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ", "xformOp:scale"]
    }
    def Cube "bore_col_6" (
        prepend apiSchemas = ["PhysicsCollisionAPI"]
    )
    {
        double size = 1
        uniform token purpose = "guide"
        float3 xformOp:translate = (0.004879, -0.004879, 0.01900)
        float3 xformOp:rotateXYZ = (0, 0, -135.0)
        float3 xformOp:scale = (0.002090, 0.003500, 0.02200)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ", "xformOp:scale"]
    }
    def Cube "bore_col_7" (
        prepend apiSchemas = ["PhysicsCollisionAPI"]
    )
    {
        double size = 1
        uniform token purpose = "guide"
        float3 xformOp:translate = (0.002641, -0.006375, 0.01900)
        float3 xformOp:rotateXYZ = (0, 0, -157.5)
        float3 xformOp:scale = (0.002090, 0.003500, 0.02200)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ", "xformOp:scale"]
    }
    def Cube "bore_col_8" (
        prepend apiSchemas = ["PhysicsCollisionAPI"]
    )
    {
        double size = 1
        uniform token purpose = "guide"
        float3 xformOp:translate = (0.000000, -0.006900, 0.01900)
        float3 xformOp:rotateXYZ = (0, 0, -180.0)
        float3 xformOp:scale = (0.002090, 0.003500, 0.02200)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ", "xformOp:scale"]
    }
    def Cube "bore_col_9" (
        prepend apiSchemas = ["PhysicsCollisionAPI"]
    )
    {
        double size = 1
        uniform token purpose = "guide"
        float3 xformOp:translate = (-0.002641, -0.006375, 0.01900)
        float3 xformOp:rotateXYZ = (0, 0, -202.5)
        float3 xformOp:scale = (0.002090, 0.003500, 0.02200)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ", "xformOp:scale"]
    }
    def Cube "bore_col_10" (
        prepend apiSchemas = ["PhysicsCollisionAPI"]
    )
    {
        double size = 1
        uniform token purpose = "guide"
        float3 xformOp:translate = (-0.004879, -0.004879, 0.01900)
        float3 xformOp:rotateXYZ = (0, 0, -225.0)
        float3 xformOp:scale = (0.002090, 0.003500, 0.02200)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ", "xformOp:scale"]
    }
    def Cube "bore_col_11" (
        prepend apiSchemas = ["PhysicsCollisionAPI"]
    )
    {
        double size = 1
        uniform token purpose = "guide"
        float3 xformOp:translate = (-0.006375, -0.002641, 0.01900)
        float3 xformOp:rotateXYZ = (0, 0, -247.5)
        float3 xformOp:scale = (0.002090, 0.003500, 0.02200)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ", "xformOp:scale"]
    }
    def Cube "bore_col_12" (
        prepend apiSchemas = ["PhysicsCollisionAPI"]
    )
    {
        double size = 1
        uniform token purpose = "guide"
        float3 xformOp:translate = (-0.006900, -0.000000, 0.01900)
        float3 xformOp:rotateXYZ = (0, 0, -270.0)
        float3 xformOp:scale = (0.002090, 0.003500, 0.02200)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ", "xformOp:scale"]
    }
    def Cube "bore_col_13" (
        prepend apiSchemas = ["PhysicsCollisionAPI"]
    )
    {
        double size = 1
        uniform token purpose = "guide"
        float3 xformOp:translate = (-0.006375, 0.002641, 0.01900)
        float3 xformOp:rotateXYZ = (0, 0, -292.5)
        float3 xformOp:scale = (0.002090, 0.003500, 0.02200)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ", "xformOp:scale"]
    }
    def Cube "bore_col_14" (
        prepend apiSchemas = ["PhysicsCollisionAPI"]
    )
    {
        double size = 1
        uniform token purpose = "guide"
        float3 xformOp:translate = (-0.004879, 0.004879, 0.01900)
        float3 xformOp:rotateXYZ = (0, 0, -315.0)
        float3 xformOp:scale = (0.002090, 0.003500, 0.02200)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ", "xformOp:scale"]
    }
    def Cube "bore_col_15" (
        prepend apiSchemas = ["PhysicsCollisionAPI"]
    )
    {
        double size = 1
        uniform token purpose = "guide"
        float3 xformOp:translate = (-0.002641, 0.006375, 0.01900)
        float3 xformOp:rotateXYZ = (0, 0, -337.5)
        float3 xformOp:scale = (0.002090, 0.003500, 0.02200)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ", "xformOp:scale"]
    }
    def Cube "body_col_0" (
        prepend apiSchemas = ["PhysicsCollisionAPI"]
    )
    {
        double size = 1
        uniform token purpose = "guide"
        float3 xformOp:translate = (0.000000, 0.017900, 0.01000)
        float3 xformOp:rotateXYZ = (0, 0, -0.0)
        float3 xformOp:scale = (0.011424, 0.006000, 0.01000)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ", "xformOp:scale"]
    }
    def Cube "body_col_1" (
        prepend apiSchemas = ["PhysicsCollisionAPI"]
    )
    {
        double size = 1
        uniform token purpose = "guide"
        float3 xformOp:translate = (0.008950, 0.015502, 0.01000)
        float3 xformOp:rotateXYZ = (0, 0, -30.0)
        float3 xformOp:scale = (0.011424, 0.006000, 0.01000)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ", "xformOp:scale"]
    }
    def Cube "body_col_2" (
        prepend apiSchemas = ["PhysicsCollisionAPI"]
    )
    {
        double size = 1
        uniform token purpose = "guide"
        float3 xformOp:translate = (0.015502, 0.008950, 0.01000)
        float3 xformOp:rotateXYZ = (0, 0, -60.0)
        float3 xformOp:scale = (0.011424, 0.006000, 0.01000)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ", "xformOp:scale"]
    }
    def Cube "body_col_3" (
        prepend apiSchemas = ["PhysicsCollisionAPI"]
    )
    {
        double size = 1
        uniform token purpose = "guide"
        float3 xformOp:translate = (0.017900, 0.000000, 0.01000)
        float3 xformOp:rotateXYZ = (0, 0, -90.0)
        float3 xformOp:scale = (0.011424, 0.006000, 0.01000)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ", "xformOp:scale"]
    }
    def Cube "body_col_4" (
        prepend apiSchemas = ["PhysicsCollisionAPI"]
    )
    {
        double size = 1
        uniform token purpose = "guide"
        float3 xformOp:translate = (0.015502, -0.008950, 0.01000)
        float3 xformOp:rotateXYZ = (0, 0, -120.0)
        float3 xformOp:scale = (0.011424, 0.006000, 0.01000)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ", "xformOp:scale"]
    }
    def Cube "body_col_5" (
        prepend apiSchemas = ["PhysicsCollisionAPI"]
    )
    {
        double size = 1
        uniform token purpose = "guide"
        float3 xformOp:translate = (0.008950, -0.015502, 0.01000)
        float3 xformOp:rotateXYZ = (0, 0, -150.0)
        float3 xformOp:scale = (0.011424, 0.006000, 0.01000)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ", "xformOp:scale"]
    }
    def Cube "body_col_6" (
        prepend apiSchemas = ["PhysicsCollisionAPI"]
    )
    {
        double size = 1
        uniform token purpose = "guide"
        float3 xformOp:translate = (0.000000, -0.017900, 0.01000)
        float3 xformOp:rotateXYZ = (0, 0, -180.0)
        float3 xformOp:scale = (0.011424, 0.006000, 0.01000)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ", "xformOp:scale"]
    }
    def Cube "body_col_7" (
        prepend apiSchemas = ["PhysicsCollisionAPI"]
    )
    {
        double size = 1
        uniform token purpose = "guide"
        float3 xformOp:translate = (-0.008950, -0.015502, 0.01000)
        float3 xformOp:rotateXYZ = (0, 0, -210.0)
        float3 xformOp:scale = (0.011424, 0.006000, 0.01000)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ", "xformOp:scale"]
    }
    def Cube "body_col_8" (
        prepend apiSchemas = ["PhysicsCollisionAPI"]
    )
    {
        double size = 1
        uniform token purpose = "guide"
        float3 xformOp:translate = (-0.015502, -0.008950, 0.01000)
        float3 xformOp:rotateXYZ = (0, 0, -240.0)
        float3 xformOp:scale = (0.011424, 0.006000, 0.01000)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ", "xformOp:scale"]
    }
    def Cube "body_col_9" (
        prepend apiSchemas = ["PhysicsCollisionAPI"]
    )
    {
        double size = 1
        uniform token purpose = "guide"
        float3 xformOp:translate = (-0.017900, -0.000000, 0.01000)
        float3 xformOp:rotateXYZ = (0, 0, -270.0)
        float3 xformOp:scale = (0.011424, 0.006000, 0.01000)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ", "xformOp:scale"]
    }
    def Cube "body_col_10" (
        prepend apiSchemas = ["PhysicsCollisionAPI"]
    )
    {
        double size = 1
        uniform token purpose = "guide"
        float3 xformOp:translate = (-0.015502, 0.008950, 0.01000)
        float3 xformOp:rotateXYZ = (0, 0, -300.0)
        float3 xformOp:scale = (0.011424, 0.006000, 0.01000)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ", "xformOp:scale"]
    }
    def Cube "body_col_11" (
        prepend apiSchemas = ["PhysicsCollisionAPI"]
    )
    {
        double size = 1
        uniform token purpose = "guide"
        float3 xformOp:translate = (-0.008950, 0.015502, 0.01000)
        float3 xformOp:rotateXYZ = (0, 0, -330.0)
        float3 xformOp:scale = (0.011424, 0.006000, 0.01000)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ", "xformOp:scale"]
    }
    def Cube "hub_col_0" (
        prepend apiSchemas = ["PhysicsCollisionAPI"]
    )
    {
        double size = 1
        uniform token purpose = "guide"
        float3 xformOp:translate = (0.000000, 0.015750, 0.02250)
        float3 xformOp:rotateXYZ = (0, 0, -0.0)
        float3 xformOp:scale = (0.014999, 0.004000, 0.01500)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ", "xformOp:scale"]
    }
    def Cube "hub_col_1" (
        prepend apiSchemas = ["PhysicsCollisionAPI"]
    )
    {
        double size = 1
        uniform token purpose = "guide"
        float3 xformOp:translate = (0.011137, 0.011137, 0.02250)
        float3 xformOp:rotateXYZ = (0, 0, -45.0)
        float3 xformOp:scale = (0.014999, 0.004000, 0.01500)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ", "xformOp:scale"]
    }
    def Cube "hub_col_2" (
        prepend apiSchemas = ["PhysicsCollisionAPI"]
    )
    {
        double size = 1
        uniform token purpose = "guide"
        float3 xformOp:translate = (0.015750, 0.000000, 0.02250)
        float3 xformOp:rotateXYZ = (0, 0, -90.0)
        float3 xformOp:scale = (0.014999, 0.004000, 0.01500)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ", "xformOp:scale"]
    }
    def Cube "hub_col_3" (
        prepend apiSchemas = ["PhysicsCollisionAPI"]
    )
    {
        double size = 1
        uniform token purpose = "guide"
        float3 xformOp:translate = (0.011137, -0.011137, 0.02250)
        float3 xformOp:rotateXYZ = (0, 0, -135.0)
        float3 xformOp:scale = (0.014999, 0.004000, 0.01500)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ", "xformOp:scale"]
    }
    def Cube "hub_col_4" (
        prepend apiSchemas = ["PhysicsCollisionAPI"]
    )
    {
        double size = 1
        uniform token purpose = "guide"
        float3 xformOp:translate = (0.000000, -0.015750, 0.02250)
        float3 xformOp:rotateXYZ = (0, 0, -180.0)
        float3 xformOp:scale = (0.014999, 0.004000, 0.01500)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ", "xformOp:scale"]
    }
    def Cube "hub_col_5" (
        prepend apiSchemas = ["PhysicsCollisionAPI"]
    )
    {
        double size = 1
        uniform token purpose = "guide"
        float3 xformOp:translate = (-0.011137, -0.011137, 0.02250)
        float3 xformOp:rotateXYZ = (0, 0, -225.0)
        float3 xformOp:scale = (0.014999, 0.004000, 0.01500)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ", "xformOp:scale"]
    }
    def Cube "hub_col_6" (
        prepend apiSchemas = ["PhysicsCollisionAPI"]
    )
    {
        double size = 1
        uniform token purpose = "guide"
        float3 xformOp:translate = (-0.015750, -0.000000, 0.02250)
        float3 xformOp:rotateXYZ = (0, 0, -270.0)
        float3 xformOp:scale = (0.014999, 0.004000, 0.01500)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ", "xformOp:scale"]
    }
    def Cube "hub_col_7" (
        prepend apiSchemas = ["PhysicsCollisionAPI"]
    )
    {
        double size = 1
        uniform token purpose = "guide"
        float3 xformOp:translate = (-0.011137, 0.011137, 0.02250)
        float3 xformOp:rotateXYZ = (0, 0, -315.0)
        float3 xformOp:scale = (0.014999, 0.004000, 0.01500)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ", "xformOp:scale"]
    }
}
EOF

cat > "$DEST/gear_base_body.usda" <<'EOF'
#usda 1.0
(
    defaultPrim = "gear_base"
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "gear_base" (
    prepend references = @./factory_gear_base.usd@</factory_gear_base_loose/factory_gear_base_loose>
)
{
}
EOF

echo "[fetch] done: $(ls "$DEST" | tr '\n' ' ')"
