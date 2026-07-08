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
    # Convex hub ring: 8 guide-purpose box colliders forming an octagon at the
    # Ø35.5 mm hub band (z 0.015..0.030). The factory SDF mesh handles bore/
    # teeth/shaft contact, but pad<->hub SDF contacts drop nondeterministically
    # at >100 parallel envs (10% hold at 512 envs, 100% at 4). Convex-vs-convex
    # pad contact is scale-proof; the ring's inner radius (13.8 mm) clears the
    # Ø10 mm bore so the shaft never touches it.
    def Cube "hub_col_0" (
        prepend apiSchemas = ["PhysicsCollisionAPI"]
    )
    {
        double size = 1
        uniform token purpose = "guide"
        float3 xformOp:translate = (0.000000, 0.015750, 0.0225)
        float3 xformOp:rotateXYZ = (0, 0, -0.0)
        float3 xformOp:scale = (0.014999, 0.004000, 0.015)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ", "xformOp:scale"]
    }
    def Cube "hub_col_1" (
        prepend apiSchemas = ["PhysicsCollisionAPI"]
    )
    {
        double size = 1
        uniform token purpose = "guide"
        float3 xformOp:translate = (0.011137, 0.011137, 0.0225)
        float3 xformOp:rotateXYZ = (0, 0, -45.0)
        float3 xformOp:scale = (0.014999, 0.004000, 0.015)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ", "xformOp:scale"]
    }
    def Cube "hub_col_2" (
        prepend apiSchemas = ["PhysicsCollisionAPI"]
    )
    {
        double size = 1
        uniform token purpose = "guide"
        float3 xformOp:translate = (0.015750, 0.000000, 0.0225)
        float3 xformOp:rotateXYZ = (0, 0, -90.0)
        float3 xformOp:scale = (0.014999, 0.004000, 0.015)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ", "xformOp:scale"]
    }
    def Cube "hub_col_3" (
        prepend apiSchemas = ["PhysicsCollisionAPI"]
    )
    {
        double size = 1
        uniform token purpose = "guide"
        float3 xformOp:translate = (0.011137, -0.011137, 0.0225)
        float3 xformOp:rotateXYZ = (0, 0, -135.0)
        float3 xformOp:scale = (0.014999, 0.004000, 0.015)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ", "xformOp:scale"]
    }
    def Cube "hub_col_4" (
        prepend apiSchemas = ["PhysicsCollisionAPI"]
    )
    {
        double size = 1
        uniform token purpose = "guide"
        float3 xformOp:translate = (0.000000, -0.015750, 0.0225)
        float3 xformOp:rotateXYZ = (0, 0, -180.0)
        float3 xformOp:scale = (0.014999, 0.004000, 0.015)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ", "xformOp:scale"]
    }
    def Cube "hub_col_5" (
        prepend apiSchemas = ["PhysicsCollisionAPI"]
    )
    {
        double size = 1
        uniform token purpose = "guide"
        float3 xformOp:translate = (-0.011137, -0.011137, 0.0225)
        float3 xformOp:rotateXYZ = (0, 0, -225.0)
        float3 xformOp:scale = (0.014999, 0.004000, 0.015)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ", "xformOp:scale"]
    }
    def Cube "hub_col_6" (
        prepend apiSchemas = ["PhysicsCollisionAPI"]
    )
    {
        double size = 1
        uniform token purpose = "guide"
        float3 xformOp:translate = (-0.015750, -0.000000, 0.0225)
        float3 xformOp:rotateXYZ = (0, 0, -270.0)
        float3 xformOp:scale = (0.014999, 0.004000, 0.015)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ", "xformOp:scale"]
    }
    def Cube "hub_col_7" (
        prepend apiSchemas = ["PhysicsCollisionAPI"]
    )
    {
        double size = 1
        uniform token purpose = "guide"
        float3 xformOp:translate = (-0.011137, 0.011137, 0.0225)
        float3 xformOp:rotateXYZ = (0, 0, -315.0)
        float3 xformOp:scale = (0.014999, 0.004000, 0.015)
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
