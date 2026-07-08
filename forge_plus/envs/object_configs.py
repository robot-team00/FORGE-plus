"""Object configuration registry.

Each ObjectConfig carries:
  - The identity the LLM can read (name, material, geometry_tags, etc.)
  - The HIDDEN breaking-force distribution (class mean + per-instance spread)

F_break is sampled once per episode from the distribution and stored in the
simulator. It is ONLY visible to the evaluator's breakage check. The policy,
encoder, budget-setter, and recovery-selector never observe it.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from forge_plus.llm.budget_setter import ObjectIdentity


@dataclass
class ObjectConfig:
    """Full object specification including hidden fragility model."""

    # --- LLM-visible identity ---
    identity: ObjectIdentity

    # --- Evaluator-only fragility model (NEVER exposed to the agent) ---
    f_break_mean_n: float         # class mean breaking force (N)
    f_break_std_n: float          # per-instance std deviation (N)
    f_break_min_n: float          # hard minimum (physical plausibility)

    # --- Task compatibility ---
    compatible_tasks: list[str] = field(default_factory=list)

    # --- Gripper-specific properties ---
    grasp_width_mm: float = 30.0  # nominal gripper opening for grasping

    def sample_f_break(self, rng: random.Random | None = None) -> float:
        """Draw one instance's hidden breaking force from the class distribution."""
        _rng = rng or random
        value = _rng.gauss(self.f_break_mean_n, self.f_break_std_n)
        return max(value, self.f_break_min_n)


# ---------------------------------------------------------------------------
# Object registry — Task 1 (single insertion)
# ---------------------------------------------------------------------------

ABS_ROUND_CONNECTOR = ObjectConfig(
    identity=ObjectIdentity(
        name="Ø8mm ABS round connector",
        material="ABS",
        **{"class": "round_connector"},
        nominal_mass_g=6.2,
        geometry_tags=["thin_wall", "press_fit"],
    ),
    f_break_mean_n=38.0,
    f_break_std_n=5.0,
    f_break_min_n=20.0,
    compatible_tasks=["task1_single_insertion"],
    grasp_width_mm=12.0,
)

STEEL_PEG = ObjectConfig(
    identity=ObjectIdentity(
        name="Ø10mm steel precision peg",
        material="steel",
        **{"class": "precision_peg"},
        nominal_mass_g=42.0,
        geometry_tags=["solid", "clearance_fit"],
    ),
    f_break_mean_n=230.0,
    f_break_std_n=20.0,
    f_break_min_n=180.0,
    compatible_tasks=["task1_single_insertion"],
    grasp_width_mm=14.0,
)

# ---------------------------------------------------------------------------
# Object registry — Task 2 (multi-step assembly / planetary gearbox)
# ---------------------------------------------------------------------------

RESIN_PLANET_GEAR = ObjectConfig(
    identity=ObjectIdentity(
        name="Ø22mm resin planet gear (12T)",
        material="photopolymer resin",
        **{"class": "planet_gear"},
        nominal_mass_g=8.5,
        geometry_tags=["tooth_mesh", "press_fit_bore", "brittle"],
    ),
    f_break_mean_n=48.0,
    f_break_std_n=8.0,
    f_break_min_n=28.0,
    compatible_tasks=["task2_multi_step_assembly"],
    grasp_width_mm=24.0,
)

METAL_PLANET_GEAR = ObjectConfig(
    identity=ObjectIdentity(
        name="Ø22mm aluminium planet gear (12T)",
        material="aluminium",
        **{"class": "planet_gear"},
        nominal_mass_g=18.0,
        geometry_tags=["tooth_mesh", "press_fit_bore"],
    ),
    f_break_mean_n=210.0,
    f_break_std_n=18.0,
    f_break_min_n=165.0,
    compatible_tasks=["task2_multi_step_assembly"],
    grasp_width_mm=24.0,
)

# ---------------------------------------------------------------------------
# Object registry — Task 3 (fragile place / stack)
# ---------------------------------------------------------------------------

GLASS_BOWL = ObjectConfig(
    identity=ObjectIdentity(
        name="Ø120mm borosilicate glass bowl",
        material="borosilicate glass",
        **{"class": "bowl"},
        nominal_mass_g=210.0,
        geometry_tags=["brittle", "curved_rim", "no_hole"],
    ),
    f_break_mean_n=22.0,
    f_break_std_n=4.0,
    f_break_min_n=12.0,
    compatible_tasks=["task3_fragile_place"],
    grasp_width_mm=80.0,
)

CERAMIC_PLATE = ObjectConfig(
    identity=ObjectIdentity(
        name="Ø200mm ceramic plate",
        material="stoneware ceramic",
        **{"class": "plate"},
        nominal_mass_g=380.0,
        geometry_tags=["brittle", "flat_seating", "no_hole"],
    ),
    f_break_mean_n=26.0,
    f_break_std_n=5.0,
    f_break_min_n=14.0,
    compatible_tasks=["task3_fragile_place"],
    grasp_width_mm=120.0,
)

METAL_PLATE = ObjectConfig(
    identity=ObjectIdentity(
        name="200×150mm aluminium tray",
        material="aluminium",
        **{"class": "tray"},
        nominal_mass_g=320.0,
        geometry_tags=["flat_seating", "robust", "no_hole"],
    ),
    f_break_mean_n=180.0,
    f_break_std_n=25.0,
    f_break_min_n=120.0,
    compatible_tasks=["task3_fragile_place"],
    grasp_width_mm=120.0,
)

STURDY_MUG = ObjectConfig(
    identity=ObjectIdentity(
        name="Ø90mm stoneware mug (thick-walled)",
        material="stoneware",
        **{"class": "mug"},
        nominal_mass_g=290.0,
        geometry_tags=["thick_wall", "robust_for_ceramics"],
    ),
    f_break_mean_n=160.0,
    f_break_std_n=20.0,
    f_break_min_n=110.0,
    compatible_tasks=["task3_fragile_place"],
    grasp_width_mm=95.0,
)

# ---------------------------------------------------------------------------
# Registry and helpers
# ---------------------------------------------------------------------------

OBJECT_REGISTRY: dict[str, ObjectConfig] = {
    "abs_round_connector": ABS_ROUND_CONNECTOR,
    "steel_peg": STEEL_PEG,
    "resin_planet_gear": RESIN_PLANET_GEAR,
    "metal_planet_gear": METAL_PLANET_GEAR,
    "glass_bowl": GLASS_BOWL,
    "ceramic_plate": CERAMIC_PLATE,
    "metal_plate": METAL_PLATE,
    "sturdy_mug": STURDY_MUG,
}


def sample_f_break(obj_key: str, rng: random.Random | None = None) -> float:
    """Sample a hidden F_break for one episode instance of the given object."""
    cfg = OBJECT_REGISTRY[obj_key]
    return cfg.sample_f_break(rng)


def get_object_identity(obj_key: str) -> ObjectIdentity:
    """Return the LLM-visible identity for an object (no F_break information)."""
    return OBJECT_REGISTRY[obj_key].identity
