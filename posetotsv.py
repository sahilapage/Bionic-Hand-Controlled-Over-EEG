import time
import mujoco
import numpy as np
import mujoco.viewer
from scipy.interpolate import interp1d
from scipy.spatial import KDTree

# ────────────────────────────────────────
MODEL_PATH = "mjcf/scene.xml"

model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data  = mujoco.MjData(model)

tip_names = ["tip1", "tip2", "tip3", "tip4"]
tip_ids   = [model.site(name).id for name in tip_names]

palm_body_id = model.body("r_wrist_interface").id

print_requested = False


# ────────────────────────────────────────
def compute_robot_tsv():
    palm_pos = data.xpos[palm_body_id]
    palm_rot = data.xmat[palm_body_id].reshape(3, 3)

    tsvs = []
    tip_world_positions = []

    for tid in tip_ids:
        tip_pos = data.site_xpos[tid]
        tip_world_positions.append(np.round(tip_pos, 4))

        v_world = tip_pos - palm_pos
        v_local = palm_rot.T @ v_world
        tsvs.append(np.round(v_local, 4))

    return np.array(tsvs), np.array(tip_world_positions)


# ════════════════════════════════════════
# FINGERS 1 / 2 / 3  (shared calibration)
# ════════════════════════════════════════
# ════════════════════════════════════════
# FINGER 1 — full lookup table + KD-tree IK
# ════════════════════════════════════════
#
# Three measured sweeps (ctrl[0], ctrl[1]):
#   SWEEP A — ctrl_a = -ctrl_b  (flexion, tip_x varies widely, tip_y ~0.024)
#   SWEEP B — both ctrl positive (abduction+, tip_y increases to 0.059)
#   SWEEP C — both ctrl negative (abduction−, tip_y decreases to -0.016)
#
# y is the disambiguating axis — scale it 3× in KD-tree distance.
#
# Columns: ctrl_a, ctrl_b, tip_x, tip_y, tip_z

FINGER1_TABLE = np.array([
    # ── SWEEP A: ctrl_a = -ctrl_b (flexion, tip_x sweeps) ─────────────
    (-1.5710,  1.5710, -0.0667,  0.0236,  0.1670),
    (-1.4453,  1.4610, -0.0664,  0.0237,  0.1673),
    (-1.3196,  1.3196, -0.0632,  0.0237,  0.1703),
    (-1.1782,  1.1782, -0.0558,  0.0238,  0.1761),
    (-1.0369,  1.0369, -0.0448,  0.0238,  0.1823),
    (-0.8955,  0.8955, -0.0324,  0.0240,  0.1867),
    (-0.7541,  0.7541, -0.0169,  0.0239,  0.1891),
    (-0.6127,  0.6127, -0.0027,  0.0241,  0.1886),
    (-0.4399,  0.4399,  0.0150,  0.0241,  0.1841),
    (-0.2514,  0.2514,  0.0319,  0.0240,  0.1750),
    ( 0.0000,  0.0000,  0.0483,  0.0240,  0.1590),  # neutral
    ( 0.1885, -0.1100,  0.0551,  0.0250,  0.1479),
    ( 0.2671, -0.2514,  0.0584,  0.0241,  0.1404),
    ( 0.6755, -0.7069,  0.0626,  0.0238,  0.1123),
    ( 0.9269, -0.9269,  0.0610,  0.0238,  0.1004),
    ( 1.0997, -1.0997,  0.0592,  0.0238,  0.0937),
    ( 1.2725, -1.2568,  0.0575,  0.0238,  0.0888),
    ( 1.3825, -1.3511,  0.0566,  0.0238,  0.0867),
    ( 1.5710, -1.5710,  0.0556,  0.0238,  0.0844),

    # ── SWEEP B: both ctrl positive (abduction+, tip_y increases) ──────
    ( 0.1257,  0.1257,  0.0487,  0.0284,  0.1583),
    ( 0.2828,  0.2671,  0.0493,  0.0333,  0.1570),
    ( 0.5184,  0.4870,  0.0497,  0.0404,  0.1547),
    ( 0.6755,  0.6598,  0.0494,  0.0452,  0.1534),
    ( 0.8483,  0.8326,  0.0492,  0.0497,  0.1517),
    ( 1.0526,  1.0369,  0.0490,  0.0540,  0.1497),
    ( 1.2254,  1.2254,  0.0485,  0.0570,  0.1485),
    ( 1.3825,  1.3825,  0.0482,  0.0587,  0.1477),
    ( 1.5710,  1.5710,  0.0482,  0.0593,  0.1473),

    # ── SWEEP C: both ctrl negative (abduction−, tip_y decreases) ──────
    (-0.2042, -0.1728,  0.0470,  0.0173,  0.1602),
    (-0.4242, -0.3770,  0.0460,  0.0096,  0.1600),
    (-0.6441, -0.6127,  0.0454,  0.0021,  0.1583),
    (-0.9112, -0.8483,  0.0435, -0.0058,  0.1570),
    (-1.1468, -1.1468,  0.0433, -0.0114,  0.1538),
    (-1.3982, -1.3668,  0.0419, -0.0151,  0.1528),
    (-1.5710, -1.5710,  0.0419, -0.0160,  0.1521),
])

_f1_ctrl = FINGER1_TABLE[:, 0:2]
_f1_tips  = FINGER1_TABLE[:, 2:5]
# y separates the 3 sweeps — boost it 3× in KD-tree distance
_F1_SCALE   = np.array([1.0, 3.0, 1.0])
_f1_kdtree  = KDTree(_f1_tips * _F1_SCALE)


# ════════════════════════════════════════
# FINGER 2 — full lookup table + KD-tree IK
# ════════════════════════════════════════
#
# Sweep measured: ctrl_a > 0, ctrl_b < 0 (flexion, tip_z decreases).
# tip_y barely moves (~-0.010 ± 0.007) — z is the main varying axis.
# Add abduction sweeps here when measured (same pattern as finger 1).
#
# Columns: ctrl_a, ctrl_b, tip_x, tip_y, tip_z

FINGER2_TABLE = np.array([
    # ── SWEEP A: ctrl_a positive, ctrl_b negative (flexion+, tip_z drops) ─
    ( 0.0000,  0.0000,  0.0484, -0.0147,  0.1525),  # neutral
    ( 0.1100, -0.1100,  0.0535, -0.0139,  0.1446),
    ( 0.2042, -0.2042,  0.0568, -0.0132,  0.1379),
    ( 0.3142, -0.3142,  0.0597, -0.0125,  0.1301),
    ( 0.4399, -0.4242,  0.0616, -0.0115,  0.1220),
    ( 0.5813, -0.5498,  0.0626, -0.0106,  0.1134),
    ( 0.6598, -0.6127,  0.0627, -0.0102,  0.1093),
    ( 0.7541, -0.7069,  0.0624, -0.0097,  0.1039),
    ( 0.8640, -0.8483,  0.0616, -0.0091,  0.0975),
    ( 0.9426, -0.8483,  0.0613, -0.0090,  0.0957),
    ( 0.9426, -0.9583,  0.0607, -0.0087,  0.0933),
    ( 1.0211, -1.0840,  0.0597, -0.0082,  0.0893),
    ( 1.1940, -1.2097,  0.0581, -0.0077,  0.0844),
    ( 1.2882, -1.3039,  0.0572, -0.0075,  0.0821),
    ( 1.4610, -1.4610,  0.0560, -0.0072,  0.0793),
    ( 1.5710, -1.5710,  0.0556, -0.0071,  0.0783),

    # ── SWEEP B: both ctrl positive (abduction+, tip_y increases) ─────────
    ( 0.1257,  0.1257,  0.0486, -0.0103,  0.1525),
    ( 0.2199,  0.2199,  0.0488, -0.0070,  0.1522),
    ( 0.2671,  0.2828,  0.0486, -0.0052,  0.1525),
    ( 0.3613,  0.4242,  0.0473, -0.0009,  0.1538),
    ( 0.5341,  0.5341,  0.0490,  0.0032,  0.1508),
    ( 0.6441,  0.6284,  0.0494,  0.0061,  0.1495),
    ( 0.6755,  0.7541,  0.0473,  0.0092,  0.1516),
    ( 0.8640,  0.8640,  0.0490,  0.0121,  0.1481),
    ( 0.9112,  0.9269,  0.0486,  0.0136,  0.1480),
    ( 1.0211,  1.0054,  0.0490,  0.0154,  0.1467),
    ( 1.0997,  1.0997,  0.0486,  0.0172,  0.1464),
    ( 1.2097,  1.1940,  0.0487,  0.0187,  0.1456),
    ( 1.3039,  1.3196,  0.0482,  0.0204,  0.1454),
    ( 1.3825,  1.3196,  0.0487,  0.0203,  0.1446),
    ( 1.3825,  1.3982,  0.0482,  0.0210,  0.1450),
    ( 1.4453,  1.4610,  0.0482,  0.0213,  0.1448),
    ( 1.5710,  1.5710,  0.0482,  0.0216,  0.1446),

    # ── SWEEP C: both ctrl negative (abduction−, tip_y decreases) ─────────
    (-0.1257, -0.1257,  0.0482, -0.0191,  0.1521),
    (-0.2671, -0.1885,  0.0455, -0.0234,  0.1546),
    (-0.3456, -0.2828,  0.0456, -0.0263,  0.1536),
    (-0.3927, -0.3456,  0.0459, -0.0281,  0.1526),
    (-0.4870, -0.4242,  0.0452, -0.0313,  0.1523),
    (-0.5498, -0.5341,  0.0464, -0.0335,  0.1498),
    (-0.5813, -0.6127,  0.0470, -0.0346,  0.1483),
    (-0.6598, -0.6441,  0.0455, -0.0370,  0.1492),
    (-0.7698, -0.7227,  0.0444, -0.0403,  0.1488),
    (-0.8169, -0.8012,  0.0449, -0.0416,  0.1474),
    (-0.8640, -0.8955,  0.0455, -0.0428,  0.1456),
    (-0.9583, -0.9269,  0.0437, -0.0453,  0.1463),
    (-1.0526, -1.0369,  0.0436, -0.0475,  0.1448),
    (-1.1311, -1.0840,  0.0426, -0.0492,  0.1448),
    (-1.1940, -1.1468,  0.0424, -0.0503,  0.1441),
    (-1.2725, -1.2097,  0.0419, -0.0516,  0.1436),
    (-1.3668, -1.3039,  0.0417, -0.0528,  0.1429),
    (-1.4139, -1.4139,  0.0421, -0.0531,  0.1420),
    (-1.5710, -1.5710,  0.0418, -0.0539,  0.1416),
])

_f2_ctrl   = FINGER2_TABLE[:, 0:2]
_f2_tips   = FINGER2_TABLE[:, 2:5]
# y separates the 3 sweeps (same pattern as finger 1) — boost it 4×
_F2_SCALE  = np.array([1.0, 4.0, 1.0])
_f2_kdtree = KDTree(_f2_tips * _F2_SCALE)


# ════════════════════════════════════════
# FINGER 3 — full lookup table + KD-tree IK
# ════════════════════════════════════════
#
# Three measured sweeps (ctrl[4], ctrl[5]):
#   SWEEP A — ctrl_a > 0, ctrl_b < 0  (flexion+, tip_z drops 0.134→0.068)
#   SWEEP B — both ctrl positive       (abduction+, tip_y rises -0.054→-0.017)
#   SWEEP C — both ctrl negative       (abduction−, tip_y drops -0.058→-0.092)
#
# y separates B vs C cleanly; A sits in a distinct z-region.
# Scale y 4× (same as finger 2) to disambiguate sweeps.
#
# Columns: ctrl_a, ctrl_b, tip_x, tip_y, tip_z

FINGER3_TABLE = np.array([
    # ── SWEEP A: ctrl_a positive, ctrl_b negative (flexion+, tip_z drops) ─
    ( 0.0628, -0.1100,  0.0526, -0.0535,  0.1339),
    ( 0.1257, -0.1728,  0.0549, -0.0524,  0.1298),
    ( 0.2042, -0.2828,  0.0579, -0.0512,  0.1231),
    ( 0.3142, -0.3142,  0.0596, -0.0491,  0.1186),
    ( 0.3613, -0.3613,  0.0605, -0.0483,  0.1153),
    ( 0.4556, -0.4399,  0.0618, -0.0469,  0.1095),
    ( 0.5499, -0.5184,  0.0624, -0.0455,  0.1040),
    ( 0.6441, -0.6127,  0.0627, -0.0442,  0.0983),
    ( 0.7227, -0.7069,  0.0625, -0.0432,  0.0935),
    ( 0.7698, -0.8012,  0.0621, -0.0424,  0.0899),
    ( 0.8640, -0.8326,  0.0617, -0.0416,  0.0869),
    ( 0.9269, -0.8640,  0.0613, -0.0411,  0.0848),
    ( 1.0054, -0.9269,  0.0606, -0.0404,  0.0817),
    ( 1.1468, -0.9426,  0.0599, -0.0399,  0.0789),
    ( 1.3039, -1.0054,  0.0588, -0.0393,  0.0755),
    ( 1.4139, -1.1311,  0.0576, -0.0385,  0.0723),
    ( 1.4453, -1.2882,  0.0567, -0.0378,  0.0702),
    ( 1.5710, -1.5710,  0.0556, -0.0370,  0.0677),

    # ── SWEEP B: both ctrl positive (abduction+, tip_y rises) ─────────────
    ( 0.0157,  0.0314,  0.0481, -0.0535,  0.1407),
    ( 0.0628,  0.0786,  0.0483, -0.0519,  0.1408),
    ( 0.1885,  0.1728,  0.0492, -0.0478,  0.1403),
    ( 0.2514,  0.2671,  0.0485, -0.0453,  0.1415),
    ( 0.3456,  0.3613,  0.0486, -0.0421,  0.1415),
    ( 0.4556,  0.4399,  0.0495, -0.0390,  0.1403),
    ( 0.5341,  0.5184,  0.0495, -0.0366,  0.1402),
    ( 0.6127,  0.6284,  0.0487, -0.0335,  0.1409),
    ( 0.6755,  0.6755,  0.0492, -0.0321,  0.1401),
    ( 0.8169,  0.8012,  0.0492, -0.0284,  0.1394),
    ( 0.9269,  0.8640,  0.0501, -0.0267,  0.1376),
    ( 0.9897,  0.9269,  0.0499, -0.0252,  0.1375),
    ( 1.0526,  1.0054,  0.0495, -0.0235,  0.1376),
    ( 1.0840,  1.1311,  0.0479, -0.0213,  0.1391),
    ( 1.1940,  1.1940,  0.0486, -0.0202,  0.1378),
    ( 1.2725,  1.2882,  0.0482, -0.0189,  0.1378),
    ( 1.3196,  1.3196,  0.0484, -0.0186,  0.1374),
    ( 1.3982,  1.3196,  0.0488, -0.0185,  0.1368),
    ( 1.3982,  1.4139,  0.0482, -0.0177,  0.1374),
    ( 1.5710,  1.5710,  0.0482, -0.0172,  0.1371),

    # ── SWEEP C: both ctrl negative (abduction−, tip_y drops) ─────────────
    (-0.0943, -0.1257,  0.0490, -0.0577,  0.1382),
    (-0.1885, -0.1728,  0.0474, -0.0606,  0.1396),
    (-0.2828, -0.2514,  0.0467, -0.0638,  0.1392),
    (-0.3456, -0.3456,  0.0473, -0.0660,  0.1373),
    (-0.3927, -0.4399,  0.0482, -0.0676,  0.1351),
    (-0.5184, -0.5341,  0.0470, -0.0716,  0.1347),
    (-0.6127, -0.6284,  0.0466, -0.0744,  0.1335),
    (-0.6598, -0.6755,  0.0463, -0.0757,  0.1330),
    (-0.7227, -0.7384,  0.0460, -0.0775,  0.1321),
    (-0.7541, -0.8012,  0.0464, -0.0782,  0.1308),
    (-0.8955, -0.8955,  0.0447, -0.0822,  0.1301),
    (-0.9426, -0.9426,  0.0444, -0.0832,  0.1295),
    (-1.0526, -1.0526,  0.0438, -0.0856,  0.1280),
    (-1.0997, -1.1311,  0.0440, -0.0864,  0.1269),
    (-1.1940, -1.1940,  0.0430, -0.0882,  0.1265),
    (-1.2725, -1.2725,  0.0426, -0.0894,  0.1257),
    (-1.3039, -1.3825,  0.0432, -0.0895,  0.1246),
    (-1.4453, -1.4767,  0.0421, -0.0911,  0.1244),
    (-1.5710, -1.5710,  0.0419, -0.0916,  0.1242),
])

_f3_ctrl   = FINGER3_TABLE[:, 0:2]
_f3_tips   = FINGER3_TABLE[:, 2:5]
# y cleanly separates B vs C; boost 4× (same pattern as finger 2)
_F3_SCALE  = np.array([1.0, 4.0, 1.0])
_f3_kdtree = KDTree(_f3_tips * _F3_SCALE)


# ════════════════════════════════════════
# THUMB — full lookup table + KD-tree IK
# ════════════════════════════════════════
#
# Every measured sample: (ctrl_a, ctrl_b, tip_x, tip_y, tip_z)
# Three sweep types recorded:
#   SWEEP 1 — ctrl_a increases, ctrl_b decreases  (flexion+ / opposition)
#   SWEEP 2 — both ctrl increase together          (abduction+)
#   SWEEP 3 — both ctrl decrease together          (abduction−)
#
# IK strategy:
#   Given target (tip_x, tip_y, tip_z), find the K nearest measured
#   tip positions in the table (weighted by z since z disambiguates
#   overlapping regions), then do inverse-distance-weighted (IDW)
#   interpolation of ctrl_a and ctrl_b.

# fmt: off
# Columns: ctrl_a, ctrl_b, tip_x, tip_y, tip_z
THUMB_TABLE = np.array([

    # ── SWEEP 1a: ctrl_a negative, ctrl_b positive (flex-back) ────────
    (-1.5710,  1.5710,  0.0487, 0.0328, -0.0317),
    (-1.4139,  1.3982,  0.0488, 0.0328, -0.0312),
    (-1.3511,  1.3196,  0.0490, 0.0329, -0.0304),
    (-1.1782,  1.1782,  0.0499, 0.0332, -0.0271),
    (-1.0054,  1.0211,  0.0514, 0.0336, -0.0222),
    (-0.9112,  0.9112,  0.0524, 0.0340, -0.0186),
    (-0.7384,  0.6598,  0.0540, 0.0352, -0.0102),
    (-0.5813,  0.5341,  0.0548, 0.0353, -0.0042),
    (-0.4556,  0.4242,  0.0552, 0.0353,  0.0009),
    (-0.2042,  0.2042,  0.0546, 0.0348,  0.0109),
    ( 0.0000,  0.0000,  0.0529, 0.0342,  0.0191),  # neutral

    # ── SWEEP 1b: ctrl_a positive, ctrl_b negative (opposition) ───────
    ( 0.1100, -0.1100,  0.0515, 0.0337,  0.0233),
    ( 0.2671, -0.2828,  0.0490, 0.0329,  0.0288),
    ( 0.4399, -0.4556,  0.0461, 0.0318,  0.0337),
    ( 0.6598, -0.6755,  0.0421, 0.0304,  0.0388),
    ( 0.8326, -0.8326,  0.0392, 0.0293,  0.0417),
    ( 0.9269, -0.9269,  0.0376, 0.0287,  0.0430),
    ( 0.9897, -1.0054,  0.0364, 0.0283,  0.0439),
    ( 1.1468, -1.1468,  0.0343, 0.0275,  0.0453),
    ( 1.1468, -1.2568,  0.0336, 0.0274,  0.0456),
    ( 1.3511, -1.3511,  0.0322, 0.0267,  0.0464),
    ( 1.4610, -1.4453,  0.0315, 0.0265,  0.0467),

    # ── SWEEP 2: both ctrl positive together (abduction+) ─────────────
    ( 0.0000,  0.0000,  0.0811, 0.0443,  0.0896),  # neutral
    ( 0.1100,  0.1100,  0.0821, 0.0406,  0.0899),
    ( 0.2514,  0.2828,  0.0841, 0.0355,  0.0893),
    ( 0.4242,  0.4242,  0.0837, 0.0301,  0.0902),
    ( 0.5499,  0.5499,  0.0839, 0.0261,  0.0902),
    ( 0.6598,  0.6755,  0.0843, 0.0225,  0.0899),
    ( 0.8012,  0.8169,  0.0841, 0.0185,  0.0898),
    ( 0.9426,  0.9426,  0.0834, 0.0152,  0.0900),
    ( 1.0526,  1.0997,  0.0840, 0.0121,  0.0892),
    ( 1.2097,  1.1940,  0.0826, 0.0102,  0.0899),
    ( 1.3668,  1.3668,  0.0825, 0.0081,  0.0895),
    ( 1.4610,  1.4610,  0.0824, 0.0074,  0.0894),
    ( 1.5710,  1.5710,  0.0823, 0.0072,  0.0894),

    # ── SWEEP 3: both ctrl negative together (abduction−) ─────────────
    (-0.1571, -0.1728,  0.0790, 0.0497,  0.0895),
    (-0.2985, -0.2985,  0.0776, 0.0542,  0.0888),
    (-0.4713, -0.4713,  0.0749, 0.0595,  0.0882),
    (-0.6598, -0.6284,  0.0728, 0.0652,  0.0866),
    (-0.8483, -0.8326,  0.0691, 0.0699,  0.0860),
    (-1.0526, -1.0054,  0.0664, 0.0745,  0.0843),
    (-1.2254, -1.2097,  0.0635, 0.0771,  0.0839),
    (-1.3668, -1.3668,  0.0620, 0.0787,  0.0834),
    (-1.4924, -1.5082,  0.0613, 0.0795,  0.0832),
    (-1.5710, -1.5710,  0.0611, 0.0797,  0.0831),  # endpoint (was missing)

    # ── SWEEP 4: ctrl_a negative, ctrl_b positive, large (flex+) ──────
    (-0.2042,  0.2042,  0.0937, 0.0489,  0.0768),
    (-0.3613,  0.3142,  0.1004, 0.0524,  0.0657),
    (-0.5498,  0.5813,  0.1085, 0.0535,  0.0439),
    (-0.7069,  0.7069,  0.1097, 0.0548,  0.0287),
    (-0.8483,  0.8326,  0.1084, 0.0546,  0.0145),
    (-0.9112,  0.9426,  0.1068, 0.0532,  0.0062),
    (-1.0526,  1.0526,  0.1022, 0.0522, -0.0055),
    (-1.1782,  1.1782,  0.0972, 0.0504, -0.0147),
    (-1.3511,  1.2882,  0.0917, 0.0488, -0.0220),
    (-1.4453,  1.3982,  0.0894, 0.0478, -0.0247),
    (-1.5710,  1.5710,  0.0889, 0.0476, -0.0252),  # endpoint (was missing)

    # ── SWEEP 5: opposition extreme (ctrl_a >> 0, ctrl_b << 0) ────────
    # This region has high z (~0.097) and very low x/y — distinct from sweep 1b
    ( 1.5710, -1.5710,  0.0111, 0.0190,  0.0968),  # measured endpoint
])
# fmt: on

# ── Separate ctrl and tip columns ─────────────────────────────────────
_thumb_ctrl = THUMB_TABLE[:, 0:2]   # (N, 2): ctrl_a, ctrl_b
_thumb_tips = THUMB_TABLE[:, 2:5]   # (N, 3): tip_x, tip_y, tip_z

# ── Scale factors: z carries disambiguation weight, x/y carry pose ────
# z range ~0.12, x range ~0.08, y range ~0.06
# We upscale z so it contributes proportionally in nearest-neighbour search
_THUMB_SCALE = np.array([1.0, 1.0, 2.0])   # z upweighted — separates overlapping sweeps

_thumb_tips_scaled = _thumb_tips * _THUMB_SCALE
_thumb_kdtree = KDTree(_thumb_tips_scaled)


def ik_thumb(tsv_xyz, k_neighbors: int = 4):
    """IK for thumb — KD-tree lookup + inverse-distance-weighted interpolation.

    Finds the K nearest measured tip positions to the target TSV (in
    scaled 3-D space so z disambiguates overlapping sweep regions), then
    blends the corresponding ctrl values by inverse distance weighting.

    Parameters
    ----------
    tsv_xyz : array-like (3,)
        Target tip position in palm-local frame (x, y, z).
    k_neighbors : int
        Number of nearest neighbours to blend (default 4).
    """
    q = np.array(tsv_xyz) * _THUMB_SCALE
    dists, idxs = _thumb_kdtree.query(q, k=min(k_neighbors, len(_thumb_tips)))

    # Exact hit
    if dists[0] < 1e-9:
        ca, cb = _thumb_ctrl[idxs[0]]
        return float(np.clip(ca, -1.571, 1.571)), float(np.clip(cb, -1.571, 1.571))

    # Inverse-distance weighting
    weights = 1.0 / dists
    weights /= weights.sum()

    ca = float(np.dot(weights, _thumb_ctrl[idxs, 0]))
    cb = float(np.dot(weights, _thumb_ctrl[idxs, 1]))

    return float(np.clip(ca, -1.571, 1.571)), float(np.clip(cb, -1.571, 1.571))


# ════════════════════════════════════════
# Finger → (ctrl_a_idx, ctrl_b_idx, tip_index)
# ════════════════════════════════════════
FINGER_CTRL = {
    1: (0, 1, 0),
    2: (2, 3, 1),
    3: (4, 5, 2),
    4: (6, 7, 3),   # thumb
}


# ════════════════════════════════════════
# IK routines
# ════════════════════════════════════════
def ik_finger(tsv_xyz, finger_id: int = 1):
    """IK for fingers 1/2/3 — KD-tree lookup + IDW interpolation.

    Each finger has its own calibration table and scale factors.
    Fingers without dedicated data fall back to finger 1's table.
    """
    if finger_id == 2:
        table_ctrl, table_tips, scale, tree = _f2_ctrl, _f2_tips, _F2_SCALE, _f2_kdtree
    elif finger_id == 3:
        table_ctrl, table_tips, scale, tree = _f3_ctrl, _f3_tips, _F3_SCALE, _f3_kdtree
    else:
        # finger 1
        table_ctrl, table_tips, scale, tree = _f1_ctrl, _f1_tips, _F1_SCALE, _f1_kdtree

    q = np.array(tsv_xyz) * scale
    dists, idxs = tree.query(q, k=min(4, len(table_tips)))
    if dists[0] < 1e-9:
        ca, cb = table_ctrl[idxs[0]]
        return float(np.clip(ca, -1.571, 1.571)), float(np.clip(cb, -1.571, 1.571))
    weights = 1.0 / dists
    weights /= weights.sum()
    ctrl_a = float(np.clip(np.dot(weights, table_ctrl[idxs, 0]), -1.571, 1.571))
    ctrl_b = float(np.clip(np.dot(weights, table_ctrl[idxs, 1]), -1.571, 1.571))
    return ctrl_a, ctrl_b


def ik_dispatch(finger_id, tsv_xyz):
    """Route to correct IK based on finger."""
    if finger_id == 4:
        return ik_thumb(tsv_xyz)
    else:
        return ik_finger(tsv_xyz, finger_id)


# ════════════════════════════════════════
# Helpers
# ════════════════════════════════════════
def print_state():
    tsv, _ = compute_robot_tsv()

    print("\n" + "="*60)
    print("MOTOR CTRL:")
    for f, (a, b, _) in FINGER_CTRL.items():
        label = "thumb" if f == 4 else f"finger{f}"
        print(f"  {label}: ctrl[{a}]={data.ctrl[a]:+.4f}  ctrl[{b}]={data.ctrl[b]:+.4f}"
              f"   α={(data.ctrl[a]+data.ctrl[b])/2:+.4f}"
              f"   β={(data.ctrl[a]-data.ctrl[b])/2:+.4f}")

    print("\nFINGERTIP TSV (local wrt palm):")
    for i, (name, t) in enumerate(zip(tip_names, tsv)):
        label = "thumb" if i == 3 else name
        print(f"  {label}: x={t[0]:+.4f}  y={t[1]:+.4f}  z={t[2]:+.4f}")
    print("="*60 + "\n")


def run_ik_for_finger(f):
    """Shared IK entry: ask TSV, apply, report."""
    label = "thumb" if f == 4 else f"finger{f}"
    print(f"  Enter target TSV for {label} (palm-local frame):")
    x = float(input("  target x: "))
    y = float(input("  target y: "))
    z = float(input("  target z: "))

    ctrl_a, ctrl_b = ik_dispatch(f, [x, y, z])

    a_idx, b_idx, tip_idx = FINGER_CTRL[f]
    data.ctrl[a_idx] = ctrl_a
    data.ctrl[b_idx] = ctrl_b
    mujoco.mj_forward(model, data)

    tsv, _ = compute_robot_tsv()
    actual = tsv[tip_idx]
    err_mm = np.linalg.norm(np.array([x, y, z]) - actual) * 1000

    print(f"\n  ctrl[{a_idx}] = {ctrl_a:+.4f}   ctrl[{b_idx}] = {ctrl_b:+.4f}")
    print(f"  alpha = {(ctrl_a+ctrl_b)/2:+.4f}   beta = {(ctrl_a-ctrl_b)/2:+.4f}")
    print(f"  target TSV: x={x:+.4f}  y={y:+.4f}  z={z:+.4f}")
    print(f"  actual TSV: x={actual[0]:+.4f}  y={actual[1]:+.4f}  z={actual[2]:+.4f}")
    print(f"  IK error: {err_mm:.2f} mm\n")


# ════════════════════════════════════════
# Keyboard callback
# ════════════════════════════════════════
def keyboard_callback(keycode):
    global print_requested

    if keycode == 80:   # P — print state
        print_requested = True

    if keycode == 73:   # I — IK single finger
        try:
            print("\n--- IK: which finger? (1 / 2 / 3 / 4=thumb) ---")
            f = int(input("  finger: "))
            if f not in FINGER_CTRL:
                print("  Invalid. Choose 1, 2, 3, or 4.")
                return
            run_ik_for_finger(f)
        except Exception as e:
            print("Invalid input:", e)

    if keycode == 65:   # A — IK all 4 fingers
        try:
            print("\n--- IK ALL fingers (1, 2, 3, 4=thumb) ---")
            for f in [1, 2, 3, 4]:
                run_ik_for_finger(f)
        except Exception as e:
            print("Invalid input:", e)


# ════════════════════════════════════════
# Main
# ════════════════════════════════════════
data.ctrl[:] = np.zeros(model.nu)
mujoco.mj_forward(model, data)

print("\nInstructions:")
print("  P — print current motor angles + TSV for all fingers")
print("  I — IK single finger (1/2/3/4=thumb): enter target TSV")
print("  A — IK all 4 fingers at once")
print()

with mujoco.viewer.launch_passive(
    model=model,
    data=data,
    show_left_ui=False,
    show_right_ui=True,
    key_callback=keyboard_callback
) as viewer:

    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True

    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()

        if print_requested:
            print_state()
            print_requested = False

        time.sleep(0.002)

print("Viewer closed.")