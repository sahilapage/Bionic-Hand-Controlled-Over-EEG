"""Continuous in-hand cube rotation for the Amazing Hand (4 fingers, 8 DoF, 50 Hz).

WHY THIS ENVIRONMENT EXISTS
---------------------------
An earlier face-target environment asked for a discrete
reorientation: bring a named cube face to +Z and hold it. Three trained runs
(PPO `9ilrj53u`, SAC run 1, SAC run 2) all converged to the same degenerate
policy -- freeze and hold -- with like-for-like solve rates of 0.027 / 0.060 /
0.070. The measured cause was not exploration and not the optimiser:

  * The reward paid freezing better than rotating. Measured on run-2 weights:
    do-nothing scored -0.48 per episode; a rollout that genuinely rotated the
    cube 70.9 deg scored -1.41.
  * Undirected rotation earned nothing. During that 70.9 deg rotation the
    face-alignment angle theta moved 91.4 -> 88.3 deg, so the potential-based
    shaping term telescoped to -0.24.
  * The task distribution had no easy instance: face-to-face targets make the
    minimum task a 90 deg roll, and 90 deg is at the very edge of what the hand
    can do in one motion.

A discrete goal is the wrong shape for this hand. This module replaces it with
the task the in-hand-manipulation literature actually solves: keep rotating the
object about one fixed axis, for as many revolutions as possible, without
dropping it (Qi et al., "In-Hand Object Rotation via Rapid Motor Adaptation",
CoRL 2022; Yang et al., "AnyRotate", CoRL 2024).

WHY +Z, AND WHY THE NUMBERS BELOW ARE WHAT THEY ARE
---------------------------------------------------
Measured, not assumed. 120 open-loop random per-joint sinusoids pushed through
this codebase's own action pipeline (axis probe, 2026-08-26):

    survived the full 8 s: 116/120        <- the hand does hold the cube
    accumulated rotation, world frame, deg
        axis      mean       p90       max
          +X      -0.3      10.7      33.9
          +Y      -0.1      10.1      68.4
          +Z     +12.4      38.3      88.0
    principal axis of accumulated spin (SVD): [-0.03, 0.01, 1.00]
    singular values: 4.84 (Z) vs 2.70, 1.68

Rotation about world +Z is the axis this hand drives, by a factor of ~2 over
the others, and random actions already produce a *positive mean* about it --
the reward signal is reachable by exploration on step one, which is exactly
what the face-target task never gave. Geometrically this is unsurprising:
fingers 1-3 are spread along y and press along +x, so differential finger force
is a torque about z; the thumb presses down from +z and acts as the pivot.

A CEM search over a 24-parameter per-joint sinusoid gait gives the open-loop
floor a closed-loop policy has to clear. The gait that search produced ships as
`checkpoints/gait_cem.npy`; replayed on this scene it measures +0.061 +- 0.009
revolutions per 20 s episode (0.014 rad/s, n=60) -- an order of magnitude below
the 0.10-0.12 rad/s reported when it was searched, so it was evidently tuned
against a configuration this repository does not ship. Take the measured
number, not the historical one: it is still 20x what a do-nothing policy
accumulates, and one full revolution in a 20 s episode needs 0.314 rad/s, which
is 22x the shipped gait.

`angvel_clip` is set to 0.8 rad/s, which leaves the reward linear across the
whole reachable range.

REWARD DESIGN
-------------
Hora's reward, adapted. The one structural change is a hard invariant, imposed
because violating it is precisely what broke the previous three runs:

    every term except the rotation term is <= 0.

So a policy that does nothing scores <= 0, and any net rotation about +Z scores
strictly more than doing nothing. There is no configuration of the weights that
can make freezing the optimum. The rotation term is signed, so turning one way
and back nets zero: it pays for progress, not for motion.
"""

import os
from dataclasses import dataclass, replace

import numpy as np
import mujoco
from gymnasium import spaces

from sohand.envs.mujoco_env import MujocoEnv
from sohand.hand import (
    ACTUATORS, JOINTS, TIP_SITES, N_JOINTS, N_FINGERS,
    JVEL_SCALE, REACH_NORM, FACE_NORMALS,
    HAND_CLOSE_FRAC, HAND_CLOSE_JITTER, GRASP_OPEN_FRAC, CLOSE_PROBE_FRAC,
    SETTLE_CTRL_STEPS,
)
from sohand.paths import CUBE_SCENE
from sohand.rotations import (
    qmul, q_from_axis_angle, q_from_vecs, quat_to_mat, so3_log,
)


# Angular-velocity normalisation for the observation. The reachable range is
# ~0.1-0.5 rad/s, so dividing by 10 (as the face-target env did) would squash
# the entire signal into +-0.05 of the observation range.
ANGVEL_OBS_SCALE = 2.0
LINVEL_OBS_SCALE = 0.5

# Named spin axes, in the world frame. The hand base is fixed to the world in
# this scene, so world == hand frame; the distinction only matters once the
# hand is mounted on a moving wrist, at which point k_hat must be rotated into
# the world frame before it is compared against the tracked cube angular
# velocity.
SPIN_AXES = {
    "+Z": np.array([0.0, 0.0, 1.0]), "-Z": np.array([0.0, 0.0, -1.0]),
    "+X": np.array([1.0, 0.0, 0.0]), "-X": np.array([-1.0, 0.0, 0.0]),
    "+Y": np.array([0.0, 1.0, 0.0]), "-Y": np.array([0.0, -1.0, 0.0]),
}


@dataclass(frozen=True)
class SpinCfg:
    # --- episode ---------------------------------------------------------
    max_steps: int = 1000                 # 20 s at 50 Hz, matching Hora's 400 @ 20 Hz
    spin_axis: str = "+Z"                 # measured to be the only well-driven axis
    randomize_axis_sign: bool = False     # True -> also train the -Z direction

    # --- drop detection --------------------------------------------------
    # Unchanged from the face-target env: measured maxima under 1000-step
    # random-action episodes were 0.033 m lateral / 0.004 m vertical.
    z_drop_m: float = 0.035
    xy_drop_m: float = 0.055
    drop_persist_steps: int = 3

    # --- start-state randomisation ---------------------------------------
    pos_jitter_m: float = 0.004
    yaw_jitter_rad: float = float(np.radians(180.0))   # spin phase is irrelevant
    tilt_jitter_rad: float = float(np.radians(10.0))
    randomize_start_face: bool = True

    # A reset is only accepted if the settle actually produced a grasp. The
    # previous env accepted whatever the settle happened to leave, which made
    # the start state a large uncontrolled variance source -- it is what
    # destroyed levels 1-2 of the reverse curriculum (measured: an intended
    # 40-80 deg offset came out as 90.4 +- 3.1 deg because the settle tipped
    # the cube back onto a stable face).
    min_start_contacts: int = 2
    max_start_drift_m: float = 0.015
    max_reset_tries: int = 6

    # How far the settle curls the fingers. A config field rather than the
    # imported constant because the right value depends on where the cube is:
    # at the v2 geometry the kinematic sweep puts the fingertips closest at
    # ~0.0-0.15, and leaving it at 0.50 makes the settle fail its own grasp
    # validation on all six attempts, every episode.
    hand_close_frac: float = HAND_CLOSE_FRAC

    # --- action mapping (unchanged; calibrated against the real servos) ---
    grasp_band_frac: float = 0.50
    action_lpf: float = 0.42
    # 0.08 frac/step = 5.6 rad/s of commanded joint speed. Measured joint
    # velocity under runs 1-2 was 2.3 rad/s mean and 3.3 p95, i.e. the policy
    # spends much of its time near the limit, which is what reads as frantic.
    # 0.05 = 3.5 rad/s: still well above what the task needs at a capped
    # 0.8 rad/s cube rate, and closer to what the real STS3215 sustains
    # under load rather than its no-load figure.
    max_ctrl_rate_frac: float = 0.05

    # --- reward ----------------------------------------------------------
    # WEIGHTS ARE MEASURED, NOT CHOSEN. A calibration sweep runs do-nothing,
    # random and a CEM-optimised rotator gait through this env and reports each
    # term's UNWEIGHTED per-episode sum. The first pass (2026-08-26) rejected a
    # set of plausible-looking Hora-derived weights outright:
    #
    #       term        do-nothing   uniform random   CEM rotator
    #       rot_clipped      0.75            2.20          4.51
    #       offaxis          2.58          202.6         505.1
    #       pose_sq          0.002         157.6         681.7
    #       work_sq          0.41        23527.0       14805.4
    #       torque_sq        0.84         2730.6        1779.2
    #       act_rate_sq      0.00         5228.9          33.2
    #       contact_loss   621.8            647.0         679.6
    #
    # Three of those terms are *anti-task* on this hand: pose, work and offaxis
    # each charge the rotator 100-1000x what they charge a frozen policy,
    # because on an 8-DoF hand whose entire action band is +-0.5 around the
    # grasp, moving away from the grasp pose IS the task. Hora can afford a
    # -0.3 pose penalty because a 16-DoF Allegro gaits *around* a canonical
    # grasp; this hand cannot. And contact_loss came out at ~0.63 for every
    # policy alike -- a constant offset carrying no gradient, so it is off.
    #
    # Invariant, re-verified after every weight change: a rotator must score
    # strictly above do-nothing. That is the single thing all three previous
    # runs got wrong.

    # The rotation term is the *increment*, w_rot * omega.k_hat * dt, so the
    # episode sum is exactly w_rot x (radians turned). No clipping artefact, no
    # farmable oscillation, and the return has a unit: one full revolution is
    # worth w_rot * 2pi = 50.
    w_rot: float = 8.0
    # Was 2.0, a pure spike guard, which left the rotation reward linear all the
    # way up and told the policy that faster is always better with no ceiling.
    # Run 2 took that literally: 1.586 rad/s, cube airborne 15.6% of the time,
    # only 1.22 fingers in contact, dropping 42.5% of episodes. Capping the
    # PAID rate means anything above `angvel_clip` earns nothing extra while
    # still paying the work and torque penalties, so a deliberate 0.8 rad/s
    # beats a frantic 1.6. Hora clips at 0.5 for the same reason.
    # 0.8 rad/s sustained = 2.5 revolutions per 20 s episode.
    angvel_clip: float = 0.8

    # Measured: the instantaneous per-control-step angular velocity is ~0.5
    # rad/s of contact-solver jitter riding on ~0.02-0.10 rad/s of actual spin,
    # a signal-to-noise ratio of about 1:25. A one-pole low-pass at ~2 Hz
    # removes the jitter and leaves the spin. It is linear with unit DC gain,
    # so the telescoping identity above survives up to one boundary term, and
    # it is exactly what a deployed pose tracker has to do anyway.
    angvel_lpf: float = 0.25

    # Off-axis tumbling, on the *filtered* angular velocity. Second calibration
    # pass: even filtered, the rotator's off-axis rate is 0.34 rad/s against a
    # mean on-axis rate of 0.0004 rad/s. That is not noise -- it is how this
    # hand turns the cube. With three 2-DoF fingers and a thumb there is no
    # clean spin available; the cube is rolled from face to face, and every
    # roll is off-axis motion. Hora and AnyRotate can charge -0.3 here because
    # a 16-DoF Allegro really can spin an object cleanly about one axis.
    # Weighted at 0.02 this term alone cost the rotator -6.79 against a +3.20
    # rotation reward. Kept at a whisper: a 5% nudge toward cleaner rotation,
    # never a reason not to rotate. Raise it for a polish phase once the task
    # is being solved, not before.
    w_offaxis: float = 0.0005
    offaxis_clip: float = 2.0

    # Cube-centre velocity, finite-differenced. NOT qvel[0:3]: that is the
    # velocity of the body *origin*, which sits 3.5 cm from the cube's centre,
    # so pure rotation at 0.5 rad/s shows up in it as 1.8 cm/s of phantom
    # translation and the penalty would charge the policy for rotating.
    w_linvel: float = 0.005
    w_drift: float = 0.01         # sustained displacement from the start pose

    # Hora uses -0.3 here. On this hand it is anti-task (see the table above).
    w_pose: float = 0.0
    w_work: float = 5e-6          # (sum tau * qdot)^2
    w_torque: float = 3e-5        # ||tau||^2
    # 0.004 was the original calibration, but measured at 6k steps it charges an
    # exploring policy -20.3 per episode against a do-nothing policy's 0.0 --
    # i.e. freezing would outscore exploring, which is precisely the failure
    # that produced three dead runs. 0.002 keeps twice the old deterrent
    # (-10 exploring, ~-1.3 for a converged gait) without inverting the
    # incentive. The slew limit above is the honest lever for "slower": it is a
    # physical constraint rather than a reward the policy can trade against.
    w_action_rate: float = 0.002  # command reversals: -0.03 for a smooth gait,
                                  # -5.2 for uniform random. A thrash deterrent
                                  # that costs a real gait essentially nothing.
    # Every weight above is set so that the penalties together come to ~19% of
    # what the rotation term pays a working gait -- a tax, not a competitor.
    # Measured episode totals at these weights (1000 steps):
    #   CEM rotator  +2.58     do-nothing  +0.10
    #   small random -0.90     uniform random -5.50
    w_contact_loss: float = 0.0   # no gradient; see the table above

    # Measured on the trained policies: finger1 bears force on 24% of steps
    # (run 1) or 29% (run 2) while finger2 manages 56%/41%. The geometry change
    # narrowed that gap but by lowering everyone, not raising finger1 --
    # fingers-in-contact fell 1.58 -> 1.22 and the cube spends 15.6% of the
    # episode airborne.
    #
    # This charges for the LEAST engaged finger being idle, using a ~1 s
    # exponential average of each finger's contact so it measures a gait rather
    # than an instant. It is a penalty, never a bonus, so the invariant holds:
    # a policy that grips with all four and does not rotate still scores ~0,
    # not more. Bounded by construction at -w_finger_idle per step.
    w_finger_idle: float = 0.02
    finger_ema: float = 0.02      # ~1 s window at 50 Hz

    # Terminal. One revolution is worth +50, so a drop costs a fifth of a turn
    # -- enough to matter, nowhere near enough to make freezing the safe play.
    # The real deterrent is termination: a drop forfeits the rest of the
    # episode's rotation reward.
    drop_penalty: float = 10.0
    terminate_on_drop: bool = True

    # --- reporting -------------------------------------------------------
    # "Success" is now a real, interpretable quantity: a full revolution about
    # the target axis inside one episode. Partial bars are logged too so the
    # learning curve is legible long before the first full turn.
    success_revolutions: float = 1.0

    # --- domain randomisation --------------------------------------------
    dr_friction: float = 0.30
    dr_mass: float = 0.20
    dr_damping: float = 0.25
    dr_gain: float = 0.15
    dr_ctrl_rate: float = 0.20
    noise_jpos_rad: float = 0.010
    noise_jvel: float = 0.15
    noise_quat_rad: float = 0.035
    # The reachable spin rate is ~0.1-0.5 rad/s. The face-target env injected
    # 0.30 rad/s of angular-velocity noise, which would bury the entire signal
    # this task is built on. A real tracker differentiating a 2 deg pose error
    # at 50 Hz is far noisier than this -- the deployment-side estimator has to
    # low-pass it, and 0.05 rad/s is what a 5 Hz filter leaves.
    noise_angvel: float = 0.05
    noise_linvel: float = 0.02
    noise_tip_m: float = 0.006
    action_dropout_p: float = 0.02


SPIN_CFG = SpinCfg()


# Observation layout, 74 dims. Every entry is measurable on the real rig:
# joint encoders -> jpos/jvel; commanded setpoints -> action history and
# filtered_ctrl; a camera pose tracker -> the cube blocks; forward kinematics
# on the encoders plus the tracked cube position -> finger_to_cube; contact is
# the one block that needs either tactile pads or a force-threshold proxy.
SPIN_OBS_SLICES = {
    "jpos": 8, "jvel": 8, "last_action": 8, "prev_action": 8, "filtered_ctrl": 8,
    "tilt_cos": 1, "tilt_vec": 3, "phase4": 2,
    "angvel": 3, "linvel": 3, "center_offset": 3,
    "finger_to_cube": 12, "contacts": 4, "axis": 3,
}
SPIN_OBS_DIM = sum(SPIN_OBS_SLICES.values())  # 74


def cube_spin_features(R, k_hat):
    """Pose features for a 4-fold-symmetric object spinning about `k_hat`.

    Absolute yaw about the spin axis is *irrelevant* to this task and, for a
    cube, only meaningful modulo 90 deg. Handing the policy a raw rotation
    matrix would make the observation non-stationary in the one coordinate the
    task is periodic in. Instead:

      tilt_cos : how squarely a cube axis lines up with the spin axis (1.0 =
                 a pair of faces exactly perpendicular to k_hat)
      tilt_vec : which way it is tipped away from that, in the plane normal to
                 k_hat -- the signal for "the cube is about to be lost"
      phase4   : (cos 4psi, sin 4psi) of the roll about k_hat, so the policy can
                 time a push against a face rather than a corner, and the
                 feature repeats every quarter turn exactly as the cube does
    """
    axes = R.T                                   # rows: cube's x, y, z in world
    dots = axes @ k_hat
    i = int(np.argmax(np.abs(dots)))
    a_up = axes[i] * np.sign(dots[i])            # cube axis nearest +k_hat
    tilt_cos = float(np.clip(np.dot(a_up, k_hat), -1.0, 1.0))
    tilt_vec = a_up - tilt_cos * k_hat           # perpendicular component

    # Roll phase: take a cube axis perpendicular to a_up and measure its angle
    # in the plane normal to k_hat.
    j = (i + 1) % 3
    e1 = np.array([k_hat[1], -k_hat[2], k_hat[0]])       # any vector off k_hat
    e1 = e1 - np.dot(e1, k_hat) * k_hat
    n1 = np.linalg.norm(e1)
    e1 = e1 / n1 if n1 > 1e-8 else np.array([1.0, 0.0, 0.0])
    e2 = np.cross(k_hat, e1)
    a_p = axes[j] - np.dot(axes[j], k_hat) * k_hat
    psi = float(np.arctan2(np.dot(a_p, e2), np.dot(a_p, e1)))
    return tilt_cos, tilt_vec.astype(np.float32), np.array(
        [np.cos(4 * psi), np.sin(4 * psi)], dtype=np.float32)


class AmazingHandSpinEnv(MujocoEnv):
    """Rotate the cube continuously about one fixed axis, without dropping it."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, model_path=CUBE_SCENE, render_mode=None, randomize=True,
                 sensor_noise=True, cfg=SPIN_CFG, **kw):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"MuJoCo scene not found: {model_path}")
        super().__init__(model_path, frame_skip=10, render_mode=render_mode)
        self.cfg = cfg
        self.randomize = randomize
        self.sensor_noise = sensor_noise

        self.actids = np.array([self.model.actuator(n).id for n in ACTUATORS])
        self.qposids = np.array([self.model.joint(n).qposadr[0] for n in JOINTS])
        self.qvelids = np.array([self.model.joint(n).dofadr[0] for n in JOINTS])
        self.cubeid = self.model.body("cube").id
        cj = self.model.body("cube").jntadr[0]
        self.cube_qpos = self.model.jnt_qposadr[cj]
        self.cube_dof = self.model.jnt_dofadr[cj]

        self.tip_sites = [self.model.site(n).id for n in TIP_SITES]
        self.cube_geoms = {g for g in range(self.model.ngeom)
                           if self.model.geom_bodyid[g] == self.cubeid}
        self.finger_bodies = [self._finger_subtree(f) for f in range(N_FINGERS)]
        self.finger_geoms = [{g for g in range(self.model.ngeom)
                              if self.model.geom_bodyid[g] in bodies}
                             for bodies in self.finger_bodies]
        self.finger_body_ids = [np.array(sorted(b)) for b in self.finger_bodies]

        self._base = {
            "geom_friction": self.model.geom_friction.copy(),
            "dof_damping": self.model.dof_damping.copy(),
            "dof_frictionloss": self.model.dof_frictionloss.copy(),
            "dof_armature": self.model.dof_armature.copy(),
            "body_mass": self.model.body_mass.copy(),
            "body_inertia": self.model.body_inertia.copy(),
            "actuator_gainprm": self.model.actuator_gainprm.copy(),
            "actuator_biasprm": self.model.actuator_biasprm.copy(),
        }

        # Cube geometry is READ FROM THE MODEL, not hardcoded. The module
        # constants HALF = 0.0235 and CUBE_LOCAL_CENTER were baked in at the
        # scene's original scale, so scaling the mesh (the only way to resize a
        # mesh geom -- MuJoCo silently ignores `size` on one) left the contact
        # test, the reward and the drop detector all reading the old size. That
        # is what corrupted the first geometry sweep: a cube lying on the floor
        # scored as "held".
        #
        # half-extent: the mesh is stored rotated in its geom frame, so its AABB
        # overstates the cube. The farthest vertex is a corner at HALF*sqrt(3),
        # which is rotation-invariant and therefore safe to invert.
        # The cube body carries seven geoms: the mesh plus six AprilTag decals
        # of size 0.005. Picking an arbitrary one returns a tag and reports the
        # cube as 1 cm across.
        mesh_geoms = [g for g in self.cube_geoms
                      if self.model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH]
        if not mesh_geoms:
            raise RuntimeError("cube body has no mesh geom; cannot infer its size")
        g = mesh_geoms[0]
        mid = self.model.geom_dataid[g]
        if mid >= 0:
            v0 = self.model.mesh_vertadr[mid]
            nv = self.model.mesh_vertnum[mid]
            verts = self.model.mesh_vert[v0:v0 + nv]
            self.cube_half = float(np.max(np.linalg.norm(verts, axis=1)) / np.sqrt(3.0))
        else:
            self.cube_half = float(np.max(self.model.geom_size[g]))
        # the mesh is not centred on its body origin; the inertial frame is
        self.cube_local_center = self.model.body_ipos[self.cubeid].copy()

        cube_xyz = self.init_qpos[self.cube_qpos:self.cube_qpos + 3]
        self.nominal_cube_center = cube_xyz.copy() + self.cube_local_center

        lims = self.model.actuator_ctrlrange[self.actids]
        self.ctrl_lo, self.ctrl_hi = lims[:, 0], lims[:, 1]
        self.ctrl_mid = (self.ctrl_lo + self.ctrl_hi) / 2
        self.ctrl_half = (self.ctrl_hi - self.ctrl_lo) / 2

        self.action_space = spaces.Box(-1.0, 1.0, shape=(N_JOINTS,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(SPIN_OBS_DIM,),
                                            dtype=np.float32)

        self.k_hat = SPIN_AXES[cfg.spin_axis].copy()
        self.last_action = np.zeros(N_JOINTS, np.float32)
        self.prev_action = np.zeros(N_JOINTS, np.float32)
        self.filtered_ctrl = np.zeros(N_JOINTS, np.float32)
        self.grasp_frac = np.zeros(N_JOINTS, np.float32)
        self.grasp_qpos = np.zeros(N_JOINTS)
        self.start_pos = np.zeros(3)
        self.step_count = 0
        self._ctrl_rate = cfg.max_ctrl_rate_frac
        self._xy_over_steps = 0
        self._z_over_steps = 0
        self._rot_acc = 0.0
        self._rot_peak = 0.0
        self._reward_sums = {}
        self._prev_R = np.eye(3)
        self._prev_center = np.zeros(3)
        self._omega_f = np.zeros(3)
        self._finger_ema = np.zeros(N_FINGERS)
        self._reset_tries = 0

        self.close_sign = self._detect_close_sign()

    # ------------------------------------------------------------------
    # Model introspection (shared with the face-target env)
    # ------------------------------------------------------------------
    @property
    def dt(self):
        """Control-step duration. The base MujocoEnv does not define this, and
        every angular-velocity quantity in this file is a per-control-step
        finite difference, so it has to be right: 10 x 2 ms = 20 ms = 50 Hz."""
        return self.model.opt.timestep * self.frame_skip

    def _finger_subtree(self, finger_idx):
        """All bodies descending from either of finger `finger_idx`'s motors."""
        roots = {self.model.jnt_bodyid[self.model.joint(JOINTS[2 * finger_idx + k]).id]
                 for k in (0, 1)}
        out = set(roots)
        changed = True
        while changed:
            changed = False
            for b in range(self.model.nbody):
                if b not in out and self.model.body_parentid[b] in out:
                    out.add(b)
                    changed = True
        return out

    def _detect_close_sign(self):
        """Which control sign drives the fingers toward the cube."""
        qpos_save = self.data.qpos.copy()
        dists = {}
        for sign in (1.0, -1.0):
            self.data.qpos[self.qposids] = (self.ctrl_mid
                                            + sign * CLOSE_PROBE_FRAC * self.ctrl_half)
            mujoco.mj_forward(self.model, self.data)
            cpos = self._cube_center_world()
            dists[sign] = float(np.mean([np.linalg.norm(self.data.site_xpos[s] - cpos)
                                         for s in self.tip_sites]))
        self.data.qpos[:] = qpos_save
        mujoco.mj_forward(self.model, self.data)
        return 1.0 if dists[1.0] < dists[-1.0] else -1.0

    # ------------------------------------------------------------------
    # State readout
    # ------------------------------------------------------------------
    def _cube_R(self):
        return self.data.xmat[self.cubeid].reshape(3, 3)

    def _cube_center_world(self):
        return self.data.xpos[self.cubeid] + self._cube_R() @ self.cube_local_center

    # -- public accessors -------------------------------------------------
    # Evaluation, replay and diagnostics all need the cube's orientation and
    # the contact mask. They are exposed here so those tools do not reach into
    # private methods.

    def cube_rotation(self):
        """The cube's current world-frame rotation matrix."""
        return self._cube_R()

    def fingers_touching(self):
        """Boolean mask, one entry per finger, of which are on the cube."""
        return self._fingers_touching_mask()

    def _finger_gaps(self):
        cpos = self._cube_center_world()
        gaps = np.empty(N_FINGERS, dtype=np.float32)
        for f, bids in enumerate(self.finger_body_ids):
            d = np.linalg.norm(self.data.xpos[bids] - cpos[None, :], axis=1)
            gaps[f] = float(np.min(d)) - self.cube_half
        return gaps

    def _tip_to_cube(self):
        cpos = self._cube_center_world()
        return np.array([cpos - self.data.site_xpos[sid] for sid in self.tip_sites],
                        dtype=np.float32)

    def _fingers_touching_mask(self):
        c_force = 0.015
        touched = np.zeros(N_FINGERS, dtype=bool)
        f6 = np.zeros(6)
        for i in range(self.data.ncon):
            con = self.data.contact[i]
            if con.geom1 not in self.cube_geoms and con.geom2 not in self.cube_geoms:
                continue
            other = con.geom2 if con.geom1 in self.cube_geoms else con.geom1
            for f, geoms in enumerate(self.finger_geoms):
                if other in geoms and not touched[f]:
                    mujoco.mj_contactForce(self.model, self.data, i, f6)
                    if abs(f6[0]) > c_force:
                        touched[f] = True
        return touched

    def _measured_angvel(self):
        """World-frame cube angular velocity, by finite-differencing the pose
        over one *control* step.

        Deliberately not `qvel[3:6]`. That is a body-frame substep value which
        has to be rotated into the world (getting this wrong is what inverted
        the old spin reward once the cube had actually turned), and it is not
        what a deployed system can measure. Differencing the tracked pose over
        the control interval is both correct and reproducible on hardware --
        it is also what Hora does.
        """
        R = self._cube_R()
        return so3_log(R @ self._prev_R.T) / self.dt

    def _cube_center_vel(self):
        """Cube *centre* velocity, finite-differenced over the control step.

        `qvel[cube_dof:cube_dof+3]` is the velocity of the body origin, and the
        cube mesh's centre sits 3.5 cm away from it, so rotating the cube at
        0.5 rad/s registers there as ~1.8 cm/s of translation that is not
        happening. Penalising that would be penalising the task.
        """
        return (self._cube_center_world() - self._prev_center) / self.dt

    # ------------------------------------------------------------------
    # Domain randomisation
    # ------------------------------------------------------------------
    def _apply_domain_randomization(self):
        c, rng = self.cfg, self.np_random
        b = self._base
        self.model.geom_friction[:] = b["geom_friction"]
        self.model.dof_damping[:] = b["dof_damping"]
        self.model.dof_frictionloss[:] = b["dof_frictionloss"]
        self.model.dof_armature[:] = b["dof_armature"]
        self.model.body_mass[:] = b["body_mass"]
        self.model.body_inertia[:] = b["body_inertia"]
        self.model.actuator_gainprm[:] = b["actuator_gainprm"]
        self.model.actuator_biasprm[:] = b["actuator_biasprm"]
        self._ctrl_rate = c.max_ctrl_rate_frac
        if not self.randomize:
            mujoco.mj_setConst(self.model, self.data)
            return

        def jit(spread, size=None):
            return rng.uniform(1.0 - spread, 1.0 + spread, size)

        self.model.geom_friction[:, 0] *= jit(c.dr_friction, self.model.ngeom)
        self.model.geom_friction[:, 1] *= jit(c.dr_friction, self.model.ngeom)
        self.model.dof_damping[:] *= jit(c.dr_damping, self.model.nv)
        self.model.dof_frictionloss[:] *= jit(c.dr_damping, self.model.nv)
        self.model.dof_armature[:] *= jit(c.dr_damping, self.model.nv)
        m = float(jit(c.dr_mass))
        self.model.body_mass[self.cubeid] *= m
        self.model.body_inertia[self.cubeid] *= m
        # A position actuator's stiffness lives in two places; changing only
        # gainprm leaves the bias term inconsistent and silently detunes the
        # servo instead of stiffening it.
        g = jit(c.dr_gain, len(self.actids))
        self.model.actuator_gainprm[self.actids, 0] *= g
        self.model.actuator_biasprm[self.actids, 1] *= g
        self._ctrl_rate = c.max_ctrl_rate_frac * float(jit(c.dr_ctrl_rate))
        mujoco.mj_setConst(self.model, self.data)

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        self._reset_options = options or {}
        return super().reset(seed=seed, options=options)

    def reset_model(self):
        """Place the cube, close the hand, and *verify* the grasp took.

        Hora keeps a cache of pre-validated grasps and samples from it. The
        equivalent here is to re-roll the settle until it produces a real grasp:
        at least `min_start_contacts` fingers bearing force and the cube still
        near where it was placed. Without this check the start state is a large
        uncontrolled variance source -- measured on the face-target env, a
        settle intended to leave the cube 40-80 deg from a face instead tipped
        it back onto a stable face at 90.4 +- 3.1 deg, every time.
        """
        c, rng = self.cfg, self.np_random
        opts = getattr(self, "_reset_options", None) or {}

        if c.randomize_axis_sign and "spin_axis" not in opts:
            self.k_hat = SPIN_AXES[c.spin_axis].copy() * float(rng.choice([-1.0, 1.0]))
        elif "spin_axis" in opts:
            self.k_hat = SPIN_AXES[opts["spin_axis"]].copy()
        else:
            self.k_hat = SPIN_AXES[c.spin_axis].copy()

        for attempt in range(c.max_reset_tries):
            self._place_and_settle(rng, opts)
            touching = self._fingers_touching_mask()
            drift = float(np.linalg.norm(self._cube_center_world()
                                         - self._placed_center))
            if (touching.sum() >= c.min_start_contacts
                    and drift <= c.max_start_drift_m):
                break
        self._reset_tries = attempt + 1

        frac_init = np.clip(
            (self.data.qpos[self.qposids] - self.ctrl_mid) / self.ctrl_half, -1.0, 1.0)
        self.grasp_frac[:] = frac_init
        self.filtered_ctrl[:] = frac_init
        self.grasp_qpos[:] = self.data.qpos[self.qposids]
        self.last_action[:] = 0.0
        self.prev_action[:] = 0.0
        self.start_pos = self._cube_center_world().copy()
        self._prev_R = self._cube_R().copy()
        self._prev_center = self.start_pos.copy()
        self._omega_f = np.zeros(3)
        # start at 1.0 so an episode is not charged for idleness it has not had
        # time to demonstrate
        self._finger_ema[:] = 1.0

        self.step_count = 0
        self._xy_over_steps = 0
        self._z_over_steps = 0
        self._rot_acc = 0.0
        self._rot_peak = 0.0
        self._reward_sums = {}

    def _place_and_settle(self, rng, opts):
        c = self.cfg
        # Retries have to start from the same clean state, or attempt 2
        # inherits attempt 1's passive linkage angles and settles somewhere
        # different for reasons that have nothing to do with the new draw.
        mujoco.mj_resetData(self.model, self.data)
        self._apply_domain_randomization()

        qpos_open = np.clip(
            self.ctrl_mid + self.close_sign * GRASP_OPEN_FRAC * self.ctrl_half,
            self.ctrl_lo, self.ctrl_hi)
        self.data.qpos[self.qposids] = qpos_open
        self.data.qvel[self.qvelids] = 0.0
        self.data.ctrl[self.actids] = qpos_open

        base_quat = self.init_qpos[self.cube_qpos + 3:self.cube_qpos + 7].copy()
        face = int(opts.get("start_face",
                            int(rng.integers(6)) if c.randomize_start_face else 0))
        quat = qmul(q_from_vecs(FACE_NORMALS[face], np.array([0.0, 0.0, 1.0])),
                    base_quat)
        # Full 180 deg of roll jitter: for a continuous-rotation task the phase
        # at which the episode starts carries no information, and pinning it
        # would let the policy memorise one entry point into the gait.
        quat = qmul(q_from_axis_angle([0, 0, 1],
                                        float(rng.uniform(-c.yaw_jitter_rad,
                                                          c.yaw_jitter_rad))), quat)
        tilt_axis = rng.normal(size=3)
        tilt_axis[2] = 0.0
        quat = qmul(q_from_axis_angle(tilt_axis + np.array([1e-6, 0, 0]),
                                        float(rng.uniform(-c.tilt_jitter_rad,
                                                          c.tilt_jitter_rad))), quat)
        quat /= np.linalg.norm(quat)

        target_center = self.nominal_cube_center.copy()
        target_center[:2] += rng.uniform(-c.pos_jitter_m, c.pos_jitter_m, 2)
        self.data.qpos[self.cube_qpos:self.cube_qpos + 3] = (
            target_center - quat_to_mat(quat) @ self.cube_local_center)
        self.data.qpos[self.cube_qpos + 3:self.cube_qpos + 7] = quat
        self.data.qvel[self.cube_dof:self.cube_dof + 6] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self._placed_center = self._cube_center_world().copy()

        qpos_closed = (self.ctrl_mid
                       + self.close_sign * c.hand_close_frac * self.ctrl_half)
        qpos_closed = np.clip(qpos_closed + rng.uniform(-HAND_CLOSE_JITTER,
                                                        HAND_CLOSE_JITTER, N_JOINTS),
                              self.ctrl_lo, self.ctrl_hi)
        full = np.zeros(self.model.nu)
        for i in range(SETTLE_CTRL_STEPS):
            frac = (i + 1) / SETTLE_CTRL_STEPS
            full[self.actids] = qpos_open + frac * (qpos_closed - qpos_open)
            self.do_simulation(full, self.frame_skip)

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------
    def _get_obs(self):
        c, rng = self.cfg, self.np_random
        noisy = self.sensor_noise

        jpos_raw = self.data.qpos[self.qposids].copy()
        jvel_raw = self.data.qvel[self.qvelids].copy()
        if noisy:
            jpos_raw = jpos_raw + rng.normal(0.0, c.noise_jpos_rad, N_JOINTS)
            jvel_raw = jvel_raw + rng.normal(0.0, c.noise_jvel, N_JOINTS)
        jpos = np.clip((jpos_raw - self.ctrl_mid) / self.ctrl_half, -1.0, 1.0)
        jvel = np.clip(jvel_raw / JVEL_SCALE, -1.0, 1.0)

        R = self._cube_R()
        if noisy:
            ax = rng.normal(size=3)
            tilt = q_from_axis_angle(ax + np.array([1e-9, 0, 0]),
                                     float(rng.normal(0.0, c.noise_quat_rad)))
            R = quat_to_mat(tilt) @ R
        tilt_cos, tilt_vec, phase4 = cube_spin_features(R, self.k_hat)

        angvel = self._omega_f.copy()   # filtered: the raw signal is ~1:25 SNR
        linvel = self._cube_center_vel()
        if noisy:
            angvel = angvel + rng.normal(0.0, c.noise_angvel, 3)
            linvel = linvel + rng.normal(0.0, c.noise_linvel, 3)
        angvel = np.clip(angvel / ANGVEL_OBS_SCALE, -3.0, 3.0)
        linvel = np.clip(linvel / LINVEL_OBS_SCALE, -3.0, 3.0)

        offset = np.clip((self._cube_center_world() - self.nominal_cube_center) / 0.05,
                         -3.0, 3.0)

        tip_vecs = self._tip_to_cube()
        if noisy:
            tip_vecs = tip_vecs + rng.normal(0.0, c.noise_tip_m, tip_vecs.shape)
        tip_to_cube = np.clip(tip_vecs / REACH_NORM, -3.0, 3.0).flatten()

        contacts = self._fingers_touching_mask().astype(np.float32)

        return np.concatenate([
            jpos, jvel, self.last_action, self.prev_action, self.filtered_ctrl,
            [tilt_cos], tilt_vec, phase4,
            angvel, linvel, offset, tip_to_cube, contacts, self.k_hat,
        ]).astype(np.float32)

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------
    def _reward(self, act, dropped):
        """r = w_rot * (omega . k_hat) * dt  -  (penalties, all <= 0).

        The rotation term is an increment, so the episode sum is *exactly*
        w_rot x (net radians turned about k_hat): one revolution = +50. Signed,
        so turning one way and back nets zero -- it pays for progress, not for
        motion -- and there is no clipping artefact to farm.

        The invariant that separates this from every previous version of this
        project's reward: nothing except the rotation term is ever positive, so
        a policy that does nothing scores ~0 and any net rotation beats it.
        Verified numerically by measuring every term over do-nothing,
        random and CEM-rotator rollouts -- see docs/in-hand-rotation.md --
        rather than asserted.
        """
        c = self.cfg
        omega_raw = self._measured_angvel()
        omega = self._omega_f                      # jitter removed; see angvel_lpf
        proj = float(np.clip(np.dot(omega, self.k_hat), -c.angvel_clip, c.angvel_clip))

        r_rot = c.w_rot * proj * self.dt

        offaxis = float(np.linalg.norm(omega - np.dot(omega, self.k_hat) * self.k_hat))
        r_offaxis = -c.w_offaxis * min(offaxis, c.offaxis_clip)

        vel = self._cube_center_vel()
        r_linvel = -c.w_linvel * float(np.sum(np.abs(vel)))

        drift = float(np.linalg.norm(self._cube_center_world() - self.start_pos))
        r_drift = -c.w_drift * drift

        if c.w_pose > 0.0:
            dq = self.data.qpos[self.qposids] - self.grasp_qpos
            r_pose = -c.w_pose * float(np.sum(dq * dq))
        else:
            r_pose = 0.0

        tau = self.data.actuator_force[self.actids]
        qd = self.data.qvel[self.qvelids]
        r_work = -c.w_work * float(np.sum(tau * qd)) ** 2
        r_torque = -c.w_torque * float(np.sum(tau * tau))

        d_act = act - self.last_action
        r_action = -c.w_action_rate * float(np.sum(d_act * d_act))

        touching = self._fingers_touching_mask()
        n_touch = int(touching.sum())
        r_contact = (-c.w_contact_loss * (N_FINGERS - n_touch) / N_FINGERS
                     if c.w_contact_loss > 0.0 else 0.0)

        self._finger_ema += c.finger_ema * (touching.astype(float) - self._finger_ema)
        idle = 1.0 - float(self._finger_ema.min())
        r_finger = -c.w_finger_idle * idle if c.w_finger_idle > 0.0 else 0.0

        r_drop = -c.drop_penalty if dropped else 0.0

        total = (r_rot + r_offaxis + r_linvel + r_drift + r_pose + r_work
                 + r_torque + r_action + r_contact + r_finger + r_drop)

        # The metric, deliberately computed from the RAW measurement: "the cube
        # rotated N degrees" has to mean the cube actually rotated N degrees,
        # not that a filtered estimate of it did.
        self._rot_acc += float(np.dot(omega_raw, self.k_hat)) * self.dt
        self._rot_peak = max(self._rot_peak, abs(self._rot_acc))

        info = {
            "r_rot": r_rot, "r_offaxis": r_offaxis, "r_linvel": r_linvel,
            "r_drift": r_drift, "r_pose": r_pose, "r_work": r_work,
            "r_torque": r_torque, "r_action": r_action, "r_contact": r_contact,
            "r_finger": r_finger, "r_drop": r_drop,
            "min_finger_ema": float(self._finger_ema.min()),
            "spin_rate": proj, "offaxis_rate": offaxis, "n_touch": n_touch,
            "rot_acc_rad": self._rot_acc, "drift_m": drift,
        }
        for k, v in info.items():
            if k.startswith("r_"):
                self._reward_sums[k] = self._reward_sums.get(k, 0.0) + float(v)
        return float(total), info

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------
    def step(self, action):
        c = self.cfg
        act = np.clip(np.nan_to_num(action, nan=0.0), -1.0, 1.0).astype(np.float32)
        if self.sensor_noise and self.np_random.random() < c.action_dropout_p:
            act = self.last_action.copy()

        target_frac = np.clip(self.grasp_frac + act * c.grasp_band_frac, -1.0, 1.0)
        lpf_ctrl = c.action_lpf * target_frac + (1 - c.action_lpf) * self.filtered_ctrl
        step_delta = np.clip(lpf_ctrl - self.filtered_ctrl,
                             -self._ctrl_rate, self._ctrl_rate)
        self.filtered_ctrl = self.filtered_ctrl + step_delta

        self._prev_R = self._cube_R().copy()
        self._prev_center = self._cube_center_world().copy()
        full = np.zeros(self.model.nu)
        full[self.actids] = self.ctrl_mid + self.filtered_ctrl * self.ctrl_half
        self.do_simulation(full, self.frame_skip)
        self.step_count += 1

        cpos = self._cube_center_world()
        z_drop = float(self.start_pos[2] - cpos[2])
        xy_drift = float(np.linalg.norm(cpos[:2] - self.start_pos[:2]))
        self._xy_over_steps = self._xy_over_steps + 1 if xy_drift > c.xy_drop_m else 0
        self._z_over_steps = self._z_over_steps + 1 if z_drop > c.z_drop_m else 0
        dropped = (self._xy_over_steps >= c.drop_persist_steps
                   or self._z_over_steps >= c.drop_persist_steps)

        self._omega_f = ((1.0 - c.angvel_lpf) * self._omega_f
                         + c.angvel_lpf * self._measured_angvel())
        reward, info = self._reward(act, dropped)
        self.prev_action = self.last_action.copy()
        self.last_action = act.copy()

        terminated = bool(dropped and c.terminate_on_drop)
        truncated = self.step_count >= c.max_steps

        if terminated or truncated:
            revs = self._rot_acc / (2 * np.pi)
            info.update({f"ep_{k}": v for k, v in self._reward_sums.items()})
            info.update({
                "ep_rotation_rad": self._rot_acc,
                "ep_revolutions": revs,
                "ep_rot_per_sec": self._rot_acc / max(self.step_count * self.dt, 1e-9),
                "ep_steps": self.step_count,
                "ep_dropped": float(dropped),
                "ep_reset_tries": self._reset_tries,
                # A ladder of bars, so the curve is readable long before the
                # first full turn. `success` is the headline number.
                "success": float(revs >= c.success_revolutions),
                "reached_quarter_turn": float(revs >= 0.25),
                "reached_half_turn": float(revs >= 0.5),
                "reached_full_turn": float(revs >= 1.0),
                "reached_two_turns": float(revs >= 2.0),
            })

        if self.render_mode == "human":
            self.render()
        return self._get_obs(), reward, terminated, truncated, info


def make_spin_cfg(**overrides):
    return replace(SPIN_CFG, **overrides)


def make_eval_env(**kw):
    """Deterministic evaluation: no domain randomisation, no sensor noise."""
    kw.setdefault("randomize", False)
    kw.setdefault("sensor_noise", False)
    return AmazingHandSpinEnv(**kw)
