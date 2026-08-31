# MuJoCo models

Two models of the [Pollen Robotics Amazing Hand](https://github.com/pollen-robotics/AmazingHand),
both exported from the same Onshape document with
[`onshape-to-robot`](https://github.com/Rhoban/onshape-to-robot). They are
**not interchangeable** — the naming and the contact parameters differ, and
each is tuned for a different job.

| | `hand/` | `cube/` |
|---|---|---|
| contents | hand only, four mocap fingertip targets | hand + 4.7 cm cube with AprilTag decals |
| actuator names | `finger1_motor1` … (same as the joints) | `motor_finger1_1` … |
| position servo | class `mjcf`, `kp` 50, `inheritrange` | class `sts3215_147`, `kp` 17.11, `forcerange` ±3.23 N·m |
| collisions | visual meshes are `contype=0` — nothing collides | collision group + fingertip grasp pads |
| solver | MuJoCo defaults | Newton, 30 iterations, `timestep` 2 ms |
| used by | `sohand.retarget` | `sohand.envs`, `sohand.rl` |

Joint names (`finger1_motor1` … `finger4_motor2`) are identical in both, so
anything that addresses joints rather than actuators works against either.

## Files

```
hand/
  robot.xml               kinematics, meshes, actuators
  scene.xml               robot.xml + floor, lighting, fingertip mocap targets
  keyframes.xml           the settled zero pose
  joints_properties.xml   servo classes: perfect_actuator, sts3215_345/147, xc330m288t
  additional.xml          contact exclusions and equality-constraint solver settings
  config.json             the onshape-to-robot export config, incl. the source document
  assets/                 STL meshes

cube/
  robot.xml               as above, with collision geoms and fingertip grasp pads
  scene.xml               robot.xml + a 4.7 cm cube on a free joint, six AprilTags
  scene_spin.xml          generated: cube scaled to 6.1 cm and shifted toward finger1
  joints_properties.xml
  additional.xml
  tag36h11_0*.png         AprilTag textures, one per cube face
  assets/                 STL meshes, including cube.stl
```

`scene_spin.xml` is generated, not hand-edited:

```bash
python -m sohand.rl.make_scene          # reproduces the shipped file exactly
```

Its header records the exact command that produced it. Regenerating with
different arguments changes the task — the run 2 checkpoint is tied to this
geometry.

## Two things that bite

**MuJoCo ignores `size` on a mesh geom.** The only way to resize the cube is
the mesh asset's `scale`. Mass then scales as *s*³ and inertia as *s*⁵ (or *s*²
for a hollow shell of fixed mass), and the inertial offset scales linearly.
`sohand.rl.make_scene` handles all four.

**The fingers are passive four-bar linkages.** Their constrained pose does not
exist until the solver has run, so `mj_forward` alone leaves every fingertip at
its unconstrained position. Any measurement — forward kinematics, a Jacobian
sweep, a grasp settle — has to step physics first. This was the original cause
of "the motors do not move".

## Provenance

The mechanical design is Pollen Robotics' open-source Amazing Hand. `config.json`
records the Onshape document and microversion each mesh was exported from.
