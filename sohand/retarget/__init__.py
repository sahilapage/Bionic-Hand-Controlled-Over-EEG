"""Human hand pose -> Amazing Hand motor commands.

The Amazing Hand has four fingers; a human has five. The pinky is dropped and
the rest map index->tip1, middle->tip2, ring->tip3, thumb->tip4. Per finger,
motor1 flexes and motor2 abducts.

    python -m sohand.retarget.retarget      # retarget TSVs and play them back
    python -m sohand.retarget.probe_fk      # per-motor fingertip sensitivity

Import the pieces from `sohand.retarget.retarget` directly -- re-exporting them
here would make `python -m sohand.retarget.retarget` import the module twice.
"""
