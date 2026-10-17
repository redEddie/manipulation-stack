# GELLO calibration tools

These scripts establish the leader's joint mapping and check its pose against
the follower before collection. NumPy handles joint values, Dynamixel SDK reads
the leader, ZeroMQ reads the robot node, and Tyro provides the command-line options.

| Script | Purpose |
| --- | --- |
| [`gello_get_offset.py`](gello_get_offset.py) | Estimate joint offsets at a known reference pose and report gripper endpoints. |
| [`gello_match_pose.py`](gello_match_pose.py) | Display leader/follower joint differences while the operator aligns the leader by hand; sends no robot motion commands. |

Run from the repository root in the robot-node environment. Inspect the options
before choosing the serial port, reference angles and joint signs:

```bash
python scripts/calib/gello_get_offset.py --help
python scripts/calib/gello_match_pose.py --help
```

The offset tool defaults to six arm joints. For FR3, supply all seven reference
angles and matching signs. The runtime mapping is currently stored in
`PORT_CONFIG_MAP` in [`gello_agent.py`](../../mstack/agents/gello_agent.py).
The pose monitor uses that mapping and the selected `--grip` convention.
Stop the monitor before collection so the GUI can acquire the leader's serial port.

## Hardware references

- [Leader/follower joint correspondence](images/gello_matching_joints.jpg)
- [FR3 and GELLO calibration pose](images/fr3_gello_calib_pose.jpeg)
- [Single-arm wiring and joint order](images/franka_gello_single_v05_annotated.jpg)
- [Dual-arm wiring and joint order](images/franka_gello_duo_v05_annotated.jpg)
- [Robot reference configuration](images/robot_known_configuration.jpg)

Return to the [GUI guide](../../mstack/gui/README.md) for collection setup.
