"""Pinocchio model with the station's calibrated flange-to-EE transform.

This is kinematics only: no collision geometry or collision-free guarantee.
"""
import numpy as np
import pinocchio as pin

from mstack.robots.fr3_kinematics import _MDH, X_EE


class FR3Kinematics:
    def __init__(self):
        self.model = pin.Model()
        parent = 0
        for i, (a, d, alpha) in enumerate(_MDH[:7]):
            ca, sa = np.cos(alpha), np.sin(alpha)
            placement = pin.SE3(np.array([[1, 0, 0], [0, ca, -sa], [0, sa, ca]]),
                                np.array([a, -d * sa, d * ca]))
            parent = self.model.addJoint(parent, pin.JointModelRZ(), placement, f"joint{i+1}")
        a, d, alpha = _MDH[7]
        tool = np.eye(4)
        tool[:3, 3] = [a, -d*np.sin(alpha), d*np.cos(alpha)]
        tool[:3, :3] = pin.exp3(np.array([alpha, 0., 0.]))
        self.frame = self.model.addFrame(pin.Frame("ee", parent, 0, pin.SE3(tool @ X_EE), pin.OP_FRAME))
        self.data = self.model.createData()

    def pose(self, q):
        pin.framesForwardKinematics(self.model, self.data, q)
        return self.data.oMf[self.frame].homogeneous.copy()

    def residual_jacobian(self, q, target):
        """Derivative of log(T_target^-1 T(q)), in local SE(3) coordinates."""
        pin.computeJointJacobians(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        J = pin.getFrameJacobian(self.model, self.data, self.frame, pin.LOCAL)
        error = pin.SE3(target).inverse() * self.data.oMf[self.frame]
        return pin.log6(error).vector, pin.Jlog6(error) @ J
