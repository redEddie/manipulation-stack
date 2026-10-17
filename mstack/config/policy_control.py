"""Initial policy TCP speed caps, selected from recorded chunk replay.

See docs/policy-qp.md for the evidence and its limits. These are operational
starting values, not a learned optimum or a contact-safety certification.
"""

POLICY_EE_LINEAR_VELOCITY = .4  # m/s at the configured TCP origin
POLICY_EE_ANGULAR_VELOCITY = .8  # rad/s, angular velocity norm
