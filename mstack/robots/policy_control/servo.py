"""Deterministic command-clock servo; no solver, IK, filtering, or clipping.

The owner synchronizes snapshot/install/step. Polynomial time advances by one
command tick, never by a delayed wall-clock interval (no catch-up jumps).
"""
from dataclasses import dataclass

import numpy as np

from mstack.robots.policy_control.trajectory import Plan, Settings


@dataclass(frozen=True)
class ScheduledPlan:
    start_tick: int
    plan: Plan
    sequence: int
    gripper: float


def state_at(plans, tick, dt):
    chosen = plans[0]
    for p in plans[1:]:
        if p.start_tick > tick:
            break
        chosen = p
    return chosen.plan.sample((tick-chosen.start_tick)*dt)


class CommonServo:
    def __init__(self, q, epoch, settings=Settings(), gripper=0.):
        self.s, self.epoch = settings, epoch
        rest = Plan.rest(q, settings)
        rest.validate(settings)
        self.plans = [ScheduledPlan(0, rest, -1, gripper)]
        self.tick = 0
        self.held = False
        self.reason = ""
        self.accepted = self.rejected = 0
        self.last_sequence = -1

    def snapshot(self):
        return self.tick, tuple(self.plans)

    def install(self, plan, start_tick, sequence, epoch, gripper):
        """Caller validates the full plan outside the servo lock first."""
        if (self.held or epoch != self.epoch or start_tick <= self.tick
                or sequence <= self.last_sequence
                or not isinstance(start_tick, int) or not np.isfinite(gripper)):
            self.rejected += 1
            return False
        initial = state_at(self.plans, start_tick, self.s.servo_dt)
        if np.max(abs(initial-plan.states[0])) > 1e-8:
            self.rejected += 1
            return False
        # A newer optimization may replace a not-yet-started plan at the same
        # knot. It cannot erase a plan whose initial state it depended on.
        self.plans = [p for p in self.plans if p.start_tick < start_tick]
        self.plans.append(ScheduledPlan(start_tick, plan, sequence, float(gripper)))
        self.accepted += 1
        self.last_sequence = sequence
        return True

    def hold(self, reason="requested"):
        self.held = True
        self.reason = reason
        # Cancel future activations. The active trajectory already ends at rest
        # within horizon*knot_dt; never zero the current velocity/acceleration.
        self.plans = [self.plans[0]]

    def step(self):
        self.tick += 1
        while len(self.plans) > 1 and self.plans[1].start_tick <= self.tick:
            self.plans.pop(0)
        p = self.plans[0]
        return p.plan.sample((self.tick-p.start_tick)*self.s.servo_dt), p.gripper

    @property
    def stopped(self):
        return (self.tick-self.plans[0].start_tick)*self.s.servo_dt >= self.plans[0].plan.duration
