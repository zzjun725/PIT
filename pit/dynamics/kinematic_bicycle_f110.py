from . import Dynamics

import torch
from torch import nn


# Reference: https://github.com/ETH-PBL/MAP-Controller/blob/main/steering_lookup/src/LUT_Generation/dynamics/vehicle_dynamics_stown.py

class UncertaintyModel(nn.Module):
    """
    Out put the diag covariance matrix.

    input: history state:(x, y, v, ...)*state_steps
    output: diag of covariance matrix
    """
    def __init__(self, state_dim, state_steps):
        super().__init__()



class DynamicDistribution(nn.Module):
    def __init__(self, dynamic):
        super().__init__()
        pass


class KinematicBicycle_Steer(Dynamics, nn.Module):
    """
    """

    def __init__(self, lwb, st_delay, accel_delay) -> None:
        super().__init__()
        self.lwb = torch.nn.Parameter(torch.tensor(lwb, dtype=torch.float32))
        self.steer_delay_time = torch.nn.Parameter(torch.tensor(st_delay, dtype=torch.float32))
        self.accel_delay_time = torch.nn.Parameter(torch.tensor(accel_delay, dtype=torch.float32))
        # real U: steer, velocity
        # model U: steer_vel, accel.
        # Define indexes for states and control inputs
        self.X = 0
        self.Y = 1
        self.DELTA = 2
        self.YAW = 3
        self.V = 4
        self.STEER = 0
        self.VEL = 1

    def forward(self, states, control_inputs):
        """ Get the evaluated ODEs of the state at this point

        Args:
            states (): Shape of (B, 5) or (5)
            control_inputs (): Shape of (B, 2) or (2)
        """
        # print(states.shape, control_inputs.shape)
        batch_mode = True if len(states.shape) == 2 else False
        X, Y, DELTA, YAW, V = 0, 1, 2, 3, 4
        STEER, VEL = 0, 1
        diff = torch.zeros_like(states)

        if batch_mode:
            desired_vel = control_inputs[:, VEL]
            current_vel = states[:, V]
            accel = (desired_vel - current_vel) / self.accel_delay_time
            diff[:, X] = (states[:, V] + self.lwb / 2) * torch.cos(states[:, YAW])
            diff[:, Y] = (states[:, V] + self.lwb / 2) * torch.sin(states[:, YAW])
            diff[:, DELTA] = (control_inputs[:, STEER] - states[:, DELTA]) / self.steer_delay_time
            # diff[:, YAW] = (states[:, V] * torch.tan(control_inputs[:, STEER])) / self.lwb
            diff[:, YAW] = (states[:, V] * torch.tan(control_inputs[:, STEER])) / self.lwb
            diff[:, V] = accel
        else:
            desired_vel = control_inputs[VEL]
            current_vel = states[V]
            accel = (desired_vel - current_vel) / self.accel_delay_time
            diff[X] = states[V] * torch.cos(states[YAW])
            diff[Y] = states[V] * torch.sin(states[YAW])
            diff[DELTA] = (control_inputs[STEER] - states[DELTA]) / self.steer_delay_time
            # diff[YAW] = (states[V] * torch.tan(control_inputs[STEER])) / self.lwb
            diff[YAW] = (states[V] * torch.tan(control_inputs[STEER])) / self.lwb
            diff[V] = accel
        return diff
