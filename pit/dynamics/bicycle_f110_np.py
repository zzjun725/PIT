import numpy as np
from numba import njit
from scipy.integrate import solve_ivp
import math

X, Y, YAW, VX, VY, YAW_RATE, STEERING_ANGLE = 0, 1, 2, 3, 4, 5, 6
DRIVE_FORCE, STEER_SPEED = 0, 1
FRX, FFY, FRY = 0, 1, 2


############## Dynamic Models for [steer, velocity] control input ####################
@njit(cache=True)
def steer_pid(current_steer, desired_steer, sv_min, sv_max, dt):
    steer_diff = desired_steer - current_steer
    if np.fabs(steer_diff) > 1e-4:
        steer_v = steer_diff / dt
    else:
        steer_v = 0.0
    # steer_v = max(sv_min, min(sv_max, steer_v))
    return steer_v


@njit(cache=True)
def accel_pid(current_speed, desired_speed, switch_v, min_accel, max_accel, dt):
    vel_diff = desired_speed - current_speed
    accel = vel_diff / dt
    accel = min(max_accel, max(min_accel, accel))
    if current_speed > switch_v:
        accel = accel * switch_v / current_speed
    return accel


@njit(cache=True)
def njit_kinematic_st(x, u, lf, lr, sv_min, sv_max, steer_delay_time, a_min, a_max, accel_delay_time):
    sv = steer_pid(x[2], u[0], sv_min, sv_max, steer_delay_time)
    accel = accel_pid(x[4], u[1], a_min, a_max, accel_delay_time)

    lwb = lf + lr
    f = np.array([x[4]*np.cos(x[3]),
                  x[4]*np.sin(x[3]),
                  sv,
                  x[4] / lwb * np.tan(x[2]),
                  accel])
    return f


def kinematic_st(x, u, p):
    return njit_kinematic_st(x, u, p["lf"], p["lr"], p["sv_min"], p["sv_max"], p["steer_delay_time"], p["a_min"], p["a_max"], p["accel_delay_time"])

@njit(cache=True)
def njit_dynamic_st_pacejka(x, u, lf, lr, h, m, I, steer_delay_time, g, mu, B_f, C_f, D_f, E_f, B_r, C_r, D_r, E_r, sv_min, sv_max, v_switch, a_min, a_max, accel_delay_time):
    sv = steer_pid(x[2], u[0], sv_min, sv_max, steer_delay_time)
    accel = accel_pid(x[4], u[1], v_switch, a_min, a_max, accel_delay_time)

    lwb = lf + lr
    if abs(x[4]) < 1.0:
        # Simplified low-speed kinematic model
        f_ks = np.array([x[4] * np.cos(x[3]),
                         x[4] * np.sin(x[3]),
                         sv,
                         x[4] / lwb * np.tan(x[2]),
                         accel])
        f = np.hstack((f_ks, np.array([u[1] / lwb * np.tan(x[2]) + x[4] / (lwb * np.cos(x[2])**2) * u[0], 0])))
        return f

    # Compute slip angles and vertical tire forces
    alpha_f = -math.atan((x[5] + x[6] * lf) / x[4]) + x[2]
    alpha_r = -math.atan((x[5] - x[6] * lr) / x[4])
    F_zf = m * (-u[1] * h + g * lr) / (lr + lf)
    F_zr = m * (u[1] * h + g * lf) / (lr + lf)

    # Calculate lateral forces using Pacejka's magic formula
    F_yf = mu * F_zf * D_f * math.sin(C_f * math.atan(B_f * alpha_f - E_f * (B_f * alpha_f - math.atan(B_f * alpha_f))))
    F_yr = mu * F_zr * D_r * math.sin(C_r * math.atan(B_r * alpha_r - E_r * (B_r * alpha_r - math.atan(B_r * alpha_r))))

    f = np.array([x[4] * np.cos(x[3]) - x[5] * math.sin(x[3]),
                  x[4] * np.sin(x[3]) + x[5] * math.cos(x[3]),
                  sv,
                  x[6],
                  accel,
                  1 / m * (F_yr + F_yf) - x[4] * x[6],
                  1 / I * (-lr * F_yr + lf * F_yf * math.cos(x[2]))])
    return f

@njit(cache=True)
def njit_dynamic_st_linear(x, u, lf, lr, h, m, I, steer_delay_time, g, mu, C_Sf, C_Sr, sv_min, sv_max, v_switch, a_min, a_max, accel_delay_time):
    sv = steer_pid(x[2], u[0], sv_min, sv_max, steer_delay_time)
    accel = accel_pid(x[4], u[1], v_switch, a_min, a_max, accel_delay_time)

    lwb = lf + lr
    if abs(x[4]) < 1.0:
        # Simplified low-speed kinematic model
        f_ks = np.array([x[4] * np.cos(x[3]),
                         x[4] * np.sin(x[3]),
                         sv,
                         x[4] / lwb * np.tan(x[2]),
                         accel])
        f = np.hstack((f_ks, np.array([u[1] / lwb * np.tan(x[2]) + x[4] / (lwb * np.cos(x[2])**2) * u[0], 0])))
        return f

    # Compute slip angles and vertical tire forces
    alpha_f = -math.atan((x[5] + x[6] * lf) / x[4]) + x[2]
    alpha_r = -math.atan((x[5] - x[6] * lr) / x[4])
    F_zf = m * (-u[1] * h + g * lr) / (lr + lf)
    F_zr = m * (u[1] * h + g * lf) / (lr + lf)

    # Calculate lateral forces using linear tire model
    F_yf = mu * F_zf * C_Sf * alpha_f
    F_yr = mu * F_zr * C_Sr * alpha_r

    f = np.array([x[4] * np.cos(x[3]) - x[5] * math.sin(x[3]),
                  x[4] * np.sin(x[3]) + x[5] * math.cos(x[3]),
                  sv,
                  x[6],
                  accel,
                  1 / m * (F_yr + F_yf) - x[4] * x[6],
                  1 / I * (-lr * F_yr + lf * F_yf * math.cos(x[2]))])
    return f

# Wrapper functions that prepare parameters and choose the correct JIT function
def dynamic_st(x, u, p, tire_type):
    if tire_type == "pacejka":
        return njit_dynamic_st_pacejka(
            x, u, p["lf"], p["lr"], p["h"], p["m"], p["I"], p["steer_delay_time"], 9.81, p["mu"],
            p["B_f"], p["C_f"], p["D_f"], p["E_f"], p["B_r"], p["C_r"], p["D_r"], p["E_r"],
            p["sv_min"], p["sv_max"], p["v_switch"], p["a_min"], p["a_max"], p["accel_delay_time"]
        )
    elif tire_type == "linear":
        return njit_dynamic_st_linear(
            x, u, p["lf"], p["lr"], p["h"], p["m"], p["I"], p["steer_delay_time"], 9.81, p["mu"],
            p["C_Sf"], p["C_Sr"], p["sv_min"], p["sv_max"], p["v_switch"], p["a_min"], p["a_max"], p["accel_delay_time"]
        )

############## Dynamic Models for [steer, velocity] control input ####################


### Those two functions are not used. In practice, we use equivalents above with njit decorator ###
def kinematic_st_steer(x, u, p):
    """
    Single Track Kinematic Vehicle Dynamics.
        Args:
            x (numpy.ndarray (5, )): vehicle state vector
                x0: x position in global coordinates
                x1: y position in global coordinates
                x2: steering angle of front wheels
                x3: yaw angle
                x4: velocity in x direction

            u (numpy.ndarray (2, )): control input vector
                u0: steering angle
                u1: longitudinal acceleration
        Returns:
            f (numpy.ndarray): right hand side of differential equations
    """
    # pid for sv and accl. NOTE: can not modify u in-place! Maybe because u will be used repeatedly in solve_ivp？
    # u[0] = steer_pid(x[2], u[0], p["sv_min"], p["sv_max"], p["steer_delay_time"])
    sv = steer_pid(x[2], u[0], p["sv_min"], p["sv_max"], p["steer_delay_time"])
    accel = accel_pid(x[4], u[1], p["v_switch"], p["a_min"], p["a_max"], p["accel_delay_time"])

    lf = p["lf"]
    lr = p["lr"]
    lwb = lf + lr
    steer_delay_time = p["steer_delay_time"]
    f = np.array([x[4]*np.cos(x[3]),
         x[4]*np.sin(x[3]),
         # (u[0] - x[2]) / steer_delay_time,  # steer_delay_time = 0.2 in MAP
         sv,
         x[4] / lwb * np.tan(x[2]),
         accel])
    return f

def dynamic_st_steer(x, u, p, type):
    """
    Args:
        x (numpy.ndarray (5, )): vehicle state vector
            x0: x-position in a global coordinate system
            x1: y-position in a global coordinate system
            x2: steering angle of front wheels
            x3: yaw angle
            x4: velocity in x-direction
            x5: velocity in y direction
            x6: yaw rate

        u: (numpy.ndarray (2, )) control input vector
            u0: steering angle
            u1: velocity
        p:
        type:

    Returns:

    """
    # pid for sv and accl. NOTE: can not modify u in-place! Maybe because u will be used repeatedly in solve_ivp？
    # u[0] = steer_pid(x[2], u[0], p["sv_min"], p["sv_max"], p["steer_delay_time"])
    # u[1] = accl_constraints(x[4], u[1], p["v_switch"], p["a_max"], p["v_min"], p["v_max"])
    sv = steer_pid(x[2], u[0], p["sv_min"], p["sv_max"], p["steer_delay_time"])
    accel = accel_pid(x[4], u[1], p["v_switch"], p["a_min"], p["a_max"], p["accel_delay_time"])

    lf = p["lf"]
    lr = p["lr"]
    h = p["h"]
    m = p["m"]
    I = p["I"]
    steer_delay_time = p["steer_delay_time"]
    lwb = lf + lr

    ## In low speed, switch to kinematic model
    # mix expert model here
    if abs(x[4]) < 1.0:
        f_ks = kinematic_st_steer(x[:5], u, p)
        # u[1]/lwb*np.tan(x[2])+x[3]/(lwb*np.cos(x[2])**2)*u[0]
        f = np.hstack((f_ks, np.array([u[1]/lwb*np.tan(x[2])+x[4]/(lwb*np.cos(x[2])**2)*u[0], 0])))
        return f

    # set gravity constant
    g = 9.81  # [m/s^2]

    # create equivalent bicycle parameters
    mu = p["mu"]

    if type == "pacejka":
        B_f = p["B_f"]
        C_f = p["C_f"]
        D_f = p["D_f"]
        E_f = p["E_f"]
        B_r = p["B_r"]
        C_r = p["C_r"]
        D_r = p["D_r"]
        E_r = p["E_r"]
    elif type == "linear":
        C_Sf = p["C_Sf"]  # -p.tire.p_ky1/p.tire.p_dy1
        C_Sr = p["C_Sr"]  # -p.tire.p_ky1/p.tire.p_dy1


    # compute lateral tire slip angles
    alpha_f = -math.atan((x[5] + x[6] * lf) / x[4]) + x[2]
    alpha_r = -math.atan((x[5] - x[6] * lr) / x[4])

    # compute vertical tire forces
    F_zf = m * (-u[1] * h + g * lr) / (lr + lf)
    F_zr = m * (u[1] * h + g * lf) / (lr + lf)
    F_yf = F_yr = 0

    # combined slip lateral forces
    if type == "pacejka":
        F_yf = mu * F_zf * D_f * math.sin(
            C_f * math.atan(B_f * alpha_f - E_f * (B_f * alpha_f - math.atan(B_f * alpha_f))))
        F_yr = mu * F_zr * D_r * math.sin(
            C_r * math.atan(B_r * alpha_r - E_r * (B_r * alpha_r - math.atan(B_r * alpha_r))))
    elif type == "linear":
        F_yf = mu * F_zf * C_Sf * alpha_f
        F_yr = mu * F_zr * C_Sr * alpha_r

    f = [x[4] * math.cos(x[3]) - x[5] * math.sin(x[3]),
         x[4] * math.sin(x[3]) + x[5] * math.cos(x[3]),
         # (u[0] - x[2]) / steer_delay_time,  # steer_delay_time = 0.2 in MAP
         sv,
         x[6],
         accel,
         1 / m * (F_yr + F_yf) - x[4] * x[6],
         1 / I * (-lr * F_yr + lf * F_yf * math.cos(x[2]))]
    return f


############## Dynamic Models for [steer, velocity] control input ####################

dynamic_f110_idx = {
    'X': 0, 'Y': 1, 'DELTA': 2, 'YAW': 3, 'VX': 4, 'VY': 5, 'YAWRATE': 6
}


dynamic_idx = {
    'X': 0, 'Y': 1, 'DELTA': 2, 'VX': 3, 'YAW': 4, 'YAWRATE': 5, 'BETA': 6
}

### Those two functions are not used. In practice, we use equivalents above with njit decorator ###
class VehicleDynamicModel:
    def __init__(self, model: str, params: dict):
        self.model = model
        self.params = params
        self.tire_model = self.params["tire_model"]
        if self.model == 'dynamic_st_v1':
            self.idx = dynamic_f110_idx

    def dynamics(self, t, s, u):
        # This wrapper adapts the vehicle dynamics to the solve_ivp format
        # We assume 'u' is constant over each interval `dt`
        if self.model == "kinematic_st_v1":
            # return kinematic_st_steer(s, u, self.params)
            return kinematic_st(s, u, self.params)
        if self.model == "dynamic_st_v1":
            # return dynamic_st_steer(s, u, self.params, self.tire_model)
            return dynamic_st(s, u, self.params, self.tire_model)

    def forward_trajectory(self, s0, u_list: np.ndarray, dt):
        if isinstance(s0, dict):
            s0 = np.array([s0['x'], s0['y'], s0['steer'], s0['yaw'], s0['vx'], s0['vy'], s0['yaw_rate']])
        # NOTE: self-implementation Explicit Euler or RK4 are not stable for small timestep.
        s_list = [s0]
        t_span = [0, dt]
        for u in u_list:
            # Using solve_ivp with method 'RK45'
            sol = solve_ivp(self.dynamics, t_span, s_list[-1], args=(u,), method='RK45', rtol=1e-6, atol=1e-9)
            if sol.success:
                s_list.append(sol.y[:, -1])
            else:
                raise RuntimeError("Integration failed")
        return s_list

    def forward_single_step(self, s0, u, dt):
        if isinstance(s0, dict):
            s0 = np.array([s0['x'], s0['y'], s0['steer'], s0['yaw'], s0['vx'], s0['vy'], s0['yaw_rate']])
        sol = solve_ivp(self.dynamics, [0, dt], s0, args=(u,), method='RK45', rtol=1e-6, atol=1e-9)
        next_state = sol.y[:, -1]
        x, y, steer, yaw, vx, vy, yaw_rate = next_state
        return x, y, steer, yaw, vx, vy, yaw_rate
