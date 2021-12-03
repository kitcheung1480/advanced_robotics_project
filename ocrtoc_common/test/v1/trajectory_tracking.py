#! /usr/bin/env python

import pdb
import numpy as np
import math
from test_control import *

from numpy.linalg import pinv
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

# from test_fk_ik import get_fk
# from test_get_jacobian import get_jacobian

class DesiredTraj:
    def __init__(self, timespan, delta_t, r=None, num=None, shift=None):
        self.x, self.y, self.z, self.theta, self.phi, self.psi = np.zeros(timespan), np.zeros(timespan), np.zeros(
            timespan), np.zeros(timespan), np.zeros(timespan), np.zeros(timespan)
        self.xt, self.yt, self.zt, self.thetat, self.phit, self.psit = 0.125, 0.125, 0.5, 0, 0, 0

        self.r = r
        self.num = num
        self.shift = shift
        self.timespan = timespan
        self.delta_t = delta_t

    def get_instant_pose_circle(self):
        vy = 0
        vz = 0.5
        vth = 7
        delta_y = (vy * math.cos(self.thetat) - vz * math.sin(self.thetat)) * self.delta_t
        delta_z = (vy * math.sin(self.thetat) + vz * math.cos(self.thetat)) * self.delta_t
        delta_th = vth * self.delta_t
        self.yt += delta_y
        self.zt += delta_z
        self.thetat += delta_th

        # traj_t = np.array([self.xt, self.yt, self.zt, 0, 0, 0]) # end effector point upward
        traj_t = np.array([self.xt, self.yt, self.zt, 0, math.pi, 0]) # end effector point downward

        trajd_t = np.array([0, 0, 0, 0, 0, 0])
        return traj_t, trajd_t


class KineControl:
    def __init__(self, desired_traj):
        self.num_joints = 7
        self.dsr_traj = desired_traj

        # self.q0 = np.array([0, math.pi / 3, 0, math.pi / 6, 0, 0, 0])
        self.q0 = np.array([0, 0, 0, -math.pi/2, 0, math.pi/2, 0])
        self.q_dot0 = np.zeros(7)
        
        self.error = None
        self.joint_list = None
        self.joint_vel_list = None
        self.manipulability_list = None

    def dh(self, q):
        a = np.array([0, 0, 0.0825, -0.0825, 0, 0.088, 0, 0])
        d = np.array([0.333, 0, 0.316, 0, 0.384, 0, 0, 0.107])
        alpha = np.array([-math.pi/2, math.pi/2, math.pi/2, -math.pi/2, math.pi/2, math.pi/2, 0, 0])
        theta = np.array([q[0], q[1], q[2], q[3], q[4], q[5], q[6], 0])
        T = np.empty((self.num_joints + 2, 4, 4))
        T[0] = np.eye(4)
        for i in range(self.num_joints + 1):
            sin_alpha = np.sin(alpha[i])
            cos_alpha = np.cos(alpha[i])
            sin_theta = np.sin(theta[i])
            cos_theta = np.cos(theta[i])
            T[i+1] = T[i] @ np.round(np.array([
                [cos_theta, -cos_alpha * sin_theta,  sin_alpha*sin_theta, a[i]*cos_theta],
                [sin_theta,  cos_alpha * cos_theta, -sin_alpha*cos_theta, a[i]*sin_theta],
                [        0,              sin_alpha,            cos_alpha,           d[i]],
                [        0,                      0,                    0,              1]
            ]), 9)
        return T
        
    def fk(self, q):
        """
        Need to be implemented
        You can use the 'get_fk' function I provided to verify your results, but do not use it for your project.
        """
        offset = np.array([-0.42, 0, 0])
        M = self.dh(q)[self.num_joints + 1]
        pose_value = np.hstack((M[0:3, 3] + offset, R.from_matrix(M[0:3,0:3]).as_rotvec()))
        # fk = get_fk(q)
        # fk = np.hstack((fk[0:3], R.from_quat(fk[3:7]).as_rotvec()))
        # np.testing.assert_allclose(pose_value, fk, atol=1e-6)
        return pose_value

    def jacobian(self, q):
        """
        Need to be implemented
        You can use the 'get_jacobian' function I provided to verify your results, but do not use it for your project.
        """
        T = self.dh(q)
        J = np.zeros((6, self.num_joints))
        for i in range(self.num_joints):
            J[0:3, i] = np.cross(T[i, 0:3, 2], T[self.num_joints+1, 0:3, 3] - T[i, 0:3, 3])
            J[3:6, i] = T[i, 0:3, 2]
        # np.testing.assert_allclose(J, np.round(get_jacobian(q), 9), atol=1e-6)
        return J

    def control(self):
        """
        Need to be implemented
        """
        dt = self.dsr_traj.delta_t
        ts = self.dsr_traj.timespan

        self.error = []
        self.joint_list = []
        self.joint_vel_list = []
        self.manipulability_list = []
        joint_limits = np.array([
            [-166.0, 166.0],
            [-101.0, 101.0],
            [-166.0, 166.0],
            [-176.0,  -4.0],
            [-166.0, 166.0],
            [  -1.0, 215.0],
            [-166.0, 166.0]
        ]) / 180.0 * math.pi
        '''
        test = np.linspace(joint_limits[:,0], joint_limits[:,1], 500, True).reshape((-1,))
        print(test)
        execute_joint_traj_goal(test)
        '''
        joint_centre = np.mean(joint_limits, axis=1)
        K = 200 # 300
        k0 = -100 # 0
        for t in np.arange(0, ts, dt).reshape(-1):
            print('=' * 80)

            x_dsr, xd_dsr = self.dsr_traj.get_instant_pose_circle()
            
            self.q0 = np.clip(self.q0, joint_limits[:,0], joint_limits[:,1])
            self.joint_list.append(self.q0)
            
            x_cur = self.fk(self.q0)
            J = self.jacobian(self.q0)
            self.manipulability_list.append(np.sqrt(np.linalg.det(J @ J.T)))
            error = np.hstack((
                x_dsr[0:3] - x_cur[0:3],
                R.from_matrix(
                   R.from_rotvec(x_dsr[3:6]).as_matrix() @ R.from_rotvec(x_cur[3:6]).as_matrix().T
                ).as_rotvec()
            ))
            self.error.append(np.hstack((error[0:3], R.from_rotvec(error[3:6]).as_euler("ZYX"))))
            
            J_inv = pinv(J)
            self.q_dot0 = J_inv @ (K * error + xd_dsr)
            self.joint_vel_list.append(self.q_dot0)
            self.q_dot0 += (np.eye(7) - J_inv @ J) @ (k0 * (self.q0 - joint_centre))
            self.q0 = self.q0 + self.q_dot0 * dt
            
        self.error = np.array(self.error)
        self.joint_list = np.array(self.joint_list)
        self.joint_vel_list = np.array(self.joint_vel_list)
        self.manipulability_list = np.array(self.manipulability_list)
                
        # convert the whole trajectory with a shape of [ts/dt, 7]
        # into a 1-dim array [ts/dt * 7, ] by using '.reshape((-1,))'
        joint_goal_lst = self.joint_list.reshape((-1,))

        execute_joint_traj_goal(joint_goal_lst)

    def traj_vis(self):
        """
        Need to be implemented
        Plot and save the following figures:
        1. tracking_error(time, e_r)
        2. joint_angles(time, q_r)
        3. joint_velocities(time, qd_r)
        4. manipulability(time,W)
        """
        time = np.arange(0, self.dsr_traj.timespan, self.dsr_traj.delta_t)
        plt.cla()
        ax1 = plt.subplot(2, 1, 1)
        ax2 = plt.subplot(2, 1, 2)
        for i in range(3):
            ax1.plot(time, self.error[:,i], color="rgb"[i], label=["x", "y", "z"][i], alpha=0.3)
        for i in range(3):
            ax2.plot(time, self.error[:,i+3], color="rgb"[i], label=["yaw", "pitch", "roll"][i], alpha=0.3)
        ax1.legend()
        ax2.legend()
        plt.savefig("tracking_error")
        plt.delaxes(ax1)
        plt.delaxes(ax2)
        
        plt.cla()
        for i in range(self.num_joints):
            plt.plot(time, self.joint_list[:,i], color="rgbcmyk"[i], label=f"q{i}", alpha=0.5)
        plt.legend()
        plt.savefig("joint_angles")
        
        plt.cla()
        for i in range(self.num_joints):
            plt.plot(time, self.joint_vel_list[:,i], color="rgbcmyk"[i], label=f"q{i}", alpha=0.5)
        plt.legend()
        plt.savefig("joint_velocities")
        
        plt.cla()
        plt.plot(time, self.manipulability_list, color="k", label="manipulability")
        plt.legend()
        plt.savefig("manipulability")
        
if __name__ == '__main__':
    # You can define any trajectory you like, there is no limit.
    # If you define a new trajectory, the visualization script 'test_dsr_traj_vis.py' need to modify correspondingly.
    DsrTraj = DesiredTraj(timespan=3, delta_t=0.005, r=1, num=10, shift=10)

    kinecontroller = KineControl(DsrTraj)
    kinecontroller.control()
    kinecontroller.traj_vis()
