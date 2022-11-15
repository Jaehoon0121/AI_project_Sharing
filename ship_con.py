#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""                 
References: 

    D. Clarke, P. Gedling and G. Hine (1983). The Application of Manoeuvring 
        Criteria in Hull Design using Linear Theory. Transactions of the Royal 
        Institution of Naval Architects (RINA), 125, pp. 45-68.
    T. I. Fossen (2021). Handbook of Marine Craft Hydrodynamics and Motion 
         Control. 2nd. Edition, Wiley. URL: www.fossen.biz/wiley            

Author:     Thor I. Fossen
"""
import numpy as np
import math
from ShipModellib.models import clarke83

# Class Vehicle
class ship_con:
    def __init__(
        self,
        L = 50.0,
        B = 7.0,
        T = 5.0,
        Cb = 0.7,
        tau_X = 1e5,
    ):
        
        # Constants
        self.rho = 1026         # density of water (kg/m^3)

        # Initialize the ship model
        self.name = "Linear ship maneuvering model (see 'shipClarke83.py' for more details)"
        self.L = L  # length (m)
        self.B = B  # beam (m)
        self.T = T  # Draft (m)
        self.Cb = Cb  # block coefficient
        self.Lambda = 0.7  # rudder aspect ratio:  Lambda = b**2 / AR
        self.tau_X = tau_X  # surge force (N), pilot input
        self.deltaMax = 30  # max rudder angle (deg)
        self.T_delta = 5.0  # rudder time constants (s)
        self.nu = np.array([2, 0, 0], float)  # velocity vector
        self.u_actual = np.array([0], float)  # control input vector

        if self.L > 100:
            self.R66 = 0.27 * self.L  # approx. radius of gyration in yaw (m)
        else:
            self.R66 = 0.25 * self.L

    def dynamics(self, nu, u_actual, u_control, sampleTime):

        if len(u_control) == 3:
            self.tau_X = u_control[0]
            if u_control[2] == 'deg':
                u_control[1] = np.deg2rad(u_control[1])
                u_control[2] = 'rad'

        vel = np.array(nu)
        s = np.sqrt(vel.dot(vel))

        # Rudder command and actual rudder angle
        delta_c = u_control[1]
        delta = u_actual[0]

        # Rudder forces and moment (Fossen 2021, Chapter 9.5.1)
        b = 0.7 * self.T  # rudder height
        AR = b ** 2 / self.Lambda  # aspect ratio: Lamdba = b**2/AR
        CN = 6.13 * self.Lambda / (self.Lambda + 2.25)  # normal coefficient
        t_R = 1 - 0.28 * self.Cb - 0.55
        a_H = 0.4
        x_R = -0.45 * self.L
        x_H = -1.0 * self.L

        Xdd = -0.5 * (1 - t_R) * self.rho * s ** 2 * AR * CN
        Yd = -0.25 * (1 + a_H) * self.rho * s ** 2 * AR * CN
        Nd = -0.25 * (x_R + a_H * x_H) * self.rho * s ** 2 * AR * CN

        # Control forces and moment
        delta_R = -delta  # physical rudder angle (rad)
        T = self.tau_X  # thrust (N)
        t_deduction = 0.1  # thrust deduction number
        tau1 = (1 - t_deduction) * T - Xdd * math.sin(delta_R) ** 2
        tau2 = -Yd * math.sin(2 * delta_R)
        tau6 = -Nd * math.sin(2 * delta_R)
        tau = np.array([tau1, tau2, tau6], float)

        # Linear maneuvering model
        T_surge = self.L  # approx. time constant in surge (s)
        xg = 0  # approx. x-coordinate, CG (m)

        # 3-DOF ship model
        [M, N] = clarke83(s, self.L, self.B, self.T, self.Cb, self.R66, xg, T_surge)
        Minv = np.linalg.inv(M)
        nu3_dot = np.matmul(Minv, tau - np.matmul(N, vel))

        # Rudder angle saturation
        if abs(delta) >= self.deltaMax * math.pi / 180:
            delta = np.sign(delta) * self.deltaMax * math.pi / 180

        # Rudder dynamics
        delta_dot = (delta_c - delta) / self.T_delta

        # Forward Euler integration [k+1]
        nu = nu + sampleTime * nu3_dot
        delta = delta + sampleTime * delta_dot

        u_actual = np.array([delta], float)

        return nu, u_actual 