#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
shipClarke83.py:  

   Class for a generic ship parametrized using the main dimensions L, B, and T.
   The ship model is based on the linear maneuvering coefficients by 
   Clarke (1983).  
       
   shipClarke83()                           
       Step input, rudder angle    
       
   shipClarke83('headingAutopilot',psi_d,L,B,T,Cb,V_c,beta_c,tau_X)
        psi_d: desired yaw angle (deg)
        L: ship length (m)
        B: ship beam (m)
        T: ship draft (m)
        Cb: block coefficient (-)
        V_c: current speed (m/s)
        beta_c: current direction (deg)
        tau_X: surge force, pilot input (N)                    

Methods:
        
    [nu,u_actual] = dynamics(eta,nu,u_actual,u_control,sampleTime ) returns 
        nu[k+1] and u_actual[k+1] using Euler's method랴ㅜㅇ . The control input is:

        u_control = delta_r (rad) is for the ship rudder.

    u = headingAutopilot(eta,nu,sampleTime) 
        PID controller for automatic heading control based on pole placement.
       
    u = stepInput(t) generates rudder step inputs.   
       
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
class shipClarke83:
    """
    shipClarke83()
        Rudder angle step inputs
    shipClarke83('headingAutopilot', psi_d, L, B, T, Cb, V_c, beta_c, tau_X)
        Heading autopilot
        
    Inputs:
        psi_d: desired yaw angle (deg)
        L: ship length (m)
        B: ship beam (m)
        T: ship draft (m)
        Cb: block coefficient (-)
        V_c: current speed (m/s)
        beta_c: current direction (deg)
        tau_X: surge force, pilot input (N) 
    """

    def __init__(
        self,
        controlSystem="stepInput",
        r = 0,
        L = 50.0,
        B = 7.0,
        T = 5.0,
        Cb = 0.7,
        V_current = 0,
        beta_current = 0,
        tau_X = 1e5,
    ):
        
        # Constants
        D2R = math.pi / 180     # deg2rad
        self.rho = 1026         # density of water (kg/m^3)

        self.controlDescription = "Step input for delta_r"
        controlSystem = "stepInput"

        self.V_c = V_current
        self.beta_c = beta_current * D2R
        self.controlMode = controlSystem

        # Initialize the ship model
        self.name = "Linear ship maneuvering model (see 'shipClarke83.py' for more details)"
        self.L = L  # length (m)
        self.B = B  # beam (m)
        self.T = T  # Draft (m)
        self.Cb = Cb  # block coefficient
        self.Lambda = 0.7  # rudder aspect ratio:  Lambda = b**2 / AR
        self.tau_X = tau_X  # surge force (N), pilot input
        self.deltaMax = 30  # max rudder angle (deg)
        self.T_delta = 1.0  # rudder time constants (s)
        self.nu = np.array([2, 0, 0], float)  # velocity vector
        self.u_actual = np.array([0], float)  # control input vector

        if self.L > 100:
            self.R66 = 0.27 * self.L  # approx. radius of gyration in yaw (m)
        else:
            self.R66 = 0.25 * self.L

        self.controls = ["Rudder angle (deg)"]
        self.dimU = len(self.controls)

        # controller parameters m, d and k
        U0 = 3  # cruise speed
        [M, N] = clarke83(U0, self.L, self.B, self.T, self.Cb, self.R66, 0, self.L)

        # Rudder yaw moment coefficient (Fossen 2021, Chapter 9.5.1)
        b = 0.7 * self.T  # rudder height
        AR = b ** 2 / self.Lambda  # aspect ratio: Lamdba = b**2/AR
        CN = 6.13 * self.Lambda / (self.Lambda + 2.25)  # normal coefficient
        t_R = 1 - 0.28 * self.Cb - 0.55
        a_H = 0.4
        x_R = -0.45 * self.L
        x_H = -1.0 * self.L

    def dynamics(self, eta, nu, u_actual, sampleTime):
        """
        nu = dynamics(eta,nu,u_actual,sampleTime) integrates
        the ship equations of motion using Euler's method.
        """

        if len(u_actual) == 2:
            if u_actual[1] == 'deg':
                u_actual[0] = np.deg2rad(u_actual[0])
                u_actual[1] = 'rad'

        # Current velocities
        u_c = self.V_c * math.cos(self.beta_c - eta[2])  # current surge velocity
        v_c = self.V_c * math.sin(self.beta_c - eta[2])  # current sway velocity

        nu_c = np.array([u_c, v_c, 0], float)  # current velocity vector
        nu_r = nu - nu_c  # relative velocity vector

        U_r = math.sqrt(nu_r[0] ** 2 + nu_r[1] ** 2)  # relative speed

        # Rudder command and actual rudder angle
        delta = u_actual[0]

        # Rudder forces and moment (Fossen 2021, Chapter 9.5.1)
        b = 0.7 * self.T  # rudder height
        AR = b ** 2 / self.Lambda  # aspect ratio: Lamdba = b**2/AR
        CN = 6.13 * self.Lambda / (self.Lambda + 2.25)  # normal coefficient
        t_R = 1 - 0.28 * self.Cb - 0.55
        a_H = 0.4
        x_R = -0.45 * self.L
        x_H = -1.0 * self.L

        Xdd = -0.5 * (1 - t_R) * self.rho * U_r ** 2 * AR * CN
        Yd = -0.25 * (1 + a_H) * self.rho * U_r ** 2 * AR * CN
        Nd = -0.25 * (x_R + a_H * x_H) * self.rho * U_r ** 2 * AR * CN

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
        [M, N] = clarke83(U_r, self.L, self.B, self.T, self.Cb, self.R66, xg, T_surge)
        Minv = np.linalg.inv(M)
        nu3 = np.array([nu_r[0], nu_r[1], nu_r[2]])
        nu3_dot = np.matmul(Minv, tau - np.matmul(N, nu3))

        # Rudder angle saturation
        if abs(delta) >= self.deltaMax * math.pi / 180:
            delta = np.sign(delta) * self.deltaMax * math.pi / 180

        nu = nu + sampleTime * nu3_dot

        return nu