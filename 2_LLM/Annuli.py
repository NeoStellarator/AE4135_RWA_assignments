from pathlib import Path
from typing import Dict, List, Callable, Tuple
import numpy as np
from LiftingLine import LiftingLine
from scipy.spatial.transform import Rotation
from globals import main_dir,data_dir



class Annuli:
    def __init__(self, polar_path:Path|str,
                 r:float,
                 chord:float,
                 beta: float,
                 Vinf:np.ndarray,
                 Omega:float,
                 tv_inner:LiftingLine,
                 tv_outer:LiftingLine,
                 bv:LiftingLine ,
                 cp_x:float,
                 cp_y:float,
                 cp_z:float):
        self.polar_path = polar_path
        self.r = r
        self.beta = beta
        self.n_azim = np.array([0,0,-1])
        self.n_axial = np.array([1,0,0])
        self.n_tan = Rotation.from_euler("y",beta,degrees=True).apply(self.n_azim)
        self.chord = chord
        self.is_prop = True

        self.Vinf = Vinf
        self.Omega = Omega
        self.phi=0
        self.alpha=0


        self.V_i = np.zeros(3)
        # lifting lines
        self.tv_inner:LiftingLine = tv_inner
        self.tv_outer:LiftingLine = tv_outer
        
        # Wake
        self.wake:LiftingLine = []
        if Omega == 0:
            n_wake = 1
            period  = 3/self.Vinf[0]
        else:
            n_wake = 10
            period = 2*np.pi/self.Omega

        inner_wake_position = self.tv_inner.p1
        outer_wake_position = self.tv_outer.p2
        angles = np.linspace(0,-2*np.pi,n_wake+1).reshape((n_wake+1,1))
        rotations = Rotation.from_euler('x', angles,degrees=False)  
        
        inner_rotated_positions = rotations.apply(inner_wake_position)  
        outer_rotated_positions = rotations.apply(outer_wake_position)
        
        axial_offsets = np.linspace(np.zeros(3), self.Vinf*period,n_wake+1)
        
        self.inner_wake_points = inner_rotated_positions+axial_offsets
        self.inner_wake_lines:List[LiftingLine] = []

        self.outer_wake_points = outer_rotated_positions+axial_offsets
        self.outer_wake_lines:List[LiftingLine] = []

        for i in range(n_wake):
            end_inner = self.inner_wake_points[i]
            start_inner = self.inner_wake_points[i+1]
            ll_inner = LiftingLine(start_inner[0],start_inner[1],start_inner[2],end_inner[0],end_inner[1],end_inner[2])
            self.inner_wake_lines.append(ll_inner)

            start_outer = self.outer_wake_points[i]
            end_outer = self.outer_wake_points[i+1]
            ll_outer = LiftingLine(start_outer[0],start_outer[1],start_outer[2],end_outer[0],end_outer[1],end_outer[2])
            self.outer_wake_lines.append(ll_outer)


        self.bv:LiftingLine = bv
        self.cp_x = cp_x
        self.cp_y = cp_y
        self.cp_z = cp_z
        # read & store polar data
        self.polar_data = self._load_polar_data(polar_path)
    
    def calculate_Cl(self, alpha:np.ndarray|float) -> np.ndarray|float:
        """Method to compute Cl at a given angle of attack"""
        if self.is_prop:
            return np.interp(alpha, 
                             self.polar_data["alpha"], 
                             self.polar_data["Cl"])
        else:
            return -np.interp(-alpha, 
                             self.polar_data["alpha"], 
                             self.polar_data["Cl"])
    
    def calculate_Cd(self, alpha:np.ndarray|float) -> np.ndarray|float:
        """Method to compute Cd at a given angle of attack"""
        if self.is_prop:
            return np.interp(alpha, 
                             self.polar_data["alpha"], 
                             self.polar_data["Cd"])
        else:
            return np.interp(-alpha, 
                             self.polar_data["alpha"], 
                             self.polar_data["Cd"])

    def calculate_performance(self, 
                              V_i:np.ndarray,
                              rho:float):
        if np.any(np.isnan(V_i)):
            V_i = np.zeros(3)
        V_omega = -self.n_azim * self.Omega*self.r
        self.V = self.Vinf+V_i+V_omega
        V_norm = np.linalg.norm(self.V)

        # print(V)
        if self.V[2] == 0:
            self.phi = 90    
        else:
            self.phi=np.rad2deg(np.atan2(self.V[0],self.V[2]))
        # print(f"phi = {phi}")
        if self.is_prop:
            self.alpha = self.beta-self.phi
        else:
            self.alpha = self.phi-self.beta

        # print(alpha)
        # print(f"alpha = {alpha:.2f}")
        Cl = self.calculate_Cl(self.alpha)
        Cd = self.calculate_Cd(self.alpha)
        lift = 0.5*self.chord*rho*V_norm**2*Cl
        drag = 0.5*self.chord*rho*V_norm**2*Cd

        phi_rad = np.deg2rad(self.phi)
        if self.is_prop:
            F_azim = lift*np.sin(phi_rad)-drag*np.cos(phi_rad)
            F_axial = lift*np.cos(phi_rad)+drag*np.sin(phi_rad)
        else:
            F_azim = lift*np.sin(phi_rad)+drag*np.cos(phi_rad)
            F_axial = lift*np.cos(phi_rad)-drag*np.sin(phi_rad)

        gamma = 0.5*self.chord*V_norm*Cl
        return gamma,F_azim,F_axial,Cl,Cd
        # n_phi = Rotation.from_euler("z",phi,degrees=True).apply(self.n_axial)
        # print(n_phi)
        # local_coefficients = np.array([Cl,Cd,0])
        # global_coefficients = local_coefficients * n_phi
        # print(global_coefficients)
        # print(f"Cl = {Cl:.2f}")
        # print(f"Cd = {Cd:.2f}")
    def _load_polar_data(self, polar_path:Path|str) -> Dict[str, np.ndarray]:
        """Function to read the polar data"""

        polar_txt = np.loadtxt(polar_path,skiprows=2)
        
        polar_data = {}
        polar_data["alpha"] = polar_txt[:,0]
        polar_data["Cl"]    = polar_txt[:,1]
        polar_data["Cd"]    = polar_txt[:,2]
        return polar_data
    
if __name__ == "__main__":
    ann = Annuli(polar_path=data_dir.joinpath("ARAD8pct_polar.txt"),
                 chord=1,
                 r = 1,
                 beta = 45)
    Vinf = np.array([1,0,0])
    V_i = np.array([0,0,0])
    Omega = 1
    gamma,F_azim,F_axial=ann.calculate_performance(V_i=V_i,Vinf=Vinf,Omega = Omega,rho=1.225)
    print(f"gamma = {gamma:.2f}")
    print(f"F_azim = {F_azim:.2f}")
    print(f"F_axial = {F_axial:.2f}")
