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
                 tv_inner:LiftingLine,
                 tv_outer:LiftingLine,
                 bv:LiftingLine ,
                 cp_x:float,
                 cp_y:float,
                 cp_z:float):
        self.polar_path = polar_path
        self.r = r
        self.beta = beta
        self.n_azim = np.array([0,1,0])
        self.n_axial = np.array([1,0,0])
        self.n_tan = Rotation.from_euler("z",-beta,degrees=True).apply(self.n_azim)
        self.chord = chord
        self.is_prop = True
        self.V_i = np.zeros(3)
        # lifting lines
        self.tv_inner:LiftingLine = tv_inner
        self.tv_outer:LiftingLine = tv_outer
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
                              Vinf: np.ndarray,
                              Omega: float,
                              rho:float):
        V_omega = self.n_azim * Omega*self.r
        V = Vinf+V_i+V_omega
        V_norm = np.linalg.norm(V)

        phi=np.rad2deg(np.atan(V[0]/V[1]))
        # print(f"phi = {phi}")
        
        alpha = phi-self.beta
        # print(f"alpha = {alpha:.2f}")
        Cl = self.calculate_Cl(alpha)
        Cd = self.calculate_Cd(alpha)
        lift = 0.5*self.chord*rho*V_norm**2*Cl
        drag = 0.5*self.chord*rho*V_norm**2*Cd

        phi_rad = np.deg2rad(phi)
        F_azim = lift*np.sin(phi_rad)-drag*np.cos(phi_rad)
        F_axial = lift*np.cos(phi_rad)+drag*np.sin(phi_rad)

        gamma = 0.5*self.chord*V_norm*Cl
        return gamma,F_azim,F_axial
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
