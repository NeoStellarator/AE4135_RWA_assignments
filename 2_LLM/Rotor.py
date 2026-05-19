from typing import Callable, List, Literal
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from LiftingLine import LiftingLine
from Annuli import Annuli
from globals import main_dir,data_dir
class Rotor:
    def __init__(self,# Geometry
        B : int,
        R : float,
        r_R_H : float,
        c_R_func : Callable[[float|np.ndarray], float|np.ndarray],
        twst_func : Callable[[float|np.ndarray], float|np.ndarray],
        pitch : float,
        polar_path : Path|str,
        # Operating condition
        Omega :float,
        Vinf : np.ndarray,
        rho : float,
        # Other parameters
        n_elem:int=100,
        dist_elem:Literal['uniform','cosine']='uniform',
        isPropeller:bool=True):
        # Rotor geometry
        self.B = B                    
        self.R = R
        self.r_R_H = r_R_H            
        self.c_R_func = c_R_func      
        self.twst_func = twst_func    
        self.pitch = pitch
        self.polar_path = polar_path

        self.isPropeller = isPropeller

        # Operating Conditions                 
        self.Vinf = Vinf
        self.Omega  = Omega
        self.rho  = rho

        # Discretizaiton scheme
        self.n_elem = n_elem
        self.dist_elem = dist_elem
        self.annuli :List[Annuli] = []
        if dist_elem == "uniform":
            r_R_trailing = np.linspace(r_R_H, 1, n_elem)
            r_trailing = r_R_trailing*self.R

            c_trailing = c_R_func(r_R_trailing)*self.R
            beta_trailing = np.deg2rad(self.pitch+twst_func(r_R_trailing))

            r_R_bound = (r_R_trailing[1:]+r_R_trailing[:-1])/2
            r_bound = r_R_bound*self.R
            c_bound = c_R_func(r_R_bound)*self.R
            beta_bound = np.deg2rad(self.pitch+twst_func(r_bound))
            for idx in range(len(r_bound)):
                tv_inner_x1 = c_trailing[idx]/4*np.cos(beta_trailing[idx])
                tv_inner_y1 = c_trailing[idx]/4*np.sin(beta_trailing[idx])
                tv_inner_z1 = r_trailing[idx]
                tv_inner_x2 = c_trailing[idx]*np.cos(beta_trailing[idx])
                tv_inner_y2 = c_trailing[idx]*np.sin(beta_trailing[idx])
                tv_inner_z2 = r_trailing[idx]
                tv_inner = LiftingLine(tv_inner_x1,tv_inner_y1,tv_inner_z1,tv_inner_x2,tv_inner_y2,tv_inner_z2)

                tv_outer_x1 = c_trailing[idx+1]/4*np.cos(beta_trailing[idx+1])
                tv_outer_y1 = c_trailing[idx+1]/4*np.sin(beta_trailing[idx+1])
                tv_outer_z1 = r_trailing[idx+1]
                tv_outer_x2 = c_trailing[idx+1]*np.cos(beta_trailing[idx+1])
                tv_outer_y2 = c_trailing[idx+1]*np.sin(beta_trailing[idx+1])
                tv_outer_z2 = r_trailing[idx+1]
                tv_outer = LiftingLine(tv_outer_x1,tv_outer_y1,tv_outer_z1,tv_outer_x2,tv_outer_y2,tv_outer_z2)

                bv_x1 = tv_inner_x1
                bv_x2 = tv_outer_x1
                bv_y1 = tv_inner_y1
                bv_y2 = tv_outer_y1
                bv_z1 = tv_inner_z1
                bv_z2 = tv_outer_z1
                bv = LiftingLine(bv_x1, bv_y1, bv_z1, bv_x2,bv_y2,bv_z2)

                cp_x = 3/4*c_bound[idx]*np.cos(beta_bound[idx])
                cp_y = 3/4*c_bound[idx]*np.sin(beta_bound[idx])
                cp_z = r_bound[idx]
                ann = Annuli(polar_path=polar_path,r=r_bound[idx],chord = c_bound[idx],beta=beta_bound[idx],tv_inner=tv_inner,tv_outer=tv_outer,bv=bv,cp_x=cp_x,cp_y=cp_y,cp_z=cp_z)
                self.annuli.append(ann)
    def calculate_induced_velocities(self):
        V_i_lst = []
        for ann_i in self.annuli:
            V_i = np.zeros(3)
            for ann_j in self.annuli:
                gamma,F_azim,F_axial = ann_j.calculate_performance(ann_j.V_i,self.Vinf,self.Omega,self.rho)
                V_i += ann_j.tv_inner.calculate_induced_velocity(ann_i.cp_x,ann_i.cp_y,ann_i.cp_z,gamma)
                V_i += ann_j.tv_outer.calculate_induced_velocity(ann_i.cp_x,ann_i.cp_y,ann_i.cp_z,gamma)
                V_i += ann_j.bv.calculate_induced_velocity(ann_i.cp_x,ann_i.cp_y,ann_i.cp_z,gamma)
            V_i_lst.append(V_i)
        return V_i_lst
    def solve(self, tol=1e-6, max_iter=100):

        for iteration in range(max_iter):
            V_i_old = np.array([ann.V_i for ann in self.annuli])
            V_i_new = self.calculate_induced_velocities()
            for i in range(len(V_i_new)):
                self.annuli[i].V_i = V_i_new[i]
            # V_i_new = np.array([ann.V_i for ann in self.annuli])
            print(f"iteration {iteration} with max error: {np.max(np.abs(V_i_new - V_i_old))}")
            if np.max(np.abs(V_i_new - V_i_old)) < tol:
                print(V_i_new)
                print(f"Converged in {iteration+1} iterations")
                break
        print("Did not converge V_i:")
        print(V_i_new)
                
    def plot_blade(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        for ann in self.annuli:
            ann.tv_inner.plot_on_ax(ax)
            ann.tv_outer.plot_on_ax(ax)
            ann.bv.plot_on_ax(ax)
            ax.scatter(ann.cp_x, ann.cp_y, ann.cp_z, color='red')
        def set_equal_aspect_3d(ax):
            limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
            center = limits.mean(axis=1)
            max_range = (limits[:, 1] - limits[:, 0]).max() / 2

            ax.set_xlim3d(center[0] - max_range, center[0] + max_range)
            ax.set_ylim3d(center[1] - max_range, center[1] + max_range)
            ax.set_zlim3d(center[2] - max_range, center[2] + max_range)
            ax.set_box_aspect([1, 1, 1])
        set_equal_aspect_3d(ax)
        plt.tight_layout()
        plt.show()
            

if __name__ == "__main__":
    Vinf = np.array([60,0,0])
    c_R_func:Callable = lambda r_R : 0.18-0.06*r_R
    twst_func:Callable = lambda r_R : -50*r_R+35
    rotor = Rotor(B=2,
                  R=0.7,
                  r_R_H=0.25,
                  c_R_func=c_R_func,
                  twst_func=twst_func,
                  pitch=45,
                  polar_path=data_dir.joinpath("ARAD8pct_polar.txt"),
                  Omega = 0,
                  Vinf=Vinf,
                  rho=1.067,
                  n_elem=15,
                  dist_elem="uniform")
    rotor.solve()
    # rotor.plot_blade()