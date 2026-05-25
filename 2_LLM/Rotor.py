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
            r_R_trailing = np.linspace(r_R_H, 1, n_elem+1)
            r_trailing = r_R_trailing*self.R

            c_trailing = c_R_func(r_R_trailing)*self.R
            beta_trailing = np.deg2rad(self.pitch+twst_func(r_R_trailing))

            r_R_bound = (r_R_trailing[1:]+r_R_trailing[:-1])/2
            r_bound = r_R_bound*self.R
            c_bound = c_R_func(r_R_bound)*self.R
            beta_bound = np.deg2rad(self.pitch+twst_func(r_R_bound))

            

            for idx in range(len(r_bound)):
                tv_inner_x1 = c_trailing[idx]*np.cos(beta_trailing[idx])
                tv_inner_z1 = c_trailing[idx]*np.sin(beta_trailing[idx])
                tv_inner_y1 = r_trailing[idx]
                tv_inner_x2 = c_trailing[idx]/4*np.cos(beta_trailing[idx])
                tv_inner_z2 = c_trailing[idx]/4*np.sin(beta_trailing[idx])
                tv_inner_y2 = r_trailing[idx]
                tv_inner = LiftingLine(tv_inner_x1,tv_inner_y1,tv_inner_z1,tv_inner_x2,tv_inner_y2,tv_inner_z2)
                
                tv_outer_x1 = c_trailing[idx+1]/4*np.cos(beta_trailing[idx+1])
                tv_outer_z1 = c_trailing[idx+1]/4*np.sin(beta_trailing[idx+1])
                tv_outer_y1 = r_trailing[idx+1]
                tv_outer_x2 = c_trailing[idx+1]*np.cos(beta_trailing[idx+1])
                tv_outer_z2 = c_trailing[idx+1]*np.sin(beta_trailing[idx+1])
                tv_outer_y2 = r_trailing[idx+1]
                tv_outer = LiftingLine(tv_outer_x1,tv_outer_y1,tv_outer_z1,tv_outer_x2,tv_outer_y2,tv_outer_z2)

                bv_x1 = tv_inner_x2
                bv_x2 = tv_outer_x1
                bv_z1 = tv_inner_z2
                bv_z2 = tv_outer_z1
                bv_y1 = tv_inner_y2
                bv_y2 = tv_outer_y1
                bv = LiftingLine(bv_x1, bv_y1, bv_z1, bv_x2,bv_y2,bv_z2)

                cp_x = 3/4*c_bound[idx]*np.cos(beta_bound[idx])
                cp_y = r_bound[idx]
                cp_z = 3/4*c_bound[idx]*np.sin(beta_bound[idx])
                ann = Annuli(polar_path=polar_path,r=r_bound[idx],chord = c_bound[idx],beta=beta_bound[idx],Vinf=Vinf,Omega=Omega,tv_inner=tv_inner,tv_outer=tv_outer,bv=bv,cp_x=cp_x,cp_y=cp_y,cp_z=cp_z)
                self.annuli.append(ann)

            

    def calculate_induced_velocities(self):
        V_i_lst = []
        for ann_i in self.annuli:
            V_i = np.zeros(3)
            for ann_j in self.annuli:
                gamma,F_azim,F_axial,Cl,Cd = ann_j.calculate_performance(ann_j.V_i,self.rho)
                V_i += ann_j.tv_inner.calculate_induced_velocity(ann_i.cp_x,ann_i.cp_y,ann_i.cp_z,gamma)
                V_i += ann_j.tv_outer.calculate_induced_velocity(ann_i.cp_x,ann_i.cp_y,ann_i.cp_z,gamma)
                V_i += ann_j.bv.calculate_induced_velocity(ann_i.cp_x,ann_i.cp_y,ann_i.cp_z,gamma)
                for i_wake in range(len(ann_j.inner_wake_lines)):
                    V_i+=ann_j.inner_wake_lines[i_wake].calculate_induced_velocity(ann_i.cp_x,ann_i.cp_y,ann_i.cp_z,gamma)
                    V_i+=ann_j.outer_wake_lines[i_wake].calculate_induced_velocity(ann_i.cp_x,ann_i.cp_y,ann_i.cp_z,gamma)
            V_i_lst.append(V_i)
        return V_i_lst
    def calculate_spanwise_performance(self):
        V_i_lst = []
        F_azim_lst = []
        F_axial_lst = []
        Cl_lst = []
        Cd_lst = []
        y_lst =[]
        for ann_i in self.annuli:
            
            gamma,F_azim,F_axial,Cl,Cd = ann_i.calculate_performance(ann_i.V_i,self.rho)
            
            F_azim_lst.append(F_azim)
            F_axial_lst.append(F_axial)
            Cl_lst.append(Cl)
            Cd_lst.append(Cd)
            V_i_lst.append(ann_i.V_i)
            y_lst.append(ann_i.r)
        return V_i_lst,F_azim_lst,F_axial_lst,Cl_lst,Cd_lst,y_lst

    def solve(self, tol=1e-6, max_iter=1000,step_size = 0.8):

        for iteration in range(max_iter):
            V_i_old = np.array([ann.V_i for ann in self.annuli])
            V_i_new = self.calculate_induced_velocities()
            for i in range(len(V_i_new)):
                self.annuli[i].V_i = (V_i_new[i]*step_size+V_i_old[i]*(1-step_size))
            # V_i_new = np.array([ann.V_i for ann in self.annuli])
            print(f"iteration {iteration} with max error: {np.max(np.abs(V_i_new - V_i_old))}")
            if np.max(np.abs(V_i_new - V_i_old)) < tol:
                # print(f"V_i_new = {V_i_new}")

                print(f"Converged in {iteration+1} iterations")
                return
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
            # for wake_p in ann.inner_wake_points:
            #     ax.scatter(wake_p[0],wake_p[1],wake_p[2],color='blue')
            for wake_ll in ann.inner_wake_lines:
                wake_ll.plot_on_ax(ax)
            for wake_ll in ann.outer_wake_lines:
                wake_ll.plot_on_ax(ax)
        def set_equal_aspect_3d(ax):
            limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
            center = limits.mean(axis=1)
            max_range = (limits[:, 1] - limits[:, 0]).max() / 2

            ax.set_xlim3d(center[0] - max_range, center[0] + max_range)
            ax.set_ylim3d(center[1] - max_range, center[1] + max_range)
            ax.set_zlim3d(center[2] - max_range, center[2] + max_range)
            ax.set_box_aspect([1, 1, 1])
        set_equal_aspect_3d(ax)
        ax.set_axis_on()
        ax.set_axis_on()
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.tight_layout()
        plt.show()
            

if __name__ == "__main__":
    Vinf = np.array([60,0,0])
    Vinf_wing = np.array([1,0,0])
    c_R_func:Callable = lambda r_R : 0.18-0.06*r_R
    twst_func:Callable = lambda r_R : -50*r_R+35

    wing_c_R_func:Callable = lambda r_R: np.ones(shape=r_R.shape) #=1
    wing_twst_func:Callable = lambda r_R : r_R*0
    wing = Rotor(B=1,
                  R=1,
                  r_R_H=0,
                  c_R_func=wing_c_R_func,
                  twst_func=wing_twst_func,
                  pitch=0,
                  polar_path=data_dir.joinpath("ARAD8pct_polar.txt"),
                  Omega = 0,
                  Vinf=Vinf_wing,
                  rho=1,
                  n_elem=13,
                  dist_elem="uniform")
    wing.plot_blade()
    wing.solve(tol=0.01,step_size=0.01)
    V_i_lst,F_azim_lst,F_axial_lst,Cl_lst,Cd_lst,y_lst=wing.calculate_spanwise_performance()
    print(Cl_lst)
    print(F_azim_lst)
    plt.plot(y_lst,Cl_lst)
    plt.show()
    # rotor = Rotor(B=2,
    #               R=0.7,
    #               r_R_H=0.25,
    #               c_R_func=c_R_func,
    #               twst_func=twst_func,
    #               pitch=45,
    #               polar_path=data_dir.joinpath("ARAD8pct_polar.txt"),
    #               Omega = 225,
    #               Vinf=Vinf,
    #               rho=1.067,
    #               n_elem=15,
    #               dist_elem="uniform")

    # rotor.solve()
    # rotor.plot_blade()