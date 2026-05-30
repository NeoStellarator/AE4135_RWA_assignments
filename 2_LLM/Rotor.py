from typing import Callable, List, Literal
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial.transform import Rotation
import pickle

from LiftingLine import LiftingLine
from Annuli import Annuli
from globals import main_dir, data_dir, res_dir
from jit_vector_math import generate_induction_matrix,convert_gamma_vector

class Rotor:
    def __init__(self,
        # Geometry
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
        periods:int=1,
        n_elems_per_wake:int = 10,
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
        self.periods = periods
        self.n_elems_per_wake = n_elems_per_wake
        ## Generating lifting lines
        self.annuli :List[Annuli] = []
        
        if dist_elem == "uniform":
            r_R_trailing = np.linspace(r_R_H, 1, n_elem+1)
            r_trailing = r_R_trailing*self.R

            c_trailing = c_R_func(r_R_trailing)*self.R
            beta_trailing = np.deg2rad(self.pitch+twst_func(r_R_trailing))

            r_R_bound = (r_R_trailing[1:]+r_R_trailing[:-1])/2
            r_bound = r_R_bound*self.R

            self.c_bound = c_R_func(r_R_bound)*self.R # used later
            c_bound = self.c_bound
            beta_bound = np.deg2rad(self.pitch+twst_func(r_R_bound))
        else:
            raise NotImplementedError("Only uniform spacing is implemented!")
        n_wake = n_elems_per_wake*periods
        interval = periods*2*np.pi/self.Omega
        angles = np.linspace(0,-2*np.pi*periods,n_wake+1).reshape((n_wake+1,1))
        wake_rotations = Rotation.from_euler('x', angles,degrees=False).as_matrix()
        axial_offsets = np.linspace(np.zeros(3), self.Vinf*interval,n_wake+1)
            # tv_inner_x1_lst = 5/4*c_trailing[:-1]*np.sin(beta_trailing[:-1])
            # tv_inner_y1_lst = r_trailing[:-1]
            # tv_inner_z1_lst = 5/4*c_trailing[:-1]*np.cos(beta_trailing[:-1])
                
            # tv_inner_x2_lst = c_trailing[:-1]/4*np.sin(beta_trailing[:-1])
            # tv_inner_y2_lst = r_trailing[:-1]
            # tv_inner_z2_lst = c_trailing[:-1]/4*np.cos(beta_trailing[:-1])

            # tv_outer_x1_lst = c_trailing[1:]/4*np.sin(beta_trailing[1:])
            # tv_outer_y1_lst = r_trailing[1:]
            # tv_outer_z1_lst = c_trailing[1:]/4*np.cos(beta_trailing[1:])

            # tv_outer_x2_lst = 5/4*c_trailing[1:]*np.sin(beta_trailing[1:])
            # tv_outer_y2_lst = r_trailing[1:]
            # tv_outer_z2_lst = 5/4*c_trailing[1:]*np.cos(beta_trailing[1:])
            
            # cp_x_lst = 3/4*self.c_bound*np.sin(beta_bound)
            # cp_y_lst = r_bound
            # cp_z_lst = 3/4*self.c_bound*np.cos(beta_bound)


        # for blade in range(self.B):
        #     blade_angle = 2 * np.pi / self.B * blade
        #     rot = Rotation.from_euler("x", blade_angle)

        #     for idx in range(len(r_bound)):
        #         # Generating trailing vortices
        #         tv_inner_x1 = 5/4*c_trailing[idx]*np.sin(beta_trailing[idx])
        #         tv_inner_y1 = r_trailing[idx]
        #         tv_inner_z1 = 5/4*c_trailing[idx]*np.cos(beta_trailing[idx])
                
        #         tv_inner_x2 = c_trailing[idx]/4*np.sin(beta_trailing[idx])
        #         tv_inner_y2 = r_trailing[idx]
        #         tv_inner_z2 = c_trailing[idx]/4*np.cos(beta_trailing[idx])

        #         p_inner1 = rot.apply([tv_inner_x1, tv_inner_y1, tv_inner_z1])
        #         p_inner2 = rot.apply([tv_inner_x2, tv_inner_y2, tv_inner_z2])
        #         tv_inner = LiftingLine(*p_inner1, *p_inner2)

                
        #         tv_outer_x1 = c_trailing[idx+1]/4*np.sin(beta_trailing[idx+1])
        #         tv_outer_y1 = r_trailing[idx+1]
        #         tv_outer_z1 = c_trailing[idx+1]/4*np.cos(beta_trailing[idx+1])

        #         tv_outer_x2 = 5/4*c_trailing[idx+1]*np.sin(beta_trailing[idx+1])
        #         tv_outer_y2 = r_trailing[idx+1]
        #         tv_outer_z2 = 5/4*c_trailing[idx+1]*np.cos(beta_trailing[idx+1])

        #         p_outer1 = rot.apply([tv_outer_x1, tv_outer_y1, tv_outer_z1])
        #         p_outer2 = rot.apply([tv_outer_x2, tv_outer_y2, tv_outer_z2])
        #         tv_outer = LiftingLine(*p_outer1, *p_outer2)

                # # generating bound vortex
                # p_bv1 = p_inner2
                # p_bv2 = p_outer1
                # bv = LiftingLine(*p_bv1, *p_bv2)

                # cp = rot.apply([3/4*self.c_bound[idx]*np.sin(beta_bound[idx]),
                #                 r_bound[idx],
                #                 3/4*self.c_bound[idx]*np.cos(beta_bound[idx])])

                # ann = Annuli(polar_path=polar_path, r=r_bound[idx], chord=self.c_bound[idx],
                #             beta=np.rad2deg(beta_bound[idx]), Vinf=Vinf,rho=self.rho, Omega=Omega,
                #             tv_inner=tv_inner, tv_outer=tv_outer, bv=bv,
                #             cp_x=cp[0], cp_y=cp[1], cp_z=cp[2],periods = periods,n_elems_per_wake=n_elems_per_wake)
                # self.annuli.append(ann)
        blade_angles =np.linspace(0,2*np.pi,self.B,endpoint=False)
        blade_rotations= [Rotation.from_euler("x", blade_angle).as_matrix() for blade_angle in blade_angles]
        for axial_rot in blade_angles:
            print(axial_rot)
            axial_rot_matrix = Rotation.from_euler("x", axial_rot).as_matrix()
            for i in range(len(r_bound)):
                ann = Annuli(polar_path=polar_path, r=r_bound[i], chord=c_bound[i],axial_rot_matrix=axial_rot_matrix,
                            beta=np.rad2deg(beta_bound[i]), Vinf=Vinf,rho=self.rho, Omega=Omega)
                self.annuli.append(ann)
        self.m_induced,self.p1_lst,self.p2_lst = generate_induction_matrix(dist_elem=dist_elem,r_trailing=r_trailing,c_trailing=c_trailing,beta_trailing=beta_trailing,r_bound=r_bound,c_bound=self.c_bound,beta_bound=beta_bound,wake_rotations=wake_rotations,axial_offsets=axial_offsets,blade_rotations=blade_rotations)
        print("Generated induction matrix!!")



    def generate_induction_matrix(self):
        
        n_ann = len(self.annuli)
        n_line = n_ann*(self.periods*self.n_elems_per_wake*2+3)
        m_ind = np.zeros((n_ann, n_line,3))
        gamma = 1
        for i in range(len(self.annuli)):
            ann_i = self.annuli[i]
            j=0
            for ann_j in self.annuli:
                m_ind[i,j]=ann_j.tv_inner.calculate_induced_velocity(ann_i.cp_x,ann_i.cp_y,ann_i.cp_z,gamma)
                j+=1
                m_ind[i,j]= ann_j.tv_outer.calculate_induced_velocity(ann_i.cp_x,ann_i.cp_y,ann_i.cp_z,gamma)
                j+=1
                m_ind[i,j]= ann_j.bv.calculate_induced_velocity(ann_i.cp_x,ann_i.cp_y,ann_i.cp_z,gamma)
                j+=1
                for i_wake in range(len(ann_j.inner_wake_lines)):
                    m_ind[i,j]=ann_j.inner_wake_lines[i_wake].calculate_induced_velocity(ann_i.cp_x,ann_i.cp_y,ann_i.cp_z,gamma)
                    j+=1
                    m_ind[i,j]=ann_j.outer_wake_lines[i_wake].calculate_induced_velocity(ann_i.cp_x,ann_i.cp_y,ann_i.cp_z,gamma)
                    j+=1
        return m_ind
    
    def generate_gamma_matrix(self):
        n_ann = len(self.annuli)
        n_lines_per_annuli = self.periods*self.n_elems_per_wake*2+3
        n_line = n_ann*n_lines_per_annuli
        
        m_gamma = np.zeros(n_line)
        gamma_vector = np.zeros(n_ann)
        for i in range(n_ann):
            ann_i = self.annuli[i]
            gamma,Cy,Cx,Cl,Cd,a,aline = ann_i.calculate_performance(ann_i.V_i)
            # m_gamma[i*n_lines_per_annuli:(i+1)*n_lines_per_annuli]=gamma
            gamma_vector[i]=gamma
        m_gamma = convert_gamma_vector(gamma_vector,n_line,self.n_elems_per_wake,self.B)
        return m_gamma
 
    def calculate_induced_velocities(self):
        m_gamma = self.generate_gamma_matrix()
        print(f"m_induced shape: {self.m_induced.shape}")
        print(f"m_gamma shape: {m_gamma.shape}")
        u_ind = self.m_induced[..., 0] @ m_gamma  # (n, m) @ (m,) → (n,)
        v_ind = self.m_induced[..., 1] @ m_gamma  # (n, m) @ (m,) → (n,)
        w_ind = self.m_induced[..., 2] @ m_gamma  # (n, m) @ (m,) → (n,)

        result = np.stack([u_ind, v_ind, w_ind], axis=-1)  # → (n, 3)
        return result
    
    def calculate_spanwise_performance(self):
        perf_dict = {
            "V_i":[],
            "Cy": [],
            "Cx": [],
            "Cl": [],
            "Cd": [],
            "y": [],
            "r_R": [],
            "gamma": [],
            "alpha": [],
            "phi": [],
            "a": [],
            "aline": [],
        }
        for ann_i in self.annuli:
            
            gamma,Cy,Cx,Cl,Cd,a,aline = ann_i.calculate_performance(ann_i.V_i)
            perf_dict["V_i"].append(ann_i.V_i)
            perf_dict["Cy"].append(Cy)
            perf_dict["Cx"].append(Cx)
            perf_dict["Cl"].append(Cl)
            perf_dict["Cd"].append(Cd)
            perf_dict["a"].append(a)
            perf_dict["aline"].append(aline)
            perf_dict["y"].append(ann_i.r)
            perf_dict["r_R"].append(ann_i.r / self.R)
            perf_dict["gamma"].append(gamma)
            perf_dict["alpha"].append(ann_i.alpha)
            perf_dict["phi"].append(ann_i.phi)

        return perf_dict
    def calculate_integral_performance(self):
        perf_dict=self.calculate_spanwise_performance()
        S = np.trapezoid(self.c_bound,perf_dict["y"])
        Cl_total = np.trapezoid(perf_dict["Cl"]*self.c_bound,perf_dict["y"])/(S)
        return Cl_total

    def solve(self, tol=0.01, max_iter=1000,step_size = 0.01):
        last_max_error = np.inf
        for iteration in range(max_iter):
            V_i_old = np.array([ann.V_i for ann in self.annuli])
            V_i_new = self.calculate_induced_velocities()
            for i in range(len(V_i_new)):
                self.annuli[i].V_i = (V_i_new[i]*step_size+V_i_old[i]*(1-step_size))
            print(f"iteration {iteration} with max error: {np.max(np.abs(V_i_new - V_i_old))}")
            max_error = np.max(np.abs(V_i_new - V_i_old))
            if max_error<last_max_error:
                step_size*=1.1
            else:
                step_size*=0.5
            last_max_error=max_error
            if np.max(np.abs(V_i_new - V_i_old)) < tol:
                print(f"Converged in {iteration+1} iterations")
                break
        if iteration == max_iter-1:
            print("Did not converge")
            # print(V_i_new)

        r_lst = []
        F_azim_lst = []
        F_axial_lst = []
        annuli_per_blade = int(len(self.annuli)/self.B)
        for i in range(annuli_per_blade):
            ann_i = self.annuli[i]
            r_lst.append(ann_i.r)
            F_azim_lst.append(ann_i.F_azim)
            F_axial_lst.append(ann_i.F_axial)
        
        self.T = np.trapezoid(F_axial_lst,r_lst)*self.B
        self.Q = np.trapezoid(F_azim_lst,r_lst)*self.B
        self.P = self.T*self.Omega

        V_inf_mag = np.linalg.norm(self.Vinf)
        n = self.Omega/(2*np.pi)
        self.CT  = self.T/(self.rho*n**2*(self.R*2)**4)
        self.TC  = self.T/(self.rho*V_inf_mag**2*(self.R*2)**2)
        self.CP  = self.P/(self.rho*n**3*(self.R*2)**5)
        self.PC  = self.P/(self.rho*V_inf_mag**3*(self.R*2)**2)
        self.CQ  = self.Q/(self.rho*n**2*(self.R*2)**5)
        self.QC  = self.Q/(self.rho*V_inf_mag**2*(self.R*2)**3)
        self.eta = self.TC/self.PC
        self.iteration = iteration
    

    def make_summary(self):

        return dict(
            CT=self.CT,
            TC=self.TC,
            CP=self.CP,
            PC=self.PC,
            CQ=self.CQ,
            QC=self.QC,
            eta=self.eta,
            it=self.iteration,
        )
    
    def plot_blade(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        color = "blue"
        for i in range(len(self.p1_lst)):
            p1 = self.p1_lst[i]
            p2 = self.p2_lst[i]
            ax.quiver(p1[0], p1[1], p1[2],
                p2[0] - p1[0],
                p2[1] - p1[1],
                p2[2] - p1[2],
                color=color, arrow_length_ratio=0.05)
        # for ann in self.annuli:
        #     ann.tv_inner.plot_on_ax(ax)
        #     ann.tv_outer.plot_on_ax(ax)
        #     ann.bv.plot_on_ax(ax)
        #     ax.scatter(ann.cp_x, ann.cp_y, ann.cp_z, color='red')
        #     for wake_ll in ann.inner_wake_lines:
        #         wake_ll.plot_on_ax(ax)
        #     for wake_ll in ann.outer_wake_lines:
        #         wake_ll.plot_on_ax(ax)
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
    def export_dist(self, file_path: str | Path) -> None:
        perf_dict = self.calculate_spanwise_performance()
        n = self.n_elem  # first blade only — all blades are symmetric

        V_i = perf_dict["V_i"][:n]
        save_df = pd.DataFrame({
            "r_R"     : perf_dict["r_R"][:n],
            "alpha"   : perf_dict["alpha"][:n],
            "phi"     : perf_dict["phi"][:n],
            "Cl"      : perf_dict["Cl"][:n],
            "Cd"      : perf_dict["Cd"][:n],
            "gamma"   : perf_dict["gamma"][:n],
            "Cy"      : perf_dict["Cy"][:n],
            "Cx"      : perf_dict["Cx"][:n],
            "a"       : perf_dict["a"][:n],
            "aline"   : perf_dict["aline"][:n],
            "V_i_x"   : [v[0] for v in V_i],
            "V_i_y"   : [v[1] for v in V_i],
            "V_i_z"   : [v[2] for v in V_i],
        })
        save_df.to_csv(file_path, index=False)

    def save_performance(self,file_name="LLM_data.pkl"):
        perf_dict = self.calculate_spanwise_performance()
        with open(file_name, "wb") as f:
            pickle.dump(perf_dict, f)
    def plot_performance(self):
        perf_dict = self.calculate_spanwise_performance()
       
        V_i_x = [v[0] for v in perf_dict["V_i"]]
        V_i_y = [v[1] for v in perf_dict["V_i"]]
        V_i_z = [v[2] for v in perf_dict["V_i"]]
        phi_lst = [ann.phi for ann in self.annuli]

        V_lst = [ann.V for ann in self.annuli]
        V_x = [v[0] for v in V_lst]
        V_y = [v[1] for v in V_lst]
        V_z = [v[2] for v in V_lst]
        fig, axes = plt.subplots(4, 4, figsize=(18, 14), constrained_layout=True)
        fig.suptitle('Spanwise Performance', fontsize=14)

        plots = [
            (perf_dict["Cl"],      'Cl',              'Lift Coefficient'),
            (perf_dict["Cd"],      'Cd',              'Drag Coefficient'),
            (perf_dict["Cy"],  'Cy ',      'Azimuthal Force coeficient'),
            (perf_dict["Cx"], 'Cx',     'Axial Force coeficient'),
            (V_i_x,       'V_i_x [m/s]',    'Induced Velocity X'),
            (V_i_y,       'V_i_y [m/s]',    'Induced Velocity Y'),
            (V_i_z,       'V_i_z [m/s]',    'Induced Velocity Z'),
            (perf_dict["gamma"],   r'$\Gamma$',      'Gamma'),
            (perf_dict["alpha"],   r'$\alpha$',      'AoA'),
            (phi_lst,   r'$\phi$',      'Inflow angle'),
            (V_x,       'V_x [m/s]',    'Total Velocity X'),
            (V_y,       'V_y [m/s]',    'Total Velocity Y'),
            (V_z,       'V_z [m/s]',    'Total Velocity Z'),
        ]

        for ax, (data, ylabel, title) in zip(axes.flat, plots):
            ax.plot(perf_dict["y"][:self.n_elem], data[:self.n_elem])
            ax.set_xlabel('Span y [m]')
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.grid(True)

        for ax in axes.flat[len(plots):]:
            ax.set_visible(False)
        plt.show()
            

if __name__ == "__main__":
    j = 1.2
    R = 0.7
    Vinf = np.array([60,0,0])
    n = Vinf[0]/(j*2*R)
    Ome  = n*2*np.pi
    Vinf = np.array([60,0,0])
    Vinf_wing = np.array([1,0,0])
    c_R_func:Callable = lambda r_R : 0.18-0.06*r_R
    twst_func:Callable = lambda r_R : -50*r_R+35
    wing_c_R_func:Callable = lambda r_R: np.ones(shape=r_R.shape)/10 #=1
    wing_twst_func:Callable = lambda r_R : r_R*0
    # wing = Rotor(B=1,
    #               R=R,
    #               r_R_H=0,
    #               c_R_func=wing_c_R_func,
    #               twst_func=wing_twst_func,
    #               pitch=90,
    #               polar_path=data_dir.joinpath("ARAD8pct_polar.txt"),
    #               Omega = 0,
    #               Vinf=Vinf_wing,
    #               rho=1,
    #               n_elem=30,
    #               dist_elem="uniform")
    # wing.plot_blade()
    # wing.solve(tol=1e-2,step_size=0.01)
    # wing.plot_performance()
    
    rotor = Rotor(B=6,
                  R=R,
                  r_R_H=0.25,
                  c_R_func=c_R_func,
                  twst_func=twst_func,
                  pitch=45,
                  polar_path=data_dir.joinpath("ARAD8pct_polar.txt"),
                  Omega = Ome,
                  Vinf=Vinf,
                  rho=1.067,
                  n_elem=40,
                  dist_elem="uniform",
                  periods = 1,
                  n_elems_per_wake=10
                  )

    
    # rotor.plot_blade()
    rotor.solve(tol = 1e-6,step_size=0.01,max_iter =10000)
    # rotor.export_dist(data_dir.joinpath("LLM_distribution.csv"))
    rotor.plot_performance()
    print(f"T = {rotor.T}")
    print(f"TC = {rotor.TC}")
    print(f"Q = {rotor.Q}")
    print(f"P = {rotor.P}")
    print(f"CT = {rotor.CT}")
    print(f"CP = {rotor.CP}")
    print(f"PC = {rotor.PC}")
    print(f"CQ = {rotor.CQ}")
    print(f"QC = {rotor.QC}")
    print(f"eta = {rotor.eta}")
    rotor.save_performance()
    rotor.export_dist(res_dir.joinpath('LLM_test_mfkr.csv'))