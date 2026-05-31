import os
from typing import Callable, List, Literal
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from Annuli import Annuli

from globals import data_dir, res_dir



class Rotor:
    def __init__(
        self,
        # Geometry
        B : int,
        R : float,
        r_R_H : float,
        c_R_func : Callable[[float|np.ndarray], float|np.ndarray],
        twst_func : Callable[[float|np.ndarray], float|np.ndarray],
        pitch : float,
        polar_path : Path|str,
        # Operating condition
        J : float,
        Vinf : float,
        rho : float,
        # Other parameters
        n_elem:int=100,
        dist_elem:Literal['uniform','cosine']='uniform',
        isPropeller:bool=True
    ):
        # Rotor geometry
        self.B = B                    
        self.R    = R
        self.r_R_H = r_R_H            
        self.c_R_func = c_R_func      
        self.twst_func = twst_func    
        self.pitch = pitch
        self.polar_path = polar_path

        self.isPropeller = isPropeller

        # Operating Conditions
        self.J    = J                    
        self.Vinf = Vinf
        self.n    = self.Vinf/(self.J*2*self.R)
        self.Ome  = self.n*2*np.pi
        self.rho  = rho

        # Discretizaiton scheme
        self.n_elem = n_elem
        self.dist_elem = dist_elem
        
        if dist_elem == "uniform":
            rr = np.linspace(r_R_H, 1, n_elem+1)
            self.r1_R_lst = rr[:-1]
            self.r2_R_lst = rr[1:]
            
            self.r_R = (rr[1:] + rr[:-1])/2
            self.dr_R = rr[1:] - rr[:-1]
            self.c_R = c_R_func(self.r_R)
            self.beta = pitch + twst_func(self.r_R)
        elif dist_elem == "cosine":
            rr = (1 - np.cos(np.linspace(0, np.pi, n_elem+1)))/2 * (1-r_R_H) + r_R_H
            self.r1_R_lst = rr[:-1]
            self.r2_R_lst = rr[1:]
            self.dr_R = rr[1:] - rr[:-1]
            self.r_R = (rr[1:] + rr[:-1])/2
            
            self.c_R = c_R_func(self.r_R)
            self.beta = pitch + twst_func(self.r_R)
        self.annuli_A_R2 = np.pi*(np.array(self.r2_R_lst)**2-np.array(self.r1_R_lst)**2)


        # Evaluation of annuli
        self.annuli_lst = [Annuli(polar_path=polar_path, 
                 r_R=self.r_R[i],
                 c_R=self.c_R[i],
                 beta=self.beta[i],
                 B=self.B, 
                 J=self.J,
                 r_R_H=self.r_R_H,
                 is_prop=self.isPropeller) for i in range(n_elem)]
        

        self.phi   = np.array([(an.phi)*180/np.pi for an in self.annuli_lst])
        self.alpha = np.array([an.alpha for an in self.annuli_lst])
        self.Cl    = np.array([an.Cl for an in self.annuli_lst])
        self.Cd    = np.array([an.Cd for an in self.annuli_lst])
        self.Cx    = np.array([an.Cx for an in self.annuli_lst])
        self.Cy    = np.array([an.Cy for an in self.annuli_lst])
        self.f     = np.array([an.F  for an in self.annuli_lst])
        self.Ct    = np.array([an.CT for an in self.annuli_lst])
        self.Cq    = np.array([an.CQ for an in self.annuli_lst])
        self.a     = np.array([an.a  for an in self.annuli_lst])
        self.aline = np.array([an.aline for an in self.annuli_lst])

        self.W     = np.sqrt((self.Vinf*(1+self.a))**2 + 
                             (self.Ome*self.r_R*self.R*(1-self.aline))**2)
        self.q_loc = 0.5*self.rho*self.W**2

        # force/moment distributions
        self.Np    = self.Cx*self.q_loc*self.c_R*self.R  # normal force per unit length (for one blade)
        self.Tp    = self.Cy*self.q_loc*self.c_R*self.R  # tangential force per unit length (for one blade)
        self.Qp    = self.Tp*self.r_R*self.R             # torque moment per unit length (for one blade)
        
        # net forces/moments
        self.T     = self.B*np.trapezoid(self.Np, self.r_R*self.R)
        self.A     = self.B*np.trapezoid(self.Tp, self.r_R*self.R)
        self.Q     = self.B*np.trapezoid(self.Qp, self.r_R*self.R)
        self.P     = self.T*self.Ome


        # force/moment coefficients
        self.CT  = self.T/(self.rho*self.n**2*(self.R*2)**4)
        self.TC  = self.T/(self.rho*self.Vinf**2*(self.R*2)**2)
        self.CP  = self.P/(self.rho*self.n**3*(self.R*2)**5)
        self.PC  = self.P/(self.rho*self.Vinf**3*(self.R*2)**2)
        self.CQ  = self.Q/(self.rho*self.n**2*(self.R*2)**5)
        self.QC  = self.Q/(self.rho*self.Vinf**2*(self.R*2)**3)
        self.eta = self.TC/self.PC
    
    def plot_check(self):

        fig, axs = plt.subplots(2, 2)
        axs = list(np.ravel(axs))
        axs : List[Axes]

        # (1) phi & alpha
        ax = axs[0]
        ax.plot(self.r_R, self.phi, label=r'$\phi$')
        ax.plot(self.r_R, self.alpha, label=r'$\alpha$')
        ax.set_title("Angles")
        ax.legend()
        ax.grid()

        # (2) Cl, Cd, CT
        ax = axs[1]
        ax.plot(self.r_R, self.Cl, label=r'$C_l$')
        ax.plot(self.r_R, self.Cd, label=r'$C_d$')
        ax.plot(self.r_R, self.Ct, label=r'$C_T$')
        ax.set_title("Coefficients")
        ax.legend()
        ax.grid()

        # (3) f
        ax = axs[2]
        ax.plot(self.r_R, self.f, label='f')
        ax.set_title("f")
        ax.legend()
        ax.grid()

        # (4) a & aline
        ax = axs[3]
        ax.plot(self.r_R, self.a, label='a')
        ax.plot(self.r_R, self.aline, label="a'")
        ax.set_title("Induction Factors")
        ax.legend()
        ax.grid()

        plt.tight_layout()
        plt.show()

    def print_geometry(self, plot=False, axs:List[Axes]|None=None, **kwargs):
        for i in range(self.n_elem):
            print(self.r_R[i],self.c_R[i],self.beta[i])

        if plot:
            if axs is None:
                fig, axs = plt.subplots()

            axs[0].plot(self.r_R, self.c_R, **kwargs)
            axs[1].plot(self.r_R, self.beta, **kwargs)

            axs[0].set_ylabel(r'$c/R$ [-]')
            axs[1].set_ylabel(r'$\beta$ [deg]')

            for ax in axs:
                ax.set_xlabel(r'$r/R$ [-]')
                ax.grid(True)
                ax.legend()

            plt.show()

    
    def export_dist(self, file_path:str|Path) -> None:
        """Function to export the distribution of properties along the span"""
        # table with distribution
        save_df = pd.DataFrame()
        save_df["r_R"]     = self.r_R
        save_df["alpha"]   = self.alpha
        save_df["phi"]     = self.phi
        save_df["a"]       = self.a
        save_df["aline"]   = self.aline
        save_df["Ct"]      = self.Ct
        save_df["Cl"]      = self.Cl
        save_df["Cx"]      = self.Cx
        save_df["Cy"]      = self.Cy
        save_df["Cq"]      = self.CQ
        save_df["Cp"]      = self.CP
        save_df["N"]      = self.Np
        save_df["T"]      = self.Tp
        save_df.to_csv(file_path, index=False)

    def export_hist(self, file_path:str|Path, vname:str='CT') -> None:
        """Function to export the iteration history of a specified variable,
        by default the CT"""
        hist_df = pd.DataFrame()
        hist_df["r_R"] = self.r_R

        v_histories = [an.hist[vname] for an in self.annuli_lst]
    
        max_len = max(len(h) for h in v_histories)
        padded = [h + [np.nan] * (max_len - len(h)) for h in v_histories]

        hist_df = pd.concat([
            hist_df,
            pd.DataFrame(padded, columns=[f"iter_{i}" for i in range(max_len)])
        ], axis=1)

        hist_df.to_csv(file_path, index=False)

    def export_total(self, target_row:int, file_path:str|Path):
        """
        Writes total results to a specific row in the CSV file.
        target_row: 1-indexed row number (1 = headers, 2 = first data row)
        """
        
        # Create new row as DataFrame
        new_row = pd.DataFrame([{
            'R': self.R,
            'n_elem': self.n_elem,
            'J': self.J,
            'T': self.T,
            'A': self.A,
            'Q': self.Q,
            'P': self.P,
            'QC': self.QC,
            'TC': self.TC,
            'PC': self.PC,
        }])
        
        if os.path.isfile(file_path):
            # Read existing data
            existing = pd.read_csv(file_path)
            target_idx = target_row - 2  # Convert to 0-indexed (headers at row 0)
            
            # Ensure we have enough rows
            while len(existing) <= target_idx:
                existing = pd.concat([existing, pd.DataFrame([{col: np.nan for col in existing.columns}])], ignore_index=True)
            
            # Update or append
            if target_idx < len(existing):
                existing.iloc[target_idx] = new_row.iloc[0].values
            else:
                existing = pd.concat([existing, new_row], ignore_index=True)
        else:
            existing = new_row
        
        existing.to_csv(file_path, index=False)

if __name__ == "__main__":
    # Propeller
    #Optimized windmiling case 1
    #     === Final optimized design ===
    # Twist: -47.35·(r/R) + 67.03°
    # Chord: c/R = 0.2029 + -0.0747·(r/R)
    # c_R_func:Callable = lambda r_R : 0.2029-0.0747*r_R
    # twst_func:Callable = lambda r_R : -47.35*r_R+21.03

    c_R_func:Callable = lambda r_R : 0.18-0.06*r_R
    twst_func:Callable = lambda r_R : -50*r_R+35

    rotor = Rotor(
        # Geometry definition
        c_R_func  = c_R_func,
        twst_func = twst_func,
        pitch = 45,
        B     = 6,
        R     = 0.7,
        r_R_H = 0.25,
        polar_path = data_dir.joinpath('ARAD8pct_polar.txt'),
        # Operating condition
        J = 1.2,
        Vinf = 60,
        rho = 1.067,
        # Discretizaiton
        n_elem = 100,
        dist_elem = 'uniform',
        isPropeller = True)

    rotor.export_hist(res_dir.joinpath('propeller_CT_history.csv'), vname='CT')

    # rotor.print_geometry()
    rotor.export_total(target_row = 1,file_path=data_dir.joinpath('total_data.csv'))
    rotor.export_dist(data_dir.joinpath('propeller_radial_data.csv'))
    rotor.plot_check()
    # rotor.export("propeller_radial_data.csv")