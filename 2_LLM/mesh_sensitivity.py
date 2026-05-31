import os
from pathlib import Path
from typing import Callable, Literal

import numpy as np
import pandas as pd
from tqdm import tqdm

from Rotor import Rotor
from globals import data_dir, res_dir

res_dir : Path
R = 0.7
J = 1.2
Vinf = 60
n = Vinf/(J*2*R)

dist : Literal['uniform', 'cosine'] = 'cosine'

if dist == 'uniform': ext = ''
elif dist == 'cosine': ext = '_c'
else: raise NotImplementedError("Distribution not implemented!")

const_rotor_inputs = dict(
        B=6,
        R=R,
        r_R_H=0.25,
        c_R_func= lambda r_R : 0.18-0.06*r_R,
        twst_func= lambda r_R : -50*r_R+35 ,
        pitch=45,
        polar_path=data_dir.joinpath("ARAD8pct_polar.txt"),
        Omega = n*2*np.pi,
        Vinf = np.array([Vinf,0,0]),
        rho = 1.067,
        dist_elem=dist,
        a_wake = 0
)

df = pd.DataFrame([], columns=['i',]) # initialising summary dataframe

sens_mode : Literal['wake_length', 'wake_res', 'blade_res','a_wake','a_short_wake'] = 'blade_res'

n_elem_0 = 40
period_0 = 3
n_elem_per_wake_0 = 10

n_elem_lst = [10, 20, 40, 80]#, 120, 160, 200]
period_lst  = np.arange(1,30,2, dtype='int32')
n_elem_per_wake_lst = [10, 20, 40, 80, 120, 160, 200]

if sens_mode == 'wake_length':
    save_dir = res_dir.joinpath('sensitivity/wake_length')
    os.makedirs(save_dir, exist_ok=True)
    

    for i in tqdm(period_lst):
        rotor = Rotor(
            **const_rotor_inputs,
            n_elem = n_elem_0,
            periods = i,
            n_elems_per_wake=n_elem_per_wake_0,
        )

        rotor.solve(tol = 1e-6,step_size=0.01,max_iter =10000)
        rotor.export_dist(save_dir.joinpath(f'sens_wl_n={i:>02}{ext}.csv'))

        summary = rotor.make_summary()
        summary['i'] = i

        df = pd.concat((df,pd.DataFrame([summary])))

    df.to_csv(save_dir.joinpath(f'sens_wl_summary{ext}.csv'), index=False)

elif sens_mode == 'wake_res':

    save_dir = res_dir.joinpath('sensitivity/wake_res')
    os.makedirs(save_dir, exist_ok=True)
    
    df = pd.DataFrame([])
    
    for i in tqdm(n_elem_per_wake_lst):
        rotor = Rotor(
            **const_rotor_inputs,
            n_elem = n_elem_0,
            periods = period_0,
            n_elems_per_wake=i,
        )
    
        rotor.solve(tol = 1e-6,step_size=0.01,max_iter =10000)
        rotor.export_dist(save_dir.joinpath(f'sens_wr_n={i:>02}{ext}.csv'))

        summary = rotor.make_summary()
        summary['i'] = i

        df = pd.concat((df,pd.DataFrame([summary])))

    df.to_csv(save_dir.joinpath(f'sens_wr_summary{ext}.csv'), index=False)

elif sens_mode == 'blade_res':

    save_dir = res_dir.joinpath('sensitivity/blade_res')
    os.makedirs(save_dir, exist_ok=True)
    
    for i in tqdm(n_elem_lst):
        rotor = Rotor(
            **const_rotor_inputs,
            n_elem = i,
            periods = period_0,
            n_elems_per_wake=n_elem_per_wake_0,
        )
    
        rotor.solve(tol = 1e-6,step_size=0.01,max_iter =10000)
        rotor.export_dist(save_dir.joinpath(f'sens_br_n={i:>02}{ext}.csv'))

        summary = rotor.make_summary()
        summary['i'] = i

        df = pd.concat((df,pd.DataFrame([summary])))

    df.to_csv(save_dir.joinpath(f'sens_br_summary{ext}.csv'), index=False)
elif sens_mode == 'a_wake':
    save_dir = res_dir.joinpath('sensitivity/a_wake')
    os.makedirs(save_dir, exist_ok=True)

    R = 0.7
    J = 1.2
    Vinf = 60
    n = Vinf/(J*2*R)
    const_a_wake_rotor_inputs = dict(
            B=6,
            R=R,
            r_R_H=0.25,
            c_R_func= lambda r_R : 0.18-0.06*r_R,
            twst_func= lambda r_R : -50*r_R+35 ,
            pitch=45,
            polar_path=data_dir.joinpath("ARAD8pct_polar.txt"),
            Omega = n*2*np.pi,
            Vinf = np.array([Vinf,0,0]),
            rho = 1.067,
            dist_elem=dist,
            n_elems_per_wake = 30,
            n_elem = 30,
            periods = 20,
    )
    a_wake_lst = np.linspace(0,0.5,11)
    
    for a_wake in tqdm(a_wake_lst):
        rotor = Rotor(**const_a_wake_rotor_inputs,
                    a_wake=a_wake)
        rotor.solve()
        rotor.export_dist(save_dir.joinpath(f'sens_a_wake_n={a_wake:.2f}{ext}.csv'))
elif sens_mode == 'a_short_wake':
    save_dir = res_dir.joinpath('sensitivity/a_short_wake')
    os.makedirs(save_dir, exist_ok=True)

    R = 0.7
    J = 1.2
    Vinf = 60
    n = Vinf/(J*2*R)
    const_a_wake_rotor_inputs = dict(
            B=6,
            R=R,
            r_R_H=0.25,
            c_R_func= lambda r_R : 0.18-0.06*r_R,
            twst_func= lambda r_R : -50*r_R+35 ,
            pitch=45,
            polar_path=data_dir.joinpath("ARAD8pct_polar.txt"),
            Omega = n*2*np.pi,
            Vinf = np.array([Vinf,0,0]),
            rho = 1.067,
            dist_elem=dist,
            n_elems_per_wake = 30,
            n_elem = 30,
            periods = 1,
    )
    a_wake_lst = np.linspace(0,0.5,11)
    
    for a_wake in tqdm(a_wake_lst):
        rotor = Rotor(**const_a_wake_rotor_inputs,
                    a_wake=a_wake)
        rotor.solve()
        rotor.export_dist(save_dir.joinpath(f'sens_a_short_wake_n={a_wake:.2f}{ext}.csv'))


