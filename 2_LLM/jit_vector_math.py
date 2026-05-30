import numpy as np
from numba import jit
@jit(nopython=True)
def calculate_induced_velocity(p1x, p1y, p1z, p2x, p2y, p2z, x, y, z, gamma): #Katz, Joseph, and Allen Plotkin. Low-speed aerodynamics p.41
    p = np.array([x,y,z])
    p1 = np.array([p1x,p1y,p1z])
    p2 = np.array([p2x,p2y,p2z])
    r1 = p1-p
    r2 = p2-p
    r0 = r1-r2
    r1r2_cross = np.cross(r1,r2)
    r1r2_norm = np.linalg.norm(r1r2_cross)
    induced_velocity = gamma / (4 * np.pi) * r1r2_cross/(r1r2_norm**2)*(r0@(r1/np.linalg.norm(r1)-r2/np.linalg.norm(r2)))
    return induced_velocity
@jit(nopython=True)
def generate_induction_matrix(dist_elem, r_trailing,c_trailing,beta_trailing,r_bound,c_bound,beta_bound,wake_rotations,axial_offsets,blade_rotations):
        if dist_elem == "uniform":
            tv_inner_x1_lst = 5/4*c_trailing[:-1]*np.sin(beta_trailing[:-1])
            tv_inner_y1_lst = r_trailing[:-1]
            tv_inner_z1_lst = 5/4*c_trailing[:-1]*np.cos(beta_trailing[:-1])
                
            tv_inner_x2_lst = c_trailing[:-1]/4*np.sin(beta_trailing[:-1])
            tv_inner_y2_lst = r_trailing[:-1]
            tv_inner_z2_lst = c_trailing[:-1]/4*np.cos(beta_trailing[:-1])

            tv_outer_x1_lst = c_trailing[1:]/4*np.sin(beta_trailing[1:])
            tv_outer_y1_lst = r_trailing[1:]
            tv_outer_z1_lst = c_trailing[1:]/4*np.cos(beta_trailing[1:])

            tv_outer_x2_lst = 5/4*c_trailing[1:]*np.sin(beta_trailing[1:])
            tv_outer_y2_lst = r_trailing[1:]
            tv_outer_z2_lst = 5/4*c_trailing[1:]*np.cos(beta_trailing[1:])
            
            cp_x_lst = 3/4*c_bound*np.sin(beta_bound)
            cp_y_lst = r_bound
            cp_z_lst = 3/4*c_bound*np.cos(beta_bound)

            bv_x1_lst = tv_inner_x2_lst
            bv_y1_lst = tv_inner_y2_lst
            bv_z1_lst = tv_inner_z2_lst

            bv_x2_lst = tv_outer_x1_lst
            bv_y2_lst = tv_outer_y1_lst
            bv_z2_lst = tv_outer_z1_lst

            
        wake_x1 = []
        wake_y1 = []
        wake_z1 = []

        wake_x2 = []
        wake_y2 = []
        wake_z2 = []
        
        for i in range(len(tv_inner_x1_lst)):
            inner_rotated_positions = np.zeros((len(wake_rotations),3))
            outer_rotated_positions = np.zeros((len(wake_rotations),3))

            inner_wake_position = np.array([tv_inner_x1_lst[i],tv_inner_y1_lst[i],tv_inner_z1_lst[i]])
            outer_wake_position = np.array([tv_outer_x2_lst[i],tv_outer_y2_lst[i],tv_outer_z2_lst[i]])
            
            for rot_i in range(len(wake_rotations)):
                inner_rotated_positions[rot_i]=wake_rotations[rot_i]@inner_wake_position
                outer_rotated_positions[rot_i]=wake_rotations[rot_i]@outer_wake_position
            inner_wake_points = inner_rotated_positions+axial_offsets
            outer_wake_points = outer_rotated_positions+axial_offsets
            
            

            for i in range(len(inner_wake_points)-1):
                start_inner = inner_wake_points[i+1]
                end_inner = inner_wake_points[i]
                wake_x1.append(start_inner[0])
                wake_y1.append(start_inner[1])
                wake_z1.append(start_inner[2])

                wake_x2.append(end_inner[0])
                wake_y2.append(end_inner[1])
                wake_z2.append(end_inner[2])

                start_outer = outer_wake_points[i]
                end_outer = outer_wake_points[i+1]
                wake_x1.append(start_outer[0])
                wake_y1.append(start_outer[1])
                wake_z1.append(start_outer[2])

                wake_x2.append(end_outer[0])
                wake_y2.append(end_outer[1])
                wake_z2.append(end_outer[2])
                
        x1_lst = np.concatenate((tv_inner_x1_lst,tv_outer_x1_lst,bv_x1_lst,np.array(wake_x1)))
        y1_lst = np.concatenate((tv_inner_y1_lst,tv_outer_y1_lst,bv_y1_lst,np.array(wake_y1)))
        z1_lst = np.concatenate((tv_inner_z1_lst,tv_outer_z1_lst,bv_z1_lst,np.array(wake_z1)))

        x2_lst = np.concatenate((tv_inner_x2_lst,tv_outer_x2_lst,bv_x2_lst,np.array(wake_x2)))
        y2_lst = np.concatenate((tv_inner_y2_lst,tv_outer_y2_lst,bv_y2_lst,np.array(wake_y2)))
        z2_lst = np.concatenate((tv_inner_z2_lst,tv_outer_z2_lst,bv_z2_lst,np.array(wake_z2)))   
        
        n_blades = len(blade_rotations)
        n_lines_per_blade=len(x1_lst)
        n_line=n_lines_per_blade*n_blades
        n_cp_per_blade= len(cp_x_lst)
        p1_total_lst = np.zeros((n_line,3))
        p2_total_lst = np.zeros((n_line,3))
        cp_total_lst = np.zeros((n_cp_per_blade*n_blades,3))
        for rot_i in range(len(blade_rotations)):
            blade_rot=blade_rotations[rot_i]
            n_cp = len(cp_z_lst)
            n_p1 = len(x1_lst)

            cp_lst = np.empty((n_cp, 3))
            p1_lst = np.empty((n_p1, 3))
            p2_lst = np.empty((n_p1, 3))

            r00, r01, r02 = blade_rot[0, 0], blade_rot[0, 1], blade_rot[0, 2]
            r10, r11, r12 = blade_rot[1, 0], blade_rot[1, 1], blade_rot[1, 2]
            r20, r21, r22 = blade_rot[2, 0], blade_rot[2, 1], blade_rot[2, 2]

            for i in range(n_cp):
                x = cp_x_lst[i]
                y = cp_y_lst[i]
                z = cp_z_lst[i]

                cp_lst[i, 0] = r00*x + r01*y + r02*z
                cp_lst[i, 1] = r10*x + r11*y + r12*z
                cp_lst[i, 2] = r20*x + r21*y + r22*z

            for i in range(n_p1):
                x1, y1, z1 = x1_lst[i], y1_lst[i], z1_lst[i]
                x2, y2, z2 = x2_lst[i], y2_lst[i], z2_lst[i]

                p1_lst[i, 0] = r00*x1 + r01*y1 + r02*z1
                p1_lst[i, 1] = r10*x1 + r11*y1 + r12*z1
                p1_lst[i, 2] = r20*x1 + r21*y1 + r22*z1

                p2_lst[i, 0] = r00*x2 + r01*y2 + r02*z2
                p2_lst[i, 1] = r10*x2 + r11*y2 + r12*z2
                p2_lst[i, 2] = r20*x2 + r21*y2 + r22*z2
            
            p1_total_lst[rot_i*n_lines_per_blade:(rot_i+1)*n_lines_per_blade]=p1_lst
            p2_total_lst[rot_i*n_lines_per_blade:(rot_i+1)*n_lines_per_blade]=p2_lst
            cp_total_lst[rot_i*n_cp_per_blade:(rot_i+1)*n_cp_per_blade]=cp_lst
        n_line = len(p1_total_lst)
        n_ann = len(cp_total_lst)
        m_ind = np.zeros((n_ann, n_line,3))
        gamma = 1
        for i in range(len(cp_total_lst)):
            x=cp_total_lst[i][0]
            y=cp_total_lst[i][1]
            z=cp_total_lst[i][2]
            for i_line in range(n_line):
                p1x= p1_total_lst[i_line,0]
                p1y= p1_total_lst[i_line,1]
                p1z= p1_total_lst[i_line,2]

                p2x= p2_total_lst[i_line,0]
                p2y= p2_total_lst[i_line,1]
                p2z= p2_total_lst[i_line,2]
                m_ind[i,i_line]+=calculate_induced_velocity(p1x,p1y,p1z,p2x,p2y,p2z,x,y,z,gamma)
        return m_ind, p1_total_lst,p2_total_lst
@jit(nopython=True)
def convert_gamma_vector(gamma_vector,n_line,n_wake,n_blade):
    n_lines_per_blade = int(n_line/n_blade)

    n_ann_per_blade = int(len(gamma_vector)/n_blade)
    m_gamma = np.zeros(n_line)
    for i in range(n_blade):
        gamma_blade = gamma_vector[i*n_ann_per_blade:(i+1)*n_ann_per_blade]
        ordered_gamma_blade = order_gamma(gamma_blade,n_lines_per_blade,n_wake)
        # print(ordered_gamma_blade)
        m_gamma[i*len(ordered_gamma_blade):(i+1)*len(ordered_gamma_blade)]=ordered_gamma_blade
    return m_gamma

@jit(nopython=True)
def order_gamma(gamma_blade,n_lines_per_blade,n_wake):
    m_gamma_blade = np.zeros(n_lines_per_blade)
    m_gamma_blade[0:3*len(gamma_blade)]=np.concatenate((gamma_blade,gamma_blade,gamma_blade))
    offset=3*len(gamma_blade)
    for i in range(len(gamma_blade)):
        m_gamma_blade[offset+i*n_wake*2:offset+(i+1)*n_wake*2]=np.ones(n_wake*2)*gamma_blade[i]
    return m_gamma_blade
