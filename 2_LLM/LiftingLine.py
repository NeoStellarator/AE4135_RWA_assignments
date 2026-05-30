import numpy as np


class LiftingLine:
    def __init__(self,
                 x1:float, 
                 y1:float, 
                 z1:float, 
                 x2:float, 
                 y2:float, 
                 z2:float):
        
        self.x1 = x1
        self.y1 = y1
        self.z1 = z1

        self.x2 = x2
        self.y2 = y2
        self.z2 = z2

        self.p1 = np.array([x1,y1,z1])
        self.p2 = np.array([x2,y2,z2])
    
    def unit_V_i(self,
                x:float,
                y:float,
                z:float):
        gamma = 1
        p = np.array([x,y,z])
        r1 = self.p1-p
        r2 = self.p2-p
        r0 = r1-r2
        r1r2_cross = np.cross(r1,r2)
        r1r2_norm = np.linalg.norm(r1r2_cross)
        induced_velocity = gamma / (4 * np.pi) * r1r2_cross/(r1r2_norm**2)*(r0@(r1/np.linalg.norm(r1)-r2/np.linalg.norm(r2)))
        return induced_velocity
    
    def calculate_induced_velocity(self, 
                                   x:float,
                                   y:float,
                                   z:float,
                                   gamma:float): #Katz, Joseph, and Allen Plotkin. Low-speed aerodynamics p.41
        p = np.array([x,y,z])
        r1 = self.p1-p
        r2 = self.p2-p
        r0 = r1-r2
        r1r2_cross = np.cross(r1,r2)
        r1r2_norm = np.linalg.norm(r1r2_cross)
        induced_velocity = gamma / (4 * np.pi) * r1r2_cross/(r1r2_norm**2)*(r0@(r1/np.linalg.norm(r1)-r2/np.linalg.norm(r2)))
        return induced_velocity
    
    def plot_on_ax(self,ax,color= 'blue'):
        # ax.plot([self.x1, self.x2], 
        #         [self.y1, self.y2], 
        #         [self.z1, self.z2], color=color)
        ax.quiver(self.x1, self.y1, self.z1,
          self.x2 - self.x1,
          self.y2 - self.y1,
          self.z2 - self.z1,
          color=color, arrow_length_ratio=0.05)
if __name__ == "__main__":
    ll = LiftingLine(0,0.2,0,0,0,1)

    print(ll.calculate_induced_velocity(1,1,1,0.5))