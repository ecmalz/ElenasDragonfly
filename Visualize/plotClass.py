'''
PLOT the optimal trajectory of states and inputs
of the AWE system in pumping mode
Python Version 2.7 / Casadi version 3.5.5
-
Author: Elena Malz, elena@malz.me
Chalmers, Goeteborg Sweden, 2017, (2020 updated from casadi 2.4.1 to 3.5.5)
-
'''
import sys
sys.path.append(r"/usr/local/casadi-py27-v3.3.0/")
import casadi as ca
import casadi.tools as ca
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as a3
from matplotlib.patches import FancyArrowPatch
import matplotlib.colors as colors
import pylab as pl

class plots(object):
    """docstring for plots"""
    def __init__(self, opt, val_opt, params, nk = 20, d = 3):
        super(plots, self).__init__()
        self.opt = opt
        self.val_opt = val_opt
        self.params = params
        self.nk = nk
        self.d  = d

        self.tgrid_x, self.tgrid_u, self.tgrid_xa = self.xgrid() # generate already the grids for the  plots
        plt.rcParams['axes.grid'] = True # grids on for all plots
        plt.ion()

    def xgrid(self):
        nk = self.nk
        d  = self.d
        tf = self.opt['tf']
        tau_root = ca.collocation_points(d, 'radau')
        tau_root = ca.veccat(0, tau_root)

        Tx = np.zeros((nk,d+1))
        for k in range(nk):
            for j in range(d+1):
                Tx[k,j] = (k + tau_root[j])*tf/float(nk)

        Tu = np.zeros((nk))
        for k in range(nk):
                Tu[k] = k *tf/float(nk)

        Txa = np.zeros((nk,d))
        for k in range(nk):
            for j in range(d):
                Txa[k,j] = (k + tau_root[j+1])* tf/float(nk)

        tgrid_x  = Tx.reshape(nk*(d+1))
        tgrid_u  = Tu.reshape(nk)
        tgrid_xa = Txa.reshape(nk*(d))
        return tgrid_x, tgrid_u, tgrid_xa

    def plottraj(self,init = 0):
        # --------------------------------------------------------
        #  3D PLOT /  TRAJECTORIES
        # --------------------------------------------------------
        opt = self.opt
        if not init: init = opt
        # ____ 3D PLOT /  TRAJECTORIES
        fig = plt.figure()
        ax  = fig.add_subplot(2,2,1,projection='3d')
        ax.plot(np.concatenate(opt["Xd",:,:,'q',0]),np.concatenate(opt["Xd",:,:,'q',1]),np.concatenate(opt["Xd",:,:,'q',2]),'-')
        ax.plot(np.concatenate(init["Xd",:,:,'q',0]),np.concatenate(init["Xd",:,:,'q',1]),np.concatenate(init["Xd",:,:,'q',2]),'-')
        ax.scatter(0,0,0,'r')
        ax.plot([0, opt["Xd",0,0,'q',0]],[0, opt["Xd",0,0,'q',1]],[0,opt["Xd",0,0,'q',2]])
        ax.scatter([opt["Xd",3,0,'q',0]],[opt["Xd",3,0,'q',1]],[opt["Xd",3,0,'q',2]],'g')
        #ax.set_zlim3d(0.8, 1.1)
        plt.axis('equal')

        plt.subplot(2,2,2)
        plt.plot(np.concatenate(opt["Xd",:,:,'q',0]),np.concatenate(opt["Xd",:,:,'q',1]),'-')
        plt.plot(np.concatenate(init["Xd",:,:,'q',0]),np.concatenate(init["Xd",:,:,'q',1]),'-')
        plt.title('x-y')

        plt.subplot(2,2,3)
        plt.plot(np.concatenate(opt["Xd",:,:,'q',1]),np.concatenate(opt["Xd",:,:,'q',2]))
        plt.plot(np.concatenate(init["Xd",:,:,'q',1]),np.concatenate(init["Xd",:,:,'q',2]),'-')
        plt.title('y-z')

        plt.subplot(2,2,4)
        plt.plot(np.concatenate(opt["Xd",:,:,'q',0]),np.concatenate(opt["Xd",:,:,'q',2]),'-')
        plt.plot(np.concatenate(init["Xd",:,:,'q',0]),np.concatenate(init["Xd",:,:,'q',2]),'-')
        plt.title('x-z')

        plt.axis('equal')
        plt.show()

    def plotcontrols(self,init = 0):
        opt = self.opt
        if not init: init = opt
        nk  = self.nk
        tgrid_u = self.tgrid_u
        tgrid_x = self.tgrid_x
        plt.figure()
        for k in range(3):
            plt.subplot(3,3,k+1)
            plt.step(tgrid_u,[(np.concatenate(opt['U',:,'u', k])[o]) for o in range(0,nk)])
            plt.title('homotopy_accels_v'+str(k))

            plt.subplot(3,3,k+4)
            plt.step(tgrid_u,[(np.concatenate(opt['U',:,'T', k])[o]) for o in range(0,nk)])
            plt.title('homotopy_accels_omega'+str(k))

            plt.subplot(3,3,k+7)
            plt.plot(tgrid_x,np.concatenate(opt['Xd',:,:,'coeff',k])*180/np.pi)
            plt.plot(tgrid_x,np.concatenate(init['Xd',:,:,'coeff',k])*180/np.pi)
            plt.title('coeff'+str(k))


            # plt.subplot(4,3,k+10)
            # plt.step(tgrid_x,init['Xd',:,:,'w',k])
            # plt.step(tgrid_x,opt['Xd',:,:,'w',k])
            # plt.title('omega'+str(k))
            # plt.grid('on')
            # plt.subplot(4,3,k+10)
            # plt.step(tgrid_u,opt['U',veccat,:,'dcoeff',k])
            # plt.title('dcoeff'+str(k))
            # plt.grid('on')

        plt.show()

    def plot3Dtraj(self):
        # --------------------------------------------------------
        opt = self.opt
        nk  = self. nk
        params = self.params
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1,projection='3d')

        for k in np.arange(0,nk,1):


            A = 13

            a = 0.5*params['bref']
            b = 1 *params['cref']


            vtx = np.array([
                               [float(opt["Xd",k,0,"q",0]+b*opt["Xd",k,0,"R",0]+a*opt["Xd",k,0,"R",3]),float(opt["Xd",k,0,"q",1]+b*opt["Xd",k,0,"R",1]+a*opt["Xd",k,0,"R",4]),float(opt["Xd",k,0,"q",2]+b*opt["Xd",k,0,"R",2]+a*opt["Xd",k,0,"R",5]) ],
                               [float(opt["Xd",k,0,"q",0]-b*opt["Xd",k,0,"R",0]+a*opt["Xd",k,0,"R",3]),float(opt["Xd",k,0,"q",1]-b*opt["Xd",k,0,"R",1]+a*opt["Xd",k,0,"R",4]),float(opt["Xd",k,0,"q",2]-b*opt["Xd",k,0,"R",2]+a*opt["Xd",k,0,"R",5]) ],
                               [float(opt["Xd",k,0,"q",0]-b*opt["Xd",k,0,"R",0]-a*opt["Xd",k,0,"R",3]),float(opt["Xd",k,0,"q",1]-b*opt["Xd",k,0,"R",1]-a*opt["Xd",k,0,"R",4]),float(opt["Xd",k,0,"q",2]-b*opt["Xd",k,0,"R",2]-a*opt["Xd",k,0,"R",5]) ],
                               [float(opt["Xd",k,0,"q",0]+b*opt["Xd",k,0,"R",0]-a*opt["Xd",k,0,"R",3]),float(opt["Xd",k,0,"q",1]+b*opt["Xd",k,0,"R",1]-a*opt["Xd",k,0,"R",4]),float(opt["Xd",k,0,"q",2]+b*opt["Xd",k,0,"R",2]-a*opt["Xd",k,0,"R",5]) ],
                               ])

            ax.plot([float(opt["Xd",k,0,"q",0]-b*opt["Xd",k,0,"R",0]), float(opt["Xd",k,0,"q",0]-0.5*a*opt["Xd",k,0,"R",0])],
                [float(opt["Xd",k,0,"q",1]-b*opt["Xd",k,0,"R",1]), float(opt["Xd",k,0,"q",1]-0.5*a*opt["Xd",k,0,"R",1])],
                [float(opt["Xd",k,0,"q",2]-b*opt["Xd",k,0,"R",2]), float(opt["Xd",k,0,"q",2]-0.5*a*opt["Xd",k,0,"R",2])],color='black',linewidth = 4)

            tri = a3.art3d.Poly3DCollection([vtx])
            tri.set_color(colors.rgb2hex([.5, .8, .4]))
            tri.set_edgecolor('k')
            tri.set_zorder=1
            ax.add_collection3d(tri)



        ax.plot(np.concatenate(opt["Xd",:,:-1,'q',0]),
                np.concatenate(opt["Xd",:,:-1,'q',1]),
                np.concatenate(opt["Xd",:,:-1,'q',2]),linestyle = '--', color = 'grey',)
        plt.xlabel('x position [m]', size = 15,labelpad=10)
        plt.ylabel('y position [m]', size = 15,labelpad=10)
        ax.set_zlabel('altitude (-z position) [m]', size = 15,labelpad=10)
        plt.tick_params(labelsize=13)


        plt.show()

    def plot3Dtraj_vectors(self):
        # --------------------------------------------------------
        # KITE FLYING
        # --------------------------------------------------------
        opt = self.opt
        nk  = self. nk

        fig = plt.figure()
        ax  = fig.add_subplot(1,1,1,projection='3d')
        for k in np.arange(0,nk,1):

        #    b = 0.1*A
        #    a = 0.3*A
            A = 30
            b = 0.5*0.2*A
            a = 0.5*A
            # a = params['bref']*3
            # b = params['cref']*3
            ax.plot([float(opt["Xd",k,0,"q",0]), float(opt["Xd",k,0,"q",0]+a*opt["Xd",k,0,"R",0])],
                    [float(opt["Xd",k,0,"q",1]), float(opt["Xd",k,0,"q",1]+a*opt["Xd",k,0,"R",1])],
                    [float(opt["Xd",k,0,"q",2]), float(opt["Xd",k,0,"q",2]+a*opt["Xd",k,0,"R",2])],color='red')


            ax.plot([float(opt["Xd",k,0,"q",0]), float(opt["Xd",k,0,"q",0]+a*opt["Xd",k,0,"R",3])],
                    [float(opt["Xd",k,0,"q",1]), float(opt["Xd",k,0,"q",1]+a*opt["Xd",k,0,"R",4])],
                    [float(opt["Xd",k,0,"q",2]), float(opt["Xd",k,0,"q",2]+a*opt["Xd",k,0,"R",5])],color='blue')


            ax.plot([float(opt["Xd",k,0,"q",0]), float(opt["Xd",k,0,"q",0]+a*opt["Xd",k,0,"R",6])],
                    [float(opt["Xd",k,0,"q",1]), float(opt["Xd",k,0,"q",1]+a*opt["Xd",k,0,"R",7])],
                    [float(opt["Xd",k,0,"q",2]), float(opt["Xd",k,0,"q",2]+a*opt["Xd",k,0,"R",8])],color='green')

            ax.plot(np.concatenate(opt["Xd",:,0,'q',0])[0],
                    np.concatenate(opt["Xd",:,0,'q',1])[0],
                    np.concatenate(opt["Xd",:,0,'q',2])[0],linestyle = '--', color = 'grey',)


            vtx = np.array([
                               [float(opt["Xd",k,0,"q",0]+b*opt["Xd",k,0,"R",0]+a*opt["Xd",k,0,"R",3]),float(opt["Xd",k,0,"q",1]+b*opt["Xd",k,0,"R",1]+a*opt["Xd",k,0,"R",4]),float(opt["Xd",k,0,"q",2]+b*opt["Xd",k,0,"R",2]+a*opt["Xd",k,0,"R",5]) ],
                               [float(opt["Xd",k,0,"q",0]-b*opt["Xd",k,0,"R",0]+a*opt["Xd",k,0,"R",3]),float(opt["Xd",k,0,"q",1]-b*opt["Xd",k,0,"R",1]+a*opt["Xd",k,0,"R",4]),float(opt["Xd",k,0,"q",2]-b*opt["Xd",k,0,"R",2]+a*opt["Xd",k,0,"R",5]) ],
                               [float(opt["Xd",k,0,"q",0]-b*opt["Xd",k,0,"R",0]-a*opt["Xd",k,0,"R",3]),float(opt["Xd",k,0,"q",1]-b*opt["Xd",k,0,"R",1]-a*opt["Xd",k,0,"R",4]),float(opt["Xd",k,0,"q",2]-b*opt["Xd",k,0,"R",2]-a*opt["Xd",k,0,"R",5]) ],
                               [float(opt["Xd",k,0,"q",0]+b*opt["Xd",k,0,"R",0]-a*opt["Xd",k,0,"R",3]),float(opt["Xd",k,0,"q",1]+b*opt["Xd",k,0,"R",1]-a*opt["Xd",k,0,"R",4]),float(opt["Xd",k,0,"q",2]+b*opt["Xd",k,0,"R",2]-a*opt["Xd",k,0,"R",5]) ],
                               ])

            ax.plot([float(opt["Xd",k,0,"q",0]-b*opt["Xd",k,0,"R",0]), float(opt["Xd",k,0,"q",0]-a*opt["Xd",k,0,"R",0])],
                [float(opt["Xd",k,0,"q",1]-b*opt["Xd",k,0,"R",1]), float(opt["Xd",k,0,"q",1]-a*opt["Xd",k,0,"R",1])],
                [float(opt["Xd",k,0,"q",2]-b*opt["Xd",k,0,"R",2]), float(opt["Xd",k,0,"q",2]-a*opt["Xd",k,0,"R",2])],color='black',linewidth = 4)

            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            tri = a3.art3d.Poly3DCollection([vtx])
            tri.set_color(colors.rgb2hex([.5, .8, .4]))
            tri.set_edgecolor('k')
            tri.set_zorder=1
            ax.add_collection3d(tri)
        plt.show()

    def rotationmatrix(self,init = 0):
        opt = self.opt
        if not init: init = opt
        tgrid_x = self.tgrid_x
        plt.figure()
        for i in range(3):
            for j in range(3):
                plt.subplot(3,3,j+3*i+1)
                plt.plot(tgrid_x,np.concatenate(init["Xd",:,:,"R",i+3*j]), label = 'init')
                plt.plot(tgrid_x,np.concatenate(opt["Xd",:,:,"R",i+3*j]) , label = 'opt')
                plt.title('R'+str(i+1)+str(j+1))
        plt.legend()

    def draw(self,x1,y1,x2,y2,label1, label2, titlename='noname'):
        '''(x1, y1, x1, x2, label1, label2, titlename='noname')'''
        plt.figure()
        plt.plot(x1,y1, label = label1)
        plt.plot(x2,y2, label = label2)
        plt.legend()
        plt.title(titlename)

    def draw3(self,x1,y1,x2,y2,x3,y3,label1, label2, label3,titlename='noname'):
        plt.figure()
        plt.plot(x1,y1, label = label1)
        plt.plot(x2,y2,  label = label2)
        plt.plot(x3,y3, label = label3)
        plt.legend()
        plt.title(titlename)

    def drawdot(self,x1,y1,x2,y2,titlename='noname'):
        plt.figure()
        plt.plot(x1,y1,'.')
        plt.plot(x2,y2)
        plt.title(titlename)
        plt.show()

    def draw3D(self,x1,y1,x2,y2,x3,y3,x4,y4,x5,y5,x6,y6,label1, label2,titlename='noname'):
        plt.figure()
        plt.subplot(3,1,1)
        plt.plot(x1,y1, label = label1)
        plt.plot(x2,y2, label = label2)
        plt.legend()
        plt.title(titlename)
        plt.subplot(3,1,2)
        plt.plot(x3,y3)
        plt.plot(x4,y4)
        plt.subplot(3,1,3)
        plt.plot(x5,y5)
        plt.plot(x6,y6)
        plt.show()

    def vis_force_vectors(self,type = 'drag', k=3, j=1):
        """ Inputs: type, k=3, j=1\n
            Chose type = 'drag' or 0. \n
            Choose k, d as time interval and coll point\n
            -- Visualizes all the forces at time instance k and collocation point j --. """
        try:
            self.val_opt['Dragcoeff']
        except Exception:
            print('val_opt[ Dragcoeff ] and  val_opt[ Liftcoeff ]  are probably not available for plotting the lift forces')
            return 0

        opt     = self.opt
        val_opt = self.val_opt
        params  = self.params



        fig = plt.figure(figsize = (9,9))
        ax  = fig.add_subplot(1,1,1,projection='3d')

        # NOTE ####
        # - val_opt is concatenated to nk*d instances, while opt is of size nk*(d+1) and organised in [k]|j] segments.
        # The last segment in each interval nk at j = d+1 equals the value of the next interval at j = 0.  Due to that different expressions of the solutions
        # the plotting of val_opt is adjusted to k*3+j in order to end up at the correct time instance.
        # - The wing is plotted double the size in order to visualize the forces better
        # opt["Xd",k,j,"R"] =  opt["Xd",k,j,"R"].T
        a = params['bref']
        b = params['cref']
        A = b*a*2
        scale = 1/(np.mean(val_opt['speed'])**2)
        scaleDrag = scale * 10
        x = float(opt["Xd",k,j,"q",0])
        y = float(opt["Xd",k,j,"q",1])
        z = float(opt["Xd",k,j,"q",2])


        # bo = opt['Xd',k,j,'R']
        # BodyX = Arrow3D([x,  x + bo[0]*a],
        #                 [y,  y + bo[1]*a],
        #                 [z,  z + bo[2]*a],
        #             mutation_scale=5, lw=1, arrowstyle="-|>", color = 'red')
        # BodyY = Arrow3D([x,  x + bo[3]*a],
        #                 [y,  y + bo[4]*a],
        #                 [z,  z + bo[5]*a],
        #             mutation_scale=5, lw=1, arrowstyle="-|>", color = 'red')
        # BodyZ = Arrow3D([x,  x + bo[6]*a],
        #                 [y,  y + bo[7]*a],
        #                 [z,  z + bo[8]*a],
        #             mutation_scale=5, lw=1, arrowstyle="-|>", color = 'red')
        # ax.add_artist(BodyX); ax.text(x + bo[0], y + bo[1], z + bo[2], 'BODY X', color = 'red' )
        # ax.add_artist(BodyY); ax.text(x + bo[3], y + bo[4], z + bo[5], 'BODY Y', color = 'red' )
        # ax.add_artist(BodyZ); ax.text(x + bo[6], y + bo[7], z + bo[8], 'BODY Z', color = 'red' )


        ax.plot([float(opt["Xd",k,j,"q",0]), float(opt["Xd",k,j,"q",0]+a*opt["Xd",k,j,"R",0])],
                [float(opt["Xd",k,j,"q",1]), float(opt["Xd",k,j,"q",1]+a*opt["Xd",k,j,"R",1])],
                [float(opt["Xd",k,j,"q",2]), float(opt["Xd",k,j,"q",2]+a*opt["Xd",k,j,"R",2])],color='gray', linestyle= '--')
        ax.text(x + a*opt["Xd",k,j,"R",0], y + a*opt["Xd",k,j,"R",1], z + a*opt["Xd",k,j,"R",2], 'Rx (1:3)', color = 'gray' )


        ax.plot([float(opt["Xd",k,j,"q",0]), float(opt["Xd",k,j,"q",0]+a*opt["Xd",k,j,"R",3])],
                [float(opt["Xd",k,j,"q",1]), float(opt["Xd",k,j,"q",1]+a*opt["Xd",k,j,"R",4])],
                [float(opt["Xd",k,j,"q",2]), float(opt["Xd",k,j,"q",2]+a*opt["Xd",k,j,"R",5])],color='gray',linestyle=  '--')
        ax.text(x + a*opt["Xd",k,j,"R",3], y + a*opt["Xd",k,j,"R",4], z + a*opt["Xd",k,j,"R",5], 'Ry (3:6)', color = 'gray' )


        ax.plot([float(opt["Xd",k,j,"q",0]), float(opt["Xd",k,j,"q",0]+a*opt["Xd",k,j,"R",6])],
                [float(opt["Xd",k,j,"q",1]), float(opt["Xd",k,j,"q",1]+a*opt["Xd",k,j,"R",7])],
                [float(opt["Xd",k,j,"q",2]), float(opt["Xd",k,j,"q",2]+a*opt["Xd",k,j,"R",8])],color='gray',linestyle=  '--')
        ax.text(x + a*opt["Xd",k,j,"R",6], y + a*opt["Xd",k,j,"R",7], z + a*opt["Xd",k,j,"R",8], 'Rz (6:9)', color = 'gray' )


        # F_drag from Props
        if type == 'Drag':
            F_prop =  val_opt['F_drag'][k*3+j]*scaleDrag
            Prop_drag = Arrow3D([x, x + F_prop[0]],
                                [y, y + F_prop[1]],
                                [z, z + F_prop[2]],
                                mutation_scale=5, lw=1, arrowstyle="-|>", color = 'red')
            ax.add_artist(Prop_drag)
            ax.text(x + F_prop[0],y + F_prop[1], z + F_prop[2], 'prop drag', color = 'red' )

        # Tether constraint
        F_tether = val_opt['F_tether'][k*3+j+1]*scale
        tetcstr = Arrow3D(  [x, x - F_tether[0]],
                            [y, y - F_tether[1]],
                            [z, z - F_tether[2]],
                            mutation_scale=5, lw=1, arrowstyle="-|>", color = 'b')

        # Tether drag
        Tether_drag =  val_opt['Tether_drag'][k*3+j]*scaleDrag
        tet_drag = Arrow3D( [x, x + Tether_drag[0]],
                            [y, y + Tether_drag[1]],
                            [z, z + Tether_drag[2]],
                            mutation_scale=5, lw=1, arrowstyle="-|>", color = 'green')

        v_app  = val_opt['v_app'][k*3+j]
        # Aero coefficient projection on body x direction and y direction in WIND frame.
        alpha = val_opt['AoA'][k*3+j]
        FW =  np.dot(self.NEDBody2W_C(alpha), np.array([ val_opt['Liftcoeff'][k*3+j]  , val_opt['Dragcoeff'][k*3+j]   ]) )
        # Get correct wind vectors for wind frame, but reverse signs for correct visualization
        FL = -ca.mtimes(self.Nav2W(k,j)[6:9].T,FW[0]) *scale
        FD = -ca.mtimes(self.Nav2W(k,j)[0:3].T,FW[1]) *scaleDrag# reverse signs for correct visualization


        F_lift = Arrow3D(   [x, x + FL[0]],
                            [y, y + FL[1]],
                            [z, z + FL[2]],
                            mutation_scale=5, lw=1, arrowstyle="-|>", color = 'purple')


        F_drag = Arrow3D(   [x, x + FD[0]],
                            [y, y + FD[1]],
                            [z, z + FD[2]],
                            mutation_scale=5, lw=1, arrowstyle="-|>", color = 'purple')



        F_aero = val_opt['F_aero'][k*3+j]*scale
        FAero    = Arrow3D( [x, x + F_aero[0]],
                            [y, y + F_aero[1]],
                            [z, z + F_aero[2]],
                            mutation_scale=5, lw=1, arrowstyle="-|>", color = 'purple')


        vapp = Arrow3D(     [x, x + v_app[0]*0.1 ],
                            [y, y + v_app[1]*0.1 ],
                            [z, z + v_app[2]*0.1 ],
                            mutation_scale=5, lw=1, arrowstyle="-|>", color = 'turquoise')

        dqvector = Arrow3D( [x, x + 1*float(opt["Xd",k,j+1,"dq",0])],
                            [y, y + 1*float(opt["Xd",k,j+1,"dq",1])],
                            [z, z + 1*float(opt["Xd",k,j+1,"dq",2])],
                            mutation_scale=5, lw=1, arrowstyle="-|>", color = 'magenta')




        ax.add_artist(tetcstr)
        ax.text(x - F_tether[0],y - F_tether[1], z - F_tether[2], 'tet cstr', color = 'blue' )

        ax.add_artist(tet_drag)
        ax.text(x + Tether_drag[0], y + Tether_drag[1], z + Tether_drag[2], 'Tether drag', color = 'green')

        ax.add_artist(vapp)
        ax.text(x + v_app[0]*0.1, y + v_app[1]*0.1, z + v_app[2]*0.1, 'v_app', color = 'turquoise')

        # ax.add_artist(dqvector)
        # ax.text(x + 1*float(opt["Xd",k,j+1,"dq",0]),y + 1*float(opt["Xd",k,j+1,"dq",1]),z + 1*float(opt["Xd",k,j+1,"dq",2]), 'flight direction',color = 'magenta')

        ax.add_artist(F_lift)
        ax.text(x + FL[0],y + FL[1], z + FL[2], 'Lift', color = 'purple' )

        ax.add_artist(F_drag)
        ax.text(x + FD[0],y + FD[1], z + FD[2], 'Drag', color = 'purple' )


        # ax.plot([0, x],
        #         [0, y],
        #         [0, z],linestyle = '-.')

        # Plot airplane, which is double the size here
        vtx = np.array([
                           [float(x+b*opt["Xd",k,j,"R",0]+a*opt["Xd",k,j,"R",3]),float(y+b*opt["Xd",k,j,"R",1]+a*opt["Xd",k,j,"R",4]),float(z+b*opt["Xd",k,j,"R",2]+a*opt["Xd",k,j,"R",5]) ], #Punkt links vorne aussen.
                           [float(x-b*opt["Xd",k,j,"R",0]+a*opt["Xd",k,j,"R",3]),float(y-b*opt["Xd",k,j,"R",1]+a*opt["Xd",k,j,"R",4]),float(z-b*opt["Xd",k,j,"R",2]+a*opt["Xd",k,j,"R",5]) ], #Punkt links hinten aussen.
                           [float(x-b*opt["Xd",k,j,"R",0]-a*opt["Xd",k,j,"R",3]),float(y-b*opt["Xd",k,j,"R",1]-a*opt["Xd",k,j,"R",4]),float(z-b*opt["Xd",k,j,"R",2]-a*opt["Xd",k,j,"R",5]) ],
                           [float(x+b*opt["Xd",k,j,"R",0]-a*opt["Xd",k,j,"R",3]),float(y+b*opt["Xd",k,j,"R",1]-a*opt["Xd",k,j,"R",4]),float(z+b*opt["Xd",k,j,"R",2]-a*opt["Xd",k,j,"R",5]) ],
                           ])


        # ax.scatter(x,y,z, c = 'k', marker = 'o')
        ax.scatter( float(opt["Xd",k,j+1,"q",0]),float(opt["Xd",k,j+1,"q",1]),float(opt["Xd",k,j+1,"q",2]), c = 'k', marker = 'o')
        ax.text(    float(opt["Xd",k,j+1,"q",0]),float(opt["Xd",k,j+1,"q",1]),float(opt["Xd",k,j+1,"q",2]), 'next point', color = 'black')


        ax.plot([float(x-b*opt["Xd",k,j,"R",0]), float(x-a/2*opt["Xd",k,j,"R",0])],
                [float(y-b*opt["Xd",k,j,"R",1]), float(y-a/2*opt["Xd",k,j,"R",1])],
                [float(z-b*opt["Xd",k,j,"R",2]), float(z-a/2*opt["Xd",k,j,"R",2])],color='black',linewidth = 2)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_xlim3d(float(x-1.5*a), float(x + 1.5*a))
        ax.set_ylim3d(float(y-1.5*a), float(y + 1.5*a))
        ax.set_zlim3d(float(z-1.5*a), float(z + 1.5*a))
        # if not type == 'Drag':
        #     ax.invert_zaxis()
        tri = a3.art3d.Poly3DCollection([vtx])
        tri.set_color(colors.rgb2hex([.5, .8, .4]))
        tri.set_edgecolor('k')
        tri.set_zorder=1
        ax.add_collection3d(tri)
        # ax.axis('equal')
        plt.show(block=False)
        from decimal import Decimal
        plt.legend(['lift forces scale: ' + "{:.2E}".format(Decimal(scale)) + '      drag forces scale: ' + "{:.2E}".format(Decimal(scaleDrag))])

    def get_CLCD(self, frame = 'body'):
        """Compute lift and drag forces in the wind frame to create an LoD over the time.
        For this is has to be rotated by alpha and both coeff rotated by 180."""

        val_opt = self.val_opt
        # Rotate Cl and Cd by alpha in orde to convert the body frame coefficients to the wind frame
        L = []; D = []

        if frame == 'body':
            for k in range(0,self.nk*self.d):
                alpha = val_opt['AoA'][k]
                C =  np.dot(self.NEDBody2W_C(alpha), np.array([ val_opt['CL'][k]  , val_opt['CD'][k]   ]) )
                L.append(C[0]); D.append(C[1])

        if frame == 'wind':
            L = np.array( val_opt['CL'])
            D = np.array( val_opt['CD'])


        plt.figure('LoD , only kite')
        plt.subplot(1,3,1)
        plt.plot(self.tgrid_xa, L, label = 'CL')
        plt.ylim([0,3.5])
        plt.legend()
        plt.subplot(1,3,2)
        plt.plot(self.tgrid_xa, D, label = 'CD')
        plt.ylim([0,0.4])
        plt.legend()
        plt.subplot(1,3,3)
        plt.plot(self.tgrid_xa, np.divide(L,D), label = 'LoD')
        # plt.ylim([0,40])
        plt.legend()
        plt.show()
        return L, D

    def costweighting(self,E_final, Lifting, Tracking, Cost, Reg):
        # --------------------------------------
        # Check cost function....
        # --------------------------------------

        plt.ion()
        plt.figure('Cost function weighting')
        ax = plt.subplot(111)
        track0 = ax.bar(1,np.array(Tracking)[0],0.1, color = 'r')
        regu0 = ax.bar(1+0.1,np.array(Reg)[0],0.1, color = 'b')
        lifting0 = ax.bar(1+0.2,np.array(Lifting)[0],0.1, color = 'k')
        energy0 = ax.bar(1+0.3,np.array(E_final*-1)[0],0.1, color = 'g')
        cost0 = ax.bar(1+0.4,np.array(Cost)[0],0.1, color = 'y')
        ax.legend((track0,regu0,lifting0,energy0,cost0), (('tracking','regularisation','lifting','energy*-1','total cost')))
        plt.grid(True)
        # ax.set_ylim([0,3000])
        ax.set_xlim([1,1.5])

        plt.show()

    def savePDF_all(self,dir = '',name = 'plots.pdf'):
        import matplotlib.backends.backend_pdf
        pdf = matplotlib.backends.backend_pdf.PdfPages(dir + name)
        for fig in xrange(1, plt.figure().number): ## will open an empty extra figure :(
            pdf.savefig( fig )
        pdf.close()

    def savePDF(self,dir,name,fig):
        import matplotlib.backends.backend_pdf
        pdf = matplotlib.backends.backend_pdf.PdfPages(dir + name)
        pdf.savefig(fig)
        pdf.close()
