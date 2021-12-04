
import sys

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy.interpolate import griddata
from pyDOE import lhs
from mpl_toolkits.mplot3d import Axes3D
import time
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import interpolate


class PhysicsInformedNN:
    # Initialize the class
    def __init__(self, x0, tem0, tb, X_f, X_tem, layers, lb, ub):
        
        ltem = 298.15
        utem = 973.15

        X0 = np.concatenate((x0, 0*x0+5.0), 1) # (x0, 5)
        X_lb = np.concatenate((0*tb + lb[0], tb), 1) # (lb[0], tb)
        X_ub = np.concatenate((0*tb + ub[0], tb), 1) # (ub[0], tb)

        tem_lb = np.concatenate((0*tb + ltem, tb), 1)
        tem_ub = np.concatenate((0*tb + utem, tb), 1)
        
        self.lb = lb
        self.ub = ub
               
        self.x0 = X0[:,0:1]
        self.t0 = X0[:,1:2]

        self.x_lb = X_lb[:,0:1]
        self.t_lb = X_lb[:,1:2]

        self.x_ub = X_ub[:,0:1]
        self.t_ub = X_ub[:,1:2]
        
        self.x_f = X_f[:,0:1]
        self.t_f = X_f[:,1:2]

        self.X_tem = X_tem[:,0:1]
        
        self.tem0 = tem0
        self.tem_lb = tem_lb[:,0:1]
        self.tem_ub = tem_ub[:,0:1]
        
        # Initialize NNs
        self.layers = layers
        self.weights, self.biases = self.initialize_NN(layers)
        
        # tf Placeholders        
        self.x0_tf = tf.placeholder(tf.float32, shape=[None, self.x0.shape[1]])
        self.t0_tf = tf.placeholder(tf.float32, shape=[None, self.t0.shape[1]])
        
        self.tem0_tf = tf.placeholder(tf.float32, shape=[None, self.tem0.shape[1]])
        
        self.x_lb_tf = tf.placeholder(tf.float32, shape=[None, self.x_lb.shape[1]])
        self.t_lb_tf = tf.placeholder(tf.float32, shape=[None, self.t_lb.shape[1]])
        self.tem_lb_tf = tf.placeholder(tf.float32, shape=[None, self.tem_lb.shape[1]])
        
        self.x_ub_tf = tf.placeholder(tf.float32, shape=[None, self.x_ub.shape[1]])
        self.t_ub_tf = tf.placeholder(tf.float32, shape=[None, self.t_ub.shape[1]])
        self.tem_ub_tf = tf.placeholder(tf.float32, shape=[None, self.tem_ub.shape[1]])
        
        self.x_f_tf = tf.placeholder(tf.float32, shape=[None, self.x_f.shape[1]])
        self.t_f_tf = tf.placeholder(tf.float32, shape=[None, self.t_f.shape[1]])

        self.X_tem_tf = tf.placeholder(tf.float32, shape=[None, self.X_tem.shape[1]])

        self.global_step=tf.placeholder(tf.int32)

        # tf Graphs
        self.tem0_pred    = self.net_uv(self.x0_tf, self.t0_tf)
        self.tem_lb_pred  = self.net_uv(self.x_lb_tf, self.t_lb_tf)
        self.tem_ub_pred  = self.net_uv(self.x_ub_tf, self.t_ub_tf)
        self.f_tem_pred   = self.net_f_uv(self.x_f_tf, self.t_f_tf)
        self.X_tem_pred   = self.net_uv(self.x_f_tf, self.t_f_tf)
        self.ftem_pred    = self.net_f_uv(self.x0_tf, self.t0_tf)

        # Loss
        self.loss_pre = tf.reduce_mean(tf.square(self.X_tem_pred - self.X_tem_tf))

        self.loss     = tf.reduce_mean(tf.square(self.tem0_pred - self.tem0_tf))     + \
                        1.0e-3*tf.reduce_mean(tf.square(self.f_tem_pred))
        
        # Optimizers
        self.optimizer_pre = tf.contrib.opt.ScipyOptimizerInterface(self.loss_pre, 
                                                                method = 'L-BFGS-B', 
                                                                options = {'maxiter': 50000,
                                                                           'maxfun': 50000,
                                                                           'maxcor': 50,
                                                                           'maxls': 50,
                                                                           'ftol' : 1.0 * np.finfo(float).eps})

        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss, 
                                                                method = 'L-BFGS-B', 
                                                                options = {'maxiter': 50000,
                                                                           'maxfun': 50000,
                                                                           'maxcor': 50,
                                                                           'maxls': 50,
                                                                           'ftol' : 1.0 * np.finfo(float).eps})
    
        self.optimizer_Adam_pre = tf.train.AdamOptimizer()
        self.train_op_Adam_pre = self.optimizer_Adam_pre.minimize(self.loss_pre)

        decayed_lr = tf.train.exponential_decay( 0.001,self.global_step, 1000, 0.95, staircase=True)

        self.optimizer_Adam = tf.train.AdamOptimizer(learning_rate=0.001)
        #self.optimizer_Adam = tf.train.AdamOptimizer(learning_rate = decayed_lr)
        self.train_op_Adam  = self.optimizer_Adam.minimize(self.loss)
                
        # tf session
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))
        
        init = tf.global_variables_initializer()
        self.sess.run(init)
              
    def initialize_NN(self, layers):        
        weights = []
        biases = []
        num_layers = len(layers) 
        for l in range(0,num_layers-1):
            W = self.xavier_init(size=[layers[l], layers[l+1]])
            b = tf.Variable(tf.zeros([1,layers[l+1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)        
        return weights, biases
        
    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]        
        xavier_stddev = np.sqrt(2/(in_dim + out_dim))
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)
    
    def neural_net(self, X, weights, biases):
        num_layers = len(weights) + 1
        
        H = 2.0*(X - self.lb)/(self.ub - self.lb) - 1.0
        for l in range(0,num_layers-2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y
   
    def cal_H(self, x):
        eps= 1e-3
        x1 = (-0.4 + eps)*tf.ones_like(x)
        x2 = ( 0.4 - eps)*tf.ones_like(x)
        one = tf.ones_like(x)

        d1 =  x - x1
        d2 = x2 - x
        dist = tf.math.minimum(d1, d2)

        Hcal = 0.5*( one + dist/eps + 1.0/np.pi*tf.sin(dist*np.pi/eps) )

        #xtmp = tf.where(tf.greater(dist, eps), tf.ones_like(x), Hcal)
        xout = tf.where(tf.less(dist, -eps), tf.zeros_like(x), tf.where(tf.greater(dist, eps), tf.ones_like(x), Hcal))

        return xout
 
    def net_uv(self, x, t):
        X = tf.concat([x,t],1)
        
        Hcal = self.cal_H(x)
        one  = tf.ones_like(x)

        T1   = 298.15*one
        T2   = 973.15*one
        xlen = 0.8
        dx   = x + 0.4*one
        Tbc  = T1 + (T2-T1)/xlen*dx

        tem = self.neural_net(X, self.weights, self.biases)
        tem = Tbc*(one-Hcal) + tem*Hcal

        return tem

    def net_f_uv(self, x, t):
        tem = self.net_uv(x,t)

        one  = tf.ones_like(tem)
        zero = tf.zeros_like(tem)

        rho_Al_liquid = 2555.0*one
        rho_Al_solid  = 2555.0*one
        rho_grap      = 2200.0*one

        kappa_Al_liquid = 91.0*one
        kappa_Al_solid  = 211.0*one
        kappa_grap      = 100.0*one

        cp_Al_liquid = 1190.0*one
        cp_Al_solid  = 1190.0*one
        cp_grap      = 1700.0*one

        cl_Al_liquid = 3.98e5*one
        cl_Al_solid  = 3.98e5*one
        cl_grap      = 3.98e5*one

        # what is value of Ts #
        Ts = 913.15*one
        Tl = 933.15*one

        tem_t = tf.gradients(tem, t)[0]
        tem_x = tf.gradients(tem, x)[0]
        fL    = (tem-Ts)/(Tl-Ts)
        fL    = tf.maximum(tf.minimum((tem-Ts)/(Tl-Ts),one), zero)
        fL_t  = tf.gradients(fL, t)[0]
        
        rho   = tf.where(tf.greater(x, zero), rho_Al_liquid*fL   + rho_Al_solid*(one-fL),   rho_grap)
        kappa = tf.where(tf.greater(x, zero), kappa_Al_liquid*fL + kappa_Al_solid*(one-fL), kappa_grap)
        cp    = tf.where(tf.greater(x, zero), cp_Al_liquid*fL    + cp_Al_solid*(one-fL),    cp_grap)
        cl    = tf.where(tf.greater(x, zero), cl_Al_liquid*fL    + cl_Al_solid*(one-fL),    cl_grap)

        lap   = tf.gradients(kappa*tem_x, x)[0]
        f_tem = (rho*cp*tem_t + rho*cl*fL_t - lap)/( rho_Al_solid*kappa_Al_solid ) 
        #lap   = kappa_Al_liquid*tf.gradients(tem_x, x)[0]
        #f_tem = rho_Al_liquid*cp_Al_liquid*tem_t - lap
        return f_tem
    
    def callback(self, loss):
        print('Loss:', loss)
        
    def train(self, nIter_pre, nIter):

        it = 0
        tf_dict = {self.x0_tf: self.x0, self.t0_tf: self.t0,
                   self.tem0_tf: self.tem0,
                   self.tem_lb_tf: self.tem_lb, self.tem_ub_tf: self.tem_ub,
                   self.x_lb_tf: self.x_lb, self.t_lb_tf: self.t_lb,
                   self.x_ub_tf: self.x_ub, self.t_ub_tf: self.t_ub,
                   self.x_f_tf: self.x_f, self.t_f_tf: self.t_f,
                   self.X_tem_tf: self.X_tem,
                   self.global_step: it }
        
        start_time = time.time()
        for it in range(nIter_pre):

            self.sess.run(self.train_op_Adam_pre, tf_dict)

            # Print
            if it % 10 == 0:
                elapsed = time.time() - start_time
                loss_value = self.sess.run(self.loss_pre, tf_dict)
                print('It: %d, Loss: %.3e, Time: %.2f' % 
                      (it, loss_value, elapsed))
                start_time = time.time()
                                                                                                         
                                                                                                         
        if(nIter_pre > 0):
            self.optimizer_pre.minimize(self.sess, 
                                    feed_dict = tf_dict,         
                                    fetches = [self.loss_pre], 
                                    loss_callback = self.callback)

        start_time = time.time()
        for it in range(nIter):
            self.sess.run(self.train_op_Adam, tf_dict)
            
            # Print
            if it % 10 == 0:
                elapsed = time.time() - start_time
                loss_value = self.sess.run(self.loss, tf_dict)
                print('It: %d, Loss: %.3e, Time: %.2f' % 
                      (it, loss_value, elapsed))
                sys.stdout.flush()
                start_time = time.time()
                                                                                                         
                                                            
        if(nIter > 0):
            self.optimizer.minimize(self.sess, 
                                    feed_dict = tf_dict,         
                                    fetches = [self.loss], 
                                    loss_callback = self.callback)
                                    
    
    def predict(self, X_star):
        
        tf_dict = {self.x0_tf: X_star[:,0:1], self.t0_tf: X_star[:,1:2]}
        
        tem_star = self.sess.run(self.tem0_pred, tf_dict) 

        ftem_star = self.sess.run(self.ftem_pred, tf_dict) 
               
        return tem_star,ftem_star
    
if __name__ == "__main__": 
     
    noise = 0.0        

    ltem = 298.15
    utem = 973.15
    eps  = 0.02
    
    # Doman bounds
    lb = np.array([-0.4, 5.0])
    ub = np.array([ 0.4, 10.0])

    lbr = np.array([-0.05, 5.0])
    ubr = np.array([ 0.05, 10.0])

    N0 = 300
    N_b = 100
    N_f = 30000
    num_hidden = 4
    layers = [2] + num_hidden*[200] + [1]
    #layers = [2, num_hidden, num_hidden, num_hidden, num_hidden, 1]
        
    data = scipy.io.loadmat('../../dat/thermal_fine.mat')
    
    x = data['x'].flatten()[:,None]
    t = data['tt'].flatten()[:,None]
    Exact = data['Tem']
    Exact_tem = np.real(Exact)
    #print(x.shape)
    #print(t.shape)
    #print(Exact.shape)
    ftem = interpolate.interp2d(x,t,Exact.T)
    #print(np.diag(ftem(x[:,0],x[:,0]*0+1)))
    
    X, T = np.meshgrid(x,t)
    
    X_star = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))
    
    ###########################
    
    idx_x = np.random.choice(x.shape[0], N0, replace=False)
    x0 = x[idx_x,:]
    tem0 = Exact_tem[idx_x,0:1]
    #np.savetxt('x0.txt', x0)
    #np.savetxt('tem0.txt',  tem0)
    
    idx_t = np.random.choice(t.shape[0], N_b, replace=False)
    tb = t[idx_t,:]
    
    X_f   = lb + (ub -lb )*lhs(2,   N_f)
    #X_fr  = lbr+ (ubr-lbr)*lhs(2, 3*N_f)
    #X_f   = np.concatenate((X_f,X_fr),0)
    #print(X_f.shape)

    X_f   = X_f[np.argsort(X_f[:, 0])]
    X_tem = ftem(X_f[:,0], 0).flatten()[:,None]
    #print(X_tem.shape)
    #np.savetxt('X_f.txt',    X_f)
    #np.savetxt('X_tem.txt',  X_tem)

    model = PhysicsInformedNN(x0, tem0, tb, X_f, X_tem, layers, lb, ub)
             
    start_time = time.time()                
    model.train(-1,50000)
    elapsed = time.time() - start_time                
    print('Training time: %.4f' % (elapsed))
    tem_pred,ftem_pred = model.predict(X_star)   
    np.savetxt('predict_xT.txt',    X_star)
    np.savetxt('predict_tem.txt',   tem_pred)
    np.savetxt('predict_ftem.txt',  ftem_pred)



