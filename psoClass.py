# -*- coding:utf-8 -*-
import numpy as np
import random

class PSO:
    def __init__(self):
        # 参数初始化
        self.c1 = 0.4 # 粒子自身过去的位置对粒子迁移速度的影响
        self.c2 = 1.4 # 粒子群中最好的粒子的位置对粒子迁移速度的影响
        self.maxgen = 200 # 进化代数
        self.popsize = 100 # 粒子群大小
        self.psize = 3 # 粒子大小--> 指的是维度么？？
        self.Vmax = 5
        self.Vmin = -5 # 粒子移动速度
        self.lb = np.ones((self.psize,1))*(-10)
        self.ub = np.ones((self.psize,1))*10
        self.bound = np.hstack((self.lb,self.ub)) # 粒子移动范围
        self.wstart = 0.9;self.wend = 0.4 # 惯性权重
    # 返回0~1二维数组
    def rand(self,r,c):
        mat = np.zeros((r,c)) # 创建初始r*c数组
        for i in range(r):
            for j in range(c):
                mat[i][j] = random.random()
        return mat
    # 返回一个二维数组的最小值和索引
    def amin_index(self,a,axis=0):
        value_min = np.amin(a,axis=axis)
        index_min = []
        if axis == 0:
            for i in range(a.shape[1]):
                index_min.append(list(a[:,i]).index(value_min[i]))
        else:
            for i in range(a.shape[0]):
                index_min.append(list(a[i,:]).index(value_min[i]))
        index_min = np.array(index_min)
        return value_min,index_min
    # 返回一个二维数组的最大值和索引
    def amax_index(self,a,axis=0):
        value_max = np.amax(a,axis=axis)
        index_max = []
        if axis == 0:
            for i in range(a.shape[1]):
                index_max.append(list(a[:,i]).index(value_max[i]))
        else:
            for i in range(a.shape[0]):
                index_max.append(list(a[i,:]).index(value_max[i]))
        index_min = np.array(index_max)
        return value_max,index_min
    # 适应度函数,x为一维数组
    def fun(self,x):
        x1 = x[0]
        x2 = x[1]
        x3 = x[2]
        fitness = x1**2+x2**2-x3**2+3*np.sin(x1*x2*x3)
        fitness = -1.0*fitness # 求函数最小值时注释掉这行代码
        return fitness
    # 种群初始化
    def initial(self):
        # 初始化种群
        group = {}
        group['pop'] = np.zeros((self.popsize,self.psize)) # 种群
        group['v'] = np.zeros((self.popsize,self.psize)) # 粒子速度
        group['fitness'] = np.zeros(self.popsize) # 粒子适应度值，一维数组
        for i in range(self.popsize):
            group['pop'][i] = self.bound[:,0]+(self.bound[:,1]-self.bound[:,0])*self.rand(1,self.psize)[0]
            group['v'][i] = np.ones(3)*self.Vmin+(np.ones(3)*self.Vmax-np.ones(3)*self.Vmin)*self.rand(1,3)[0]
            group['fitness'][i] = self.fun(group['pop'][i])
        return group
    # 限制粒子的速度范围
    def f_av(self,av): # 对向量av进行边界限制
        def f_v(v): # 对单一的v进行边界限制
            if v > self.Vmax:
                return self.Vmax
            elif v < self.Vmin:
                return self.Vmin
            else:
                return v
        av = np.array(map(f_v,av))
        return av
    # 限制粒子的速度范围
    def f_ap(self,ap): # 对向量ap进行边界限制
        def f_p(p,p_index): # 对单一的p进行边界限制
            if p > self.bound[p_index,1]:
                return self.bound[p_index,1]
            elif p < self.bound[p_index,0]:
                return self.bound[p_index,0]
            else:
                return p
        for i in range(len(ap)):
            ap[i] = f_p(ap[i],i)
        return ap


