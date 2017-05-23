# -*- coding:utf-8 -*-
import numpy as np
from matplotlib import pyplot as plt
from psoClass import PSO
import copy
from time import clock

# 计时
time_start = clock()
# 实例化PSO对象
pso = PSO()
# 种群初始化
group = pso.initial()
# 记录种群平均适应度值和最优适应度值
trace = []
# 个体最优值和种群最优值（这里为求函数最小值）
bestfitness,best_index = pso.amin_index(np.array(
    np.mat(group['fitness'])),axis=1)
bestfitness = bestfitness[0]
best_index = best_index[0]
# 全局最优个体
gbest0 = group['pop'][best_index,:] # 产生一维数组
gbest = gbest0 # 由全局最优值形成的popsize*psize的矩阵
for i in range(pso.popsize)[1:]:
    gbest = np.vstack((gbest,gbest0))

gbestfitness = group['fitness'][best_index]
# 整个种群的每个粒子的最优值组成的种群
popbest = copy.deepcopy(group['pop']) # 因为后面更新的时候会进行子级修改，所以使用深拷贝
popbestfitness = copy.deepcopy(group['fitness']) # 同上
# 迭代寻优
for i in range(pso.maxgen):
    # 惯性权重由大变小，最开始有较强的全局寻优能力，后来具有较强的局部寻优能力
    w = pso.wstart-(pso.wstart-pso.wend)*((float(i)/pso.maxgen)**2)
    # 粒子群迁移
    # 速度更新
    group['v'] = w*group['v']+pso.c1*(popbest-group['pop'])+pso.c2*(gbest-group['pop'])
    # 速度范围限制
    group['v'] = np.array(np.mat(map(pso.f_av,group['v'])))
    # 位置更新
    group['pop'] = group['pop']+group['v']
    # 位置范围限制
    group['pop'] = np.array(np.mat(map(pso.f_ap,group['pop'])))
    # 适应度值更新
    group['fitness'] = np.array(map(pso.fun,group['pop']))
    # 更新由全局最优值形成的popsize*psize的种群
    bestfitness,best_index = pso.amin_index(np.array(
    np.mat(group['fitness'])),axis=1)
    bestfitness = bestfitness[0]
    best_index = best_index[0]
    if gbestfitness >= bestfitness:
        gbestfitness = bestfitness
        gbest0 = group['pop'][best_index,:]
    gbest = gbest0 # 由全局最优值形成的popsize*psize的矩阵
    for j in range(pso.popsize)[1:]:
        gbest = np.vstack((gbest,gbest0))
    # 更新整个种群的每个粒子的最优值组成的种群
    for j in range(pso.popsize):
        if popbestfitness[j] >= group['fitness'][j]:
            popbest[j] = group['pop'][j]
            popbestfitness[j] = group['fitness'][j]
    avgfitness = np.mean(group['fitness'])
    trace.append([avgfitness,gbestfitness])
    print u'进化代数：',i

# 结果分析
time_end = clock()
print u'计算用时：',time_end-time_start
gbestfitness = -1.0*gbestfitness # 求函数最小值时注释掉这行代码
print u'函数目标值：',gbestfitness
print u'对应的x1:',gbest0[0],u'对应的x2:',gbest0[1],u'对应的x3:',gbest0[2]

trace = np.array(trace)
trace = -1.0*trace # 求函数最小值时注释掉这行代码
x = np.arange(pso.maxgen)
plt.plot(x,trace[:,0],'b',x,trace[:,1],'r')
# plt.title('粒子群适应度值-进化代数')
plt.title('PSO fitness value-evolution generations')
# plt.legend(['平均适应度值','最优适应度值'])
plt.legend(['Mean-fitness','Best-fitness'])
# plt.xlabel('进化代数')
plt.xlabel('evolution-generations')
# plt.ylabel('适应度值')
plt.ylabel('fitness')
plt.show()




