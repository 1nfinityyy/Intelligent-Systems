import numpy as np

MAX=100
Delta=1e-10
gamma=0.9
P = np.array([[0.7], [0.1], [0.1], [0.1]])
award_pos = [(7, 8), (2, 7), (4, 3), (7, 3)]
award = [10, 3, -5, -10]
direction = [8, 2, 4, 6] #上8下2左4右6 任意是5 用小键盘对应

class Grid_World():
    def toward_wall(self, i, j): #是向着墙方向走 是<=!
        if (0 <= i and i < 10 and 0 <= j and j < 10):
            return False
        return True
    def act(self, i, j, direction, U):  # 在状态(i,j)处采取行动后得到的效用值
        ret = 0
        pos = np.array([[i,j],[i,j],[i,j],[i,j]])  # pos的四个新位置分别代表朝向当前方向下的前、后、左、右，转移概率为0.7,0.1,0.1,0.1
        R = np.zeros((4, 1)) #存储四个方向的奖赏
        U_s_prime = np.zeros((4, 1))
        if (i, j) in [(7, 8), (2, 7)]:  # 该位置执行任何动作都终止
            index = award_pos.index((i, j))
            return award[index]
        if direction == 8:
            pos = np.array([[i - 1, j], [i + 1, j], [i, j - 1], [i, j + 1]])
        elif direction == 2:
            pos = np.array([[i + 1, j], [i - 1, j], [i, j + 1], [i, j - 1]])
        elif direction == 4:
            pos = np.array([[i, j - 1], [i, j + 1], [i + 1, j], [i - 1, j]])
        elif direction == 6:
            pos = np.array([[i, j + 1], [i, j - 1], [i - 1, j], [i + 1, j]])
        for k in range(4):
            if self.toward_wall(pos[k, 0], pos[k, 1]):
                pos[k] = [i, j] #撞墙
                R[k] = [-1]
            if (i, j) in award_pos: #有奖赏
                index = award_pos.index((i, j))
                R[k] = [award[index]]
            U_s_prime[k] = U[pos[k, 0], pos[k, 1]] #上一个状态的的奖赏
        ret = P.transpose().dot(R) + gamma * P.transpose().dot(U_s_prime)
        return ret

    def best_policy(self, U):
        best_policy = np.array([[5 for i in range(10)] for j in range(10)])
        for i in range(10):
            for j in range(10):
                if (i, j) in [(7, 8), (2, 7)]:#这两个位置任意方向都可
                    continue
                lst = []
                for k in range(4):
                    lst.append(self.act(i, j, direction[k], U))
                best_policy[i, j] = direction[np.argmax(lst)]
        return best_policy

    def value_iteration(self):
        iter = 0
        U_old = np.zeros((10,10))
        U_new = np.zeros((10,10))
        while iter < MAX:
            for i in range(10):
                for j in range(10):
                    lst = []
                    for k in range(4):
                        lst.append(self.act(i, j, direction[k], U_old))
                    U_new[i, j] = max(lst)
            iter += 1
            result = U_new - U_old
            if (np.linalg.norm(result, ord=np.inf) < Delta):
                print(f'迭代次数为{iter},值迭代收敛')
                break
            U_old = U_new.copy()
        best_policy = self.best_policy(U_new)
        print("值迭代效用矩阵")
        print(U_new)
        print("最优策略是")
        for i in range(10):
            for j in range(10):
                if best_policy[i][j]==8:
                    print("Up",end=' ')
                elif best_policy[i][j] == 5:
                    print("Any", end=' ')
                elif best_policy[i][j] == 4:
                    print("Left", end=' ')
                elif best_policy[i][j] == 6:
                    print("Right", end=' ')
                elif best_policy[i][j] == 2:
                    print("Down", end=' ')
            print()

    def gauss_iteration(self):
        iter=0
        U=np.zeros((10, 10))
        U_old = np.zeros((10, 10))
        while iter<MAX:
            for i in range(0,10):
                for j in range(0,10):
                    lst=[]
                    for k in range(4):
                        lst.append(self.act(i,j,direction[k],U))
                    U[i,j]=max(lst)
            iter+=1

            result=U-U_old
            if np.linalg.norm(result,ord=np.inf)<Delta :
                print(f'高斯迭代次数为{iter},效用矩阵为')
                break
            U_old = U.copy()
        best_policy=self.best_policy(U)
        print(U)
        print("最优策略是")
        for i in range(10):
            for j in range(10):
                if best_policy[i][j] == 8:
                    print("Up", end=' ')
                elif best_policy[i][j] == 5:
                    print("Any", end=' ')
                elif best_policy[i][j] == 4:
                    print("Left", end=' ')
                elif best_policy[i][j] == 6:
                    print("Right", end=' ')
                elif best_policy[i][j] == 2:
                    print("Down", end=' ')
            print()

    def policy_evaluation(self,pi,n):
        U_old=np.zeros((10, 10))
        U=np.zeros((10, 10))
        iter=0
        while iter<n:
            for i in range(10):
                for j in range(10):
                    U_d=np.zeros((4,1))
                    total=0
                    for k in range(4):
                        if(pi[i,j,k]==0):
                            U_d[k,0]=0
                        else:
                            total+=1
                    for k in range(4):
                        if(pi[i,j,k]==0):
                            U_d[k,0]=0
                        else:
                            U_d[k,0]=self.act(i,j,direction[k],U_old)/total
                    U[i,j]=pi[i,j].dot(U_d)
            iter+=1
            result = U - U_old
            if (np.linalg.norm(result, ord=np.inf) < Delta):
                break
            U_old = U.copy()
        return U

    def policy_best(self,U,Policy):
        best=np.ones((10, 10, 4))
        for i in range(10):
            for j in range(10):
                if (i,j) in [(7,8),(2,7)]:
                    continue
                for d in range(4):
                    if Policy[i,j,d]==max(Policy[i,j]):
                        best[i,j,d]=1
                    else:
                        best[i, j, d] = 0
        return best

    def policy_iteration(self):
        # 定义一个3维矩阵，其中一个维度存放每个位置的策略 0 1 2 3 分别代表上下左右 为1即为可以采取该策略
        iter=0
        Policy = np.ones((10, 10, 4))
        Policy_old = np.ones((10, 10, 4))
        U_pi = np.zeros((10, 10)) #效用矩阵
        while iter < MAX:
            U_pi=self.policy_evaluation(Policy,MAX)
            for i in range(10):
                for j in range(10):
                    lst=[]
                    for d in range(4):
                        lst.append(self.act(i, j, direction[d], U_pi))
                    #可能有多个最佳动作，argmax可能出问题
                    index=[]
                    for d in range(4):
                        if lst[d]==max(lst):
                            index.append(d)
                    for k in range(4):
                        if k in index:
                            Policy[i,j,k]=1
                        else:
                            Policy[i,j,k]=0
            iter+=1
            if np.all(Policy == Policy_old):
                print(f"策略迭代收敛,迭代次数为{iter}")
                break
            Policy_old = Policy.copy()
        best_policy = self.policy_best(U_pi, Policy)
        print("策略迭代的效用矩阵为")
        print(U_pi)
        print("最优策略为：")
        Any = [1 , 1 , 1 , 1 ]
        for i in range(10):
            for j in range(10):
                if list(best_policy[i, j]) == Any:
                    print("Any", end=' ')
                else:  # 都是唯一动作，不用判断多个方向
                    # if sum(list( best_policy[i,j] )) < 1:
                    if best_policy[i, j, 0] != 0:
                        print("Up", end=' ')
                    elif best_policy[i, j, 1] != 0:
                        print("Down", end=' ')
                    elif best_policy[i, j, 2] != 0:
                        print("Left", end=' ')
                    elif best_policy[i, j, 3] != 0:
                        print("Right", end=' ')
                    # else:#都是唯一动作，这个判断没什么用
                    #     if best_policy[i,j,0]==1:
                    #         print("Up,",end='')
                    #     elif best_policy[i, j, 1] == 1:
                    #         print("Down,", end='')
                    #     elif best_policy[i, j, 2] == 1:
                    #         print("Left,", end='')
                    #     elif best_policy[i, j, 3] == 1:
                    #         print("Right,", end='')
                    # print('',end=' ')
            print()





dp = Grid_World()
np.set_printoptions(precision=2)
dp.value_iteration()
dp.gauss_iteration()
dp.policy_iteration()
