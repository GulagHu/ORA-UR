import numpy as np
from My_algos import *
from gurobipy import *
import xlsxwriter as xw

if __name__ == '__main__':
    Services = [5, 10, 20, 50, 100, 200, 500]
    Types = [5, 10, 20, 50, 100, 200]
    num_samples = 30
    file_path = str('/home/??/')
    algo_names_n = ['Offline', 'Greedy', 'DPOL', 'Bayes']
    num_algo = len(algo_names_n)
    num_Ser = len(Services)
    num_Type = len(Types)
    Ru2 = 5

    T = 400

    workbook8 = xw.Workbook(file_path + 'test 8.xlsx')

    for name in algo_names_n:
        exec('worksheet_' + name + ' = workbook8.add_worksheet(\'' + name + '\')')

    for i in range(num_Ser):
        I = Services[i]
        c1 = np.full((I, 1), fill_value=50, dtype=int)
        for j in range(num_Type):
            J = Types[j]
            A = Create_A(I, J)
            r = Create_r(J, 20)
            p = Create_p(J)
            Rwd = np.zeros((num_samples, num_algo))
            for k in range(num_samples):
                R = Create_R(I, T, Ru2)
                Sequence = Create_Seq(p, T)
                for l in range(num_algo):
                    name = algo_names_n[l]
                    print('Workbook 8, I =', str(I), ', J =', str(J), ',sample', str(k + 1), ',Algo', name)
                    exec('Rwd[' + str(k) + ', ' + str(l) + '] = ' + name + '(A, c1, Sequence, R, \'n\', r)')
            Reward = np.mean(Rwd, axis=0)
            for l in range(num_algo):
                name = algo_names_n[l]
                exec('worksheet_' + name + '.write(' + str(i) + ',' + str(j) + ', Reward[' + str(l) + '])')

    workbook8.close()

    print('Test 8 completed!')
