import numpy as np
from My_algos import *
from gurobipy import *
import xlsxwriter as xw

if __name__ == '__main__':
    delta_T = 10
    I = 10
    J = 15
    num_samples = 30
    file_path = str('/home/??/')
    algo_names_n = ['Offline', 'Greedy', 'DPOL', 'Bayes']

    A = np.array([[3, 4, 8, 5, 0, 15, 0, 1, 0, 0, 4, 9, 5, 1, 5],
                  [5, 1, 7, 9, 0, 0, 0, 9, 5, 3, 0, 5, 5, 2, 2],
                  [6, 9, 0, 8, 0, 8, 4, 13, 0, 16, 9, 10, 20, 8, 6],
                  [1, 1, 0, 0, 0, 12, 0, 2, 0, 1, 0, 2, 1, 0, 0],
                  [5, 0, 6, 1, 0, 1, 7, 9, 0, 9, 11, 4, 0, 4, 0],
                  [3, 5, 4, 1, 0, 10, 8, 0, 3, 5, 3, 7, 4, 1, 4],
                  [2, 6, 9, 0, 0, 15, 8, 2, 0, 2, 8, 15, 10, 4, 4],
                  [7, 0, 7, 5, 0, 0, 5, 3, 6, 1, 9, 15, 0, 8, 6],
                  [2, 2, 3, 0, 0, 3, 0, 0, 2, 0, 0, 12, 1, 1, 0],
                  [2, 8, 7, 0, 0, 14, 14, 11, 8, 1, 8, 6, 14, 7, 2]])

    r1 = [7, 5, 16, 1, 0, 20, 10, 18, 7, 14, 17, 19, 14, 1, 2]
    c1 = np.full((I, 1), fill_value=50, dtype=int)

    '''-------------------------Test 2: Unbalanced replenishment---------------------------'''

    workbook2 = xw.Workbook(file_path + 'test 2.xlsx')

    for name in algo_names_n:
        exec('worksheet_' + name + ' = workbook2.add_worksheet(\'' + name + '\')')

    for counter in range(1, 101):  # num_samples is scale
        T = delta_T * counter
        for i in range(num_samples):
            for name in algo_names_n:
                exec('time_' + name + ' = 0')  # time_Offline = 0

            R = Create_R(I, T, 20)
            Sequence = Create_Seq(p, T)  # Sequence is a list

            for name in algo_names_n:
                start = time.time()
                exec('reward_' + name + ' = ' + name + '(A, c1, Sequence, R, \'n\', r1)')
                exec('worksheet_' + name + '.write(' + str(counter - 1) + ',' + str(i) + ', reward_' + name + ')')
                stop = time.time()
                exec('time_' + name + ' = time_' + name + ' + ' + str(stop - start))
                print('Workbook 2,', name, ', T =', str(T), ', sample ', str(i + 1))

        for name in algo_names_n:
            exec('worksheet_' + name + '.write(' + str(counter - 1) + ',' + str(num_samples + 1) + ', time_' + name + ')/num_samples')

    workbook2.close()

    print('Test 2 completed!')
