import numpy as np
import gurobipy as gp
from gurobipy import *

'''--------------------Run Offline Benchmarks------------------------'''


def Offline(A, c, Sequence, R, p, r):
    model = Model("Offline")
    I = A.shape[0]
    T = len(Sequence)  # Sequence is a list

    Service = range(I)
    Horizon = range(T)
    Y = model.addVars(T, ub=1.0)
    model.update()

    model.setObjective(quicksum(r[Sequence[t]] * Y[t] for t in Horizon), GRB.MAXIMIZE)

    for i in Service:
        for t in Horizon:
            History = range(t + 1)
            model.addConstr(sum(A[i, Sequence[t_]] * Y[t_] for t_ in History) <= c[i, 0] + sum(R[i, t_] for t_ in History))

    model.update()
    model.setParam('OutputFlag', 0)
    model.optimize()

    objval = model.objVal

    return objval


'''-------------------------Run Greedy--------------------------------'''


def Greedy(A, c, Sequence, R, p, r):
    reward = 0
    T = len(Sequence)
    Horizon = range(T)
    decision = np.full((T,), fill_value=False)
    I = A.shape[0]
    C_t = c + np.zeros((I, 1), dtype=int)
    for t in Horizon:
        j = Sequence[t]
        C_t = C_t + R[:, [t]]
        if (A[:, [j]] <= C_t).all():  # Enough resources?
            C_t = C_t - A[:, [j]]
            reward = reward + r[j]
            decision[t] = True

    return reward


'''-------------------------Run DPOL --------------------------------'''


def DPOL(A, c, Sequence, R, p, r):
    reward = 0
    I = A.shape[0]
    T = len(Sequence)

    Horizon = range(T)

    decision = np.full((T,), fill_value=False)
    C_t = c + np.zeros((I, 1), dtype=int)

    for t in Horizon:
        j = Sequence[t]
        C_t = C_t + R[:, [t]]

        P = LP_SAA(A, c, Sequence, R, t, r)
        if np.dot(A[:, j], P) < r[j] and (A[:, [j]] <= C_t).all():
            C_t = C_t - A[:, [j]]
            reward = reward + r[j]
            decision[t] = True

    return reward


'''--------------------------Run Bayes--------------------------------'''


def Bayes(A, c, Sequence, R, p, r):
    I = A.shape[0]
    J = A.shape[1]
    T = len(Sequence)

    Service = range(I)
    Horizon = range(1, T+1)
    Type = range(J)

    reward = 0
    C_t = c + np.zeros((I, 1), dtype=int)
    decision = np.full((T,), fill_value=False)

    if p == 'n':
        p = [0] * J
        counter = [0] * J
        for t in Horizon:
            j = Sequence[t - 1]
            counter[j] = counter[j] + 1
            for j_ in Type:
                p[j_] = counter[j_]/sum(counter)   # Update request arrival probability by learning
            C_t = C_t + R[:, [t - 1]]
            if not all(C_t[i] >= A[i, j] for i in Service):
                continue  # Services are not enough

            y = LP_2(A, C_t, T, t, p, r)

            if y[j] / t * p[j] >= 0.5:  # Accept or Reject
                decision[t - 1] = True
                reward = reward + r[j]
                C_t = C_t - A[:, [j]]

        return reward
    else:
        for t in Horizon:
            j = Sequence[t - 1]
            C_t = C_t + R[:, [t - 1]]
            if not all(C_t[i] >= A[i, j] for i in Service):
                continue  # no enough services

            y = LP_2(A, C_t, T, t, p, r)

            if y[j] >= (T-t) * p[j]/2:  # Accept
                decision[t - 1] = True
                reward = reward + r[j]
                C_t = C_t - A[:, [j]]

        return reward


'''-------------------------------Auxiliary Algorithms--------------------------------'''


def Create_A(I, J):
    A_bar = np.random.randint(3, 10, (I, 1))
    A = np.zeros([I, J], dtype=int)
    for i in range(I):
        A_temp = np.random.randint(0, A_bar[i], (1, J))
        A[[i], :] = A_temp

    return A


def Create_R(I, T, Ru):
    R = np.random.randint(0, Ru, (I, T))

    return R


def Create_r(J, r_bar):
    r = np.random.randint(0, r_bar, (J, ))
    r = r.tolist()

    return r


def Create_p(J):
    p = np.random.randint(0, 100, (J, ))
    p = p / sum(p)
    p = p.tolist()

    return p


def LP_2E(A, c, T, E_R, p, r):
    model = Model("LP_2E")
    I = A.shape[0]
    J = A.shape[1]

    Service = range(I)
    Type = range(J)
    Horizon = range(T)
    K = c / T + E_R
    y = model.addVars(J, T, ub=1.0)
    model.setObjective(quicksum(r[j] * y[j, t] for t in Horizon for j in Type), GRB.MAXIMIZE)
    model.update()

    for i in Service:
        for t in Horizon:
            # R is I-by-T matrices denoting replenishment
            model.addConstr(quicksum(p[j] * A[i, j] * y[j, t] for j in Type) <= K[i])
    for j in Type:
        for t in Horizon:
            model.addConstr(y[j, t] <= p[j])
    model.update()
    model.setParam('OutputFlag', 0)
    model.optimize()

    solution = np.zeros((J, T))
    for j in Type:
        for t in Horizon:
            solution[j, t] = y[j, t].X

    return solution


def LP_2(A, C_t, T, t, p, r):
    model = Model("LP_2")
    I = A.shape[0]
    J = A.shape[1]

    Service = range(I)
    Type = range(J)
    y = model.addVars(J)
    model.setObjective(quicksum(r[j] * y[j] for j in Type), GRB.MAXIMIZE)
    model.update()

    for i in Service:
        model.addConstr(quicksum(A[i, j] * y[j] for j in Type) <= C_t[i])
    for j in Type:
        model.addConstr(y[j] <= (T-t)*p[j])   # I is from 1 to T
    model.update()
    model.setParam('OutputFlag', 0)
    model.optimize()

    solution = np.zeros((J, ))
    for j in Type:
        solution[j] = y[j].X

    return solution


def LP_SAA(A, c, Sequence, R, t, r):
    model = Model("LP_SAA")
    I = A.shape[0]
    T = len(Sequence)

    Service = range(I)

    p = model.addVars(I, ub=max(r))
    z = model.addVars(T)
    model.update()

    term_1 = 0
    term_2 = 0
    term_3 = 0

    for i in Service:
        term_1 = term_1 + c[i][0] * p[i] / (t+1)
        for t_ in range(t+1):
            B_it = R[i, t_]
            term_2 = term_2 + B_it * p[i] / (t+1)

    for t_ in range(t+1):
        term_3 = term_3 + z[t_] / (t + 1)
        j = Sequence[t_]
        model.addConstr(quicksum(A[i, j] * p[i] for i in Service) + z[t_] >= r[j])

    model.setObjective(term_1 + term_2 + term_3, GRB.MINIMIZE)
    model.update()
    model.setParam('OutputFlag', 0)
    model.optimize()

    solution = np.zeros((I, ))
    for i in Service:
        solution[i] = p[i].X

    return solution


def Create_Seq(p, T):
    J = len(p)
    p_cum = np.cumsum(p)
    seed = np.random.rand(1, T)[0]
    Sequence = []
    for t in range(T):
        for k in range(J):
            if seed[t] <= p_cum[k]:
                Sequence = Sequence + [k]
                break

    return Sequence
