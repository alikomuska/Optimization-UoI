import math
import random
import numpy as np
import copy
from bitarray import bitarray
from sympy import symbols, solve, Eq, sqrt
from scipy.optimize import fsolve

input_matrix = []
initial_points = []
r_penalty = 10000

#B = [ b0, b1, b2, b3, b4] to match the row columns 

#############################################################
####                 Newton                             #####
#############################################################

def check_conditions(b):
    limits = [[10,100], [-10, 10], [50,200], [0.01, 0.1], [0.1, 1]]
    for index in range(0, len(b)):
        if(b[index] < limits[index][0]):
            b[index] = limits[index][0]
        elif(b[index] > limits[index][1]):
            b[index] = limits[index][1]
    return b


def bstar_to_b(bstar):
    bounds = [[10,100], [-10, 10], [50,200], [0.01, 0.1], [0.1, 1]]
    b = []
    for index in range(0,5):
        b.append(bounds[index][0] + bounds[index][1]*bstar[index])
    return b


def b_to_bstar(b):
    bounds = [[10,100], [-10, 10], [50,200], [0.01, 0.1], [0.1, 1]]
    bstar = []
    for index in range(0,5):
        bstar.append((b[index] - bounds[index][0])/bounds[index][1])
    return bstar


def penalty_function(b):
    bounds = [[10,100], [-10, 10], [50,200], [0.01, 0.1], [0.1, 1]]
    penalty_value = 0
    for bi in range(0,5):
        if(b[bi] < bounds[bi][0]):
            penalty_value += r_penalty*(b[bi] - bounds[bi][0])**2 
        elif(b[bi] > bounds[bi][1]):
            penalty_value += r_penalty*(b[bi] - bounds[bi][1])**2 
    return penalty_value


def dPenaltyFunction_db(b):
    bounds = [[10,100], [-10, 10], [50,200], [0.01, 0.1], [0.1, 1]]
    penalty_vector = []
    for bi in range(0,5):
        if(b[bi] < bounds[bi][0]):
            penalty_vector.append(2*r_penalty*(b[bi] - bounds[bi][0]))
        elif(b[bi] > bounds[bi][1]):
            penalty_vector.append(2*r_penalty*(b[bi] - bounds[bi][1]))
        else:
            penalty_vector.append(0.0)

    return np.array(penalty_vector)


def e(b, row):
    #b = bstar_to_b(b) 
    #u = row[0]  theta = row[1],  T = row[2],  P = row[3],  E = row[4]
    return b[0]*(row[0]**2) + b[1]*math.sin(row[1]) + b[2]*math.exp(b[3]*row[2]) + b[4]*math.log(row[3])


def dE_db(b, row):
    #u = row[0]  theta = row[1],  T = row[2],  P = row[3],  E = row[4]
    #                u^2             sin(theta)        exp(b3*T)         b2*T+exp(b3*T)                           log(P)
    return np.array([row[0]**2, math.sin(row[1]), math.exp(b[3]*row[2]), b[2]*row[2]*math.exp(b[3]*row[2]),  math.log(row[3])])


def dE_dbi(bi, b, row):
    match bi:
        case 0:
            return row[0]**2
        case 1:
            return math.sin(row[1])
        case 2:
            return math.exp(b[3]*row[2])
        case 3:
            return b[2]*row[2]*math.exp(b[3]*row[2])
        case 4:
            return math.log(row[3])
    return


def d2E_dbi_dbj(bi, bj, b, row):
    if(bi == 3 and bj == 3):
        return b[2]*row[2]**2*math.exp(b[3]*row[2])
    elif((bi == 3 and bj == 2) or (bi == 2 and bj == 2)):
        return row[2]*math.exp(b[3]*row[2])
    else:
        return 0


def mse(b):
    mse_value = 0
    for row in input_matrix:
        mse_value += (e(b, row) - row[4])**2
    return mse_value/576 + penalty_function(b)


def dMse_db(b):
    vector = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    for row in input_matrix:
        vector += (e(b,row)-row[4]) * dE_db(b,row)
    return vector/288 + dPenaltyFunction_db(b)

def d2MSE_dbi_dbj(bi, bj, b):
    value = 0
    for row in input_matrix:
        value += dE_dbi(bi, b, row) * dE_dbi(bj, b, row) + (e(b,row) - row[4]) * d2E_dbi_dbj(bi, bj, b, row) 
    return value/288



def d2_mse(b):
    hessian = np.zeros((5,5))

    for i in range(0,5):
        for j in range(i,5):
            hessian[i][j] = hessian[j][i] = d2MSE_dbi_dbj(i, j, b) 
    return hessian + 2*r_penalty*np.eye(5)


def inverse_d2_mse(b):
    return np.linalg.inv(d2_mse(b))


#quadratic model
def mk(current_potition, p):
    return mse(current_potition) + np.dot(dMse_db(current_potition), p) + np.dot(p, d2_mse(current_potition).dot(p))/2


def get_rk(current_potition, pk):                              #mk(0) = f(xk)
    return (mse(current_potition)-mse(current_potition + pk))/(mse(current_potition) - mk(current_potition, pk))


def equation(t, pU, pB, Dk):
    v = pU + (t - 1) * (pB - pU)
    return np.linalg.norm(v) - Dk


#Dogleg
def get_direction(Dk, current_point):
    g = dMse_db(current_point)
    p_b = (-1)*inverse_d2_mse(current_point).dot(g)
    if(np.linalg.norm(p_b) <= Dk):
        return p_b
    
    p_u = (-1)*(np.dot(g,g))/(np.dot(g, d2_mse(current_point).dot(g))) * g
    if(np.linalg.norm(p_u) > Dk):
        p = (-1)*(Dk/np.linalg.norm(g))*g
    else:
        #t value seams ok. Check the algorithm
        t = symbols('t')
        pU, pB = symbols('pU pB', real=True, vector=True)
        delta_k = symbols('Delta_k', positive = True)

        t_initial_guess = 1.0
        t_solution = fsolve(equation, t_initial_guess, args = (p_u, p_b, Dk))
        p = p_u + (t_solution[0] - 1)*(p_b - p_u)

    return p


def NewtonTR(initial_point, Dk, error):
    current_point = initial_point
    maxD = Dk
    
    while(np.linalg.norm(dMse_db(current_point))>error and mse(current_point)>error):
        pk = get_direction(Dk,current_point)
        rk = get_rk(current_point, pk)

        #change Dk depending on the value rk
        if(rk < 0.0):
            Dk = Dk/4
        elif(rk > 0.75): #maybe i have to add something here 
            Dk = min(2*Dk, maxD)

        print("rk", str(rk))
        print("Dk", str(Dk))

        #check if the step is accepted
        if(rk > 0.0):
            current_point = current_point + pk

        #current_point = check_conditions(current_point)
        print("Value: " + str(mse(current_point)))
        print(current_point)

    print(current_point)
    return

###############################################################
####                     BFGS                            ######
###############################################################
#For BFGS use
def line_search(c1, r, current_point, p):
    a = 0.5
    while(mse(current_point + a*p) > (mse(current_point) + c1*a*np.dot(dMse_db(current_point),p))):
        a = a*r
    return a


def BFGS_update(H, s, y):
    I = np.eye(5) 
    r = 1/np.dot(y, s)
    H = (I - r*np.outer(s,y))*H*(I - r*np.outer(y,s)) + r*np.outer(s,s) 
    return H


def BFGSWolfe(current_point):
    c1 = 0.0001
    r = 0.8
    error = 0.01
    previews_point = current_point
    H = inverse_d2_mse(current_point)

    while(np.linalg.norm(dMse_db(current_point))>error and mse(current_point)>error):
        p =  (-1) * H.dot(dMse_db(current_point))
        a = line_search(c1, r, current_point, p)
        previews_point = current_point
        current_point = current_point + a*p
        s = current_point - previews_point
        y = dMse_db(current_point) - dMse_db(previews_point) 
        H = BFGS_update(H, s, y)
        print(mse(current_point))
    return



##############################################################
##########             Nelder-Mead                   #########
##############################################################

def initialize_simplex():
    return initialize_potitions_pso(6)


def worst_point(simplex):
    worst_b_index = 0
    worst_mse = mse(simplex[worst_b_index])
        
    for index in range(1, len(simplex)):
        if(mse(simplex[index]) > worst_mse):
            worst_b_index = index
            worst_mse = mse(simplex[worst_b_index])
    return worst_b_index


def best_point(simplex):
    best_b = simplex[0]
    best_mse = mse(best_b)
    for point in simplex[1:]:
        if(mse(point) < best_mse):
            best_b = point
            best_mse = mse(best_b)
    return best_b


def get_mean_b(simplex):
    mean_b = np.array([0.0 for i in range(0,5)])
    for point in simplex:
        mean_b += point
    return mean_b/len(simplex)


def update_points(r, worst_b_index, simplex):
    b_worst = simplex[worst_b_index]
    mse_worst = mse(b_worst)

    simplex.pop(worst_b_index)
    mean_b = get_mean_b(simplex)

    #Reflection
    ref_point = (1 + r)*mean_b - r*b_worst 
    ref_point_value = mse(ref_point)
    if(ref_point_value > mse(best_point(simplex)) and ref_point_value < mse(simplex[worst_point(simplex)])):
        print("Reflection")
        simplex.append(ref_point)
        return simplex

    #Expansion
    if(ref_point_value < mse(best_point(simplex))):
        r_exp = 1
        exp_point = (1 + r_exp)*mean_b - r_exp*b_worst 
        exp_point_value = mse(exp_point)
        if(exp_point_value < ref_point_value):
            simplex.append(exp_point)
            print("Expansion")
            return simplex
        else: 
            print("Expansion")
            simplex.append(ref_point)
            return simplex

    #External contraction
    if(mse(best_point(simplex)) < ref_point_value and ref_point_value < mse_worst):
        r_exc = 0.5 # to be checked
        exc_point = (1 + r_exc)*mean_b - r_exc*mse_worst
        if(mse(exc_point) < ref_point_value):
            print("External contraction")
            simplex.append(exc_point)
            return simplex
        else:
            print("External contraction")
            simplex.append(ref_point)
            return simplex

    #Internal contraction
    if(mse_worst < ref_point_value):
        r_inc = -0.5 #to be checked
        inc_point = (1 + r_inc)*mean_b - r_inc*b_worst 
        if(mse(inc_point) < mse_worst):
            print("Internal contraction")
            simplex.append(inc_point)
            return simplex

    #Shrink
    best_b = best_point(simplex)
    simplex.append(b_worst)
    for index in range(0, len(simplex)):
        simplex[index] = best_b -(simplex[index] - best_b)/2
    print("Shrink")
    return simplex

    return simplex


def NelderMead():
    r_ref = 0
    r_exp = 1


    simplex = initialize_simplex()
    best_point_value = mse(best_point(simplex))
    
    while(best_point_value > 1):
        worst_b_index = worst_point(simplex)
        simplex = update_points(r_ref, worst_b_index, simplex)
        simplex_best_point = best_point(simplex)
        best_point_value = mse(simplex_best_point)
        print(best_point_value)
        print(simplex_best_point)
        print()
    return





##############################################################
#########              GA                            #########
##############################################################


def initialize_population(N, chromosome_size):
    population = []
    for i in range(0,N):
        chromosome = bitarray(chromosome_size)
        for index in range(0, chromosome_size):
            chromosome[index] = random.choice([0, 1])
        population.append(chromosome)
    return population


def chromosome_to_bvalue(chromosome):
    #            b0, b1, b2, b3, b4
    bits_size = [20, 18, 18, 12, 15]
    u =  [90, 20, 150, 0.09, 0.9]
    umin = [10, -10, 50, 0.01, 0.1]
    b = []
    offset = 0

    for index in range(0,5):
        zi = 0
        for bit in range(offset, bits_size[index]+offset):
            zi += chromosome[bit]*2**(bit - offset)
        bi = (u[index]*zi)/(2**(bits_size[index])-1) + umin[index] 
        b.append(bi)
        offset += bits_size[index]
    return b


def roulette_wheel_selection(population, selection_number, s):
    selections = []
    ps = []
    f = []
    ftotal = 0
    
    population.sort()
    for index in range(1, len(population)):
        fi = 2 - s + 2*(s-1)*(index/(len(population)-1))
        f.append(fi)
        ftotal += f[-1]

    ps.append(f[0]/ftotal)
    intervals = [ps[0]]
    for fi in f[1:]:
        ps.append(fi/ftotal)
        intervals.append(intervals[-1] + ps[-1])

    for rand_selection in range(0, selection_number):
        rand_num = random.random()
        for interval in range(0,len(intervals)):
            if(rand_num < intervals[interval]): 
                selections.append(population[interval])
                break
    return selections


def crossover(S, cp, kc, uniform_crossover, uniform_prob):
    crossover_set = []
    crossovered_set = []

    for chromosome in S:
        if(random.random() < cp):
            crossover_set.append(chromosome)

    if(len(crossover_set)%2 == 1):
        crossover_set.append(S[random.randint(0,len(S)-1)])

    while(len(crossover_set)):
        crossover_pair = []
        first_chromosome = crossover_set.pop(random.randint(0,len(crossover_set)-1)) 
        second_chromosome = crossover_set.pop(random.randint(0,len(crossover_set)-1)) 
        crossover_pair.append(second_chromosome)
        crossover_pair.append(first_chromosome)
        crossovered_set += _do_crossover(crossover_pair, uniform_crossover, uniform_prob, kc)

    return crossovered_set


def _do_crossover(crossover_pair, uniform_crossover, uniform_prob, kc):
    #Uniform Crossover with probability uniform_prob
    if(uniform_crossover):
        crossover_points = []
        for index in range(1, len(crossover_pair[0])-2):
            if(random.random() < uniform_prob):
                crossover_points.append(index)        
        return _k_point_crossover(crossover_pair, crossover_points)

    #kc point Crossover
    crossover_points = []
    for index in range(0,kc):
        random_kc = random.randint(1, len(crossover_pair[0]))
        while(random_kc in crossover_points):
            random_kc = random.randint(1, len(crossover_pair[0]))
        crossover_points.append(random_kc)

    crossover_points.sort()
    return _k_point_crossover(crossover_pair, crossover_points)


def _k_point_crossover(crossover_pair, crossover_points):
    if(len(crossover_points) == 0):
        return crossover_pair

    crossing = 0
    o1 = crossover_pair[crossing][:crossover_points[0]]
    o2 = crossover_pair[1 - crossing][0:crossover_points[0]]
    crossing = 1 - crossing

    for kc in crossover_points[1:]:
        o1 = o1 + crossover_pair[crossing][len(o1):kc]
        o2 = o2 + crossover_pair[1 - crossing][len(o2):kc]
        crossing = 1 - crossing

    o1 = o1 + crossover_pair[crossing][len(o1):]
    o2 = o2 + crossover_pair[1 - crossing][len(o2):]
    return [o1, o2]


def mutation(C, mp):
    for chromosome in C:
        for index in range(0, len(chromosome)):
            if(random.random()<mp):
                chromosome[index] = 1 - chromosome[index]
    return C


def evaluate(P):
    bvalues = []
    for chromosome in P:
        bvalues.append(mse(chromosome_to_bvalue(chromosome)))
    return bvalues

# isos tha eprepe na ginei find_nth_chromosome(P,n) oste na briskei to n kalitero
def find_best_chromosome(P):
    b_best = chromosome_to_bvalue(P[0])
    mse_value_best = mse(b_best)

    for chromosome in P[1:]:
        if(mse(chromosome_to_bvalue(chromosome)) < mse_value_best):
            b_best = chromosome_to_bvalue(chromosome)
            mse_value_best = mse(b_best)
    return b_best, mse_value_best


def new_population(P, M):
    for m in M:
        #find min value of P to compare with each m of M if mse(minP)< mse(m): swap(minP, m)
        worst_chrom = P[0]
        worst_mse = mse(chromosome_to_bvalue(worst_chrom))
        for p in P[1:]:
            if(mse(chromosome_to_bvalue(p)) > worst_mse):
                worst_chrom = p
                worst_mse = mse(chromosome_to_bvalue(p))

        if(mse(chromosome_to_bvalue(m))<worst_mse):
            P.remove(worst_chrom)
            P.append(m)

    return P


def GA():
    N = 25
    s = 1.5
    roulette_wheel_selection_num = 14
    
    cp = 1
    uniform_crossover = 0 # 0 = kc crossover    1 = uniform_crossover
    uniform_prob = 0.2

    kc = 4
    mp = 0.6
    
    P = initialize_population(N, 83)
    b_best, mse_value_best = find_best_chromosome(P) 
    print(mse_value_best)
    while (mse_value_best > 0.01):
        S = roulette_wheel_selection(P, roulette_wheel_selection_num, s)
        C = crossover(S, cp, kc, uniform_crossover, uniform_prob)
        M = mutation(C, mp)
        P = new_population(P, M)
        b_best, mse_value_best = find_best_chromosome(P) 
        print(mse_value_best)
    return best_mse_value


#############################################################
####                      PSO                           #####
#############################################################


def initialize_potitions_pso(N):
    #should i make this better?
    #add the initial point 

    intervals = [[10, 100], [-10, 10], [50, 200], [0.01,0.1], [0.1, 1]]
    S = []

    for index in range(0,N):
        b = []
        for bi in range(0,5):
            if(bi == 3):
                b.append(random.randint(100, 1000)/10000)
                
            elif(bi == 4):
                b.append(random.randint(100, 1000)/1000)
            else:
                b.append(random.randint(intervals[bi][0], intervals[bi][1]))
        S.append(np.array(b))
    return S


def find_neighborhood_best(i, radius, best_potitions):
    N = len(best_potitions)
    best_b = best_potitions[(i-radius)%N]
    best_neighborhood_value = mse(best_b) 
    for index in range(i-radius+1, i+radius+1):
        if(mse(best_potitions[index%N]) < best_neighborhood_value):
            best_b = best_potitions[(i-radius)%N]
            best_neighborhood_value = mse(best_b) 
    return best_b


def update_velocities(c1, c2, constriction, radius, potitions, velocities, best_potitions):
    for index in range(0, len(potitions)):
        #find neighborhood best
        neighborhood_best = find_neighborhood_best(index, radius, best_potitions)
        velocities[index] = constriction*(velocities[index] + random.random()*c1*(best_potitions[index] - potitions[index])+ random.random()*c2*neighborhood_best)
    return velocities


def update_best_positions(best_potitions, potitions):
    for index in range(0, len(potitions)):
        if(mse(potitions[index]) < mse(best_potitions[index])):
            best_potitions[index] = copy.deepcopy(potitions[index])
    return best_potitions


def get_best_value(best_potitions):
    best_value = mse(best_potitions[0])
    for potition in best_potitions[1:]:
        if(mse(potition) < best_value):
            best_value = mse(potition)
    return best_value


def check_potitions_bounds(potitions):
    intervals = [[10, 100], [-10, 10], [50, 200], [0.01,0.1], [0.1, 1]]
    for potition in range(0, len(potitions)):
        for bi in range(0,4):
            if(potitions[potition][bi] < intervals[bi][0]):
                potitions[potition][bi] = intervals[bi][0]
            elif(potitions[potition][bi] > intervals[bi][1]):
                potitions[potition][bi] = intervals[bi][1]
    return potitions


def check_velocities_bounds(velocities, vmax):
    for velocity in range(0, len(velocities)):
        velocity_value = np.linalg.norm(velocities[velocity])
        if(velocity_value > vmax):
            velocities[velocity] = (vmax/velocity_value) * velocities[velocity]
    return velocities


def PSO():
    N = 12
    radius = 7
    constriction = 0.729
    c1 = 2.05
    c2 = 2.05
    vmax = 1.2

    potitions = initialize_potitions_pso(N)
    best_potitions = copy.deepcopy(potitions)

    velocities = [np.array([0.0 for i in range(0,5)]) for n in range(0,N)]
    best_value = get_best_value(best_potitions)
    
    while(best_value > 0.1):
        for index in range(0,N):
            velocities = update_velocities(c1, c2, constriction, radius, potitions, velocities, best_potitions)
            #check bounds
            velocities = check_velocities_bounds(velocities, vmax)

            potitions[index] += velocities[index]
            #check potition bounds
            potitions = check_potitions_bounds(potitions)

        best_potitions = update_best_positions(best_potitions, potitions)
        best_value = get_best_value(best_potitions)
        print(best_value)
    return



#load values to memory 
with open('data_train.txt') as file:
    for line in file:
        row = []
        for i in range(0,5):
            row.append(float(line.split()[i]))
        input_matrix.append(row)

#load initial points to memory 
with open('initial_points.txt') as file:
    for line in file:
        row = []
        for i in range(0, 5):
            row.append(float(line.split()[i]))
        initial_points.append(row)
            

#NewtonTR(initial_points[3], 0.7, 0.001)
BFGSWolfe(initial_points[0])
#NelderMead()
#GA()
#PSO()
