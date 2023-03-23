import numpy as np
import pandas as pd

def PROMETHEE_2(Decision_Matrix, q_thresholds, p_thresholds, scurve_thresholds, weights, pref_functions):
    '''
    This function returns the flows of PROMETHEE I and PROMETHEE II

    inputs: 
        Decision_Matrix: a np.array(shape=(m,n)) with the performances of m alternatives regarding n criteria
        q_thresholds: a n-dimensional vector receiving indifference thresholds for the criteria
        p_thresholds: a n-dimensional vector receiving preference thresholds for the criteria
        scurve_thresholds: a n-dimensional vector receiving the scurve thresholds to be used in case the Gaussian function is chosen
        weights: a n-dimensional vector receiving weights for the criteria
        pref_function: a n-dimensional vector receiving as input values between 1 and 5 indicating the preference function for each criterion 

    outputs:
        results: pandas DataFrame with net, positive and negative flows for each alternative
        net_flows: m-dimensional vector of net flows
        pos_flows: m-dimensional vector of positive flows
        neg_flows: m-dimensional vector of negative flows
    '''
    
    # Get a local copy for each input
    X= np.copy(Decision_Matrix)
    w = np.copy(weights)
    q = np.copy(q_thresholds)
    p = np.copy(p_thresholds)
    s_curve = np.copy(scurve_thresholds)
    pref = np.copy(pref_functions)

    m, n = X.shape #m is the number of alternatives, n is the number of criteria
    D = np.zeros(shape=(n,m,m)) # tensor that will receive n matrices (mxm) of pairwise differences 
    P = np.zeros(shape=(n,m,m)) # tensor that will receive n matrices (mxm) of preference functions according with the given p_thresholds
    for j in range (n): #loop over criteria
        criterion_fun = pref[j] # local variable receives the information about the current preference function
        for k in range (m):
            for l in range (m):
                D[j,k,l] = X[k,j] - X[l,j]  #pairwise difference             
                # According to criterion_fun, apply the rule to go from D to P
                if criterion_fun == 1: P[j,k,l] = 0 if D[j,k,l] <=0 else 1
                if criterion_fun == 2: P[j,k,l] = 0 if D[j,k,l] <= q[j] else 1
                if criterion_fun == 3: P[j,k,l] = 0 if D[j,k,l] <= 0 else D[j,k,l] / p[j] if D[j,k,l] <= p[j] else 1
                if criterion_fun == 4: P[j,k,l] = 0 if D[j,k,l] <= q[j] else 0.5 if D[j,k,l] <= p[j] else 1
                if criterion_fun == 5: P[j,k,l] = 0 if D[j,k,l] <= q[j] else (D[j,k,l] - q[j])/(p[j]-q[j]) if D[j,k,l] <= p[j] else 1
                if criterion_fun == 6: P[j,k,l] = 0 if D[j,k,l] <=0 else 1 - np.exp(-(D[j,k,l]**2 / 2*s_curve[j]**2))

    Aggregated_P =  np.zeros(shape=(m,m)) # Aggregated_P gets the overal weighted pairwise value  
    for i in range (m):
        for j in range (m):
            Aggregated_P[i,j] = np.dot(P[:,i,j], w)

    pos_flows = np.sum(Aggregated_P, axis=1) / (m-1) # Vector of positive flows
    neg_flows = np.sum(Aggregated_P, axis=0) / (m-1) # Vector of negative flows
    net_flows = pos_flows - neg_flows # Vector of net flows
    
    # Construct a DataFrame of results
    results = pd.DataFrame({"Net Flows": net_flows, "Positive Flows": pos_flows, "Negative Flows": neg_flows})

    return results, net_flows, pos_flows, neg_flows


import pyomo.environ as pyo
def optimize_PROM2_original (unit_sigmamu, Fr_sigmamu):
    #Instanciando Modelo
    M = pyo.ConcreteModel() # instancia do modelo

    # Criando índices
    I = M.I = pyo.RangeSet(1) # range para a unidade em análise
    I_fr = M.I_fr = pyo.RangeSet(Fr_sigmamu.shape[0]) # Range for alternatives in fr
    
    #Variáveis de Decisão
    alpha = M.alpha = pyo.Var(within=pyo.NonNegativeReals)
    beta = M.beta = pyo.Var(within=pyo.NonNegativeReals)
    zeta = M.zeta = pyo.Var()

    # Parâmetros
    Sigma_i = M.Sigma_i = pyo.Param (I, initialize = lambda M, i: unit_sigmamu[i-1 , 0])
    Mu_i = M.Mu_i = pyo.Param (I, initialize = lambda M, i: unit_sigmamu[i-1 , 1])
    Sigma_fri = M.Sigma_fri = pyo.Param (I_fr, initialize = lambda M, i: Fr_sigmamu[i-1 , 0])
    Mu_fri = M.Mu_fri = pyo.Param (I_fr, initialize = lambda M, i: Fr_sigmamu[i-1 , 1])

    # Função objetivo
    obj = M.obj = pyo.Objective(rule = M.zeta, sense= pyo.maximize)

    # Restrição de eficiência
    M.R1 = pyo.Constraint(I, I_fr, \
        rule= lambda M, i, i_fr: (alpha * Mu_fri[i_fr] - beta * Sigma_fri[i_fr] + zeta) <= (alpha * Mu_i[i] - beta * Sigma_i[i]))

    # Restrição tipo 2 -> alfa+beta=1
    M.R2 = pyo.Constraint(rule = alpha + beta == 1)
    
    # Restrição de alpha >= 2beta
    #modelo.R3 = pyo.Constraint(rule = modelo.alfa >= modelo.beta)

    # Resolução
    glpk = pyo.SolverFactory('glpk') # Construindo o solver gurobi
    result = glpk.solve(M)

    return M, result


def optimize_newconstraint (unit_sigmamu, Fr_sigmamu):
    
    #Instanciando Modelo
    M = pyo.ConcreteModel() # instancia do modelo
    
    # Criando índices
    I = M.I = pyo.RangeSet(1) # range para a unidade em análise
    I_fr = M.I_fr = pyo.RangeSet(Fr_sigmamu.shape[0]) # Range for alternatives in fr
    
    #Variáveis de Decisão
    alpha = M.alpha = pyo.Var(within=pyo.NonNegativeReals)
    beta = M.beta = pyo.Var(within=pyo.NonNegativeReals)
    zeta = M.zeta = pyo.Var()

    # Parâmetros
    Sigma_i = M.Sigma_i = pyo.Param (I, initialize = lambda M, i: unit_sigmamu[i-1 , 0])
    Mu_i = M.Mu_i = pyo.Param (I, initialize = lambda M, i: unit_sigmamu[i-1 , 1])
    Sigma_fri = M.Sigma_fri = pyo.Param (I_fr, initialize = lambda M, i: Fr_sigmamu[i-1 , 0])
    Mu_fri = M.Mu_fri = pyo.Param (I_fr, initialize = lambda M, i: Fr_sigmamu[i-1 , 1])

    # Função objetivo
    obj = M.obj = pyo.Objective(rule = M.zeta, sense= pyo.maximize)

    # Restrição de eficiência
    M.R1 = pyo.Constraint(I, I_fr, \
        rule= lambda M, i, i_fr: (alpha * Mu_fri[i_fr] - beta * Sigma_fri[i_fr] + zeta) <= (alpha * Mu_i[i] - beta * Sigma_i[i]))

    # Restrição tipo 2 -> alfa+beta=1
    M.R2 = pyo.Constraint(rule = alpha + beta == 1)
    
    # Restrição de alpha >= 2beta
    M.R3 = pyo.Constraint(rule = alpha >= 2 * beta)

    # Resolução
    glpk = pyo.SolverFactory('glpk') # Construindo o solver gurobi
    result = glpk.solve(M)

    return M, result

def optimize_PROM1_original_pos_flow (unit_sigmamu, Fr_sigmamu):
    #Instanciando Modelo
    M = pyo.ConcreteModel() # instancia do modelo

    # Criando índices
    I = M.I = pyo.RangeSet(1) # range para a unidade em análise
    I_fr = M.I_fr = pyo.RangeSet(Fr_sigmamu.shape[0]) # Range for alternatives in fr
    
    #Variáveis de Decisão
    alpha_pos = M.alpha_pos = pyo.Var(within=pyo.NonNegativeReals)
    beta_pos = M.beta_pos = pyo.Var(within=pyo.NonNegativeReals)
    zeta_pos = M.zeta_pos = pyo.Var()

    # Parâmetros
    Sigma_i_pos = M.Sigma_i_pos = pyo.Param (I, initialize = lambda M, i: unit_sigmamu[i-1 , 0])
    Mu_i_pos = M.Mu_i_pos = pyo.Param (I, initialize = lambda M, i: unit_sigmamu[i-1 , 1])
    Sigma_fri_pos = M.Sigma_fri_pos = pyo.Param (I_fr, initialize = lambda M, i: Fr_sigmamu[i-1 , 0])
    Mu_fri_pos = M.Mu_fri_pos = pyo.Param (I_fr, initialize = lambda M, i: Fr_sigmamu[i-1 , 1])

    # Função objetivo
    obj = M.obj = pyo.Objective(rule = M.zeta_pos, sense= pyo.maximize)

    # Restrição de eficiência
    M.R1 = pyo.Constraint(I, I_fr, \
        rule= lambda M, i, i_fr: (alpha_pos * Mu_fri_pos[i_fr] - beta_pos * Sigma_fri_pos[i_fr] + zeta_pos) <= (alpha_pos * Mu_i_pos[i] - beta_pos * Sigma_i_pos[i]))

    # Restrição tipo 2 -> alfa+beta=1
    M.R2 = pyo.Constraint(rule = alpha_pos + beta_pos == 1)
    
    # Restrição de alpha >= 2beta
    #modelo.R3 = pyo.Constraint(rule = modelo.alfa >= modelo.beta)

    # Resolução
    glpk = pyo.SolverFactory('glpk') # Construindo o solver gurobi
    result = glpk.solve(M)

    return M, result

def optimize_newconstraint_pos_flow (unit_sigmamu, Fr_sigmamu):
    
    #Instanciando Modelo
    M = pyo.ConcreteModel() # instancia do modelo
    
    # Criando índices
    I = M.I = pyo.RangeSet(1) # range para a unidade em análise
    I_fr = M.I_fr = pyo.RangeSet(Fr_sigmamu.shape[0]) # Range for alternatives in fr
    
    #Variáveis de Decisão
    alpha_pos = M.alpha_pos = pyo.Var(within=pyo.NonNegativeReals)
    beta_pos = M.beta_pos = pyo.Var(within=pyo.NonNegativeReals)
    zeta_pos = M.zeta_pos = pyo.Var()

    # Parâmetros
    Sigma_i_pos = M.Sigma_i_pos = pyo.Param (I, initialize = lambda M, i: unit_sigmamu[i-1 , 0])
    Mu_i_pos = M.Mu_i_pos = pyo.Param (I, initialize = lambda M, i: unit_sigmamu[i-1 , 1])
    Sigma_fri_pos = M.Sigma_fri_pos = pyo.Param (I_fr, initialize = lambda M, i: Fr_sigmamu[i-1 , 0])
    Mu_fri_pos = M.Mu_fri_pos = pyo.Param (I_fr, initialize = lambda M, i: Fr_sigmamu[i-1 , 1])

    # Função objetivo
    obj = M.obj = pyo.Objective(rule = M.zeta_pos, sense= pyo.maximize)

    # Restrição de eficiência
    M.R1 = pyo.Constraint(I, I_fr, \
        rule= lambda M, i, i_fr: (alpha_pos * Mu_fri_pos[i_fr] - beta_pos * Sigma_fri_pos[i_fr] + zeta_pos) <= (alpha_pos * Mu_i_pos[i] - beta_pos * Sigma_i_pos[i]))

    # Restrição tipo 2 -> alfa+beta=1
    M.R2 = pyo.Constraint(rule = alpha_pos + beta_pos == 1)
    
    # Restrição de alpha >= 2beta
    M.R3 = pyo.Constraint(rule = alpha_pos >= 2 * beta_pos)

    # Resolução
    glpk = pyo.SolverFactory('glpk') # Construindo o solver gurobi
    result = glpk.solve(M)

    return M, result

def optimize_PROM1_original_neg_flow (unit_sigmamu, Fr_sigmamu):
    #Instanciando Modelo
    M = pyo.ConcreteModel() # instancia do modelo

    # Criando índices
    I = M.I = pyo.RangeSet(1) # range para a unidade em análise
    I_fr = M.I_fr = pyo.RangeSet(Fr_sigmamu.shape[0]) # Range for alternatives in fr
    
    #Variáveis de Decisão
    alpha_neg = M.alpha_neg = pyo.Var(within=pyo.NonNegativeReals)
    beta_neg = M.beta_neg = pyo.Var(within=pyo.NonNegativeReals)
    zeta_neg = M.zeta_neg = pyo.Var()

    # Parâmetros
    Sigma_i_neg = M.Sigma_i_neg = pyo.Param (I, initialize = lambda M, i: unit_sigmamu[i-1 , 0])
    Mu_i_neg = M.Mu_i_neg = pyo.Param (I, initialize = lambda M, i: unit_sigmamu[i-1 , 1])
    Sigma_fri_neg = M.Sigma_fri_neg = pyo.Param (I_fr, initialize = lambda M, i: Fr_sigmamu[i-1 , 0])
    Mu_fri_neg = M.Mu_fri_neg = pyo.Param (I_fr, initialize = lambda M, i: Fr_sigmamu[i-1 , 1])

    # Função objetivo
    obj = M.obj = pyo.Objective(rule = M.zeta_neg, sense= pyo.maximize)

    # Restrição de eficiência
    M.R1 = pyo.Constraint(I, I_fr, \
        rule= lambda M, i, i_fr: (alpha_neg * Mu_fri_neg[i_fr] - beta_neg * Sigma_fri_neg[i_fr] + zeta_neg) <= (alpha_neg * Mu_i_neg[i] - beta_neg * Sigma_i_neg[i]))

    # Restrição tipo 2 -> alfa+beta=1
    M.R2 = pyo.Constraint(rule = alpha_neg + beta_neg == 1)
    
    # Restrição de alpha >= 2beta
    #modelo.R3 = pyo.Constraint(rule = modelo.alfa >= modelo.beta)

    # Resolução
    glpk = pyo.SolverFactory('glpk') # Construindo o solver gurobi
    result = glpk.solve(M)

    return M, result


def optimize_newconstraint_neg_flow (unit_sigmamu, Fr_sigmamu):
    
    #Instanciando Modelo
    M = pyo.ConcreteModel() # instancia do modelo
    
    # Criando índices
    I = M.I = pyo.RangeSet(1) # range para a unidade em análise
    I_fr = M.I_fr = pyo.RangeSet(Fr_sigmamu.shape[0]) # Range for alternatives in fr
    
    #Variáveis de Decisão
    alpha_neg = M.alpha_neg = pyo.Var(within=pyo.NonNegativeReals)
    beta_neg = M.beta_neg = pyo.Var(within=pyo.NonNegativeReals)
    zeta_neg = M.zeta_neg = pyo.Var()

    # Parâmetros
    Sigma_i_neg = M.Sigma_i_neg = pyo.Param (I, initialize = lambda M, i: unit_sigmamu[i-1 , 0])
    Mu_i_neg = M.Mu_i_neg = pyo.Param (I, initialize = lambda M, i: unit_sigmamu[i-1 , 1])
    Sigma_fri_neg = M.Sigma_fri_neg = pyo.Param (I_fr, initialize = lambda M, i: Fr_sigmamu[i-1 , 0])
    Mu_fri_neg = M.Mu_fri_neg = pyo.Param (I_fr, initialize = lambda M, i: Fr_sigmamu[i-1 , 1])

    # Função objetivo
    obj = M.obj = pyo.Objective(rule = M.zeta_neg, sense= pyo.maximize)

    # Restrição de eficiência
    M.R1 = pyo.Constraint(I, I_fr, \
        rule= lambda M, i, i_fr: (alpha_neg * Mu_fri_neg[i_fr] - beta_neg * Sigma_fri_neg[i_fr] + zeta_neg) <= (alpha_neg * Mu_i_neg[i] - beta_neg * Sigma_i_neg[i]))

    # Restrição tipo 2 -> alfa+beta=1
    M.R2 = pyo.Constraint(rule = alpha_neg + beta_neg == 1)
    
    # Restrição de alpha >= 2beta
    M.R3 = pyo.Constraint(rule = alpha_neg >= 2 * beta_neg)

    # Resolução
    glpk = pyo.SolverFactory('glpk') # Construindo o solver gurobi
    result = glpk.solve(M)

    return M, result

def optimize_PROM1_original (unit_sigmamu, Fr_sigmamu):
    #Instanciando Modelo
    M = pyo.ConcreteModel() # instancia do modelo

    # Criando índices
    I = M.I = pyo.RangeSet(1) # range para a unidade em análise
    I_fr = M.I_fr = pyo.RangeSet(Fr_sigmamu.shape[0]) # Range for alternatives in fr
    
    #Variáveis de Decisão
    alpha_pos = M.alpha_pos = pyo.Var(within=pyo.NonNegativeReals)
    beta_pos = M.beta_pos = pyo.Var(within=pyo.NonNegativeReals)
    alpha_neg = M.alpha_neg = pyo.Var(within=pyo.NonNegativeReals)
    beta_neg = M.beta_neg = pyo.Var(within=pyo.NonNegativeReals)
    zeta_p1 = M.zeta_p1 = pyo.Var()

    # Parâmetros
    Sigma_i_pos = M.Sigma_i_pos = pyo.Param (I, initialize = lambda M, i: unit_sigmamu[i-1 , 0])
    Mu_i_pos = M.Mu_i_pos = pyo.Param (I, initialize = lambda M, i: unit_sigmamu[i-1 , 1])
    Sigma_fri_pos = M.Sigma_fri_pos = pyo.Param (I_fr, initialize = lambda M, i: Fr_sigmamu[i-1 , 0])
    Mu_fri_pos = M.Mu_fri_pos = pyo.Param (I_fr, initialize = lambda M, i: Fr_sigmamu[i-1 , 1])
    
    Sigma_i_neg = M.Sigma_i_neg = pyo.Param (I, initialize = lambda M, i: unit_sigmamu[i-1 , 0])
    Mu_i_neg = M.Mu_i_neg = pyo.Param (I, initialize = lambda M, i: unit_sigmamu[i-1 , 1])
    Sigma_fri_neg = M.Sigma_fri_neg = pyo.Param (I_fr, initialize = lambda M, i: Fr_sigmamu[i-1 , 0])
    Mu_fri_neg = M.Mu_fri_neg = pyo.Param (I_fr, initialize = lambda M, i: Fr_sigmamu[i-1 , 1])

    # Função objetivo
    obj = M.obj = pyo.Objective(rule = M.zeta_p1, sense= pyo.maximize)

    # Restrição de eficiência
    M.R1 = pyo.Constraint(I, I_fr, \
        rule= lambda M, i, i_fr: (alpha_pos * Mu_fri_pos[i_fr] - alpha_neg * Mu_fri_neg[i_fr] - beta_pos * Sigma_fri_pos[i_fr] - beta_neg * Sigma_fri_neg[i_fr] + zeta_p1) <= (alpha_pos * Mu_i_pos[i] - alpha_neg * Mu_i_neg[i] - beta_pos * Sigma_i_pos[i] - beta_neg * Sigma_i_neg[i]))

    # Restrição tipo 2 -> alfa_neg+beta_neg=1
    M.R2 = pyo.Constraint(rule = alpha_neg + beta_neg == 1)
    
    # Restrição tipo 3 -> alfa_pos+beta_pos=1
    M.R3 = pyo.Constraint(rule = alpha_pos + beta_pos == 1)
    
    # Restrição de alpha >= 2beta
    #modelo.R3 = pyo.Constraint(rule = modelo.alfa >= modelo.beta)

    # Resolução
    glpk = pyo.SolverFactory('glpk') # Construindo o solver gurobi
    result = glpk.solve(M)

    return M, result


def optimize_PROM1_newconstraint (unit_sigmamu, Fr_sigmamu):
    #Instanciando Modelo
    M = pyo.ConcreteModel() # instancia do modelo

    # Criando índices
    I = M.I = pyo.RangeSet(1) # range para a unidade em análise
    I_fr = M.I_fr = pyo.RangeSet(Fr_sigmamu.shape[0]) # Range for alternatives in fr
    
    #Variáveis de Decisão
    alpha_pos = M.alpha_pos = pyo.Var(within=pyo.NonNegativeReals)
    beta_pos = M.beta_pos = pyo.Var(within=pyo.NonNegativeReals)
    alpha_neg = M.alpha_neg = pyo.Var(within=pyo.NonNegativeReals)
    beta_neg = M.beta_neg = pyo.Var(within=pyo.NonNegativeReals)
    zeta_p1 = M.zeta_p1 = pyo.Var()

    # Parâmetros
    Sigma_i_pos = M.Sigma_i_pos = pyo.Param (I, initialize = lambda M, i: unit_sigmamu[i-1 , 0])
    Mu_i_pos = M.Mu_i_pos = pyo.Param (I, initialize = lambda M, i: unit_sigmamu[i-1 , 1])
    Sigma_fri_pos = M.Sigma_fri_pos = pyo.Param (I_fr, initialize = lambda M, i: Fr_sigmamu[i-1 , 0])
    Mu_fri_pos = M.Mu_fri_pos = pyo.Param (I_fr, initialize = lambda M, i: Fr_sigmamu[i-1 , 1])
    
    Sigma_i_neg = M.Sigma_i_neg = pyo.Param (I, initialize = lambda M, i: unit_sigmamu[i-1 , 0])
    Mu_i_neg = M.Mu_i_neg = pyo.Param (I, initialize = lambda M, i: unit_sigmamu[i-1 , 1])
    Sigma_fri_neg = M.Sigma_fri_neg = pyo.Param (I_fr, initialize = lambda M, i: Fr_sigmamu[i-1 , 0])
    Mu_fri_neg = M.Mu_fri_neg = pyo.Param (I_fr, initialize = lambda M, i: Fr_sigmamu[i-1 , 1])

    # Função objetivo
    obj = M.obj = pyo.Objective(rule = M.zeta_p1, sense= pyo.maximize)

    # Restrição de eficiência
    M.R1 = pyo.Constraint(I, I_fr, \
        rule= lambda M, i, i_fr: (alpha_pos * Mu_fri_pos[i_fr] - alpha_neg * Mu_fri_neg[i_fr] - beta_pos * Sigma_fri_pos[i_fr] - beta_neg * Sigma_fri_neg[i_fr] + zeta_p1) <= (alpha_pos * Mu_i_pos[i] - alpha_neg * Mu_i_neg[i] - beta_pos * Sigma_i_pos[i] - beta_neg * Sigma_i_neg[i]))

    # Restrição tipo 2 -> alfa_neg+beta_neg=1
    M.R2 = pyo.Constraint(rule = alpha_neg + beta_neg == 1)
    
    # Restrição tipo 3 -> alfa_pos+beta_pos=1
    M.R3 = pyo.Constraint(rule = alpha_pos + beta_pos == 1)
    
     # Restrição de alpha_pos >= 2beta_pos
    M.R4 = pyo.Constraint(rule = alpha_pos >= 2 * beta_pos)
    
     # Restrição de alpha_neg >= 2beta_neg
    M.R5 = pyo.Constraint(rule = alpha_neg >= 2 * beta_neg)
    
    # Resolução
    glpk = pyo.SolverFactory('glpk') # Construindo o solver gurobi
    result = glpk.solve(M)

    return M, result