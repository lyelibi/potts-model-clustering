'''Project: “Potts Model Clustering”
Author : Lionel Yelibi, 2018, University of Cape Town.
Copyright SPC, 2018

Potts Model Clustering.
Super-Paramagnetic Clustering, Maximum entropy, and Maximum Likelihood Methods.
See pre-print: https://arxiv.org/blahblah

GNU GPL

This file is part of SPC

SPC is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

SPC is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.'''

import numpy as np
from sklearn import datasets
from joblib import Parallel, delayed



def duplicates(lst, item):
    return [i for i, x in enumerate(lst) if x == item]


def Lc(S,rho):
    labels = np.unique(S)
    lc = 0
    for i in labels:
        cluster = duplicates(S,i)
        ''' n_s '''
        ns = len( cluster )
        if ns<=1:
            continue
        ''' c_s'''
        cs = 0
        for x in cluster:
            for y in cluster:
                cs += rho[x,y]

        ''' gs is between 0 & 1'''
        if cs<=ns:
            return 0
        if cs>=ns**2:
            cs=ns**2-1e-6
        B = (ns - 1)*np.log( (ns**2 - ns) / ( ns**2 - cs) )
        A = np.log(ns/cs)
        lc += A + B

    lc= 0.5*lc
    return lc


def mutation(S):

    bag = ['swap','scramble','flip','new','split','merge','bitflip','sp','per','pul']
    mutation = np.random.choice(bag)

    if mutation == 'new':
        Q = np.random.randint(2,int(N/2))
        S = np.random.randint(0,Q,N)
    elif mutation == 'split':
        labels = np.unique(S)
        s = np.random.choice(labels)
        idx = S == s
        idx = np.arange(len(S))[idx]
        e = np.random.choice(len(idx))
        idx_ = idx[:e]
        S[idx_] = max(labels)+1

    elif mutation == 'merge':
        labels = np.unique(S)
        s = np.random.choice(labels)
        labels = np.array([ x for x in labels if x!=s])
        if len(labels)!=0:
            S[S==s] = np.random.choice(labels)
    elif mutation == 'sp':
        labels = np.unique(S)
        s = np.random.choice(labels)
        idx = S == s
        idx = np.arange(len(S))[idx]
        e = np.random.choice(len(idx))
        idx_ = idx[:e]

        labels = np.array([ x for x in labels if x!=s])
        if len(labels)!=0:
            S[idx_] = np.random.choice(labels)
    elif mutation == 'per':
        S = np.random.permutation(S)
    elif mutation == 'size':
        q = len(np.unique(S))
        S = np.random.randint(0,q,len(S))
    elif mutation == 'bitflip':
        s_ = np.random.randint(N)
        s = S[s_]
        labels = np.unique(S)
        labels = [ x for x in labels if x!=s]
        if len(labels)!=0:
            S[s_] = np.random.choice(labels)
    elif mutation == 'pul':
        labels = np.unique(S)
        ex =[]
        for i in labels:
            ex.append(len(duplicates(S,i)))
        idx = np.argsort(ex)
        s = idx[-1]
        s = labels[s]
        idx = duplicates(S, s)
        n = 3
        S[idx] = np.random.randint(max(labels)+1, max(labels)+1+n, len(idx))
    elif  mutation == 'flip':
        c = np.unique(S)
        q = len(c)
        new_c = np.random.randint(0, q, len(c))
        conv = dict(zip(c, new_c))
        S = np.vectorize(conv.get)(S)
    elif mutation == 'swap':
        size = np.random.randint(N)
        idx = np.random.randint(0,N,size)
        lab = S[idx]
        lab = np.flip(lab, axis=0)
        S[idx] = lab
    elif mutation =='scramble':

        size = np.random.randint(N)
        idx = np.random.randint(0,N,size)
        lab = S[idx]
        lab = np.random.permutation(lab)
        S[idx] = lab

    ''' Make sequential'''
    for y in range(len(S)):
        n = y
        while (S[n]<n):
            n=S[n]
        S[y]=n
    un = np.unique(S)
    for i in range(len(un)-1):
        while un[i+1]-un[i] !=1:
            idx = np.where(S==un[i+1])[0]
            S[idx] -= 1
            un = np.unique(S)
    return S


def recombination(people, fitnesses, mutated, fit_):
    n = len(people)
    F = np.concatenate( (fitnesses, fit_))
    P = np.concatenate( (people,mutated),axis=0)

    idx = np.argsort(-F)
    people = P[idx[:n]]
    fitnesses = F[idx[:n]]
    return people, fitnesses


''' data to be clustered ''' 
project = 'blobs'
blob = datasets.make_blobs(n_samples=500,
                         cluster_std=[0.25,0.5,1],
                         centers = 3,
                         n_features = 500,
                         random_state=0,
                         shuffle=True)

data = blob[0]
N = data.shape[0]
rho = np.corrcoef(data)

# population
pop = 1000
#number of generation
generations = 10000
# number of cores:
cpu_ = 4

# MUTPB Mutation probability
mu = 0.2
mutants = int(mu*pop)
########### BUILD THE APPARATUS FOR THE GENETIC ALGORITHM
''' Create the population, and its fitness vector'''
people = np.random.randint(0,N,(pop,N), dtype=int)
fitnesses = np.array(  Parallel(n_jobs=cpu_)(delayed(Lc)(people[x],rho) for x in range(pop) )    )
''' select individuals to be mutated'''
idx = np.random.choice(pop,mutants,replace=False)
mutated = np.array( Parallel(n_jobs=cpu_)(delayed(mutation)(people[x]) for x in idx) )
fit_mutants = np.array(   Parallel(n_jobs=cpu_)(delayed(Lc)(x,rho) for x in mutated )    )


for i in range(generations):
    ''' select the best individuals for the next generation'''
    people, fitnesses = recombination(people, fitnesses, mutated, fit_mutants)
    best = np.argmax(fitnesses)
    alpha = people[best]
    ''' select individuals to be mutated'''
    idx = np.random.choice(pop,mutants,replace=False)
    mutated = np.array( Parallel(n_jobs=cpu_)(delayed(mutation)(people[x]) for x in idx) )
    fit_mutants = np.array(   Parallel(n_jobs=cpu_)(delayed(Lc)(x,rho) for x in mutated )    )
