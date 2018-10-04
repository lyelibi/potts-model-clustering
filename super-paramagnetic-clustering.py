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
from joblib import Parallel, delayed
from sklearn.metrics.pairwise import euclidean_distances
from sklearn import datasets
import networkx as nx


def duplicates(lst, item):
    return [i for i, x in enumerate(lst) if x == item]

def eHKnn(distance,K):
    '''Create the nodenext matrix which stores the neighbor in the neighborhood
    of size K. This needs to be supplemented by the nodes linked by the Minimum
    Spanning Tree. See pre-print'''
    N=distance.shape[0]
    nodenext=[]
    Rank=np.argsort(distance)
    Rank=Rank[:,:K+1]
    r_ = np.zeros( (N,K), dtype=int)
    for i in range(N):
        rank = list(Rank[i])
        rank.remove(i)
        r_[i] = rank[:K]
    Rank =  r_
    for i in range(N):
        e=[]
        for j in Rank[i]:
            if i in Rank[j]:
                e.append(j)
        nodenext.append(e)
    return nodenext

def eHK(link,nodenext):
    '''Extended Hoshen-Kopelman implementation. See pre-print'''

    N=len(nodenext)
    nodel=10*N*np.ones(N,dtype=int)
    label_counter=0
    nodelp=np.array([],dtype=int)
    for i in range(N):
        if (np.array(link[i])==0).all():
            nodel[i]=label_counter
            nodelp=np.append(nodelp,label_counter)
            label_counter+=1
        else:
            idx = duplicates(link[i],1) #where links are
            t=[nodel[ nodenext[i][j] ] for j in idx]
            t=np.array(t)
            if (t==10*N).all(): # all unlabeled?
                nodel[i]=label_counter
                nodelp=np.append(nodelp,label_counter)
                label_counter+=1
            else:
                w=[]
                for index in range(len(t)):
                    if t[index]!=10*N:
                        w.append(index)
                idx_ = np.array([nodenext[i][j] for j in idx])
                z = nodelp[nodel[ idx_[w] ] ]
                min_ = np.amin(z)

                nodel[i] = min_
                a = nodel[ idx_[w] ]
                nodelp[a]=min_

    # sequentialize part 1: re-order nodelp
    for y in range(len(nodelp)):
        n = y
        while (nodelp[n]<n):
            n=nodelp[n]
        nodelp[y]=n
    # sequentialize part 2: get rid of the gaps
    un = np.unique(nodelp)
    for i in range(len(un)-1):
        while un[i+1]-un[i] !=1:
            idx = np.where(nodelp==un[i+1])[0]
            nodelp[idx] -= 1
            un = np.unique(nodelp)

    # rename the labels with their root
    for i in range( len(nodelp) ):
        nodel[nodel==i]=nodelp[i]

    return nodel



def cHKlons(nodenext,G):
    ''' This function serves to perform the consensus final cluster solution
    using the Spin Spin correlation matrix G'''
    link=[]
    N=G.shape[0]
    for i in range(N):
        neighbors=nodenext[i]
        e=[]
        for j in neighbors:
            if (G[i,j]>0.5):
                e.append(1)
            else:
                e.append(0)
        idx=np.argmax(G[i,neighbors])
        e[idx]=1
        link.append(e)
    ''' make sure the neighbor with the highest correlation is linked both ways'''
    for i in range(N):
        idx = duplicates(link[i],1)
        for f in idx:
            X = nodenext[i][f]
            Y = duplicates(nodenext[X],i)
            Y = Y[0]
            link[X][Y] = 1
    return link


def kron(i, j):
    '''kronecker delta'''
    if i == j:
        return 1
    else:
        return 0

def twopc(S,cij):
    ''' two point connectedness'''
    classes=np.unique(S)
    for label in classes:
        neighbors=duplicates(S,label)
        for node in neighbors:
            cij[node,neighbors]+=1
    return cij

def Hs(S, J, nodenext):
    ''' Hamiltonian Energy'''
    E=0
    N = len(S)
    for i in range(N):
        for j in nodenext[i]:
            E += J[i, j]*(1-kron(S[i], S[j]))
    return E/N

def magnetization(S, q):
    N=len(S)
    nmax = np.amax(np.bincount(S))
    return (q*nmax-N)/((q-1)*N)

def runz(S,f,mcmc,nodenext,J,t,q,K):
    np.random.seed(0)
    def flip(S, q):
        ''' Flip clusters labels after Monte Carlo steps'''
        c = np.unique(S)  # find unique labels
        new_c = np.random.randint(0, q, len(c))  #gen new spins for clusters
        conv = dict(zip(c, new_c))  # use dic to assign new spins to clusters
        return np.vectorize(conv.get)(S)

    def eHKlons(nodenext,T,J,S):
        ''' Create link matrix, which stores the edges activation status'''
        link=[]
        N = len(nodenext)
        for i in range(N):
            e=[]
            for j in nodenext[i]:
                if (1-np.exp(-J[i,j]*kron(S[i],S[j])/T)>np.random.uniform() ):
                    e.append(1)
                else:
                    e.append(0)
            link.append(e)
        return link

    N=len(S)
    forget=int(f*mcmc)
    m=np.zeros(mcmc)
    cij=np.zeros((N,N))
    
    for i in range(forget):
        '''the number of SW steps that allow the system to reach thermal eq'''
        S1=S
        E1=Hs(S1,J,nodenext)
        LinkSnode = eHKlons(nodenext,t,J,S)
        S = eHK(LinkSnode,nodenext)
        S = flip(S, q)
        E2 = Hs(S,J,nodenext)
        if E2 >= E1:
            if np.exp(- E2 / t) < np.random.uniform() :
                S=S1

    for i in range(mcmc):
        ''' Actual SW steps we keep'''
        S1=S
        E1=Hs(S1,J,nodenext)
        LinkSnode = eHKlons(nodenext,t,J,S)
        S = eHK(LinkSnode,nodenext)
        S = flip(S, q)
        E2 = Hs(S,J,nodenext)
        if E2 >= E1:
            if np.exp(- E2 / t) < np.random.uniform() :
                S=S1
                E2 = E1

        cij=twopc(S,cij)
        m[i]=magnetization(S,q)
    ''' Compute thermodynamic averages here '''
    mbar = np.average(m)  # ''' Average magnetization'''
    su = N*np.var(m)/t # ''' Magnetic Susceptibility'''
    return su, mbar, cij, S

''' data we want to cluster'''
project = 'blobs'
blob = datasets.make_blobs(n_samples=500,
                         cluster_std=[0.25,0.5,1],
                         random_state=0, n_features=500,shuffle=True)
data = blob[0]
N = data.shape[0]
T=np.linspace(1e-6,.3,num=60,endpoint=True)
K = 10
q = 20
alpha = 4
distance = euclidean_distances(data)

''' Number of mcmc steps'''
mcmc = 200
''' Number of temperatures explored'''
k = len(T)
'''determine the Graph, and its minimal spanning tree'''

''' Determine the neighborhood and add the Minimal Spanning Tree edges on top of it'''
Tree=nx.minimum_spanning_tree(nx.from_numpy_matrix(distance))

nodenext = eHKnn(distance, K)
mst_edges = list( Tree.edges())
''' add the edges in the minimal spanning tree not in nodenext'''
for i in mst_edges:
    node = i[0]
    if i[1] not in nodenext[node]:
        nodenext[node].append(i[1])
        nodenext[node] = sorted(nodenext[node])
        nodenext[i[1]].append(node)
        nodenext[i[1]] =  sorted(nodenext[i[1]])

''' need average number of neighbors khat, and the local length scale a'''
khat = 0
for i in nodenext:
    khat+= len(i)
khat = khat / N
''' local length scale'''
a = 0
for i in range(N):
    a+=sum(distance[i,nodenext[i]])
a = alpha * a / (khat*N)
''' Interaction Strength'''
n = 2
J = (1 / khat) * np.exp(-( (n-1)/n ) * ( distance / a)**n)

''' How many mcmc steps are forgotten for every temperature t'''
f_=0.5

''' The initial spin configuration S_0 for all temperatures'''
S=np.ones(N, dtype=int)
''' SPC runs sequentially but every temperatures are ran in parallel'''
results = Parallel(n_jobs=5)(delayed( runz )(S,f_,mcmc,nodenext,J,T[y],q,K) for y in range(k))
