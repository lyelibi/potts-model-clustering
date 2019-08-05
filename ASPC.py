'''Project: “Potts Model Clustering”
Author : Lionel Yelibi, 2019, University of Cape Town.
Copyright SPC, 2019
Potts Model Clustering.
Agglomerative Fast Super-Paramagnetic Clustering
See pre-print: https://arxiv.org/abs/1908.00951
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
import networkx as nx

    
def duplicates(lst, item):
    return [i for i, x in enumerate(lst) if x == item]



''' takes two nodes, merges them, and returns the label of the merged leaf,
and the delta L_c'''        
def merge(i,j,ind, graph):
    idx0 = np.where( ind ==i)[0]
    idx1 = np.where(ind ==j)[0]
    l0 = clus_lc(idx0, graph, ind)
    l1 = clus_lc(idx1, graph, ind)
    idx01 = np.concatenate((idx0,idx1))
    l01 = clus_lc(idx01, graph, ind)    
    lc = l01-(l1+l0)
    return j, lc
''' vectorize so lists can be passed on'''
fspc = np.vectorize(merge,otypes=[int,float],excluded=['ind','graph'])

''' updates the graph by merging the nodes'''
def merge_nodes(G,nodes):
    other_nodes = list(G.nodes)
    other_nodes.remove(nodes[0])
    other_nodes.remove(nodes[1])
    for y in other_nodes:
        G[nodes[0]][y]['weight']=G[nodes[0]][y]['weight'] + G[nodes[1]][y]['weight']

    G[nodes[0]][nodes[0]]['weight'] = 2* G[nodes[0]][nodes[1]]['weight'] + G[nodes[0]][nodes[0]]['weight'] + G[nodes[1]][nodes[1]]['weight']
    G.node[nodes[0]]['ns'] = G.node[nodes[0]]['ns'] + G.node[nodes[1]]['ns']
    G.remove_node(nodes[1])
    
''' takes a cluster's index, a graph, and a spin array of labels,
and computes the cluster's likelihood'''
def clus_lc(idx, G, S):
    ''' n_s '''
    ns = sum([G.node[S[i]]['ns'] for i in idx])
    if ns<=1:
        return 0
    ''' c_s'''
    cs = 0
    for x in idx:
        for y in idx:
            cs+= G[S[x]][S[y]]['weight']        
            
    ''' A and B'''
    if cs<=ns:
        return 0
    if cs>=ns**2:
        cs=ns**2-1e-6
    B = (ns - 1)*np.log( (ns**2 - ns) / ( ns**2 - cs) )                
    A = np.log(ns/cs)
    sum_ = 0.5*(A + B)
    return sum_

''' takes an entire spin array and computes the global likelihood'''
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
            cs += sum(rho[x,cluster])

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

'''data here'''
''' CBMA uses a correlation matrix and outputs the cluster configuration
with the maximum likelihood structure'''
cor =  np.load('data')
N = cor.shape[0]

''' all spins start in their own cluster'''
S = np.arange(N, dtype=int)
''' create the graph by mapping the correlation matrix to it'''    
G = nx.from_numpy_matrix(cor)
''' initialize the cluster counts by setting all n_s values to 1'''
ns = dict( [ ( i , 1) for i in range(N)])
nx.set_node_attributes(G,ns, 'ns')
''' create the tracker array which keeps track of the cluster compositions'''
tracker = [ [x] for x in range(N)]
''' the candidates array keeps track of labels merged:
    root, leaf, delta lc'''
candidates = np.array([999,999,999]).reshape(1,-1)

''' first pass of the algorithm where every single nodes are merged to one
another. this results in ~N**2 operations, it could probably be parallelized'''
for i in range(N-1):
    labels_to_merge,dlc = fspc(S[i],S[i+1:],ind=S, graph=G)
    candidates = np.concatenate((candidates, np.array([S[i]*np.ones(len(labels_to_merge)),labels_to_merge,dlc]).T),axis=0)
candidates = candidates[candidates[:,0]!=999]

''' The next step is an iterative process which tests newly merged clusters
    against the others while avoiding re-merging clusters which have already
    been tested against each other, this provides a significant performance
    increase'''
for ttt in range(N-1):
    best = np.argmax(candidates[:,2])
    if candidates[best,2] <= 0:
        print(' time to stop ')
        break
    else:
#        print(' current max lc is %s' % candidates[best,2])

        label_a = int(candidates[best,0])
        label_b = int(candidates[best,1])
        index_a = np.where(S==label_a)[0][0]
        index_b = np.where(S==label_b)[0][0]
        xy = [index_a,index_b]
        root = np.argmin(xy)
        leaf = np.argmax(xy)
        node0 = tracker[xy[root]]
        node1 = tracker[xy[leaf]]
        node0.extend(node1)
        tracker[xy[root]] = node0
        del tracker[xy[leaf]]
        
        root = S[   xy[root]     ]
        leaf = S[   xy[leaf]     ]
    
        candidates = candidates[candidates[:,0]!=leaf]
        candidates = candidates[candidates[:,0]!=root]
        candidates = candidates[candidates[:,1]!=leaf]
        candidates = candidates[candidates[:,1]!=root]
        merge_nodes(G,[root,leaf])
        S = np.array(list(G.nodes))
    
        labels_to_merge,dlc = fspc(root,S[ S!= root ],ind=S, graph=G)
        candidates = np.concatenate((candidates, np.array([root*np.ones(len(labels_to_merge)),labels_to_merge,dlc]).T),axis=0)

''' Once the process is over, we create an array final cluster, loop over
    tracker, assign the final cluster labels, and use the resulting cluster
    configuration for the computation of the final likelihood'''
final_clusters = np.zeros(N,dtype=int)
k=0
for i in tracker:
    final_clusters[i] = k
    k+=1
final_lc = Lc(final_clusters,cor)

