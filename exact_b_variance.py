"""Exact finite-model variance of B_over40 - B_opt; run from repo root.
Uses quadrature exact for these finite trigonometric polynomials and Rademacher noise.
"""
import json
import numpy as np
import wrv_sim as w


def rectangle(prefix, left, right):
    return (prefix[right[:,None], right[None,:]] - prefix[left[:,None],right[None,:]]
            - prefix[right[:,None],left[None,:]] + prefix[left[:,None],left[None,:]])


def main():
    r=json.load(open('results.json'))
    if r['settings']['s'] != w.S or r['settings']['J_CAP'] != w.J_CAP:
        raise ValueError('Use the default main-run settings')
    ns=np.array(r['checkpoints']); m=np.arange(w.BURN,ns[-1]); i=m+1
    low=w.sizes(m,1/3); high=w.sizes(m,.4)
    bands, ids=np.unique(np.stack([low,high],axis=1),axis=0,return_inverse=True)
    q=int(high.max()); phi=w.PHI_GRID[:,:q]; theta=w.THETA[:q]
    # Integrate over X and both noise signs, avoiding random calibration.
    z=np.concatenate([(w.F0_GRID[:,None]+sign*w.SIGMA)*phi-theta for sign in [-1,1]])
    gamma=np.einsum('ij,ik->jk',z,z)/len(z)
    z2=z*z
    moment=np.einsum('ij,ik->jk',z2,z2)/len(z)
    covnorm=moment-np.outer(np.diag(gamma),np.diag(gamma))
    cumulant=covnorm-2*gamma**2
    def prefix(x):
        return np.pad(x.cumsum(0).cumsum(1),((1,0),(1,0)))
    K=rectangle(prefix(cumulant),bands[:,0],bands[:,1])
    G=rectangle(prefix(gamma**2),bands[:,0],bands[:,1])
    rec=np.array(r['rec']); names=r['candidate_names']; b=r['settings']['terms'].index('B')
    print('xi n exact_sd simulated_sd simulated/exact')
    for xk,xi in enumerate(r['settings']['xis']):
        acc1=np.zeros(len(bands));acc0=acc1.copy(); variance=0.; rows=[]; cp=set(ns)
        for mt,it,band in zip(m,i,ids):
            weight=float(it)**xi
            variance+=2*weight/mt**2*(w.matmul(K[band],acc1)+2*w.matmul(G[band],acc0))
            variance+=weight**2*(K[band,band]/mt**3+2*G[band,band]/mt**2)
            acc1[band]+=weight/mt;acc0[band]+=weight
            if it in cp: rows.append(variance)
        sd=np.sqrt(rows)
        empirical=(rec[:,:,names.index('over40'),xk,b]-rec[:,:,names.index('opt'),xk,b]).std(0,ddof=1)
        for k in [0,len(ns)-1]:
            print(f'{xi:g} {ns[k]} {sd[k]:.7g} {empirical[k]:.7g} {empirical[k]/sd[k]:.4f}')


if __name__=='__main__':
    main()
