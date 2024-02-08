import numpy as np

def get_parent(i, arr):
    if i == arr[i]:
        return i
    else:
        p = get_parent(arr[i], arr)
        arr[i] = p
        return p


def union(i, j, arr):
    p1 = get_parent(i, arr)
    p2 = get_parent(j, arr)
    arr[p2] = p1



def meanshift(features, c=0.1, r=1, max_iters=20):    
    centers = features.copy()
    groups = [i for i in range(len(features))]

    iter_centers = [i for i in range(len(centers))]
    print(f'iter centers {iter_centers}')
    
    for _ in range(max_iters):
        D = []

        for i in iter_centers:
            d = np.sqrt(np.power(features - centers[i], 2).sum(axis=1))
            D.append(d)
        
        D = np.array(D)

        F = D < r
        M = D < c

        I, J = np.where(M)

        for idx in range(len(I)):
            i, j = iter_centers[I[idx]], J[idx]

            if i == get_parent(i, groups):
                union(i, j, groups)

            elif j == get_parent(j, groups):
                union(i, j, groups)
        
        next_iter_centers = []
        for i in range(len(groups)):
            if i == get_parent(i, groups):
                next_iter_centers.append(i)
        
        next_iter_centers = list(set(groups)) 
        iter_centers = [get_parent(i, groups) for i in next_iter_centers]
        
        iter_centers = next_iter_centers

    return centers[iter_centers], np.array(groups)
