import numpy as np

def extended_det(segment,P):
    a,b = segment
    fixed = np.array([[(b-a)[1]],[-(b-a)[0]]])
    return np.dot(P-a,fixed).flatten()

def quickHull(segment,P,Hull):
    if P.size == 0 : return
    
    a,b = segment
    # Search for C
    idx_c = np.argmax(extended_det([a,b],P))
    c = P[idx_c]
    # Compute B
    cond = extended_det([a,c],P) > 0 
    B = P[ cond ]
    # Compute A from P\B
    Others = P [ ~cond ]
    A = Others[ extended_det([c,b],Others) > 0 ]
    # Create the hull recursively
    quickHull([a,c],B,Hull)
    Hull.append(c.tolist())
    quickHull([c,b],A,Hull)

def convex_hull(S):
    #Compute a,b. 
    a = np.array(min(S))
    b = np.array(max(S))
    P = np.array(S)
    #Compute lower and upper parts of the Hull
    E = extended_det([b,a],P) 
    Upper = P[E > 0]
    Lower = P[E < 0]
    #Compute the hull
    Hull = []
    Hull.append(a.tolist())
    quickHull([a,b],Lower,Hull)
    Hull.append(b.tolist())
    quickHull([b,a],Upper,Hull)
    Hull.append(a.tolist())

    return Hull

#if __name__ == "__main__":
#    P = np.random.randint(-50,50,size=(100,2))
#    t = np.linspace(-10,10,1000)
#    P = np.vstack((np.sin(t),np.cos(t))).T
#    H = np.array(convex_hull(P.tolist()))

#    plt.plot(P[:,0],P[:,1],color='blue',marker='o',linestyle='None')
#    plt.plot(H[:,0],H[:,1])
#    plt.show()
