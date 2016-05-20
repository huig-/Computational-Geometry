def ccw(p1, p2, p3):
    return (p2[0]-p1[0])*(p3[1]-p1[1]) - (p2[1]-p1[1])*(p3[0]-p1[0])

def convex_hull(L):
    P = sorted(L)
    L_upper = [P[0], P[1]]
    L_lower = [P[-1], P[-2]]
    n = len(L)

    for i in xrange(2, n):
	L_upper.append(P[i])
	L_lower.append(P[n-i-1])
	while len(L_upper) > 2 and ccw(L_upper[-3], L_upper[-2], L_upper[-1]) >= 0:
	    L_upper.pop(-2)
	while len(L_lower) > 2 and ccw(L_lower[-3], L_lower[-2], L_lower[-1]) >= 0:
	    L_lower.pop(-2)

    #L_lower.pop(-1); 
    L_lower.pop(0)
    L_upper.extend(L_lower)
    return L_upper
