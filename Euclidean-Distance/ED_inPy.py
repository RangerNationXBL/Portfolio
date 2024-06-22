def euclidean_distance(p1, p2):
    if len(p1) != len(p2):
        raise ValueError("Incompatible Points")
    sq_dist = 0
    for i in range(len(p1)):
        sq_dist += (p2[i] - p1[i]) ** 2
    return sq_dist ** 0.5