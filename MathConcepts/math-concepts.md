### Math Concepts Details

- Euclidean Distance
- Manhattan Distance   

`sum(abs(x1 - x2) for x1, x2 in zip(point1, point2))`
>point1 = (1, 2, 3)   
point2 = (1, 2, 3)

- Minkowski Distance
- Hamming Distance

`sum(c1 != c2 for c1, c2 in zip(vector1, vector2))`
> vector 1 (1, 0, 1, 0, 0, 0, 1, 1)   
vector 2 (1, 0, 0, 0, 0, 0, 0, 1)

--- The hamming distance would be 2 since only two of the values differ.