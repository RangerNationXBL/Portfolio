#### Contents:
Section 1:   
- [Euclidean Distance](#euclidean-distance)

#### __Purpose of this Note:__

<!-- This file will change and new items will be added over time. Likely to be dated from each idea, concept or new discovery. -->

- To show what my progress is and where my progress has lead me.
- To explain some of the concepts I have learned about during this course.
- To further detail complex algorithms or math equations. Which most will be written in a "Coded Manner"
- This will mostly be on the mathimatical parts of coding including machine learning algroithms I have written as lessons learned.

This github is where I put all of the things that I am learning about, most of which is going to be written in python. However, I will also most likely have some c# and c++ in the future, more nearer is the addiction of web develepment code due to the path I have taken in pursuit of my degree.

I have moved some items to this git hub from class work, and educational work that I was practicing with. Some of this code is, or was rewritten from another source. I have modified it from its original to add techniques that I have learned during the course of my path.

### Math Conepts
---

#### Euclidean Distance

*--- This is relevant to the next topic. ---*
--- ---

> Euclidean distance is the measure of the strait line distance between two points in a multi-dimensional space. 
--- ---

- Between two points

    `Distance = sqrt((x2 - x1) ^ 2 + (y2 - y1) ^ 2)`

*Using this will work just as well with multiple points its just going to take a lot more effor to code this. Example below. "..." is in between*

- Between Multiple Points

    `distance = sqrt((x2 - x1) ^ 2 ... (z2 - z1) ^ 2)`

*A better way to do this is a for loop, for loop can work in pretty much any language.

    `def euclidean_distance(point1, point2) -> float:
        distance: -> float = 0.0
        for i in range(len(point1)):
            distance += (pointt2[i] - point1[i]) ** 2
        return math.sqrt(distance)`

#### Manhatten Distance
- measure of the distance between two points in a grid based system. It's named for the grid based layout of the streets in manhattan, where the shortest path between two points incolves only horizontal and vertical movements.

*--- Not super relevant at this time in these examples ---*


    `def manhatten_distance(point1, point2):
        return sum(abs(x1 - x2) for x1, x2 in zip(point1, point2))` 

### Algorithms
---

#### K Nearest Neighbors
> The k-nearest neighbor algorithm is a non-parametric, supervised learning classifier, which uses proximity to make classifications or predictions about the groupinig of an individual data point. It is one of the most popular and simplest classification and regression classifires used in machine learning today. ~IBM

1. For classification problems, a class label is addigned on the basis of majority vote-- i.e. the label that is the most frequently represented around a given data point is used.

2. Regression problems use a similar concept as classification problem, but in this case, the average k nearest neighbors is taken to make the prediction about a classification. The main distinction here is that classification is used for discrete values, whereas regression is used with continuious ones.


