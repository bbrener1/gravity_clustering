# gravity_clustering

A basic clustering procedure. Each data point is allowed to roll downhill as if being pulled gravitationally by the other points. 

No momentum or other mechanics are implemented.

When several points land in the same place, that's a cluster. 

Subsampling is used to smooth out local maxima and to accelerate the computation.

## Tuning Parameters

#### Subsampling Rate: 

  Intuitively, roughly how many clusters do you expect to find? Subsampling rate of 1000 will find hundreds of clusters, but its going to be more aggressive about splitting larger clusters into smaller ones. A subsampling rate of 100 will find dozens of clusters, but be less aggressive. I do not recommend subsampling rates below 5. 
  
#### Smoothing:

  Intuitively, do your clusters have sharp edges? If you have highly overlapping clusters, increasing the smoothing to 3 or 4 may help to disentangle overlapping clusters. If you have "grainy" data, and are looking for larger-scale clusters, turn locality down to 2 (or even 1). 

#### Borrowing

Is your data both sparse and high dimensional? (like... more than 10 dimensional)
Are many dimensions correlated? If so, borrowing can help you overcome the sparsity of your data, when you are not guaranteed to observe a key feature in a sample that would link it to the appropriate neighboring cluster. Borrowing is a numeric value, try 1 or maybe 2. 

WARNING: Sanitizing your data becomes more important when you use borrowing, please make sure that you avoid any uniform features (eg avoid having all of one column being the same value, especially 0), and try to make sure all your features are linearly independent, (eg no column is a multiple of another column). 
  
#### Convergence: 

  Again, you probably don't need to mess with this, but this is the criteria for total displacement over 50 steps for which a point is considered to have "converged" 
  

  
  
