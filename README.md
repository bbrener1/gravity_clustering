# gravity_clustering

A basic clustering procedure. Each data point is allowed to roll downhill as if being pulled gravitationally by the other points. 

No momentum or other mechanics are implemented.

When several points land in the same place, that's a cluster. 

Subsampling is used to smooth out local maxima and to accelerate the computation.

Tune by messing with subsampling rate and step size. 
