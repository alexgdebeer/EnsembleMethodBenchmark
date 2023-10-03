# TODO:

 - Fix the noise 
 - Inflate the noise of the likelihood (to avoid inverse crimes)?
 - Add model error

# How to do POD?
Run the model a large number of times with different parameters and 
extract the model states at each time period
Generate covariance matrix by computing the covariances between 
states (irrespective of time, as in Lipponen paper?)
Perform eigendecomposition of covariance 
Need to centre at some point(s) in here, definitely when computing 
the covariance? Need to rescale the data as well?
Extract the eigenvectors corresponding to the n largest eigenvalues 
(based on the cumulative sum of the eigenvalues)
Form basis and pass into forward problem
Solve forward problem with reduced basis (making sure to project 
states back into original state space somewhere in there)
Run the reduced model on the same inputs the full model was run on and 
build the covariance of the errors (see other Lipponen paper?)