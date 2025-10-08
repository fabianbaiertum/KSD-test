# KSD-test
The code file KSD_test.py implements our proposed composite goodness of fit test, which is based on Kernel Stein Discrepancy(KSD).
It follows the algorithm 1 in our paper:  URL .
We also did a simukation study, comparing the test to the Wildbootstrap KSD-based test by Key et. al. 
For the alternative hypothesis, we compared it to two Maximum Mean Discrepancy based tests, one also proposed by Key et. al (link), 
while the other was proposed by Brueck, Ferminan and Min (BFM) in their paper ... link. The BFM test is available in R, written by Florian Brueck (https://github.com/florianbrueck/MMD_tests_for_model_selection/blob/main/BFM-TEST.R) and was later implemented in Python using GPU by Fabian Baier (https://github.com/fabianbaiertum/BFM-test).

Our KSD test is implemented using JAX, to make use of GPU, as those tests would be infeasible to run on CPU. 
For the estimation of the different theta's and the wildbootstrap based MMD and KSD tests, we are using the code of Key et. al link ... 


## Plot specifics under null hypothesis (runs=500, B=200)
l=d

d=1 sigma = 4.0

d=2 sigma = 8.0

d=10 sigma = 


## Further code/runtime improvements
As this test will most likely be used for large n, probabilistic techniques or approximations should be used.
