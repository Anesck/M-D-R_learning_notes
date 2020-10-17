import time
import numpy as np
from scipy.stats import norm

mu = 0
sigma = 1

start = time.time()
for x in range(1, 50):
    print(-(np.log(2*np.pi*sigma**2)/2 + ((x-mu)**2)/(2*sigma**2)))
end = time.time()
print("time: ", end - start)

distribution = norm(mu, sigma)
start = time.time()
for x in range(1, 50):
    #distribution.args = mu, sigma
    print(distribution.logpdf(x))
end = time.time()
print("time: ",end - start)
