




import numpy as np
from time import time

start = time() 
for i in range(0,100000):
    try :
        ind =  np.random.choice(1000,size=5,replace=False) # 15% choice
    except ValueError:
        breakpoint()

end = time()
print(f'It took {end - start} seconds for the lopp with try')



start = time() 
for i in range(0,100000):

    ind =  np.random.choice(1000,size=5,replace=False) # 15% choice


end = time()
print(f'It took {end - start} seconds for the lopp with no try')