import ray
import time
ray.init()

@ray.remote
def f(x):
    time.sleep(5)
    return x * x

futures = [f.remote(i) for i in range(40)]
print(ray.get(futures))