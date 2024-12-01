import requests

import time

start = time.time()
r = requests.get(
    url="http://localhost:7898/collections/test1",
)
elapsed = time.time() - start
print(r)
print(r.json())
print(elapsed)
print("headers")
print(r.headers)

print("---")
print(r.elapsed)
