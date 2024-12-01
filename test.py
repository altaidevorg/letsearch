import httpx
import time


def test_connection_reuse_multiple_endpoints(base_url1, base_url2, n):
    client = httpx.Client()
    try:
        print("Making initial request...")
        start = time.time()
        response = client.get(base_url1)
        response.raise_for_status()
        print("First request time: {:.4f} seconds".format(time.time() - start))
        results = httpx.post(
            "http://localhost:7898/collections/test1/search",
            json={"query": "how are you", "column_name": "user"},
        )
        print(results.json())
        exit()

        print(f"Making {n} requests to different endpoints...")
        total_time = 0
        for i in range(n):
            url = base_url1 if i % 2 == 0 else base_url2
            start = time.time()
            response = client.get(url)
            elapsed = time.time() - start
            print(f"Request to {url}: {elapsed:.4f} seconds")
            total_time += elapsed
            response.raise_for_status()
    finally:
        client.close()
        # average_time = total_time / n
        # print(f"Average time over {n} requests: {average_time:4f} seconds")


test_connection_reuse_multiple_endpoints(
    "http://localhost:7898/collections",
    "http://localhost:7898/collections",
    10,
)
