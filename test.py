import httpx
import time

ragdb_base_url = "http://localhost:7898"


def url_for(endpoint: str):
    return "{}{}".format(ragdb_base_url, endpoint)


def test_connection_reuse_multiple_endpoints(base_url1, base_url2, n):
    client = httpx.Client()
    try:
        print("Making initial request...")
        start = time.time()
        response = client.get(base_url1)
        response.raise_for_status()
        print("First request time: {:.4f} seconds".format(time.time() - start))

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
        average_time = total_time / n
        print(f"Average time over {n} requests: {average_time:4f} seconds")


def test_search(collection_name, column_name, query, limit):
    client = httpx.Client()
    # warmup
    _ = client.get(url_for("")).json()
    results = client.post(
        url_for(f"/collections/{collection_name}/search"),
        json={"column_name": column_name, "query": query, "limit": limit},
    ).json()

    return results


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("-r", "--test-reuse", action="store_true")
    ap.add_argument("-s", "--test-search", action="store_true")
    args = ap.parse_args()

    if args.test_reuse:
        test_connection_reuse_multiple_endpoints(
            "http://localhost:7898/collections",
            "http://localhost:7898/collections",
            10,
        )

    if args.test_search:
        results = test_search(
            collection_name="test2",
            column_name="passage",
            query="When was Abraham Lincoln born?",
            limit=5,
        )
        print(results)
