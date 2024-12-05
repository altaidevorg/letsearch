import httpx
import time

ragdb_base_url = "http://localhost:7898"


def url_for(endpoint: str):
    return "{}{}".format(ragdb_base_url, endpoint)


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

    if args.test_search:
        results = test_search(
            collection_name="test2",
            column_name="passage",
            query="When was Abraham Lincoln born?",
            limit=5,
        )
        print(results)
