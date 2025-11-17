#!/usr/bin/env python3
"""Quick test of the workflow scheduler"""

import requests
import time

API_URL = "http://localhost:8000"
USER_ID = "alice"

# Create a simple workflow
workflow = {
    "name": "Test Segmentation",
    "dag": {
        "branch_A": [
            {"job_id": "job1", "image": "CMU-1-Small-Region.svs", "type": "segment"}
        ]
    }
}

# Submit workflow
print("Submitting workflow...")
response = requests.post(
    f"{API_URL}/workflows",
    json=workflow,
    headers={"X-User-ID": USER_ID}
)
print(f"Response: {response.json()}")

workflow_id = response.json().get('workflow_id')

# Poll for status
if workflow_id:
    for i in range(10):
        time.sleep(2)
        status = requests.get(
            f"{API_URL}/workflows/{workflow_id}",
            headers={"X-User-ID": USER_ID}
        )
        print(f"Status: {status.json()}")
