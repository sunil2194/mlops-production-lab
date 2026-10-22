import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import boto3


PROJECT_ROOT = Path(__file__).resolve().parent.parent
VERSION_DIRECTORY = PROJECT_ROOT / "data" / "versions"


def download_dataset(bucket, key):
    """Download an S3 dataset and calculate its SHA-256 fingerprint."""

    s3 = boto3.client("s3", region_name="us-east-1")
    response = s3.get_object(Bucket=bucket, Key=key)
    data = response["Body"].read()

    dataset_hash = hashlib.sha256(data).hexdigest()

    VERSION_DIRECTORY.mkdir(parents=True, exist_ok=True)

    destination = VERSION_DIRECTORY / f"{dataset_hash}.csv"

    if destination.exists():
        if hashlib.sha256(destination.read_bytes()).hexdigest() != dataset_hash:
            raise ValueError("Existing dataset snapshot failed hash verification")
    else:
        destination.write_bytes(data)

    print(f"Downloaded s3://{bucket}/{key}")
    print(f"Dataset SHA256: {dataset_hash}")

    return destination, dataset_hash


def version_dataset_with_dvc(destination):
    """Track the dataset with DVC and upload it to the S3 remote."""

    subprocess.run(
        ["dvc", "add", str(destination)],
        cwd=PROJECT_ROOT,
        check=True,
    )

    subprocess.run(
        ["dvc", "push", str(destination) + ".dvc"],
        cwd=PROJECT_ROOT,
        check=True,
    )

    print(f"DVC versioning completed: {destination}")

    return str(destination) + ".dvc"


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--bucket", required=True)
    parser.add_argument("--key", required=True)

    args = parser.parse_args()

    destination, dataset_hash = download_dataset(
        args.bucket,
        args.key,
    )

    metadata_path = version_dataset_with_dvc(destination)

    result = {
        "source": f"s3://{args.bucket}/{args.key}",
        "dataset_sha256": dataset_hash,
        "dvc_metadata": str(
            Path(metadata_path).relative_to(PROJECT_ROOT)
        ),
    }

    # Preserve DVC metadata outside the temporary Kubernetes pod.
    s3 = boto3.client("s3", region_name="us-east-1")

    metadata_key = f"dvc-metadata/{dataset_hash}.csv.dvc"
    manifest_key = f"dvc-metadata/{dataset_hash}.json"

    s3.upload_file(
        metadata_path,
        args.bucket,
        metadata_key,
    )

    result["dvc_metadata_s3"] = f"s3://{args.bucket}/{metadata_key}"

    s3.put_object(
        Bucket=args.bucket,
        Key=manifest_key,
        Body=json.dumps(result, indent=2).encode("utf-8"),
        ContentType="application/json",
    )

    print(f"Metadata saved: s3://{args.bucket}/{metadata_key}")
    print(f"Manifest saved: s3://{args.bucket}/{manifest_key}")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
