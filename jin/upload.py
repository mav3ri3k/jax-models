import os
import io
import boto3
from rich.progress import track
from rich import print
import json
from dotenv import load_dotenv


def list_files_relative_to_top(top_dir):
    """
    Returns a list of all file paths under `top_dir`, 
    each relative to the parent of `top_dir`. 
    E.g. if top_dir='top', you get ['top/1.md', 'top/inside/3.md', …].
    """
    all_files = []
    parent = os.path.dirname(top_dir.rstrip(os.sep))
    for root, dirs, files in os.walk(top_dir):
        for filename in files:
            full_path = os.path.join(root, filename)
            relative_path = os.path.relpath(full_path, parent)
            all_files.append((full_path, relative_path))
    return all_files
def get_s3():
    load_dotenv()
    endpoint_url       = os.getenv("AWS_ENDPOINT_URL", "")
    aws_key            = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret         = os.getenv("AWS_SECRET_ACCESS_KEY")
    bucket_name        = os.getenv("R2_BUCKET_NAME")

    # initialize S3 client
    s3 = boto3.client(
        service_name = 's3',
        endpoint_url=endpoint_url,
        aws_access_key_id=aws_key,
        aws_secret_access_key=aws_secret,
        region_name="apac",
    )

    return (s3, bucket_name)

def upload(top_dir):
    # load AWS/R2 creds & config from .env
    (s3, bucket_name) = get_s3()

    # gather files
    files = list_files_relative_to_top(top_dir)
    success_files = []
    failure_files = []

    # upload each file, preserving relative paths as S3 keys
    for full_path, key in track(files, "[bold green]Uploading..."):
        try:
            with open(full_path, 'rb') as fp:
                s3.upload_fileobj(fp, bucket_name, key)
                print(f"[green]SUCCESS[/green]: {key}")
            success_files.append(key)
        except Exception as e:
            print(f"[red]FAILURE[/red]: {key}")
            failure_files.append(key)

# build summary dict
    summary = {
        "success": success_files,
        "failure": failure_files
    }

    basename    = os.path.basename(top_dir.rstrip(os.sep))
    out_parent  = os.path.dirname(top_dir.rstrip(os.sep))
    os.makedirs(out_parent, exist_ok=True)

    json_filename   = f"{basename}.json"            # "three.json"
    json_local_path = os.path.join(out_parent, json_filename)  # "one/three.json"

    with open(json_local_path, 'w') as jf:
        json.dump(summary, jf, indent=2)
    print(f"[green]Saved summary to {json_local_path}[/green]")

    try:
        with open(json_local_path, 'rb') as jf:
            s3.upload_fileobj(jf, bucket_name, json_filename)
        print(f"[green]Uploaded summary JSON to {bucket_name}/{json_filename}[/green]")
    except Exception as e:
        print(f"[red]Failed to upload summary JSON: {e}[/red]")

def delete_folder(json_name: str):
    """
    Downloads the JSON summary file `json_name` from the bucket,
    parses its "success" list, and deletes each of those objects
    from the same bucket.
    
    Assumes AWS/R2 credentials and config in a `.env`:
      AWS_ENDPOINT_URL, AWS_ACCESS_KEY_ID,
      AWS_SECRET_ACCESS_KEY, R2_BUCKET_NAME, R2_REGION
    """
    (s3, bucket) = get_s3()

    # 1) Download the JSON summary locally
    local_json = os.path.basename(json_name)
    try:
        s3.download_file(bucket, json_name, local_json)
        print(f"[green]Downloaded summary[/green] to ./{local_json}")
    except Exception as e:
        print(f"[red]Failed to download[/red] {json_name}: {e}")
        return

    # 2) Parse the JSON to get list of successful uploads
    try:
        with open(local_json, "r") as jf:
            summary = json.load(jf)
    except Exception as e:
        print(f"[red]Failed to read/parse[/red] {local_json}: {e}")
        return

    success_keys = summary.get("success", [])
    if not success_keys:
        print("No successful uploads listed in JSON; nothing to delete.")
        return

    # 3) Batch–delete all objects in the "success" list
    #    S3 allows up to 1000 keys per delete_objects call.
    CHUNK_SIZE = 10
    for i in track(range(0, len(success_keys), CHUNK_SIZE), "[bold green]Deleting..."):
        batch = success_keys[i : i + CHUNK_SIZE]
        delete_payload = {"Objects": [{"Key": k} for k in batch]}
        try:
            resp = s3.delete_objects(Bucket=bucket, Delete=delete_payload)
            deleted = resp.get("Deleted", [])
            errors  = resp.get("Errors", [])
            print(f"[green]Deleted[/green] {len(deleted)} objects (batch {i//CHUNK_SIZE+1})")
            if errors:
                print(f"[red]Errors deleting:[/red] {errors}")
        except Exception as e:
            print(f"[red]Failed to delete batch[/red] {i//CHUNK_SIZE+1}: {e}")

def download_folder(json_name: str, top_dir: str):
    """
    Downloads the summary JSON `json_name` from the bucket,  
    reads its "success" list of keys, and for each key:
      - creates any needed subdirectories under `top_dir`
      - downloads that object into the right spot
    
    Example:
      json_name = "three.json"
      top_dir   = "./one/two/three"
    
    If a key in "success" is "three/inside/3.md", it will be
    saved locally as "./one/two/three/inside/3.md".
    """
    # 1) load creds & bucket info
    (s3, bucket) = get_s3()

    # 3) download the JSON summary locally
    local_summary = os.path.basename(json_name)
    try:
        s3.download_file(bucket, json_name, local_summary)
        print(f"[green]Downloaded summary[/green] to ./{local_summary}")
    except Exception as e:
        print(f"[red]Could not download[/red] {json_name}: {e}")
        return

    # 4) parse it
    with open(local_summary, "r") as jf:
        summary = json.load(jf)
    success_keys = summary.get("success", [])
    if not success_keys:
        print("[yellow]No 'success' entries in JSON—nothing to download.[/yellow]")
        return

    # 5) figure out what prefix to strip from each key
    #    (keys were stored as "<basename>/…")
    basename = os.path.splitext(local_summary)[0]  # e.g. "three"

    # 6) download each successful object into the right local path
    for key in track(success_keys, "[bold green]Downloading..."):
        # turn "three/inside/3.md" → "inside/3.md"
        if key.startswith(basename + os.sep) or key.startswith(basename + "/"):
            internal_path = key[len(basename) + 1:]
        else:
            internal_path = key  # fallback if format differs

        dest_path = os.path.join(top_dir, internal_path)
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)

        try:
            s3.download_file(bucket, key, dest_path)
            print(f"[green]Downloaded[/green] {key} → {dest_path}")
        except Exception as e:
            print(f"[red]Failed to download[/red] {key}: {e}")

if __name__ == "__main__":
    top_dir = './checkpoints/one'
    upload(top_dir)
    # delete_folder("one.json")
    # download_folder("one.json", "./checkpoints/one")
