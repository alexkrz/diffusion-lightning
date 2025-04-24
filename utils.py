from huggingface_hub import snapshot_download

snapshot_download(
    "diffusers/dog-example",
    local_dir="./data/dog",
    repo_type="dataset",
    ignore_patterns=".gitattributes",
)
