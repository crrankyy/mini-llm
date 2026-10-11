import argparse
import os
import sys
import threading
import time
import traceback

import tiktoken
from datasets import load_dataset
from tqdm import tqdm

from mini_llm.corpus import prepare_corpus, read_manifest, write_manifest

# Both are the sample-10BT subset, pinned to one revision so that every run builds the same
# token files: a training run that is resumed in a new Colab session relies on that.
# FineWeb-Edu keeps the pages a classifier rated as educational; FineWeb is the general web.
SOURCES = {
    "fineweb-edu": {
        "repo": "HuggingFaceFW/fineweb-edu",
        "revision": "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9",
        "files": 14
    },
    "fineweb": {
        "repo": "HuggingFaceFW/fineweb",
        "revision": "9bb295ddab0e05d785b879661af7260fed5140fc",
        "files": 15
    }
}

def parse_args():
    parser = argparse.ArgumentParser(description="Download and tokenize a mix of FineWeb-Edu and FineWeb.")
    parser.add_argument(
        "--tokens", type=int, default=1_000_000_000,
        help="training tokens to prepare; each one takes 4 bytes on disk"
    )
    parser.add_argument(
        "--validation-tokens", type=int, default=1_000_000,
        help="tokens to hold out for validation, from documents that are not trained on"
    )
    parser.add_argument(
        "--edu-share", type=float, default=0.5,
        help="share of the tokens taken from FineWeb-Edu; the rest comes from FineWeb"
    )
    parser.add_argument(
        "--output-dir", default="data/fineweb-mix",
        help="where train.bin, validation.bin and manifest.json are written"
    )
    args = parser.parse_args()

    if args.tokens < 1 or args.validation_tokens < 0:
        parser.error("--tokens must be at least 1 and --validation-tokens at least 0")
    if not 0 <= args.edu_share <= 1:
        parser.error("--edu-share must be between 0 and 1")

    return args

def open_file(repo, revision, index):

    def documents():
        return load_dataset(
            repo, data_files=f"sample/10BT/{index:03d}_00000.parquet", revision=revision,
            split="train", streaming=True, columns=["id", "text"]
        )

    return documents

def build_sources(edu_share):

    shares = {"fineweb-edu": edu_share, "fineweb": 1 - edu_share}

    return [
        {
            "name": name, "share": shares[name],
            "files": [open_file(source["repo"], source["revision"], index) for index in range(source["files"])]
        }
        for name, source in SOURCES.items()
        if shares[name] > 0
    ]

def prepare_data():
    args = parse_args()

    # A thread lock for the progress bars that `datasets` draws. tqdm's own default takes a
    # system semaphore, and the exit at the bottom of this file would leave that behind for
    # Python to warn about. This script starts no other processes, so a thread lock is enough.
    tqdm.set_lock(threading.RLock())

    request = {
        "tokens": args.tokens, "validation_tokens": args.validation_tokens, "edu_share": args.edu_share,
        "revisions": {name: source["revision"] for name, source in SOURCES.items()}
    }

    manifest = read_manifest(args.output_dir)
    if manifest is not None and manifest["request"] == request:
        print(f"{args.output_dir} already holds this data: {manifest['train_tokens']:,} training tokens.")
        return

    print(f"Preparing {args.tokens:,} training tokens in {args.output_dir}...", flush=True)
    start = time.time()
    manifest = prepare_corpus(
        build_sources(args.edu_share), args.output_dir, args.tokens, args.validation_tokens,
        tiktoken.get_encoding("cl100k_base"), num_threads=os.cpu_count()
    )
    manifest["request"] = request
    write_manifest(args.output_dir, manifest)

    print(f"Saved {manifest['train_tokens']:,} training tokens and {manifest['validation_tokens']:,} "
          f"validation tokens in {(time.time() - start) / 60:.1f} minutes.")
    if manifest["train_tokens"] < args.tokens:
        # every file gives the same amount, so the smallest file of a source limits what that source can give
        print(f"That is fewer than the {args.tokens:,} asked for: a file ran out of documents before its part was full.")
    print("Done!")

if __name__ == "__main__":
    try:
        prepare_data()
        status = 0
    except Exception:
        traceback.print_exc()
        status = 1

    # Leave without the interpreter's shutdown. Closing a stream early leaves the Arrow library
    # still fetching the next part of that file in the background; if the interpreter shuts down
    # while it is, the process never exits (https://github.com/apache/arrow/issues/45214).
    # Everything this script writes is on disk by now.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(status)
