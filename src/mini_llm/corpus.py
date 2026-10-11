import json
import os
import random

import numpy as np

MANIFEST_NAME = "manifest.json"

def encode_batch(batch, tokenizer, num_threads):

    # encode_ordinary: a web page can contain the characters "<|endoftext|>", and
    # those have to stay ordinary text. Only the separator added here is the real token
    encoded = tokenizer.encode_ordinary_batch([document["text"] for document in batch], num_threads=num_threads)

    for document, ids in zip(batch, encoded):
        yield document["id"], np.array(ids + [tokenizer.eot_token], dtype=np.uint32)

def encode_documents(documents, tokenizer, batch_size=512, num_threads=8):

    batch = []
    for document in documents:
        batch.append(document)
        if len(batch) == batch_size:
            yield from encode_batch(batch, tokenizer, num_threads)
            batch = []

    if batch:
        yield from encode_batch(batch, tokenizer, num_threads)

def take_from_file(documents, validation_quota, train_quota, seen_ids, tokenizer, train_file, num_threads):

    validation = []
    counts = {"validation_tokens": 0, "train_tokens": 0, "train_documents": 0, "duplicates": 0}

    for document_id, ids in encode_documents(documents, tokenizer, num_threads=num_threads):
        if document_id in seen_ids:
            counts["duplicates"] += 1
            continue
        seen_ids.add(document_id)

        # the first documents of a file are held out, so no validation document is ever trained on
        if counts["validation_tokens"] < validation_quota:
            validation.append(ids)
            counts["validation_tokens"] += len(ids)
        elif counts["train_tokens"] < train_quota:
            ids.tofile(train_file)
            counts["train_tokens"] += len(ids)
            counts["train_documents"] += 1
        else:
            break

    return validation, counts

def prepare_corpus(sources, output_dir, train_tokens, validation_tokens, tokenizer, seed=0, num_threads=8):

    os.makedirs(output_dir, exist_ok=True)
    manifest_path = os.path.join(output_dir, MANIFEST_NAME)

    # gone first, so that files left half-written by a run that stopped are never taken for finished ones
    if os.path.exists(manifest_path):
        os.remove(manifest_path)

    seen_ids = set()
    validation = []
    summaries = []

    with open(os.path.join(output_dir, "train.bin"), "wb") as train_file:
        for source in sources:
            files = source["files"]
            summary = {
                "name": source["name"], "share": source["share"], "files": len(files),
                "validation_documents": 0, "validation_tokens": 0,
                "train_documents": 0, "train_tokens": 0, "duplicates": 0
            }

            for number, open_documents in enumerate(files, start=1):
                documents = iter(open_documents())
                try:
                    # the same amount from every file: the start of one file can be a narrow
                    # sample (in FineWeb-Edu it comes from three or four crawl dates out of dozens)
                    held_out, counts = take_from_file(
                        documents, validation_tokens * source["share"] / len(files),
                        train_tokens * source["share"] / len(files),
                        seen_ids, tokenizer, train_file, num_threads
                    )
                finally:
                    close = getattr(documents, "close", None)
                    if close is not None:
                        close()

                validation.extend(held_out)
                summary["validation_documents"] += len(held_out)
                for key, value in counts.items():
                    summary[key] += value

                print(f"{source['name']} file {number}/{len(files)}: "
                      f"{counts['train_documents']:,} documents, {counts['train_tokens']:,} training tokens",
                      flush=True)

            summaries.append(summary)

    # shuffled, because evaluation reads only the beginning of the validation file,
    # and that beginning should hold documents from every file of every source
    random.Random(seed).shuffle(validation)
    validation_ids = np.concatenate(validation) if validation else np.array([], dtype=np.uint32)
    validation_ids.tofile(os.path.join(output_dir, "validation.bin"))

    manifest = {
        "tokenizer": tokenizer.name,
        "dtype": "uint32",
        "document_separator": tokenizer.eot_token,
        "train_tokens": sum(summary["train_tokens"] for summary in summaries),
        "validation_tokens": len(validation_ids),
        "sources": summaries
    }

    return manifest

def write_manifest(output_dir, manifest):

    # written last: its presence is what says the token files are complete
    with open(os.path.join(output_dir, MANIFEST_NAME), "w") as f:
        json.dump(manifest, f, indent=2)

def read_manifest(output_dir):

    path = os.path.join(output_dir, MANIFEST_NAME)
    if not os.path.exists(path):
        return None

    with open(path) as f:
        return json.load(f)
