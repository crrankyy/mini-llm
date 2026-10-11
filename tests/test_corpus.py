import os

import numpy as np
import tiktoken

from mini_llm.corpus import prepare_corpus, read_manifest, write_manifest
from mini_llm.data import MemMapDataset

WORDS = ["river", "stone", "cloud", "lantern", "harbor", "meadow", "copper", "window"]


def make_file(name, documents=40, shared_ids=()):
    """Stands in for one remote file: a function that opens a stream of documents."""
    rows = [
        {"id": f"{name}-{index}", "text": f"{name} document {index}: " + " ".join(WORDS[index % 8:] * (1 + index % 3))}
        for index in range(documents)
    ]
    rows += [{"id": shared_id, "text": f"a page that {name} shares with another file"} for shared_id in shared_ids]

    return lambda: iter(rows)


def make_sources():
    return [
        {"name": "edu", "share": 0.5, "files": [make_file("edu-a"), make_file("edu-b")]},
        {"name": "web", "share": 0.5, "files": [make_file("web-a"), make_file("web-b"), make_file("web-c")]},
    ]


def prepare(folder, sources=None, train_tokens=1200, validation_tokens=300):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    manifest = prepare_corpus(sources or make_sources(), folder, train_tokens, validation_tokens, tokenizer)

    return manifest, tokenizer


def read_documents(folder, split, tokenizer):
    ids = np.fromfile(os.path.join(folder, f"{split}.bin"), dtype=np.uint32).tolist()
    assert ids[-1] == tokenizer.eot_token

    documents, current = [], []
    for token_id in ids:
        if token_id == tokenizer.eot_token:
            documents.append(tokenizer.decode(current))
            current = []
        else:
            current.append(token_id)

    return documents


def test_every_document_ends_with_the_separator_and_decodes_back(tmp_path):
    manifest, tokenizer = prepare(tmp_path)

    train = read_documents(tmp_path, "train", tokenizer)

    assert len(train) == sum(source["train_documents"] for source in manifest["sources"])
    assert train[0].startswith("edu-a document ")
    assert manifest["document_separator"] == tokenizer.eot_token == 100257


def test_the_separator_written_in_a_page_stays_ordinary_text(tmp_path):
    sources = [{"name": "web", "share": 1.0, "files": [lambda: iter([{"id": "1", "text": "before <|endoftext|> after"}])]}]
    _, tokenizer = prepare(tmp_path, sources, validation_tokens=0)

    ids = np.fromfile(tmp_path / "train.bin", dtype=np.uint32).tolist()

    assert ids.count(tokenizer.eot_token) == 1
    assert tokenizer.decode(ids[:-1]) == "before <|endoftext|> after"


def test_each_source_gets_its_share_spread_over_all_of_its_files(tmp_path):
    manifest, tokenizer = prepare(tmp_path)
    longest_document = 40

    assert manifest["train_tokens"] == os.path.getsize(tmp_path / "train.bin") // 4
    for source in manifest["sources"]:
        # every file overshoots its part by less than one document
        assert 600 <= source["train_tokens"] < 600 + source["files"] * longest_document

    train = read_documents(tmp_path, "train", tokenizer)
    for name in ("edu-a", "edu-b", "web-a", "web-b", "web-c"):
        assert any(document.startswith(name) for document in train)


def test_validation_documents_come_from_every_file_and_are_never_trained_on(tmp_path):
    manifest, tokenizer = prepare(tmp_path)

    train = read_documents(tmp_path, "train", tokenizer)
    validation = read_documents(tmp_path, "validation", tokenizer)

    assert manifest["validation_tokens"] >= 300
    assert not set(train) & set(validation)
    for name in ("edu-a", "edu-b", "web-a", "web-b", "web-c"):
        assert any(document.startswith(name) for document in validation)
    # shuffled: evaluation reads only the start of the file, which in file order would hold one file's documents
    origins = [document.split(" ")[0] for document in validation]
    assert origins != sorted(origins)


def test_a_document_that_appears_twice_is_used_once(tmp_path):
    sources = [
        {"name": "edu", "share": 0.5, "files": [make_file("edu-a", documents=3, shared_ids=["shared"])]},
        {"name": "web", "share": 0.5, "files": [make_file("web-a", documents=0, shared_ids=["shared"]),
                                                make_file("web-b", documents=3)]},
    ]
    manifest, tokenizer = prepare(tmp_path, sources, train_tokens=10_000, validation_tokens=0)

    train = read_documents(tmp_path, "train", tokenizer)

    assert sum("shares with another file" in document for document in train) == 1
    assert [source["duplicates"] for source in manifest["sources"]] == [0, 1]


def test_a_request_larger_than_the_sources_takes_every_document_once(tmp_path):
    manifest, tokenizer = prepare(tmp_path, train_tokens=10_000_000, validation_tokens=0)

    train = read_documents(tmp_path, "train", tokenizer)

    assert manifest["train_tokens"] < 10_000_000
    assert len(train) == len(set(train)) == 5 * 40


def test_two_runs_write_identical_files(tmp_path):
    prepare(tmp_path / "first")
    prepare(tmp_path / "second")

    for name in ("train.bin", "validation.bin"):
        assert (tmp_path / "first" / name).read_bytes() == (tmp_path / "second" / name).read_bytes()


def test_the_manifest_marks_a_finished_run_and_is_removed_when_a_new_one_starts(tmp_path):
    assert read_manifest(tmp_path) is None

    manifest, tokenizer = prepare(tmp_path)
    manifest["request"] = {"tokens": 1200}
    write_manifest(tmp_path, manifest)

    assert read_manifest(tmp_path) == manifest

    def stream_that_fails():
        raise ConnectionError("the download stopped")

    try:
        prepare_corpus([{"name": "web", "share": 1.0, "files": [stream_that_fails]}], tmp_path, 1200, 300, tokenizer)
    except ConnectionError:
        pass

    assert read_manifest(tmp_path) is None


def test_the_training_loader_can_read_the_result(tmp_path):
    manifest, _ = prepare(tmp_path)

    dataset = MemMapDataset(tmp_path / "train.bin", max_length=8, stride=8)
    x, y = dataset[0]

    assert len(dataset) == (manifest["train_tokens"] - 8) // 8
    assert x[1:].tolist() == y[:-1].tolist()
