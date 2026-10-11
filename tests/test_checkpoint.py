import os

import numpy as np
import pytest
import tiktoken
import torch

from mini_llm.data import ResumableSampler, create_memmap_dataloader
from mini_llm.model.gpt import GPTModel
from mini_llm.training import trainer
from mini_llm.training.checkpoint import (
    epoch_checkpoint_path, epoch_checkpoint_paths, load_latest_checkpoint,
    prune_epoch_checkpoints, resume_candidates, rolling_paths, rolling_write_order,
    save_rolling_checkpoint, training_progress
)
from mini_llm.training.optimizer import create_optimizer

CONFIG = {
    "vocab_size": 50,
    "context_length": 8,
    "embedding_dimension": 16,
    "n_heads": 2,
    "n_layers": 1,
    "drop_rate": 0.0,
    "qkv_bias": False
}

BATCH_SIZE = 4
BATCHES_PER_EPOCH = 6
EPOCHS = 2
SAVE_FRQ = 4


class MachineTakenAway(Exception):
    pass


def set_modified_time(path, seconds):
    os.utime(path, (seconds, seconds))


def test_the_order_depends_only_on_the_seed_and_the_epoch():
    first, second = ResumableSampler(100, seed=7), ResumableSampler(100, seed=7)

    assert list(first) == list(second)
    assert sorted(first) == list(range(100))

    second.set_position(epoch=1, start=0)

    assert list(first) != list(second)


def test_a_later_start_continues_the_same_order():
    sampler = ResumableSampler(100, seed=7)
    sampler.set_position(epoch=3, start=0)
    whole_epoch = list(sampler)

    sampler.set_position(epoch=3, start=40)

    assert list(sampler) == whole_epoch[40:]
    assert len(sampler) == 60


def test_epoch_checkpoints_are_ordered_by_number_not_by_name(tmp_path):
    for epoch in (10, 2, 1):
        torch.save({}, epoch_checkpoint_path(tmp_path, epoch))

    names = [os.path.basename(path) for path in epoch_checkpoint_paths(tmp_path)]

    assert names == [
        "gpt_wikitext_checkpoint_ep1.pth", "gpt_wikitext_checkpoint_ep2.pth", "gpt_wikitext_checkpoint_ep10.pth"
    ]


def test_pruning_deletes_the_oldest_epoch_checkpoints_and_nothing_else(tmp_path):
    for epoch in range(1, 13):
        torch.save({}, epoch_checkpoint_path(tmp_path, epoch))
    for path in rolling_paths(tmp_path) + [os.path.join(tmp_path, "gpt_wikitext_124M.pth")]:
        torch.save({}, path)

    prune_epoch_checkpoints(tmp_path, keep=10)

    assert epoch_checkpoint_paths(tmp_path) == [epoch_checkpoint_path(tmp_path, epoch) for epoch in range(3, 13)]
    assert len(os.listdir(tmp_path)) == 13


def test_saves_alternate_and_never_start_with_the_file_that_was_resumed_from(tmp_path):
    first, second = rolling_paths(tmp_path)
    torch.save({"global_step": 1}, first)
    torch.save({"global_step": 2}, second)
    # the file resumed from is the older one, as it is after the newer one turned out to be damaged
    set_modified_time(first, 1000)
    set_modified_time(second, 2000)

    order = rolling_write_order(tmp_path, resumed_from=first)
    saves = save_rolling_checkpoint({"global_step": 3}, order, 0)

    assert torch.load(first)["global_step"] == 1
    assert torch.load(second)["global_step"] == 3

    save_rolling_checkpoint({"global_step": 4}, order, saves)

    assert torch.load(first)["global_step"] == 4


def test_resume_takes_the_newest_file_that_can_be_read(tmp_path):
    first, second = rolling_paths(tmp_path)
    torch.save({"epoch": 1, "global_step": 1}, epoch_checkpoint_path(tmp_path, 1))
    torch.save({"epoch": 1, "global_step": 5}, first)
    torch.save({"epoch": 1, "global_step": 9}, second)
    set_modified_time(first, 1000)
    set_modified_time(second, 2000)

    assert load_latest_checkpoint(tmp_path)[1] == second

    # cut the newest file short, as a save that was interrupted would leave it
    os.truncate(second, os.path.getsize(second) // 2)
    set_modified_time(second, 2000)

    checkpoint, path = load_latest_checkpoint(tmp_path)

    assert path == first
    assert checkpoint["global_step"] == 5

    os.truncate(first, 0)

    assert load_latest_checkpoint(tmp_path)[1] == epoch_checkpoint_path(tmp_path, 1)


def test_resume_refuses_to_start_over_when_no_checkpoint_can_be_read(tmp_path):
    assert load_latest_checkpoint(tmp_path) == (None, None)

    with open(rolling_paths(tmp_path)[0], "wb") as f:
        f.write(b"not a checkpoint")

    with pytest.raises(RuntimeError, match="None of the checkpoints"):
        load_latest_checkpoint(tmp_path)


def make_token_file(folder):
    path = os.path.join(folder, "train.bin")
    tokens = np.random.default_rng(0).integers(0, CONFIG["vocab_size"], dtype=np.uint32,
                                               size=CONFIG["context_length"] * (BATCH_SIZE * BATCHES_PER_EPOCH + 1))
    tokens.tofile(path)
    return path


def rising_rate(update):
    # a different rate for every update: a resumed run that lost count of its updates would end elsewhere
    return 1e-3 * (update + 1)


def train(token_file, checkpoint_dir, accumulate=1, batch_size=BATCH_SIZE):
    """One run of the trainer, started the way scripts/train_wikitext.py starts it."""
    torch.manual_seed(0)
    model = GPTModel(CONFIG)
    optimizer = create_optimizer(model, lr=1e-2, weight_decay=0.1)

    def loader(**kwargs):
        return create_memmap_dataloader(
            bin_file_path=token_file, batch_size=batch_size, max_length=CONFIG["context_length"],
            stride=CONFIG["context_length"], drop_last=True, **kwargs
        )

    start_epoch, resume = 0, None
    checkpoint, path = load_latest_checkpoint(checkpoint_dir)
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"]
        resume = training_progress(checkpoint, path)

    trainer.train_model_v1(
        model, loader(seed=123), loader(shuffle=False), optimizer, torch.device("cpu"),
        num_epochs=EPOCHS - start_epoch, eval_frq=5, eval_iter=1,
        # "!" is token 0, which this small vocabulary contains
        start_context="!", tokenizer=tiktoken.get_encoding("cl100k_base"), start_epoch=start_epoch,
        checkpoint_dir=checkpoint_dir, save_frq=SAVE_FRQ, resume=resume, eval_train_loader=loader(shuffle=True),
        accumulate=accumulate, lr_schedule=rising_rate
    )
    return model


def stop_at_step(monkeypatch, step, error):
    real_loss, calls = trainer.calc_loss_batch, {"count": 0}

    def loss_until_stopped(*args):
        if calls["count"] == step:
            raise error
        calls["count"] += 1
        return real_loss(*args)

    monkeypatch.setattr(trainer, "calc_loss_batch", loss_until_stopped)


def assert_same_weights(model, expected):
    for name, tensor in expected.state_dict().items():
        assert torch.equal(model.state_dict()[name], tensor), name


# all stop inside the second epoch, between two regular saves. An interrupt saves the exact
# position; a machine that disappears leaves the last regular save, the one after 8 steps.
# With two batches per update, the ninth batch has no update yet when the run stops: its
# gradient is lost with the process, so the position saved is the one after 8 steps
@pytest.mark.parametrize("error, stopped_after, resumed_after, accumulate", [
    (KeyboardInterrupt(), 7, 7, 1),
    (MachineTakenAway(), 10, 2 * SAVE_FRQ, 1),
    (KeyboardInterrupt(), 9, 8, 2),
    (MachineTakenAway(), 11, 2 * SAVE_FRQ, 2),
])
def test_a_run_that_stops_inside_an_epoch_ends_with_the_same_weights(
        tmp_path, monkeypatch, error, stopped_after, resumed_after, accumulate):
    token_file = make_token_file(tmp_path)
    uninterrupted = train(token_file, os.path.join(tmp_path, "uninterrupted"), accumulate)

    checkpoint_dir = os.path.join(tmp_path, "stopped")
    stop_at_step(monkeypatch, stopped_after, error)
    with pytest.raises(type(error)):
        train(token_file, checkpoint_dir, accumulate)
    monkeypatch.undo()

    checkpoint, _ = load_latest_checkpoint(checkpoint_dir)
    assert checkpoint["global_step"] + 1 == resumed_after
    assert (checkpoint["epoch"], checkpoint["batches_done"]) == divmod(resumed_after, BATCHES_PER_EPOCH)
    assert checkpoint["updates_done"] == resumed_after // accumulate

    resumed = train(token_file, checkpoint_dir, accumulate)

    assert_same_weights(resumed, uninterrupted)
    assert len(resume_candidates(checkpoint_dir)) == 2 + EPOCHS


def test_an_interrupt_after_the_last_batch_of_an_epoch_repeats_only_the_end_of_epoch_work(tmp_path, monkeypatch):
    token_file = make_token_file(tmp_path)
    uninterrupted = train(token_file, os.path.join(tmp_path, "uninterrupted"))

    def interrupted_sample(*args):
        raise KeyboardInterrupt

    checkpoint_dir = os.path.join(tmp_path, "stopped")
    monkeypatch.setattr(trainer, "generate_and_print_sample", interrupted_sample)
    with pytest.raises(KeyboardInterrupt):
        train(token_file, checkpoint_dir)
    monkeypatch.undo()

    # the epoch is not counted as completed until its checkpoint has been written
    checkpoint, _ = load_latest_checkpoint(checkpoint_dir)
    assert (checkpoint["epoch"], checkpoint["batches_done"]) == (0, BATCHES_PER_EPOCH)
    assert epoch_checkpoint_paths(checkpoint_dir) == []

    resumed = train(token_file, checkpoint_dir)

    assert_same_weights(resumed, uninterrupted)
    assert len(epoch_checkpoint_paths(checkpoint_dir)) == EPOCHS


def stopped_run(token_file, checkpoint_dir, monkeypatch):
    stop_at_step(monkeypatch, 7, KeyboardInterrupt())
    with pytest.raises(KeyboardInterrupt):
        train(token_file, checkpoint_dir)
    monkeypatch.undo()


def test_a_resume_with_another_batch_size_is_refused_and_changes_no_file(tmp_path, monkeypatch):
    token_file = make_token_file(tmp_path)
    checkpoint_dir = os.path.join(tmp_path, "stopped")
    stopped_run(token_file, checkpoint_dir, monkeypatch)

    def files():
        return {name: open(os.path.join(checkpoint_dir, name), "rb").read() for name in os.listdir(checkpoint_dir)}

    before = files()

    # 7 batches of 4 are done; counted in batches of 2, the run would go over half of them again
    with pytest.raises(ValueError, match=f"batch size of {BATCH_SIZE}, and this run uses {BATCH_SIZE // 2}"):
        train(token_file, checkpoint_dir, batch_size=BATCH_SIZE // 2)

    assert files() == before


def test_a_checkpoint_that_does_not_record_its_batch_size_still_resumes(tmp_path, monkeypatch):
    token_file = make_token_file(tmp_path)
    uninterrupted = train(token_file, os.path.join(tmp_path, "uninterrupted"))

    checkpoint_dir = os.path.join(tmp_path, "stopped")
    stopped_run(token_file, checkpoint_dir, monkeypatch)
    # rewritten oldest first, so that the newest file is still the newest afterwards
    for path in reversed(resume_candidates(checkpoint_dir)):
        checkpoint = torch.load(path, weights_only=False)
        del checkpoint["batch_size"]
        torch.save(checkpoint, path)

    assert_same_weights(train(token_file, checkpoint_dir), uninterrupted)


def test_accumulated_batches_make_the_update_that_one_large_batch_makes(tmp_path):
    token_file = make_token_file(tmp_path)

    def run(batch_size, accumulate):
        torch.manual_seed(0)
        model = GPTModel(CONFIG)
        # plain gradient descent, where the size of a gradient goes straight into the
        # weights. Adam rescales every gradient and would hide one that is too large
        optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

        def loader(**kwargs):
            return create_memmap_dataloader(
                bin_file_path=token_file, batch_size=batch_size, max_length=CONFIG["context_length"],
                stride=CONFIG["context_length"], drop_last=False, **kwargs
            )

        trainer.train_model_v1(
            model, loader(seed=123), loader(shuffle=False), optimizer, torch.device("cpu"),
            num_epochs=1, eval_frq=100, eval_iter=1, start_context="!",
            tokenizer=tiktoken.get_encoding("cl100k_base"),
            checkpoint_dir=os.path.join(tmp_path, f"{batch_size}x{accumulate}"), accumulate=accumulate
        )
        return model

    # 24 samples in the same order: five updates either way, the last from the 4 samples left over
    large_batches = run(batch_size=5, accumulate=1)
    accumulated = run(batch_size=1, accumulate=5)

    for name, tensor in large_batches.state_dict().items():
        assert torch.allclose(accumulated.state_dict()[name], tensor, atol=1e-6), name


def test_the_schedule_is_asked_once_per_update_and_its_rate_is_the_one_used(tmp_path):
    token_file = make_token_file(tmp_path)
    torch.manual_seed(0)
    model = GPTModel(CONFIG)
    start = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    optimizer = create_optimizer(model, lr=1e-2, weight_decay=0.1)
    asked = []

    def no_rate(update):
        asked.append(update)
        return 0.0

    def loader(**kwargs):
        return create_memmap_dataloader(
            bin_file_path=token_file, batch_size=BATCH_SIZE, max_length=CONFIG["context_length"],
            stride=CONFIG["context_length"], drop_last=True, **kwargs
        )

    trainer.train_model_v1(
        model, loader(seed=123), loader(shuffle=False), optimizer, torch.device("cpu"),
        num_epochs=EPOCHS, eval_frq=100, eval_iter=1, start_context="!",
        tokenizer=tiktoken.get_encoding("cl100k_base"), checkpoint_dir=os.path.join(tmp_path, "checkpoints"),
        accumulate=4, lr_schedule=no_rate
    )

    # 6 batches an epoch, 4 to an update: one full update and one from the 2 batches left, twice
    assert asked == [0, 1, 2, 3]
    for name, tensor in model.state_dict().items():
        assert torch.equal(tensor, start[name]), name
