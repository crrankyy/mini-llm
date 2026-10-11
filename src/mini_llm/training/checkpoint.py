import os
import re
import torch

# Two files that are overwritten in turn. A save that is cut off damages only the file
# being written, and nothing has to be renamed or deleted: on Google Drive a deleted
# file moves to the Trash and keeps counting against the storage quota.
ROLLING_NAMES = ("gpt_wikitext_checkpoint_latest_a.pth", "gpt_wikitext_checkpoint_latest_b.pth")
EPOCH_NAME = re.compile(r"gpt_wikitext_checkpoint_ep(\d+)\.pth")

def rolling_paths(checkpoint_dir):
    return [os.path.join(checkpoint_dir, name) for name in ROLLING_NAMES]

def epoch_checkpoint_path(checkpoint_dir, epoch):
    return os.path.join(checkpoint_dir, f"gpt_wikitext_checkpoint_ep{epoch}.pth")

def epoch_checkpoint_paths(checkpoint_dir):

    if not os.path.isdir(checkpoint_dir):
        return []

    # ordered by epoch number: sorting the names would put ep10 before ep2
    numbered = sorted(
        (int(match[1]), name)
        for name in os.listdir(checkpoint_dir)
        if (match := EPOCH_NAME.fullmatch(name))
    )

    return [os.path.join(checkpoint_dir, name) for _, name in numbered]

def resume_candidates(checkpoint_dir):

    rolling = [path for path in rolling_paths(checkpoint_dir) if os.path.exists(path)]
    rolling.sort(key=os.path.getmtime, reverse=True)

    return rolling + epoch_checkpoint_paths(checkpoint_dir)[::-1]

def load_latest_checkpoint(checkpoint_dir):

    candidates = resume_candidates(checkpoint_dir)

    for path in candidates:
        try:
            # onto the CPU, so that a GPU running out of memory is not taken for a damaged file
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as error:
            print(f"Could not read {path}: {error}")
            continue

        return checkpoint, path

    if candidates:
        # starting from scratch here would train over checkpoints that may still be recoverable
        raise RuntimeError(
            f"None of the checkpoints in '{checkpoint_dir}' could be read. "
            "Move them out of the folder to start a new run."
        )

    return None, None

def training_progress(checkpoint, path):

    global_step = checkpoint.get("global_step", -1)

    # checkpoints written before the rolling files existed have only the first two keys
    return {
        "path": path,
        "batches_done": checkpoint.get("batches_done", 0),
        "global_step": global_step,
        "tokens_seen": checkpoint.get("tokens_seen", 0),
        # before gradients were accumulated, every batch was one update
        "updates_done": checkpoint.get("updates_done", global_step + 1),
        "scaler_state_dict": checkpoint.get("scaler_state_dict"),
        # None in checkpoints written while the batch size was fixed in the script
        "batch_size": checkpoint.get("batch_size"),
    }

def checkpoint_state(model, optimizer, scaler, epoch, batches_done, global_step, tokens_seen, updates_done,
                     batch_size):

    return {
        'epoch': epoch,
        'batches_done': batches_done,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict() if scaler is not None else None,
        'global_step': global_step,
        'tokens_seen': tokens_seen,
        'updates_done': updates_done,
        # the unit batches_done is counted in
        'batch_size': batch_size
    }

def rolling_write_order(checkpoint_dir, resumed_from=None):

    paths = rolling_paths(checkpoint_dir)

    if resumed_from in paths:
        # until the first save has finished, the file this run resumed from is the only good copy
        return [path for path in paths if path != resumed_from] + [resumed_from]

    return sorted(paths, key=lambda path: os.path.getmtime(path) if os.path.exists(path) else 0)

def save_rolling_checkpoint(state, write_order, saves_done):

    path = write_order[saves_done % len(write_order)]
    torch.save(state, path)
    print(f"--> Saved rolling checkpoint: {path}")

    # counted only once the file is complete, so a save that was interrupted is redone into the same file
    return saves_done + 1

def prune_epoch_checkpoints(checkpoint_dir, keep):

    if keep is None:
        return

    paths = epoch_checkpoint_paths(checkpoint_dir)

    for path in paths[:max(len(paths) - keep, 0)]:
        os.remove(path)
        print(f"--> Deleted old checkpoint: {path}")
