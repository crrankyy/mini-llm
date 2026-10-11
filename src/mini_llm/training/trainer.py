import os
import torch
from datetime import datetime
from mini_llm.data import ResumableSampler
from mini_llm.training.loss import calc_loss_batch
from mini_llm.training.evaluate import evaluate_model
from mini_llm.training.checkpoint import (
    checkpoint_state, epoch_checkpoint_path, prune_epoch_checkpoints,
    rolling_write_order, save_rolling_checkpoint
)
from mini_llm.generation import generate_and_print_sample

def train_model_v1(model, train_loader, val_loader,
                   optimizer, device, num_epochs,
                   eval_frq, eval_iter, start_context, tokenizer, start_epoch=0,
                   checkpoint_dir="checkpoints", save_frq=None, keep_checkpoints=10,
                   resume=None, eval_train_loader=None):

    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1
    completed_epochs, batches_done = start_epoch, 0

    # created up front so an unusable path fails before the first epoch, not after it
    os.makedirs(checkpoint_dir, exist_ok=True)

    device_type = 'cuda' if 'cuda' in str(device) else 'cpu'
    scaler = torch.cuda.amp.GradScaler() if device_type == 'cuda' else None

    if resume is not None:
        tokens_seen, global_step = resume["tokens_seen"], resume["global_step"]
        batches_done = resume["batches_done"]
        if scaler is not None and resume["scaler_state_dict"] is not None:
            scaler.load_state_dict(resume["scaler_state_dict"])

    if eval_train_loader is None:
        eval_train_loader = train_loader

    rolling_order = rolling_write_order(checkpoint_dir, resume["path"] if resume is not None else None)
    rolling_saves = 0

    try:
        for epoch in range(start_epoch, start_epoch + num_epochs):

            model.train()

            if isinstance(train_loader.sampler, ResumableSampler):
                train_loader.sampler.set_position(epoch, batches_done * train_loader.batch_size)
            elif batches_done:
                raise ValueError("Resuming inside an epoch needs a training loader that was created with a seed.")

            for input_batch, target_batch in train_loader:
                optimizer.zero_grad()

                if device_type == 'cuda':
                    with torch.autocast(device_type=device_type, dtype=torch.float16):
                        loss = calc_loss_batch(
                            input_batch, target_batch, model, device
                        )
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss = calc_loss_batch(
                        input_batch, target_batch, model, device
                    )
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

                tokens_seen += input_batch.numel()
                global_step += 1
                batches_done += 1

                if global_step % eval_frq == 0:
                    train_loss, val_loss = evaluate_model(
                        model, eval_train_loader, val_loader, device, eval_iter
                    )
                    train_losses.append(train_loss)
                    val_losses.append(val_loss)
                    track_tokens_seen.append(tokens_seen)

                    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    print(f"Ep {epoch+1} (Step {global_step:06d}) [{current_time}]: "
                          f"Train loss {train_loss:.3f}, "
                          f"Val loss {val_loss:.3f}"
                    )

                if save_frq is not None and (global_step + 1) % save_frq == 0:
                    rolling_saves = save_rolling_checkpoint(
                        checkpoint_state(model, optimizer, scaler, completed_epochs, batches_done, global_step, tokens_seen),
                        rolling_order, rolling_saves
                    )

            generate_and_print_sample(
                model, tokenizer, device, start_context
            )

            checkpoint_path = epoch_checkpoint_path(checkpoint_dir, epoch + 1)
            torch.save(
                checkpoint_state(model, optimizer, scaler, epoch + 1, 0, global_step, tokens_seen),
                checkpoint_path
            )
            print(f"--> Saved checkpoint: {checkpoint_path}")
            prune_epoch_checkpoints(checkpoint_dir, keep_checkpoints)

            # one statement, so that an interrupt cannot land between the two and record a
            # position that says the next epoch already has all of its batches done
            completed_epochs, batches_done = epoch + 1, 0

            if save_frq is not None:
                rolling_saves = save_rolling_checkpoint(
                    checkpoint_state(model, optimizer, scaler, completed_epochs, batches_done, global_step, tokens_seen),
                    rolling_order, rolling_saves
                )

    except KeyboardInterrupt:
        if save_frq is not None:
            print("\n[!] Interrupted. Saving the current position...")
            save_rolling_checkpoint(
                checkpoint_state(model, optimizer, scaler, completed_epochs, batches_done, global_step, tokens_seen),
                rolling_order, rolling_saves
            )
        raise

    return train_losses, val_losses, track_tokens_seen
