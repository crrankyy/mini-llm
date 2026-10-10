import torch
from datetime import datetime
from mini_llm.training.loss import calc_loss_batch
from mini_llm.training.evaluate import evaluate_model
from mini_llm.generation import generate_and_print_sample

def train_model_v1(model, train_loader, val_loader,
                   optimizer, device, num_epochs,
                   eval_frq, eval_iter, start_context, tokenizer, start_epoch=0):
    
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1
    
    device_type = 'cuda' if 'cuda' in str(device) else 'cpu'
    scaler = torch.cuda.amp.GradScaler() if device_type == 'cuda' else None

    for epoch in range(start_epoch, start_epoch + num_epochs):
        
        model.train()

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

            if global_step % eval_frq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)

                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"Ep {epoch+1} (Step {global_step:06d}) [{current_time}]: "
                      f"Train loss {train_loss:.3f}, "
                      f"Val loss {val_loss:.3f}"
                )

        generate_and_print_sample(
            model, tokenizer, device, start_context
        )

        checkpoint_path = f"gpt_wikitext_checkpoint_ep{epoch+1}.pth"
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step,
            'tokens_seen': tokens_seen
        }, checkpoint_path)
        print(f"--> Saved checkpoint: {checkpoint_path}")

    return train_losses, val_losses, track_tokens_seen