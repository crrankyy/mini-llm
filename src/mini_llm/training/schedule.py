import math

def learning_rate_at(update, total_updates, peak, warmup_updates, final):

    # `update` counts the weight updates already made, so the rate depends only on a
    # number the checkpoint holds and a resumed run continues the same curve.
    # The first update is made at peak / warmup_updates: at a rate of zero it would do nothing
    if update < warmup_updates:
        return peak * (update + 1) / warmup_updates

    if update >= total_updates:
        return final

    progress = (update - warmup_updates) / (total_updates - warmup_updates)

    return final + 0.5 * (peak - final) * (1 + math.cos(math.pi * progress))
