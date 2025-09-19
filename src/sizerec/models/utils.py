def count_params(model, trainable_only: bool = True) -> int:
    """
    Return the number of parameters in `model`.
    Set `trainable_only=True` to count only parameters with requires_grad=True.
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())
