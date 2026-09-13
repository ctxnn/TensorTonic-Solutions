import math

def memory_accountant(
    param_shapes,
    param_bytes_per_element,
    grad_bytes_per_element,
    activation_shapes,
    activation_bytes_per_element,
    optimizer,
    optimizer_bytes_per_element
):
    param_elements = 0

    for shape in param_shapes:
        param_elements += math.prod(shape)

    param_memory = param_elements * param_bytes_per_element

    grad_memory = param_elements * grad_bytes_per_element


    activation_elements = 0

    for shape in activation_shapes:
        activation_elements += math.prod(shape)

    activation_memory = (
        activation_elements * activation_bytes_per_element
    )


    if optimizer == "sgd":
        num_states = 0

    elif optimizer == "adagrad":
        num_states = 1

    elif optimizer == "adam":
        num_states = 2


    optimizer_elements = param_elements * num_states

    optimizer_memory = (
        optimizer_elements * optimizer_bytes_per_element
    )


    total = (
        param_memory
        + grad_memory
        + activation_memory
        + optimizer_memory
    )

    return {
        "parameters": param_memory,
        "gradients": grad_memory,
        "activations": activation_memory,
        "optimizer_state": optimizer_memory,
        "total": total
    }