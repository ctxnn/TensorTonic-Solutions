import torch 

def gradient_accumulation_step(param, microbatch_inputs, microbatch_targets, lr):
    # accumulator for the full gradient
    full_grad = [0.0] * len(param)

    total_examples = 0

    # go through every microbatch
    for batch_idx in range(len(microbatch_inputs)):
        inputs = microbatch_inputs[batch_idx]
        targets = microbatch_targets[batch_idx]

        # go through every example in this microbatch
        for example_idx in range(len(inputs)):
            x = inputs[example_idx]
            y = targets[example_idx]

            # prediction = dot(param, x)
            prediction = 0.0

            for j in range(len(param)):
                prediction += param[j] * x[j]

            # for MSE loss:
            # gradient contribution = 2 * (prediction - target) * x[j]
            error = prediction - y

            for j in range(len(param)):
                full_grad[j] += 2 * error * x[j]

            total_examples += 1

    # average over all examples
    for j in range(len(full_grad)):
        full_grad[j] /= total_examples

    # one SGD update
    new_param = []

    for j in range(len(param)):
        updated_value = param[j] - lr * full_grad[j]
        new_param.append(updated_value)

    return {
        "new_param": torch.tensor(new_param),
        "full_grad": torch.tensor(full_grad)
    }