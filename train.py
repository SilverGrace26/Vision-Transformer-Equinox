import jax
import jax.numpy as jnp
import jax.random as jr
import optax
import equinox as eqx
from model import VIT
from dataloading import get_dataloaders
import matplotlib.pyplot as plt

""" Hyper Parameters """
lr = 0.0003
batch_size = 64
patch_size = 4
num_steps = 30000
image_size = (32, 32, 3)
embedding_dim = 384
hidden_dim = 768
num_heads = 12
num_layers = 8
dropout_rate = 0.15
height, width, channels = image_size
num_classes = 10

num_patches = (height // patch_size) * (width // patch_size)
print(f"Number of patches: {num_patches}")  # Should be 64

trainloader, testloader = get_dataloaders(batch_size)

# Training
@eqx.filter_value_and_grad
def compute_loss(model, images, labels, key):
    # Process batch of images
    def single_forward(image, subkey):
        return model(image, key=subkey, inference=False)

    keys = jr.split(key, len(images))
    logits = jax.vmap(single_forward)(images, keys)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
    return jnp.mean(loss)

@eqx.filter_jit
def train_step(model, opt_state, images, labels, optimizer, key):
    loss, grads = compute_loss(model, images, labels, key)
    updates, new_opt_state = optimizer.update(grads, opt_state, model)
    new_model = eqx.apply_updates(model, updates)
    return new_model, new_opt_state, loss

def train_model(model, optimizer, trainloader, num_steps):
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    losses = []

    def infinite_dataloader():
        while True:
            yield from trainloader

    print("Starting training...")
    key = jr.PRNGKey(0)

    for step, (images, labels) in zip(range(num_steps), infinite_dataloader()):
        # Convert to JAX arrays and proper format
        images = jnp.array(images.numpy().transpose(0, 2, 3, 1))  # BCHW -> BHWC
        labels = jnp.array(labels.numpy())

        if step == 0:
            print(f"Image shape: {images.shape}, range: [{images.min():.3f}, {images.max():.3f}]")
            print(f"Labels shape: {labels.shape}, unique labels: {jnp.unique(labels)}")

        key, subkey = jr.split(key)
        model, opt_state, loss = train_step(model, opt_state, images, labels, optimizer, subkey)
        losses.append(float(loss))

        if step % 2000 == 0:
            print(f"Step {step:5d}: Loss = {loss:.4f}")

    return model, losses


key = jr.PRNGKey(42)
model = VIT(
    patch_size=patch_size,
    embedding_dim=embedding_dim,
    hidden_dim=hidden_dim,
    num_heads=num_heads,
    num_layers=num_layers,
    num_classes=num_classes,
    dropout_rate=dropout_rate,
    num_patches=num_patches,
    key=key
)

print("Model created successfully!")
total_params = sum(x.size for x in jax.tree_leaves(eqx.filter(model, eqx.is_array)))
print(f"Total parameters: {total_params:,}")

# Test forward pass
test_batch = next(iter(trainloader))
test_image = jnp.array(test_batch[0][0].numpy().transpose(1, 2, 0))  # Single image, CHW -> HWC
print(f"Test image shape: {test_image.shape}")

try:
    output = model(test_image, key=jr.PRNGKey(0), inference=True)
    print(f"Forward pass successful! Output shape: {output.shape}")
    print(f"Output logits: {output}")
except Exception as e:
    print(f"Forward pass failed: {e}")
    import traceback
    traceback.print_exc()

warmup_steps = 2000
total_steps = num_steps

schedule = optax.warmup_cosine_decay_schedule(
    init_value=0.0,
    peak_value=lr,
    warmup_steps=warmup_steps,
    decay_steps=total_steps - warmup_steps,
    end_value=lr * 0.01
)

optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adamw(learning_rate=schedule, weight_decay=0.01)
)

print(f"\nTraining for {num_steps} steps...")
trained_model, losses = train_model(model, optimizer, trainloader, num_steps)

def evaluate_full(model, testloader):
    """Evaluate on full test set"""
    correct = 0
    total = 0

    print("Evaluating on full test set...")
    for i, (images, labels) in enumerate(testloader):
        images = jnp.array(images.numpy().transpose(0, 2, 3, 1))
        labels = jnp.array(labels.numpy())

        def predict(image):
            logits = model(image, key=None, inference=True)
            return jnp.argmax(logits)

        predictions = jax.vmap(predict)(images)
        correct += jnp.sum(predictions == labels)
        total += len(labels)

        if (i + 1) % 20 == 0:
            current_acc = float(correct) / total
            print(f"  Progress: {i+1}/{len(testloader)} batches, Accuracy so far: {current_acc:.4f}")

    return float(correct) / total

full_accuracy = evaluate_full(trained_model, testloader)
print(f"\nFinal Results:")
print(f"Full test accuracy: {full_accuracy:.4f}")
print(f"Final loss: {losses[-1]:.4f}")
print(f"Best loss: {min(losses):.4f}")


import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Loss plot
ax1.plot(losses)
ax1.set_title('Training Loss')
ax1.set_xlabel('Step')
ax1.set_ylabel('Loss')
ax1.set_yscale('log')
ax1.grid(True)

# Learning rate schedule visualization
steps = jnp.arange(num_steps)
lrs = [schedule(step) for step in steps]
ax2.plot(lrs)
ax2.set_title('Learning Rate Schedule')
ax2.set_xlabel('Step')
ax2.set_ylabel('Learning Rate')
ax2.grid(True)

plt.tight_layout()
plt.show()

