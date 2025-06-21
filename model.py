import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
from jaxtyping import Array, PRNGKeyArray, Float
from einops import rearrange

class PatchEmbedding(eqx.Module):
    linear: eqx.nn.Linear
    patch_size: int

    def __init__(self, patch_size: int, embedding_dim: int, key: PRNGKeyArray):
        self.patch_size = patch_size
        patch_dim = patch_size * patch_size * 3
        self.linear = eqx.nn.Linear(in_features=patch_dim, out_features=embedding_dim, key=key)

    def __call__(self, x: Array) -> Array:
        H, W, C = x.shape

        x_patches = rearrange(x, '(h p1) (w p2) c -> (h w) (p1 p2 c)',
                             p1=self.patch_size, p2=self.patch_size)

        embeddings = jax.vmap(self.linear)(x_patches)
        return embeddings

class PositionalEmbedding(eqx.Module):
    pos_embed: Array

    def __init__(self, num_patches: int, embedding_dim: int, *, key: PRNGKeyArray):
        self.pos_embed = jr.normal(key, (num_patches + 1, embedding_dim)) * 0.02

    def __call__(self, x: Array) -> Array:
        return x + self.pos_embed

class Attention(eqx.Module):
    attention: eqx.nn.MultiheadAttention
    norm: eqx.nn.LayerNorm
    dropout: eqx.nn.Dropout

    def __init__(self, embedding_dim: int, num_heads: int, dropout_rate: float, key: PRNGKeyArray):
        self.attention = eqx.nn.MultiheadAttention(
            num_heads=num_heads,
            query_size=embedding_dim,
            key=key,
            dropout_p=dropout_rate
        )
        self.norm = eqx.nn.LayerNorm(shape=embedding_dim)
        self.dropout = eqx.nn.Dropout(p=dropout_rate)

    def __call__(self, x: Array, *, key: PRNGKeyArray = None, inference: bool = False) -> Array:
        normed = jax.vmap(self.norm)(x)
        attended = self.attention(normed, normed, normed, key=key, inference=inference)
        if not inference and key is not None:
            attended = self.dropout(attended, key=key)
        return x + attended

class MLP(eqx.Module):
    linear1: eqx.nn.Linear
    linear2: eqx.nn.Linear
    norm: eqx.nn.LayerNorm
    dropout: eqx.nn.Dropout

    def __init__(self, embedding_dim: int, hidden_dim: int, dropout_rate: float, key: PRNGKeyArray):
        key1, key2 = jr.split(key)
        self.linear1 = eqx.nn.Linear(in_features=embedding_dim, out_features=hidden_dim, key=key1)
        self.linear2 = eqx.nn.Linear(in_features=hidden_dim, out_features=embedding_dim, key=key2)
        self.norm = eqx.nn.LayerNorm(shape=embedding_dim)
        self.dropout = eqx.nn.Dropout(p=dropout_rate)

    def __call__(self, x: Array, *, key: PRNGKeyArray = None, inference: bool = False) -> Array:
        normed = jax.vmap(self.norm)(x)
        hidden = jax.vmap(self.linear1)(normed)
        hidden = jax.nn.gelu(hidden)
        if not inference and key is not None:
            hidden = self.dropout(hidden, key=key)
        output = jax.vmap(self.linear2)(hidden)
        if not inference and key is not None:
            output = self.dropout(output, key=key)
        return x + output

class EncoderBlock(eqx.Module):
    attention: Attention
    mlp: MLP

    def __init__(self, embedding_dim: int, hidden_dim: int, num_heads: int,
                 dropout_rate: float, key: PRNGKeyArray):
        key1, key2 = jr.split(key)
        self.attention = Attention(embedding_dim, num_heads, dropout_rate, key1)
        self.mlp = MLP(embedding_dim, hidden_dim, dropout_rate, key2)

    def __call__(self, x: Array, *, key: PRNGKeyArray = None, inference: bool = False) -> Array:
        key1, key2 = jr.split(key, 2) if key is not None else (None, None)
        x = self.attention(x, key=key1, inference=inference)
        x = self.mlp(x, key=key2, inference=inference)
        return x


class ClassificationHead(eqx.Module):
    classifier : eqx.nn.Linear
    dropout: eqx.nn.Dropout

    def __init__(self, embedding_dim: int, num_classes: int, dropout_rate: float, *, key: PRNGKeyArray):
        classifier_key = key

        self.classifier = eqx.nn.Linear(in_features=embedding_dim, out_features=num_classes, key=classifier_key)
        self.dropout = eqx.nn.Dropout(p=dropout_rate)

    def __call__(self, cls_token: Array, *, key: PRNGKeyArray | None = None, inference: bool = False) -> Array:
        if not inference and key is not None:
            cls_token = self.dropout(cls_token, key=key)
        return self.classifier(cls_token)


class VIT(eqx.Module):
    patch_embed: PatchEmbedding
    cls_token: Array
    pos_embed: PositionalEmbedding
    encoder_blocks: list
    final_norm: eqx.nn.LayerNorm
    classification_head: ClassificationHead
    dropout: eqx.nn.Dropout

    def __init__(self, patch_size: int, embedding_dim: int, hidden_dim: int,
                 num_heads: int, num_layers: int, num_classes: int,
                 dropout_rate: float, num_patches: int, *, key: PRNGKeyArray):
        keys = jr.split(key, num_layers + 6)
        self.patch_embed = PatchEmbedding(patch_size, embedding_dim, keys[0])
        self.cls_token = jr.normal(keys[1], (1, embedding_dim)) * 0.02
        self.pos_embed = PositionalEmbedding(num_patches, embedding_dim, key=keys[2])
        self.encoder_blocks = [
            EncoderBlock(embedding_dim, hidden_dim, num_heads, dropout_rate, keys[3 + i])
            for i in range(num_layers)
        ]
        self.final_norm = eqx.nn.LayerNorm(shape=embedding_dim)
        self.classification_head = ClassificationHead(embedding_dim, num_classes, dropout_rate, key=keys[3 + num_layers])
        self.dropout = eqx.nn.Dropout(p=dropout_rate)

    def __call__(self, x: Array, *, key: PRNGKeyArray = None, inference: bool = False) -> Array:
        keys = jr.split(key, len(self.encoder_blocks) + 2) if key is not None else [None] * (len(self.encoder_blocks) + 2)
        x = self.patch_embed(x)
        x = jnp.concatenate([self.cls_token, x], axis=0)
        x = self.pos_embed(x)
        if not inference:
            x = self.dropout(x, key=keys[0])
        for i, block in enumerate(self.encoder_blocks):
            x = block(x, key=keys[i + 1], inference=inference)
        cls_token = self.final_norm(x[0])
        logits = self.classification_head(cls_token, key=keys[-1], inference=inference)
        return logits
