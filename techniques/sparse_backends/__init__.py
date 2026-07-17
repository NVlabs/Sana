# Sparse-attention kernel backends vendored/integrated as first-class options,
# alongside PISA. Each backend exposes a dense-fallback attention callable plus a
# dispatch hook that model runtimes install to route real self-attention through it.
