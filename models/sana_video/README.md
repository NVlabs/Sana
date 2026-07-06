# Sana Video Model Contract

Directory-style model contract for the Sana 5B 720p193 baseline.

The runtime wrapper does not copy the private Hugging Face dataset contents or
the 17 GB checkpoint into the repository. New experiments copy only the local
wrapper, candidate manifest, model/eval profiles, and standard launch/eval
helpers. The minimal inference zip, checkpoint, VAE, and text encoder stay
reference-only shared assets.
