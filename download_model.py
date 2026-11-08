"""Explicit, one-time pretrained model download; no music leaves the device."""

from transformers import ClapFeatureExtractor, ClapModel

from core import data_directory

name = "laion/clap-htsat-unfused"
root = data_directory() / "clap"
print(
    "Descargando el codificador de audio CLAP. Puede tardar y ocupar bastante espacio."
)
processor = ClapFeatureExtractor.from_pretrained(name)
model = ClapModel.from_pretrained(name, use_safetensors=False, weights_only=True)
root.mkdir(parents=True, exist_ok=True)
processor.save_pretrained(root)
model.save_pretrained(root, safe_serialization=True)
print("Modelo guardado en", root)
