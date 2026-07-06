"""Convert MusicCoCa text encoder SavedModel to TFLite for clearer inspection."""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import tensorflow as tf
from pathlib import Path

SM = Path("/x/Hub/models--google--magenta-realtime/snapshots/c05f8d6d608afd588469b7a8ef0929d5a1f8f6bb/savedmodels/musiccoca_mv212f_cpu_novocab")
OUT = Path("/tmp/musiccoca_text_encoder.tflite")

print(f"Loading {SM}...")
with tf.device("/cpu:0"):
    mc = tf.saved_model.load(str(SM))

print(f"Available signatures: {list(mc.signatures.keys())}")
print(f"Converting embed_text signature to TFLite...")
converter = tf.lite.TFLiteConverter.from_saved_model(
    str(SM),
    signature_keys=["embed_text"],
)
# Allow custom + select TF ops if needed.
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS,
]
converter.experimental_new_converter = True
try:
    tflite_blob = converter.convert()
except Exception as e:
    print(f"Conversion error: {e}")
    raise
OUT.write_bytes(tflite_blob)
print(f"Saved {len(tflite_blob)} bytes to {OUT}")

# Inspect the model.
interp = tf.lite.Interpreter(model_path=str(OUT))
interp.allocate_tensors()
tensors = interp.get_tensor_details()
print(f"\nTotal tensors: {len(tensors)}")
# Print tensors with their names and shapes.
for t in tensors:
    if t['name']:
        print(f"  [{t['index']:4d}] {t['name'][:80]:<80s} shape={t['shape']} dtype={t['dtype'].__name__}")
