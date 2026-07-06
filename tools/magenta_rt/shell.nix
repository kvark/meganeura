# Minimal nix-shell for dump_codec_local.py — just what's needed to dump
# the SpectroStream codec from HuggingFace and run a CPU round-trip.
#
# Avoids the OOM build issue with faster-whisper → onnxruntime in the user's
# default ~/Shell/python.nix.
#
# Use:
#   nix-shell tools/magenta_rt/shell.nix --run "python tools/magenta_rt/dump_codec_local.py"

{ pkgs ? import <nixpkgs> { config.allowUnfree = true; } }:

pkgs.mkShell {
  buildInputs = [
    (pkgs.python312.withPackages (ps: [
      ps.tensorflow      # stable TF, CPU is fine
      ps.huggingface-hub
      ps.safetensors
      ps.numpy
    ]))
    pkgs.stdenv.cc.cc.lib  # libstdc++ for any C++ extension
    pkgs.zlib
  ];
  shellHook = ''
    export LD_LIBRARY_PATH="${pkgs.zlib}/lib:${pkgs.stdenv.cc.cc.lib}/lib:$LD_LIBRARY_PATH"
    export TF_CPP_MIN_LOG_LEVEL=2
    echo "Magenta-RT codec dump shell ready. Run:"
    echo "  python tools/magenta_rt/dump_codec_local.py"
  '';
}
