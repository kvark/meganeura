# Graphics-only platform qualification

These runs establish that Meganeura executes and meets the numerical gates on
a GPU without a usable PyTorch GPU path. CPU PyTorch is a correctness oracle,
not a performance competitor. Do not publish CPU/GPU timing ratios or include
these runs in GPU performance, preparation-time or portability-score aggregates.

The existing collector already supports this; no new harness is needed.
The September 25 qualification uses Inferena `7b8fcb72` and Meganeura
`0dbfcc00`, tagged `paper-p3hpc-2026-final` in both repositories. All three devices below pass all ten
workload/precision conditions at those pins. Repeat qualification after an
engine update before claiming coverage of the new revision.

## Ryzen 5 9600X integrated Radeon on rubik

The recorded native adapter is `AMD Ryzen 5 9600X 6-Core Processor
(RADV RAPHAEL_MENDOCINO)`, device ID 5056. The shared driver name is not a
reason to label this processor's integrated GPU as a separate Mendocino APU.

From the Inferena checkout, with Rust, uv and a working RADV Vulkan driver:

```sh
git fetch origin tag paper-p3hpc-2026-final
git switch --detach paper-p3hpc-2026-final
bash scripts/setup.sh cpu .venv-p3hpc-cpu
cargo run --release --locked -p inferena-meganeura -- --list-devices
.venv-p3hpc-cpu/bin/python scripts/p3hpc.py \
  --backend cpu --gpu 'AMD Ryzen 5 9600X' --qualify-only --eager
cp ../inferena-results/latest.tgz ../inferena-results/rubik-9600x-qualification.tgz
```

Reuse `.venv-p3hpc-cpu` if already installed; setup refuses to overwrite it.
The device list must show the intended hardware as available and not software
emulated. The GPU filter matches the recorded adapter name. If the list uses
a different name, use that exact name instead. If more than one matches, stop and disambiguate; do not
drop the filter. The collector selects its device ID and checks the executed
device in every result. It refuses software-renderer or ambiguous matches.

For multiple Mesa devices with identical names, first inspect
`MESA_VK_DEVICE_SELECT=list vulkaninfo`. Mesa's
[device-selection layer](https://docs.mesa3d.org/envvars.html#vulkan-mesa-device-select-layer-environment-variables)
can restrict enumeration with `MESA_VK_DEVICE_SELECT=vendor:device!`, using
the IDs actually listed on rubik. Do not guess a PCI ID from the product name.

`--eager` avoids spending time compiling the CPU oracle. It does not disable
Meganeura tuning, change precision, or relax forward/backward checks. Both
arithmetic contracts and all five models run, with one fresh qualification
process per condition rather than three 20-sample measurement processes.
No ROCm installation, architecture override or `--no-max-autotune` is needed
for this CPU-oracle/Vulkan path. Integrated GPUs remain subject to ordinary
device-memory and numerical checks; incomplete runs are not passes.

## Intel RPL-U

Use the same CPU environment and replace the collection command with:

```sh
.venv-p3hpc-cpu/bin/python scripts/p3hpc.py \
  --backend cpu --gpu RPL-U --qualify-only --eager
cp ../inferena-results/latest.tgz ../inferena-results/intel-rpl-u-qualification.tgz
```

The current archives establish ten passing workload/precision conditions,
one process each, on RPL-U, Radeon 780M, and the Ryzen iGPU. These are qualification runs,
not three-replicate timing campaigns. The paper reports no CPU-reference timings.

## Radeon 780M

The compiled ROCm campaign failed on its first strict SmolLM2-135M condition
with an unspecified HIP launch failure. Separate qualification passed:

```sh
.venv-p3hpc-cpu/bin/python scripts/p3hpc.py \
  --backend cpu --gpu 'AMD Radeon 780M' --qualify-only --eager
cp ../inferena-results/latest.tgz ../inferena-results/amd-780m-qualification.tgz
```

The supplied run used the ROCm wheel with `--backend cpu`, which is also valid.
Its architecture/copy overrides are recorded, but do not configure Vulkan or
the CPU oracle. This establishes the tested native path, not a claim that every
possible PyTorch GPU configuration must fail.

## Evidence and archive lifetime

Require `campaign.json` to finish with `status: complete`, `args.collect: false`,
the intended `native_device`, and ten valid qualification runs. Preserve the
archive, driver/hardware identity and exact source pin. A CPU-only wheel alone
does not prove that a GPU PyTorch path is unavailable: retain the GPU-backend
probe failure or vendor support evidence separately. The Ryzen qualification
uses a ROCm wheel but explicitly executes the reference on CPU. Its successful
run does not contradict the separately reported ROCm GPU bring-up failures.
Any separate Mendocino machine would need its own qualification evidence.

Each attempt creates a new timestamped directory. `latest.tgz` is atomically
replaced on **completion or failure**, so copy it before another attempt or
platform run. A failed archive is diagnostic evidence, not a qualification pass.
The timestamped directories remain available; never run two collectors sharing
the same `latest.tgz` destination at once.
