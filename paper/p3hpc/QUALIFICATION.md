# Graphics-only platform qualification

These runs establish that Meganeura executes and meets the numerical gates on
a GPU without a usable PyTorch GPU path. CPU PyTorch is a correctness oracle,
not a performance competitor. Do not publish CPU/GPU timing ratios or include
these runs in GPU performance, preparation-time or portability-score aggregates.

The existing collector already supports this; no new harness is needed.
Commands below use Inferena's `experiment/p3hpc-cuda-graphs` branch. Its current
engine pin is the submitted cohort's `428fc2d2`, **not PR #200**. A run now is
bring-up evidence at that pin. Repeat qualification at the next frozen pin
before describing it as coverage of the new engine.

## AMD Mendocino on rubik

From the Inferena checkout, with Rust, uv and a working RADV Vulkan driver:

```sh
git switch experiment/p3hpc-cuda-graphs
git pull --ff-only
bash scripts/setup.sh cpu .venv-p3hpc-cpu
cargo run --release --locked -p inferena-meganeura -- --list-devices
.venv-p3hpc-cpu/bin/python scripts/p3hpc.py \
  --backend cpu --gpu MENDOCINO --qualify-only --eager
cp ../inferena-results/latest.tgz ../inferena-results/rubik-mendocino-qualification.tgz
```

Reuse `.venv-p3hpc-cpu` if already installed; setup refuses to overwrite it.
The device list must show the intended hardware as available and not software
emulated. `--gpu MENDOCINO` matches a RADV name containing that substring
(including `RAPHAEL_MENDOCINO`). If the list uses a different name, use that
exact name instead. If more than one matches, stop and disambiguate; do not
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

The submitted archives already establish ten passing workload/precision
conditions, each in three processes, on RPL-U. The paper retains this evidence
in the qualification table and no longer reports the CPU-reference timings.

## Evidence and archive lifetime

Require `campaign.json` to finish with `status: complete`, `args.collect: false`,
the intended `native_device`, and ten valid qualification runs. Preserve the
archive, driver/hardware identity and exact source pin. A CPU-only wheel alone
does not prove that a GPU PyTorch path is unavailable: retain the GPU-backend
probe failure or vendor support evidence separately. Mendocino is still pending,
not a passing or unavailable platform inferred merely from this command.

Each attempt creates a new timestamped directory. `latest.tgz` is atomically
replaced on **completion or failure**, so copy it before another attempt or
platform run. A failed archive is diagnostic evidence, not a qualification pass.
The timestamped directories remain available; never run two collectors sharing
the same `latest.tgz` destination at once.
