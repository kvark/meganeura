Rules:
- everything flows into core from the config, no external env hooks
- minimize enums, shaders, tests
- WGSL code should in in wgsl files

Style:
- self-descripting code >> comment towers
- single `use` per crate, prefer 1-level references `like foo::Bar` in the code, no wildcards
- don't mutate an argument and return a result from a function at the same time
- don't rely on Rust's `match` sugar, match exact types and use `ref` in patterns
