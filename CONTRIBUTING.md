Rules:
- everything flows into core from the config, no external env hooks
- minimize enums, shaders, tests
- WGSL code should be in wgsl files
- data flows top to bottom: decide from the inputs, never by parsing generated text or matching names

Style:
- self-descripting code >> comment towers
- single `use` per crate, prefer 1-level references `like foo::Bar` in the code, no wildcards
- don't mutate an argument and return a result from a function at the same time
- don't rely on Rust's `match` sugar, match exact types and use `ref` in patterns
