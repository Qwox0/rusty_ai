# rusty_ai

A small Rust Library for Creating and Training Artificial Neural Networks.

## Add as a dependency

Add the `rusty_ai` crate to your project's `Cargo.toml`:

```toml
[dependencies]
rusty_ai = { git = "https://github.com/Qwox0/rusty_ai", version = "0.1.0" }
```

## `nightly` Note

This library requires the `nightly` version of Rust. To use `nightly` Rust, you can either set your toolchain globally or on per-project basis.

To set `nightly` as a default toolchain for all projects:

```bash
rustup toolchain install nightly
rustup default nightly
```

If you'd like to use `nightly` only in your project however, add [`rust-toolchain.toml`](https://rust-lang.github.io/rustup/overrides.html#the-toolchain-file) file with the following content:

```toml
[toolchain]
channel = "nightly"
```

## Examples

Run an example:

```bash
cargo run --example [<NAME>]
```

List available examples:

```bash
cargo run --example
```

## Tests

Run a test:

```bash
cargo test --test [<NAME>] -- --nocapture
```

List available tests:

```bash
cargo test --test
```
