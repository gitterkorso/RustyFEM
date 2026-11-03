# Contributing to RustyFEM

Thanks for helping out. Bug reports, small fixes, new example models and new
elements, materials or analysis types are all welcome.

## Reporting a problem

Open an [issue](https://github.com/gitterkorso/RustyFEM/issues/new/choose) and
pick the matching template. For wrong results, attach the smallest model that
shows the problem and say what you expected, ideally with a reference such as
an analytical solution.

## Building

You need [Rust](https://www.rust-lang.org/tools/install) (stable) and, for the
app, [Node.js](https://nodejs.org/) 20 or later.

```bash
cargo build --release                      # engine and command-line program
cd web && npm install && npm run tauri dev # the app
```

## Before you open a pull request

Run the checks that apply to your change:

```bash
cargo test --release -p rustyfem-session             # app core
cargo build --release -p rustyfem --lib --no-default-features  # the app's engine build
```

If you changed the user interface:

```bash
cd web && npm ci && npx tsc && npx vite build
```

`scripts/check.sh` runs the full set, but it also needs the engine test project
(`../RustyFEM tests`), which is not in this repository. Without it the script
stops at that step, so run the commands above instead. Regression cases for
engine changes are run on the maintainer's side; describe in your pull request
how you verified the results.

## Where things go

[ARCHITECTURE.md](ARCHITECTURE.md) explains how the engine is structured and
exactly how to add elements, materials, step types, `.inp` keywords,
interactions and solvers. Please read the relevant section first.

- The engine has no external dependencies. Keep it that way, and discuss it in
  an issue first if you think an exception is needed.
- Keep pull requests focused on one change.
- Match the style of the surrounding code.
- Add a line under *Unreleased* in [CHANGELOG.md](CHANGELOG.md) for changes
  users will notice.

## License

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be
dual licensed under MIT OR Apache-2.0, without any additional terms or
conditions.
