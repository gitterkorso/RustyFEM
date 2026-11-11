# Changelog

Notable changes to RustyFEM. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.1.2] - 2026-11-11

### Added

- MCP server: option to allow connections from the local network.
- iOS project and safe-area layout (not released yet).
- `*SHELL SECTION, OFFSET=` for S4, S4R, S3 and S3R.
- `*DSLOAD` surface traction TRVEC and TRSHR.
- App: beams drawn in 3D with their real cross-section and shells with their thickness and offset (toolbar toggle).

## [0.1.1] - 2026-10-30

### Added

- Installers for macOS, Linux and 32-bit Windows.
- Disclaimer about the use of results ([DISCLAIMER.md](DISCLAIMER.md)).

### Changed

- The project is translated to English.
- Licensed as open source under MIT OR Apache-2.0.

## [0.1.0] - 2026-10-30

### Added

- First release: the FEM engine, the command-line program and the desktop app,
  with installers for 64-bit Windows.

[Unreleased]: https://github.com/gitterkorso/RustyFEM/compare/v0.1.2...HEAD
[0.1.2]: https://github.com/gitterkorso/RustyFEM/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/gitterkorso/RustyFEM/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/gitterkorso/RustyFEM/releases/tag/v0.1.0
