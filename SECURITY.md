# Security policy

## Supported versions

Security fixes go into the latest release.

## Reporting a vulnerability

Please do not open a public issue. Report it privately through
[GitHub's vulnerability reporting](https://github.com/gitterkorso/RustyFEM/security/advisories/new)
and include the affected version, how to reproduce the problem and what an
attacker could do with it. You will get an answer as soon as possible; this is
a small project, so response times can vary.

## Areas to keep in mind

- **MCP server in the app.** It listens on `127.0.0.1` with a bearer token. With
  "allow from the network" switched on it listens on all interfaces, so use that
  option only on networks you trust.
- **Model files.** Treat `.inp`, `.rusty` and `.rustyproj` files from others like
  any other untrusted input.
- **Custom routines.** UMAT and UEL shared libraries are native code and run with
  your permissions. Only load libraries you trust.

Wrong analysis results are not security issues. Report those as a
[bug](https://github.com/gitterkorso/RustyFEM/issues/new/choose).
