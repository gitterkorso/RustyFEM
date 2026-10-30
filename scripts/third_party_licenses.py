# -*- coding: utf-8 -*-
# Skriver THIRD_PARTY_LICENSES.md: alla Rust-crates och npm-paket som följer
# med appen (normala beroenden, inte bygg- eller utvecklingsberoenden), med
# licens och licenstexterna från paketen själva.
#
# Körs från repots rot efter `npm ci` i web/:
#   python scripts/third_party_licenses.py
import json
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "THIRD_PARTY_LICENSES.md")
OWN = {"rustyfem", "rustyfem-app", "rustyfem-session", "rustyfem-ffi"}
LICENSE_FILE = re.compile(r"^(LICEN[CS]E|COPYING|NOTICE|UNLICENSE)([-._].*)?$", re.I)


def license_texts(directory):
    texts = []
    if not os.path.isdir(directory):
        return texts
    for name in sorted(os.listdir(directory)):
        path = os.path.join(directory, name)
        if os.path.isfile(path) and LICENSE_FILE.match(name):
            with open(path, encoding="utf-8", errors="replace") as f:
                texts.append((name, f.read().strip()))
    return texts


def rust_packages():
    tree = subprocess.run(
        ["cargo", "tree", "-p", "rustyfem-app", "-e", "normal", "--target", "all", "--prefix", "none", "--format", "{p}"],
        cwd=ROOT, capture_output=True, text=True, encoding="utf-8", check=True).stdout
    wanted = set()
    for line in tree.splitlines():
        parts = line.replace(" (*)", "").split()
        if len(parts) >= 2:
            wanted.add((parts[0], parts[1].lstrip("v")))
    meta = json.loads(subprocess.run(
        ["cargo", "metadata", "--format-version", "1"],
        cwd=ROOT, capture_output=True, text=True, encoding="utf-8", check=True).stdout)
    out = []
    for p in meta["packages"]:
        if (p["name"], p["version"]) not in wanted or p["name"] in OWN:
            continue
        lic = p.get("license") or ("se " + p["license_file"] if p.get("license_file") else "okänd")
        out.append((p["name"], p["version"], lic, p.get("repository") or "",
                    license_texts(os.path.dirname(p["manifest_path"]))))
    return sorted(out)


def npm_packages():
    web = os.path.join(ROOT, "web")
    with open(os.path.join(web, "package.json"), encoding="utf-8") as f:
        roots = list(json.load(f).get("dependencies", {}))
    seen = {}

    def walk(name, start):
        d = start
        while not os.path.isdir(os.path.join(d, "node_modules", name)):
            parent = os.path.dirname(d)
            if parent == d:
                sys.exit(f"{name} saknas i node_modules – kör `npm ci` i web/ först")
            d = parent
        pkg_dir = os.path.join(d, "node_modules", name)
        with open(os.path.join(pkg_dir, "package.json"), encoding="utf-8") as f:
            j = json.load(f)
        key = (j["name"], j["version"])
        if key in seen:
            return
        repo = j.get("repository")
        repo = repo.get("url", "") if isinstance(repo, dict) else (repo or "")
        seen[key] = (j["name"], j["version"], j.get("license", "okänd"), repo, license_texts(pkg_dir))
        for dep in j.get("dependencies", {}):
            walk(dep, pkg_dir)

    for name in roots:
        walk(name, web)
    return sorted(seen.values())


def section(title, packages):
    lines = [f"## {title}", ""]
    for name, version, lic, repo, texts in packages:
        lines.append(f"### {name} {version}")
        lines.append("")
        lines.append(f"Licens: {lic}" + (f"  \nKällkod: {repo}" if repo else ""))
        lines.append("")
        for fname, text in texts:
            lines += [f"<details><summary>{fname}</summary>", "", "```text", text, "```", "", "</details>", ""]
    return lines


def main():
    rust = rust_packages()
    npm = npm_packages()
    lines = [
        "# Tredjepartslicenser",
        "",
        "RustyFEM-appen innehåller programvara från tredje part enligt nedan.",
        "Filen genereras med `python scripts/third_party_licenses.py`; ändra den inte för hand.",
        "",
        "Crates under MPL-2.0 (cssparser, selectors m.fl.) används oförändrade; deras",
        "källkod finns på crates.io och i respektive arkiv.",
        "",
    ]
    lines += section(f"Rust ({len(rust)} crates)", rust)
    lines += section(f"JavaScript ({len(npm)} paket)", npm)
    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    missing = [f"{n} {v}" for n, v, _, _, t in rust + npm if not t]
    print(f"skrev {OUT}: {len(rust)} crates, {len(npm)} npm-paket")
    if missing:
        print(f"{len(missing)} paket utan licensfil i paketet (bara licensnamnet anges):")
        print("  " + ", ".join(missing))


if __name__ == "__main__":
    main()
