# RustyFEM

![RustyFEM](web/public/brand/rustyfem-logo.png)

**RustyFEM är ett program för finita elementanalys (FEM), skrivet i Rust.** Det
räknar ut hur konstruktioner och komponenter deformeras, vilka spänningar som
uppstår och hur de svänger, värms upp eller går sönder – från en enkel
konsolbalk till olinjära modeller med kontakt, plasticitet och stora
deformationer.

Projektet består av två delar:

- **Beräkningsmotorn** – en fristående FEM-lösare med egna glesa
  ekvationslösare. Motorn har inga externa beroenden och kan användas som
  kommandoradsprogram, som Rust-bibliotek eller via ett C-gränssnitt.
- **Skrivbordsappen** – ett grafiskt gränssnitt där man bygger modellen, löser
  den och granskar resultaten i 3D. Appen körs på Windows, macOS och Linux;
  iPad och iPhone är förberedda.

## Varför RustyFEM?

- **Snabbt att komma igång.** En modell kan beskrivas med några rader i det
  egna skriptspråket `.rusty` – geometri, nät, material, stöd och laster –
  eller byggas direkt i appen.
- **Tydliga felmeddelanden.** I stället för kryptiska numeriska fel får man
  besked i klartext, t.ex. *"nod 5 vid (1, 0, 0) kan röra sig fritt i ux –
  lägg till stöd"*.
- **Brett funktionsstöd.** Solider, skal, balkar och stänger; linjära och
  olinjära material; statik, dynamik, egenfrekvenser, knäckning, värme,
  akustik, kontakt och explicit dynamik.
- **Egen lösarteknik.** Multifrontal direktlösare med nested dissection och
  parallellism på alla kärnor, samt iterativa och indefinita lösare för stora
  eller svåra problem.
- **Öppet för egna tillägg.** Egna material- och elementrutiner kan kopplas in
  som delade bibliotek, och kärnan är byggd för att byggas ut område för
  område.
- **Import av befintliga modeller.** Indatafiler i `.inp`-format kan öppnas
  direkt.

## Ett exempel

En konsol i plan spänning, inspänd i ena änden och belastad i den andra:

```text
# Konsol i plan spänning
material steel E=210e9 nu=0.3
plate x=0..2 y=0..0.2 t=0.01 size=0.025
fix x=0
load x=2 fy=-1000
```

```bash
rustyfem examples/konsol_platta.rusty
```

Fem rader blir en färdig analys. Programmet skriver ut största förskjutning,
högsta von Mises-spänning, summan av reaktionskrafterna och en
jämviktskontroll, och hela fältet sparas för visning. Fler exempel finns i
[`examples/`](examples).

## Kom igång

**Krav:** [Rust](https://www.rust-lang.org/tools/install) (stabil) och, för
appen, [Node.js](https://nodejs.org/) 20 eller senare.

**Kommandoradsprogrammet**

```bash
cargo build --release
./target/release/rustyfem examples/konsol_platta.rusty
./target/release/rustyfem list     # element, nyckelord, steg och lösare som stöds
```

**Skrivbordsappen**

```bash
cd web
npm install
npm run tauri dev                  # utvecklingsläge
npm run tauri build                # installationspaket för den egna plattformen
```

## Filformat

| Fil | Innehåll |
|---|---|
| `.rusty` | RustyFEMs skriptspråk för modeller: geometri, nät, material, stöd och laster |
| `.rustyproj` | Projektfil från appen: modellen, redigeringarna, enhetssystemet och resultaten i en fil |
| `.inp` | Indatafiler i det vanliga nyckelordsformatet, för import av befintliga modeller |

## Projektets struktur

| Katalog | Innehåll |
|---|---|
| [`src/`](src) | Beräkningsmotorn: element, material, analyssteg, kontakt och lösare |
| [`app/`](app) | Appens kärna (`app/session`) och skrivbordsskalet (`app/src-tauri`) |
| [`web/`](web) | Användargränssnittet (React, TypeScript och three.js) |
| [`ffi/`](ffi) | C-gränssnitt till motorns ekvationslösare |
| [`examples/`](examples) | Exempelmodeller i `.rusty` och exempel på Rust-API:t |
| [`docs/`](docs) | Teknisk referens |

Testerna och de drygt 1 500 regressionsfallen ligger i ett separat
testprojekt.

## Dokumentation

- [Teknisk referens](docs/REFERENS.md) – skriptspråket, kommandoraden, alla
  element, material och analystyper, lösarna och kända begränsningar.
- [Arkitektur](ARCHITECTURE.md) – hur motorn är uppbyggd och hur nya element,
  material och steg läggs till.
- [Appen](app/README.md) – gränssnittets delar och hur appen byggs.

## Licens

RustyFEM är öppen källkod och licensieras under valfri av

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE)), eller
- MIT-licensen ([LICENSE-MIT](LICENSE-MIT)).

Copyright © 2026 Nick Westin. Programvara från tredje part omfattas av sina
egna licenser, se [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md).

### Bidrag

Bidrag är välkomna. Om du inte uttryckligen anger något annat licensieras
varje bidrag som du avsiktligt skickar in för att ingå i RustyFEM, enligt
definitionen i Apache-2.0-licensen, under båda licenserna ovan utan
ytterligare villkor.
