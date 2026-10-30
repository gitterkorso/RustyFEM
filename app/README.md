# RustyFEM-appen

Grafiskt gränssnitt ovanpå motorn.

| Del | Var | Innehåll |
|---|---|---|
| App-API | `app/session` (crate `rustyfem-session`) | Session, ångra/gör om, körning med framsteg och avbrott, minnesuppskattning och lösarval, ytextraktion, resultatfält, projektfil `.rustyproj` |
| Skal | `app/src-tauri` (crate `rustyfem-app`) | Tauri 2-kommandon, binära svar för nät och fält, fildialoger, minnesbudget per plattform |
| Gränssnitt | `web/` | React + TypeScript 7, modellfönster i ren Three.js, `Engine`/`TauriEngine` |

Motorn rapporterar framsteg via `rustyfem::progress` (steg, inkrement,
iterationer). Utan krok gör rapporterna ingenting, så resultaten ändras inte.

Motorn är förberedd för appen:

- Feature `cli` (kommandoradsprogrammet och underprocesser, `rustyfem check`) och `dynlib`
  (UMAT/UEL från delade bibliotek) är på som standard och av i appen.
- `inp::read_str_with` tar en filresolver för `*INCLUDE`; projektfilen läser
  sina inkluderade filer ur filen själv (fungerar i iOS-sandlådan).
- Miljövariablerna `RUSTYFEM_*` är samlade i `rustyfem::settings::Settings`
  (`RunOptions::settings`); miljön läses bara med feature `cli`.

## Köra

```bash
cd web && npm install && npm run tauri dev
```

Release-bygge (installationspaket för den aktuella plattformen):

```bash
cd web && npm run tauri build
```

TS-typerna för JSON-kommandona genereras från Rust (tauri-specta) till
`web/src/engine/bindings.ts` när appen startar i debugläge, eller med
`target/debug/rustyfem-app --export-bindings web/src/engine/bindings.ts`.

## Tester

Alla byggkontroller (tester, motorn utan cli/dynlib, typkontroll för iOS,
macOS och Linux, gränssnittet):

```bash
sh scripts/check.sh
```

Bara app-API:t:

```bash
cargo test --release -p rustyfem-session
```

```bash
cd web && npx tsc
```

Sessionstesterna öppnar regressionsfall, kör dem via sessionen med framsteg
och avbrott, jämför med `steps::run`, testar ångra/gör om och att
projektfilen ger samma modell och resultat efter omläsning.

Självtest i det riktiga appfönstret: öppnar två modeller, löser dem med
motorn, växlar fält och mäter bildtiden; skriver `selftest.png` och
`selftest.json` till katalogen och avslutas.

```bash
target/release/rustyfem-app --selftest selftest
```

CI (`.github/workflows/app.yml`) bygger appen på Windows, macOS och Linux och
motorn med app-API:t för iOS.

## AI-assistenter (MCP)

Appen har en inbyggd MCP-server; inget extra behöver laddas ner. Knappen med
kontakten i toppraden öppnar panelen med start/stopp och instruktioner för
Claude Desktop (en knapp lägger till RustyFEM i konfigurationen), Claude Code,
Codex och andra klienter.

- stdio: `rustyfem-app --mcp` är en brygga som klienten startar. Den startar
  appen med servern påslagen om den inte redan kör.
- HTTP: `http://127.0.0.1:7456/mcp` (Streamable HTTP, JSON-svar) med
  `Authorization: Bearer TOKEN`. Bara lokala anslutningar; förfrågningar med
  främmande `Origin` avvisas. Port och token sparas i `mcp.json` i appens
  konfigurationskatalog (`%APPDATA%\se.rustyfem.app` på Windows).
- Verktygen arbetar i modellen som är öppen i appen: ändringar, lösningar och
  framsteg syns direkt och kan ångras. Verktyg: `model_info`, `new_model`,
  `open_file`, `get_script`, `set_script`, `edit`, `undo`, `redo`, `solve`,
  `results`, `field_stats`, `node_info`, `element_info`, `save_project`,
  `rusty_help`.

## Gränssnittet

- **Modellträd** (vänster): elementtyper, set, ytor, material, tvärsnitt,
  randvillkor och steg, med filter.
- **Egenskaper** (höger): redigering av materialdata, densitet, tjocklek,
  stegets procedurdata, NLGEOM, INC, randvillkor och laster (`.inp`); skript
  och verktyg (platta, kloss, balk, stång, stöd, last, tryck, tyngd) för
  `.rusty`. Allt går att ångra.
- **Resultat**: steg, frame/mod, fält och komponent (Mises beräknas),
  automatisk eller fast deformationsskala, animering av frames och moder,
  färgskalor med band, history-kurvor.
- **Modellfönster**: panorera, zooma, rotera, 2D-läge, standardvyer
  (tangenterna 0–6, F anpassar), ytor/kanter/odeformerad form/symboler,
  GPU-plockning av element eller nod (större träffyta vid pekinmatning),
  PNG-export.
- **Körpanel**: lösarval (automatiskt enligt minnesbudgeten), framsteg,
  residualkurva, avbrott och meddelanden.
- `Ctrl+K` kommandopalett, `F5` lös, `Esc` avbryt, `Ctrl+Z/Y`, `Ctrl+S`.

## Inte gjort än

- iOS-bygget (Tauri iOS-projektet, signering, TestFlight) kräver Mac och Xcode.
  Motorn och app-API:t typkontrolleras för iOS här och byggs för iOS i CI.
- Export av redigerade `.inp`-modeller (`.inp` är import; projekt sparas som `.rustyproj`).
- Markering av set i modellfönstret.
