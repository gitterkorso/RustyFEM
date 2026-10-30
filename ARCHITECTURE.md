# RustyFEMs arkitektur

Det här dokumentet beskriver hur RustyFEM är uppbyggt och **exakt** hur man
lägger till element, material, steg-typer, .inp-nyckelord, interaktioner,
lösare och regressionsfall. Strukturen är gjord för att flera personer (eller
agenter) ska kunna arbeta parallellt i egna git-grenar med så få
sammanslagningskonflikter som möjligt.

## Dataflöde

```text
fil.inp ──inp::read_file──► FeModel ──steps::run──► AnalysisOutput ──to_result_set──► ResultSet
                               ▲                                                         │
fil.rusty ──script──► Model ──legacy::to_fe                     regression::compare ◄──────┘
                                                                        ▲
fall.ref (baslinje, rustyfem-res 1) ──ResultSet::parse──────────────────┘
```

- `FeModel` (`src/fem/model.rs`) är det gemensamma formatet: noder och element
  med nummer från indatafilen per instans, set, ytor, material (som rå beteenden),
  tvärsnitt, randvillkor och steps.
- `steps::run` (`src/steps/mod.rs`) kör stegen i följd med ärvt tillstånd
  (`SolutionState`), väljer steg-typ via registret och samlar utdata.
- Resultat lagras per step och frame (`src/fem/output.rs`) och skrivs som
  text i formatet `rustyfem-res 1`.

## Kataloger

| Katalog | Innehåll |
|---|---|
| `src/fem/` | **Kärnan** (delad): modell, frihetsgrader, assemblering, tillstånd, utdata, jämviktssystem |
| `src/elements/` | Elementbibliotek, **en fil per elementfamilj**. `api.rs` = gränssnittet, `common.rs` = kontinuumkärna |
| `src/materials/` | Materialmodeller. `api.rs` = gränssnittet |
| `src/steps/` | Steg-typer, en fil per typ. `newton.rs` = Newton–Raphson-drivern, `api.rs` = gränssnittet |
| `src/inp/` | .inp-läsaren. `reader.rs` = tokenisering och *INCLUDE, `keywords/` = hanterare, en fil per område |
| `src/interactions/` | Kontakt, tie, coupling, MPC, ekvationer, fjädrar, massor |
| `src/solver/` | Linjära ekvationslösare (`direct.rs` = gles LDLᵀ) |
| `src/regression/` | Regressionstester: fall, sidofiler, körning och jämförelse mot baslinjen |
| `src/model.rs`, `script.rs`, `legacy.rs`, `results.rs` | `.rusty`-språket och dess modell, översättning till `FeModel` |
| `src/sparse.rs`, `par.rs` | Gles multifrontal LDLᵀ och trådhjälp (används av `solver/direct.rs` och FFI) |
| `ffi/` | C-gränssnittet: `rustyfem_factor/rustyfem_solve/rustyfem_info/rustyfem_order/rustyfem_free` (signaturerna får inte ändras; används av OpenFEM) samt `rustyfem_factor_indef/rustyfem_factor_unsym/rustyfem_inertia` |
| `../RustyFEM tests/tests/cases/<område>/` | Regressionsfall: `<fall>.inp` + `<fall>.toml` + `<fall>.ref` (separat mapp, se nedan) |
| `src/user/` | Användarrutiner: UMAT, UEL, VUMAT, DLOAD, DISP, UAMP, USDFLD, UFIELD, UHARD, SIGINI, UEXPAN, CREEP, HETVAL, FILM, UMATHT; körkontext, register, delade bibliotek, exempel |
| `../RustyFEM tests/tests/` | Separat testcrate (`rustyfem-tests`, motorn via sökväg): `analytic.rs` (.rusty), `core.rs` (kärnan), ett testprogram per område (`solids.rs`, `shells_beams.rs`, `eigen.rs`, `nlgeom.rs`, `materials.rs`, `contact.rs`, `dynamics.rs`, `heat_analytic.rs`, `solver.rs`, `rigid_user.rs`) och `regression.rs` (alla fall mot `.ref`) |

## Regler för parallellt arbete

1. **Nya saker i nya filer.** Varje område har sin egen fil (eller flera) i
   rätt katalog. Platshållarna nedan finns redan och kompilerar.
2. **Delade registreringslistor ändras med EN rad per registrering, i
   alfabetisk ordning**, utan att flytta eller formatera om andra rader:

   | Lista | Fil | Rad |
   |---|---|---|
   | Elementfamiljer | `src/elements/mod.rs` | `    min_familj,` i `element_modules!` |
   | Materialmoduler | `src/materials/mod.rs` | `    mitt_material,` i `material_modules!` |
   | Steg-typer | `src/steps/mod.rs` | `    min_steg_typ,` i `step_modules!` |
   | Nyckelordsmoduler | `src/inp/keywords/mod.rs` | `    mina_nyckelord,` i `keyword_modules!` |
   | Interaktioner | `src/interactions/mod.rs` | `    min_interaktion,` i `interaction_modules!` |
   | Lösare | `src/solver/mod.rs` | `    min_losare,` i `solver_modules!` |

   Makrot skapar både `pub mod` och registreringen, så en rad räcker.
   Inom en modul listas typerna i modulens egen `TYPES`/`FACTORIES`/
   `STEP_TYPES`/`KEYWORDS`/`SOLVERS`-lista (som bara den modulen äger).
3. **Kärnfiler** (`src/fem/*`, alla `api.rs`, `elements/common.rs`,
   `steps/newton.rs`, `inp/reader.rs`, `src/regression/*`) ändras bara när ett
   gränssnitt verkligen måste utökas. Utöka då *additivt*: nya trait-metoder
   får alltid en standardimplementation (normalt `not_implemented(...)`), nya
   fält i structar får `Default`. Ändra aldrig befintliga signaturer.
4. **Befintliga element** (`continuum2d.rs`, `continuum3d.rs`, `truss.rs`,
   `beam_eb.rs`) har redan stubbar för förmågor som andra områden ska fylla i,
   markerade med en kommentar `// område <namn>: ...`. Ersätt bara den
   stubbens rad(er), och lägg själva implementationen i en egen fil (t.ex.
   `elements/large_strain.rs` för NLGEOM-kinematik eller `elements/geometric.rs`
   för geometrisk styvhet) som stubben anropar.
5. **Tester** i en egen fil per område: `tests/<område>.rs` i mappen
   `RustyFEM tests`. Regressionsfall i `tests/cases/<område>/` där. Kör
   `cargo test --release -j 2` i den mappen före commit.
6. Inga externa beroenden. Ingen omformatering av andras kod.

## Konventioner

- **Frihetsgrader** följer `.inp`-formatets numrering (`src/fem/dof.rs`): 1–3 förskjutning,
  4–6 rotation, 7 välvning, 11 temperatur NT11. Varje nod har 8 platser
  (slots 0–7) i alla nodvektorer (`NodeVec = [f64; 8]`). Porvattentrycket
  POR (frihetsgrad 8, *SOILS) delar slot 7 med NT11.
  (slots 0–7) i alla nodvektorer (`NodeVec = [f64; 8]`). Frihetsgrad 9
  (elektrisk potential EPOT) delar slot 6 med välvningen; masstransportens
  normerade koncentration (NNC11) är frihetsgrad 11.
- **Spänning/töjning** i komponentordningen 11, 22, 33, 12, 13, 23 med
  ingenjörsskjuvtöjning (se `src/materials/api.rs`). Utdata använder de
  vanliga namnen (S11, S22, S33, S12, ...). Plan spänning skrivs med S33 = 0
  och E33 = 0.
- **Namn** (set, ytor, material, instanser) lagras med versaler. Noder och element
  identifieras med (instans, nummer). En modell utan *PART har instansen
  `MODELL`, noder direkt i *ASSEMBLY instansen `ASSEMBLY`.
  Set i instanser heter `INSTANS.SET`.
- **Matriser** från element är täta, radvisa n×n `Vec<f64>`; elementvektorn är
  nodvis (`node_dofs(0)`, sedan `node_dofs(1)`, ...).
- **Tid**: steg-typer arbetar med stegtidsandel t ∈ [0, 1]; `Controls::period`
  ger stegets längd. `SolutionState::time` är total tid.
- **Laster och randvillkor** ärvs mellan steps enligt `.inp`-formatets regler (OP=MOD/NEW,
  ramp från föregående värde, släppta randvillkor ersätts av rampad reaktion,
  perturbationssteg utgår från bastillståndet). Se `src/fem/state.rs`.
- **Fel** är `FemError` med klartext på svenska som anger fil/rad eller element.

## Lägga till ett element

1. Skapa eller öppna din familjs fil, t.ex. `src/elements/solid_higher.rs`.
2. Definiera en struct och implementera `ElementType` (`src/elements/api.rs`):

   ```rust
   use super::api::{ElemCtx, ElemResponse, ElementType, Family, IpField, MassKind};
   use super::common::{self, Bbar, Ip};
   use crate::fem::dof::{Dof, U1, U2, U3};
   use crate::materials::StressMode;
   use crate::model::Result;

   pub struct Hex20;
   pub static C3D20: Hex20 = Hex20;
   pub static TYPES: &[&dyn ElementType] = &[&C3D20];

   impl ElementType for Hex20 {
       fn name(&self) -> &'static str { "C3D20" }
       fn family(&self) -> Family { Family::Continuum3D }
       fn dim(&self) -> usize { 3 }
       fn n_nodes(&self) -> usize { 20 }
       fn node_dofs(&self, _: usize) -> &'static [Dof] { &[U1, U2, U3] }
       fn n_ips(&self) -> usize { 27 }
       fn faces(&self) -> &'static [&'static [usize]] { &[/* ytor S1..S6 */] }
       fn init_state(&self, ctx: &ElemCtx) -> Result<Vec<f64>> {
           Ok(common::init_state(27, StressMode::ThreeD, ctx.material(self.name())?))
       }
       fn stiffness(&self, ctx: &ElemCtx) -> Result<Vec<f64>> { /* ips → common::stiffness */ }
       fn internal_force(&self, ctx: &ElemCtx, u: &[f64], du: &[f64], so: &[f64], sn: &mut Vec<f64>, tangent: bool)
           -> Result<ElemResponse> { /* ips → common::response */ }
       fn output(&self, ctx: &ElemCtx, u: &[f64], state: &[f64]) -> Result<Vec<IpField>> { /* common::output */ }
       // mass, geometric_stiffness, distributed_load, volume_weights efter behov
   }
   ```

   Metoder som inte implementeras ger automatiskt felet
   "... är inte implementerat för C3D20". Standardversionen av
   `internal_force` använder `stiffness` men bara för linjära material utan
   NLGEOM.
3. Integrationspunkterna ska numreras enligt `.inp`-formatets konvention och
   ytorna (`faces`) ha standardnumreringen S1..S6, annars blir S och *DLOAD Pn
   fel.
4. Om filen är ny: lägg till EN rad i `element_modules!` i `src/elements/mod.rs`.
   (Alla befintliga familjer, se tabellen "Områden och deras filer", är
   redan registrerade.)
5. Elementet kan nu användas i `*ELEMENT, TYPE=C3D20`. Lägg till ett
   regressionsfall (se nedan).

Elementets tillstånd (`state`) disponerar elementet själv. Kontinuumkärnan
lägger per integrationspunkt spänningen (ncomp värden) följt av materialets
tillståndsvariabler (`common::ip_stride`).

## Lägga till ett material

1. Skriv modellen i din fil, t.ex. `src/materials/plasticity.rs`, och
   implementera `MaterialModel` (`src/materials/api.rs`): `update` får total
   töjning, töjningsinkrement, spänning och tillstånd vid inkrementets början
   och returnerar spänning och konsistent tangent. `n_state`/`init_state` anger
   tillståndsvariablerna, `output_names`/`output_values` t.ex. `PEEQ`.
2. Registrera en fabrik i modulens `FACTORIES`:

   ```rust
   pub static FACTORIES: &[MaterialFactory] = &[MaterialFactory {
       name: "von Mises-plasticitet",
       consumes: &["ELASTIC", "PLASTIC"],
       priority: 10,               // högre än isotrop elasticitet (0)
       build: Some(build),         // fn(&MaterialDef) -> Result<Option<Arc<dyn MaterialModel>>>
   }];
   ```

   `build` returnerar `Ok(None)` om materialdefinitionen inte passar.
   Alla beteenden i ett material måste förbrukas (`consumes`) av den valda
   fabriken, annars blir det fel. Beteenden som bara läses av andra delar (t.ex.
   `CONDUCTIVITY` för värmeledning) deklareras med en passiv post
   `build: None, consumes: &["CONDUCTIVITY", ...]`.
3. Registrera nyckelordet för beteendet i en egen nyckelordsfil (se nedan) med
   `Kind::Behavior` och hanteraren `crate::inp::keywords::material::behavior`.
4. Ny modulfil? EN rad i `material_modules!` i `src/materials/mod.rs`.
5. Tidsberoende material (krypning, viskoelasticitet) returnerar
   `Some(&TimeDependence)` från `time_dependence()` (`materials/time_dependence.rs`):
   `steps::run` sätter där läget per steg (aktivt i *VISCO/*DYNAMIC, av i
   *STATIC) och *VISCO läser felmåttet för CETOL och stabilitetsgränsen.
   Material som omsluter andra (t.ex. `thermal.rs`) vidarebefordrar metoden.

Kontinuumelementen anropar redan `update` i varje integrationspunkt, så ett
nytt material fungerar direkt med CPS/CPE/C3D8/T2D2/T3D2. *STATIC väljer
automatiskt Newton–Raphson när något material inte är linjärt (`is_linear`).

Mjuknande material och osymmetriska tangenter:

- `MaterialModel::update_with_length(input, L, ...)` får elementets
  karakteristiska längd L (`elements::common::char_length`: kubikroten ur
  volymen, kvadratroten ur arean i planet/r–z-planet, halva längden för andra
  ordningens element). `common::response` anropar den; standard
  är `update`. Används för brottenergi/sprickvidd (Hillerborg) i
  `concrete_damaged.rs` och `ductile_damage.rs`.
- Materialtangenten får vara osymmetrisk. Elementmatriserna assembleras med
  `SystemMatrix::add_unsym` (hela matrisen) när lösaren är osymmetrisk och
  som symmetrisk del ½(K + Kᵀ) annars (`fem/assembly.rs`, `steps/nlgeom.rs`).
- `MaterialModel::requires_unsymmetric()` (standard `false`) gör att
  `steps::run` väljer `unsymmetric` för stegen när användaren inte valt
  lösare eller angett `UNSYMM=` (`materials::principal::auto_unsymmetric`),
  som för *CONCRETE DAMAGED PLASTICITY.
- `materials/principal.rs` har gemensamma byggstenar för returalgoritmer i
  huvudspänningsrummet (invarianter p, q, θ, skalär rotsökning, numerisk
  tangent med fram- eller centraldifferens).

## Lägga till en steg-typ

1. Skriv typen i din fil, t.ex. `src/steps/frequency.rs`, och implementera
   `StepType` (`src/steps/api.rs`):
   - `claim(step, nlgeom)` returnerar en prioritet om typen kan köra steget
     (t.ex. `step.procedure_name() == "FREQUENCY"`). Högst prioritet vinner,
     så `nlgeom.rs` kan ta *STATIC med NLGEOM=YES utan att röra `static_linear.rs`.
   - `run(ctx)` kör steget. `ctx.loads` innehåller randvillkor och laster med
     värden vid stegets början och slut, `ctx.state` tillståndet före steget
     (skriv tillståndet efter steget; vid PERTURBATION återställs det
     automatiskt), `ctx.out.frames` tar emot `FrameData`.
2. Registrera i modulens `STEP_TYPES`. Ny modulfil? EN rad i `step_modules!`.
3. Procedurnyckelordet (*FREQUENCY, *BUCKLE, *DYNAMIC, *HEAT TRANSFER, ...)
   registreras i en egen nyckelordsfil med `Kind::Step` och hanteraren
   `crate::inp::procedure`.
4. Inkrementella steg använder Newton-drivern: bygg ett `fem::system::StaticSystem`
   (eller ett eget `NonlinearSystem`) och anropa
   `newton::run(&mut sys, &Controls::from_procedure(..), &mut |sys, t| { ... })`.
   Callbacken anropas efter varje godkänt inkrement – spara frames där.
   Drivern sköter automatisk inkrementering med halvering, ökning och
   konvergenskriterier per fält (förskjutning, rotation, temperatur).
5. Egenvärdessteg skriver en frame per mod med `value` = egenvärde/frekvens och
   `mode` = modnummer och framenummer i följd, så att
   `FRAMEVALUE` och modformer kan jämföras.

## Lägga till ett .inp-nyckelord

1. Skapa en fil i `src/inp/keywords/`, t.ex. `contact.rs`:

   ```rust
   use crate::fem::model::RawKeyword;
   use crate::inp::{store_model, InpCtx, Keyword, Kind};
   use crate::model::Result;

   pub static KEYWORDS: &[Keyword] = &[
       Keyword { name: "CONTACT PAIR", kind: Kind::Step, handler: contact_pair },
       Keyword { name: "SURFACE INTERACTION", kind: Kind::Model, handler: store_model },
   ];

   fn contact_pair(ctx: &mut InpCtx, kw: &RawKeyword) -> Result<()> { /* ... */ Ok(()) }
   ```

   `RawKeyword` har parametrar (`param`, `param_upper`, `flag`, `has`) och
   datarader (`data`, `records()` som slår ihop rader som slutar med komma,
   `floats`). `InpCtx` ger `mesh()` (parten som definieras eller modellen),
   `step_mut`, `material_mut`, `node_ref`/`elem_ref` (nummer, `INST.nummer`
   eller set) och `warn`.
2. Färdiga hanterare: `store_model` (spara i `FeModel::extra`), `store_step`
   (spara i stegets `extra`), `procedure` (stegets procedur),
   `keywords::material::behavior` (materialbeteende), `ignore`.
3. `Kind` anger var nyckelordet får stå: `Model`, `Step`, `Both` eller
   `Behavior` (direkt efter *MATERIAL).
4. Ny fil: EN rad i `keyword_modules!` i `src/inp/keywords/mod.rs`. Ett
   nyckelord får bara registreras en gång (testas i `tests/core.rs`).
5. Okända nyckelord ger en varning. Nyckelord i `src/inp/unsupported.rs`
   påverkar resultatet och ger fel tills någon registrerar en hanterare – en
   registrerad hanterare går alltid före listan, så listan behöver inte ändras.

## Lägga till en interaktion eller ett villkor

Spara nyckelordet med `store_model` (se ovan) och skriv en
`InteractionFactory` i `src/interactions/<område>.rs` som bygger
`Box<dyn Interaction>` från `FeModel::extra`. En interaktion kan:

- lägga till aktiva frihetsgrader (`active_dofs`) och kopplingar för
  matrisstrukturen (`connectivity`),
- bidra med inre kraft och tangent (`contribute`, via `TangentSink`) – straff,
  kontakt, fjädrar,
- ge linjära villkor (`linear_constraints`) – dessa elimineras i
  `src/fem/mpc.rs` med master–slav-transformation,
- ge villkor som beror på läget (`constraints_depend_on_u` = true och
  `linear_constraints_at(u, nlgeom)`): NLGEOM-steget linjäriserar dem då om
  före varje utvärdering (`MpcMap::update_at`), med samma termer och slavar
  som `linear_constraints`, koefficienterna som derivator (rumsliga spinn för
  rotationer) och rhs vald så att slavens läge blir exakt (används av
  `*RIGID BODY` för ändliga rotationer),
- ge massbidrag (`mass`).

NLGEOM-steget (`steps/nlgeom.rs`) sätter villkorens slavar och kondenserar
krafterna (`MpcMap::update_at`, `condense`) precis som de linjära stegen;
implicit `*DYNAMIC` med NLGEOM linjäriserar också om villkoren i varje
iteration (`steps/dynamic.rs`), explicit dynamik eliminerar inga villkor.

Ytor från *SURFACE finns i `FeModel::surfaces` (elementytor med
beteckning S1..S6/SPOS/SNEG, eller noder). SPOS/SNEG gäller element med en
enda yta (stela element, skal, membran): normalen följer nodordningen
(högerhandsregeln i 3D, nod 1 → nod 2 vriden 90° moturs i 2D) respektive
motsatt (`interactions/geom.rs::facets`).

## Stela kroppar

Stela element (R3D4, R3D3, R2D2, RAX2 i `elements/rigid.rs`) har noder och
ytor men ingen styvhet eller massa. `*RIGID BODY, REF NODE=, ELSET=,
PIN NSET=, TIE NSET=` (`interactions/rigid_body.rs`) ger linjära villkor som
knyter nodernas translationer (och rotationer för ELSET/TIE NSET) till
referensnoden, vid NLGEOM med ändliga rotationer via `linear_constraints_at`.
En kontaktyta på stela element (`*SURFACE ... SPOS/SNEG`) blir en stel
masteryta: `interactions/contact.rs` tar straffstyvheten från slavsidan
(`elements::rigid::is_rigid`, `HARD_FACTOR_RIGID`).

Analytiska stela ytor (`*SURFACE, TYPE=SEGMENTS/CYLINDER/REVOLUTION`):
`inp/keywords/analytical.rs` (anropas från `surface.rs`) registrerar ytans namn
i `FeModel::surfaces` (typen i `kind`) och sparar nyckelordet i
`FeModel::extra`; `interactions/analytical.rs` bygger kontakten nod mot yta för
`*CONTACT PAIR` med en analytisk masteryta (`contact.rs` hoppar över sådana
par) och hittar referensnoden via `*RIGID BODY, ANALYTICAL SURFACE=`.
Kontaktegenskaperna läses med `contact::PairProps`. Nyckelord som står i en
`*PART` (`*RIGID BODY`, analytiska ytor) markeras med `_PART`; `*INSTANCE`
lägger till `_INSTANCE` (namnet) och `_PLACE` (placeringen,
`analytical::encode_placement`), och namnen tolkas med instansprefix
(`analytical::scoped`). `rigid_body::reposition` (anropas när filen är läst)
flyttar referensnoden vid POSITION=CENTER OF MASS.

## Användarrutiner (UMAT och UEL)

`src/user/` har de etablerade gränssnitten för användarmaterial och användarelement:

| Fil | Innehåll |
|---|---|
| `user/umat.rs` | `trait Umat { fn umat(&self, a: &mut UmatArgs) -> Result<()> }`, `UmatArgs` med standardargumenten (STRESS, STATEV, DDSDDE kolumnvis, SSE/SPD/SCD, STRAN, DSTRAN, TIME, DTIME, TEMP, DTEMP, CMNAME, NDI, NSHR, NTENS, NSTATV, PROPS, DROT, PNEWDT, DFGRD0, DFGRD1, JSTEP, ...), register per materialnamnsprefix (`register`, `find`), `UmatFn`/`DllUmat` för bibliotek |
| `user/uel.rs` | `trait Uel { fn uel(&self, a: &mut UelArgs) -> Result<()> }`, `UelArgs` (RHS, AMATRX kolumnvis, SVARS, ENERGY, PROPS, JPROPS, COORDS, U, DU, V, A, JTYPE, TIME, LFLAGS, PNEWDT, ...), register per typnummer (`register(Some(n))`, `None` = alla), `UelFn`/`DllUel` |
| `user/dynlib.rs` | LoadLibraryW/GetProcAddress respektive dlopen/dlsym direkt via FFI |
| `user/mod.rs` | `load_library(path)` (symbolerna `umat_`, `umat`, `UMAT`, `uel_`, ...), `RUSTYFEM_USER` |
| `user/samples.rs` | exempel: UMAT `EXEMPEL_ELASTISK` och `EXEMPEL_MISES`, UEL U9001 (stång) och U9002 (CPS4-fyrhörning) |
| `materials/user_material.rs` | *USER MATERIAL + *DEPVAR → `MaterialModel` som anropar UMAT:en |
| `user/vumat.rs`, `materials/user_vumat.rs` | VUMAT för *DYNAMIC, EXPLICIT: `trait Vumat { fn vumat(&self, a: &mut VumatArgs) -> Result<()>; fn elastic_moduli(..) }`, standardargumenten för explicit dynamik (STRAININC med tensorskjuvning, STRESSOLD/NEW, STATEOLD/NEW, ENERINTERN/INELAS, DEFGRAD, STRETCH, RELSPININC, DENSITY), en punkt per anrop, register per materialnamnsprefix; en registrerad VUMAT går före UMAT:en. Exempel `EXEMPEL_VELASTISK` och `EXEMPEL_VMISES` |
| `elements/user_element.rs` | *USER ELEMENT + *UEL PROPERTY → `ElementType` som anropar UEL:en (skapas när *ELEMENT, TYPE=Un läses, se `inp/keywords/element.rs`) |
| `interactions/user_element.rs` | gör modeller med användarelement olinjära så att *STATIC använder Newton |
| `inp/keywords/user.rs` | nyckelorden |

Semantiken är den etablerade: UMAT:en får spänning och tillstånd vid inkrementets
början, töjning vid början och töjningsinkrement (ingenjörsskjuvtöjning,
komponentordning 11, 22, 33, 12, 13, 23) och returnerar spänning, tillstånd
och Jacobian ∂Δσ/∂Δε; UEL:en returnerar RHS = −F_int och AMATRX = ∂F_int/∂u
med frihetsgraderna nodvis i stigande nummer. Båda matriserna symmetriseras.
Ett delat bibliotek anropas med Fortrans konvention (alla argument som
pekare, `int32` för heltal, CMNAME som 80 tecken med dold längd sist) och
ett anrop i taget. Elementgränssnittet (UEL) saknar stegtid (TIME(1) =
total tid, PERIOD = 0) och fördelade laster.

### Körkontext (`user/context.rs`)

Material- och elementgränssnitten skickar inte integrationspunktens läge,
element- och stegnummer eller fältvariabler. De finns i stället i
`user::context`:

- `Run` (en per analys, skapas av `steps::run` med `begin_analysis`, förs
  vidare till arbetstrådarna i `par::map`): KSTEP, KINC (`newton::run`),
  stegets starttid och längd, procedurkod, fälthistorien
  (`materials::field_vars::FieldHistory`) och begärt PNEWDT (minsta värdet i
  inkrementet; `newton::run` gör om inkrementet med PNEWDT · Δt).
- `Point` (per tråd): elementet (`Prepared::ctx` → `set_element`, med
  nodernas fältvärden) och integrationspunkten (`set_ip` från
  `elements::common::response` och `kinematics::continuum`: NPT, COORDS,
  CELENT, vikt). Fälten i punkten är elementets nodmedelvärde (som
  temperaturen).
- `record_ips(f)` spelar in integrationspunkterna (läge, vikt) medan ett
  element räknar sin inre kraft (SIGINI, energierna i `.dat`).

Nya element som vill ge användarrutiner rätt NPT/COORDS anropar `set_ip`
per integrationspunkt (annars räknas NPT som anropsnummer och COORDS är
elementets mittpunkt).

### Fler användarrutiner (`user/routines.rs`)

| Rutin | Indata | Kopplas in i |
|---|---|---|
| DLOAD | `*DLOAD` PnNU (lasttyp `PnNU@SET`) | `user/loads.rs::dload`, anropas från `fem/assembly.rs::external` och `kinematics/pressure.rs` |
| DISP | `*BOUNDARY, USER` (`_USER BOUNDARY` i stegets `extra`) | `StepLoads::user_bc`, `bc_at` (`fem/state.rs`), värdet vid stegets slut ärvs (`steps::run`) |
| UAMP | `*AMPLITUDE, DEFINITION=USER` | `Def::User` i `steps/dynamic_amplitude.rs` |
| USDFLD | `*USER DEFINED FIELD` | `materials/field_dependent.rs` |
| UFIELD | `*FIELD, USER` | `materials/field_vars.rs` |
| UHARD, UEXPAN | `*PLASTIC, HARDENING=USER`, `*EXPANSION, USER` | `materials/user_routines.rs` (`claim` före registret) |
| CREEP | `*CREEP, LAW=USER` | `Law::User` i `materials/creep.rs` |
| SIGINI | `*INITIAL CONDITIONS, TYPE=STRESS, USER` | `materials/initial_state.rs` |
| HETVAL, FILM, UMATHT | `*HEAT GENERATION`, `*FILM` FnNU, `*USER MATERIAL, TYPE=THERMAL` | `steps/heat.rs` via `user/loads.rs` |

Varje rutin är en trait med standardargumenten (`DloadArgs`, ...) och ett
register per namnprefix (`register_dload(prefix, ...)`; nyckeln är
elementsetets, nodsetets, amplitudens eller materialets namn). Exempel i
`user/samples2.rs` (namn `EXEMPEL_*`), delade bibliotek i `user/dynlib2.rs`
(`dload_`, `disp_`, `uhard_`, `sigini_`, `ufield_`, `uexpan_`, `hetval_`,
`film_`, `creep_`, `umatht_`). Fältberoende materialdata (DEPENDENCIES=) och
USDFLD löses i `materials/field_dependent.rs`, som `materials::build` frågar
först. `*PARAMETER` expanderas i `inp/parameters.rs` innan nyckelorden tolkas
(`inp::dispatch`). `*NODE PRINT`, `*EL PRINT` och `*ENERGY PRINT` skrivs till
`jobb.dat` av `fem/dat.rs` (`rustyfem run`). Fall i `tests/cases/user2/`,
tester i `tests/user2.rs`.

Egen UMAT i Rust:

```rust
use rustyfem::user::umat::{self, Umat, UmatArgs};
struct MinUmat;
impl Umat for MinUmat {
    fn umat(&self, a: &mut UmatArgs) -> rustyfem::Result<()> { /* som i Fortran */ Ok(()) }
}
umat::register("MITT_", std::sync::Arc::new(MinUmat)); // material MITT_*
```

## Lägga till en lösare

Implementera `LinearSolver` och `SystemMatrix` (`src/solver/api.rs`) i
`src/solver/<namn>.rs`, lägg den i `SOLVERS` och (för ny fil) EN rad i
`solver_modules!`. Välj med `RUSTYFEM_SOLVER=<namn>` eller `RunOptions::solver`.
Lösaren väljer ekvationsordningen (`first_dof` per block) och får den undre
triangeln via `add`.

### Registrerade lösare

| Namn | Fil | Matris | Metod |
|---|---|---|---|
| `direct` (standard) | `solver/direct.rs`, `sparse.rs` | symmetrisk, positivt definit | multifrontal LDLᵀ, fel vid icke-positiv pivot (mekanism) |
| `indefinite` | `solver/indefinite.rs`, `sparse/indef.rs` | symmetrisk, indefinit | multifrontal LDLᵀ med 1×1/2×2-pivoter (Bunch–Kaufman-tröskel u = 0,01 inom fronten), fördröjda pivoter till föräldern, tröghet |
| `unsymmetric` | `solver/indefinite.rs`, `sparse/lu.rs` | osymmetrisk, symmetriskt mönster | multifrontal LU, diagonal tröskelpivot med fördröjning, partiell pivotering i roten |
| `iterative` | `solver/iterative.rs`, `iterative/amg.rs` | symmetrisk, positivt definit | PCG med algebraisk multigrid (utjämnad aggregering, stelkroppsmoder som nollrum, symmetrisk Gauss–Seidel) |
| `iterative-ic0`, `iterative-jacobi` | `solver/iterative.rs`, `iterative/csr.rs` | symmetrisk, positivt definit | PCG med IC(0) (diagonalskift vid sammanbrott) respektive Jacobi |

Val per steg (`solver::iterative::for_step`, anropas i `steps::run`):
`*STEP, SOLVER=ITERATIVE` (`*STATIC, SOLVER=ITERATIVE` godtas
också) ger `iterative`, `*STEP, UNSYMM=YES` ger `unsymmetric`. En lösare som
valts med `--solver`/`RUSTYFEM_SOLVER`/`RunOptions::solver` går alltid före.
`tests/solver.rs` kör alla fall i `tests/cases/solver/` med varje lösare mot
baslinjen.

Osymmetriska bidrag: `SystemMatrix::add_unsym(r, c, v)` (godtyckliga r, c).
`add(r, c, v)` betyder fortfarande ett symmetriskt bidrag till både (r, c) och
(c, r). För symmetriska lösare behåller standardimplementationen av
`add_unsym` bara den undre triangeln; `is_symmetric()` säger om lösaren lagrar
hela matrisen. Kontakt med friktion och följelaster ska därför välja
`unsymmetric` (t.ex. via `*STEP, UNSYMM=YES`) och assemblera med `add_unsym`.

Tröghet: `SystemMatrix::inertia()` ger (negativa, noll, positiva) pivoter efter
`factor` för `indefinite` (annars `None`). Har matrisen negativa pivoter
skriver `indefinite` varningen "systemmatrisen har N negativa egenvärden"
; `RUSTYFEM_QUIET=1` stänger av den.

Miljövariabler: `RUSTYFEM_ITER_TOL` (relativ residual för PCG, standard 1e-10),
`RUSTYFEM_TIMING` (tider, PCG-iterationer, AMG-nivåer, fördröjda pivoter).
Konvergerar inte PCG löses systemet direkt (med varning).

### C-gränssnittet (ffi/)

De ursprungliga funktionerna `rustyfem_factor`, `rustyfem_solve`, `rustyfem_info`,
`rustyfem_order` och `rustyfem_free` är oförändrade. Nya funktioner (samma
indata: n frihetsgrader, nn noder, `loc`, `coords`, CSC `indptr`/`indices`/`data`
i anroparens numrering):

```c
/* Symmetrisk indefinit LDLᵀ. inertia_out[3] = {negativa, noll, positiva} (får vara NULL).
   NULL vid fel: *err_dof = singulär frihetsgrad, eller −1 vid ogiltig indata. */
Handle *rustyfem_factor_indef(size_t n, size_t nn, const int64_t *loc, const double *coords,
                             const int64_t *indptr, const int64_t *indices, const double *data,
                             int64_t *inertia_out, int64_t *err_dof);
/* Osymmetrisk LU; CSC med båda trianglarna. */
Handle *rustyfem_factor_unsym(size_t n, size_t nn, const int64_t *loc, const double *coords,
                             const int64_t *indptr, const int64_t *indices, const double *data,
                             int64_t *err_dof);
/* Tröghet för ett handtag; 0 = ok, −1 = finns inte (LU). rustyfem_factor ger {0, 0, n}. */
int32_t rustyfem_inertia(const Handle *h, int64_t *out3);
```

Handtagen löses med `rustyfem_solve` och frigörs med `rustyfem_free` som förut,
så en anropare (t.ex. OpenFEM) kan pröva `rustyfem_factor` och vid fel
(icke-positiv pivot) gå över till `rustyfem_factor_indef` i stället för en
extern lösare, och läsa antalet negativa egenvärden med `rustyfem_inertia`.

## Lägga till ett regressionsfall

Regressionsmodulen (`src/regression/`) kör ett fall i RustyFEM och jämför
resultatet med en baslinje som RustyFEM själv har skrivit:

| Fil | Innehåll |
|---|---|
| `regression/mod.rs` | `run_rustyfem`/`run_rustyfem_spec` (kör fallet, även med tidigare jobb för submodell och omstart), `check_case` (jämför eller skriver baslinjen) |
| `regression/case.rs` | `CaseSpec` och sidofilens TOML-delmängd, `discover` (alla fall under en katalog), `spec_path`/`ref_path` |
| `regression/compare.rs` | `compare` per steg, frame och storhet, `CaseReport` med text och största fel |

1. Skriv `tests/cases/<område>/<fall>.inp`. Inkluderade filer får vilken
   ändelse som helst men **ingen** `.toml` (bara .inp med sidofil räknas som
   fall).
2. Skriv sidofilen `tests/cases/<område>/<fall>.toml`:

   ```toml
   description = "Vad fallet testar"
   tol = 1e-4                  # största relativa fel, normerat mot största |värde| i fältet
   fields = ["U", "RF", "S"]   # nodfält (U, UR, RF, RM, NT11, ...), elementfält (S, E, PEEQ, ...),
                               # FRAMEVALUE (stegtid/egenvärde), H:NAMN (history, sista värdet)
   frames = "last"             # eller "all" (frames jämförs då med samma framenummer)
   tol.S = 1e-3                # tolerans per fält (valfritt)
   atol.RF = 1e-8              # absolut tolerans nära noll (valfritt)
   sign_free = ["U"]           # tecknet får skifta per frame (modformer)
   steps = [2, 3]              # jämför bara dessa steg (valfritt)
   global_inp = "global.inp"   # submodell: globalmodellen körs först (--global)
   old_inp = "gammal.inp"      # omstart/*IMPORT: den första analysen körs först (--oldjob)
   floor_elem = 1e-3           # elementfält normeras minst mot floor_elem · största
                               # |värde| i fältet i hela fallet (t.ex. skada
                               # och plastiska töjningar ≈ 0 före initiering)
   ```

3. Kör `rustyfem baseline tests/cases/<område>/<fall>.inp`. Kommandot kör
   fallet och skriver resultatet till `<fall>.ref` (textformatet
   `rustyfem-res 1`). Granska resultatet innan baslinjen checkas in.
4. Checka in `.inp`, `.toml` och `.ref`. `cargo test` (tests/regression.rs)
   jämför sedan alla fall mot sina baslinjer.
5. `rustyfem check <fall.inp ...>` jämför enskilda fall och skriver en
   rapport per storhet; `rustyfem check --all [katalog]` kör alla fall och
   skriver en tabell. Ändrar en ändring resultaten avsiktligt skrivs
   baslinjerna om med `rustyfem baseline <fall.inp ...|--all [katalog]>`.

Jämförelsens regler (`src/regression/compare.rs`): nodfält styrs av
baslinjens noder (saknas en nod i resultatet räknas värdet som 0); elementfält
styrs av resultatets punkter (varje punkt måste finnas i baslinjen;
baslinjepunkter utan motsvarighet redovisas men ger inte fel); ett fält som
saknas i båda redovisas som not. Komplexa resultat (*STEADY STATE DYNAMICS)
har imaginärdelen i fältet `FÄLT_IM` (U_IM, S_IM, ...). Begär utdata i
.inp-filen (`*OUTPUT, FIELD` / `*NODE OUTPUT` / `*ELEMENT OUTPUT`) om
standardutdata inte räcker. Fall med tidigare jobb körs i en temporär
arbetskatalog som tas bort efteråt.

## Områden och deras filer

Alla områden är sammanslagna i `main` (2026-10-23). Filerna nedan är de
riktiga implementationerna; inga platshållare finns kvar.

| Område | Filer |
|---|---|
| Kontinuum, första ordningen | `elements/continuum2d.rs`, `continuum3d.rs`, `common.rs` (kärna, B-bar/F-bar, NLGEOM) |
| Högre ordningens/reducerade/inkompatibla solider | `elements/solid_higher.rs`, `solid_higher_shapes.rs`, `solid_higher_tech.rs` (R/I-teknik), `solid_higher_gps.rs` (CPEG), `solid_higher_nlgeom.rs` (NLGEOM för motorn, även CAX), `solid_higher_incompat.rs` (NLGEOM för inkompatibla moder i inkrementell form; enelementsprov i `tests/cases/accuracy2/`, tester i `tests/accuracy2.rs`), `solid_higher_gps_nlgeom.rs` (NLGEOM för CPEG3/6/8/8R, inkluderas med `#[path]` från `solid_higher_gps.rs`); volymtermen i B-bar-tangenten för CPE4/C3D8 vid NLGEOM i `kinematics/bbar_tangent.rs` |
| Axisymmetri | `elements/axisym.rs` |
| Skal och membran | `elements/shell.rs` (S4/S4R, M3D4/M3D3), `shell_tri.rs` (S3/S3R), `inp/keywords/shells_beams.rs` |
| Balkar | `elements/beam_eb.rs` (B23/B33), `beam_mass.rs`, `beam_timoshenko.rs`, `beam_timoshenko_section.rs` (B21/B22/B31/B32, tvärsnitt) |
| Stänger | `elements/truss.rs` |
| Fjädrar, massor, dämpare | `elements/point.rs`, `interactions/springs.rs` |
| Geometrisk styvhet (K_σ) | `elements/geometric.rs` |
| Egenfrekvenser/knäckning | `steps/frequency.rs`, `steps/buckle.rs`, `solver/eigen.rs`, `inp/keywords/eigen.rs` |
| NLGEOM, Riks | `steps/nlgeom.rs`, `nlgeom_riks.rs`, `nlgeom_solver.rs` (huvudlösaren med den indefinita lösaren som reserv), `inp/keywords/nlgeom.rs` |
| NLGEOM för skal, membran och Timoshenkobalkar | `elements/shell_nlgeom.rs` (korotationellt, logaritmisk membrantöjning, S4R:s spänningsstyvhet), `elements/beam_timoshenko_nlgeom.rs` (inkrementell balk, geometrisk styvhet från alla snittkrafter); inkluderas med `#[path]` från `shell.rs` och `beam_timoshenko.rs`. Följande tryck P på skal i `kinematics/pressure.rs`, konsistent skalmassa i `steps/frequency.rs::mass_kind`. Fall i `tests/cases/nlgeom_struct/`, tester i `tests/nlgeom_struct.rs` |
| Skal och Timoshenkobalkar utanför linjär statik | `elements/shell_section.rs` (integrering genom tjockleken: plasticitet, temperatur med gradient, *ORIENTATION, *SHELL SECTION, COMPOSITE; även NLGEOM), `elements/beam_timoshenko_plastic.rs` (sektionspunkter för RECT/CIRC/PIPE/BOX/I, även NLGEOM); massans typ i implicit dynamik i `steps/dynamic.rs::implicit_mass_kind`. Temperaturgradienter: `ElemCtx::temp` har för skal och balkar 2·nn värden (nodtemperaturer, sedan gradienter) när modellen har gradienter (`TemperatureField::gradient_at`). Fall i `tests/cases/struct_nonlin/`, tester i `tests/struct_nonlin.rs` |
| Tunna och axisymmetriska skal | `elements/shell3.rs` (gemensamt: tvärsnitt, femfrihetsgradsskalens nodbas och tangentprojektion, utdata), `shell3_axi.rs` (SAX1, SAX2, MAX1, MAX2), `shell3_tri.rs` (STRI3), `shell3_q4.rs` (S4R5), `shell3_q9.rs` (S9R5, S8R5 med intern mittnod). Fall i `tests/cases/shells3/`tester i `tests/shells3.rs` |
| Material | `materials/elastic.rs`, `anisotropic.rs`, `plasticity.rs`, `plasticity_dp.rs`, `hyperelastic.rs`, `thermal.rs`, `thermal_elastic.rs`, `damping.rs`, `heat.rs`; `inp/keywords/material_behaviors.rs`, `temperature.rs` |
| Metallplasticitet (Hill, hastighetsberoende, Johnson–Cook) | `materials/plasticity_hill.rs` (*POTENTIAL, von Mises med anisotrop elasticitet; väljs via `plasticity_hill::claims` före `plasticity.rs`), `materials/plasticity_rate.rs` (*RATE DEPENDENT, HARDENING=JOHNSON COOK, `JcFailure` för *DAMAGE INITIATION, CRITERION=JOHNSON COOK i `ductile_damage.rs`); `inp/keywords/metal_plasticity.rs`; fall i `tests/cases/plasticity2/`, tester i `tests/plasticity2.rs` |
| Tryckberoende plasticitet och skada | `materials/mohr_coulomb.rs`, `concrete_damaged.rs`, `ductile_damage.rs`, `principal.rs`; `inp/keywords/materials_soil.rs`; fall i `tests/cases/materials_soil/`, tester i `tests/materials_soil.rs` |
| Elastomerer och skum | `materials/hyperfoam.rs` (*HYPERFOAM), `hyperelastic_vdw.rs` (VAN DER WAALS), `mullins.rs` (*MULLINS EFFECT, SENER/DMENER), `hyperelastic_testdata.rs` (TEST DATA INPUT för polynomformerna); `HyperModel` i `hyperelastic_split.rs` har `kirchhoff_full` (ej uppdelade modeller), `energy` och historievariabler (`n_hist`, `kirchhoff_hist`, `tau_dev_hist`) som F-bar (`kinematics/fbar_hyper.rs::eval_hist`), hybridelementen (`hybrid::nl_eval_hist`) och CAX4 (`hybrid::cax4_hyper`) använder; `inp/keywords/foam.rs`. Fall i `tests/cases/foam/`, tester i `tests/foam.rs` |
| Porvattentryck–förskjutning (*SOILS, *GEOSTATIC) | `elements/soils.rs` (CPE4P, CPE8P, CPE8RP, CAX4P, CAX8P, CAX8RP, C3D8P, C3D20P, C3D20RP), `steps/soils.rs`, `materials/porous.rs` (*PERMEABILITY, *POROUS BULK MODULI), `inp/keywords/soils.rs` (även *INITIAL CONDITIONS, TYPE=PORE PRESSURE/RATIO/STRESS, *CFLOW, *DFLOW, *FLOW); fall i `tests/cases/soils/`, tester i `tests/soils.rs` |
| Geomaterial och utsmetad betong | `materials/clay.rs` (*CLAY PLASTICITY, *CLAY HARDENING, *POROUS ELASTIC), `materials/mohr_coulomb_cutoff.rs` (*TENSION CUTOFF, inkluderas med `#[path]` från `mohr_coulomb.rs`), `materials/concrete_smeared.rs` (*CONCRETE, *TENSION STIFFENING, *SHEAR RETENTION, *FAILURE RATIOS), `materials/initial_state.rs` (*INITIAL CONDITIONS, TYPE=STRESS/RATIO: `MaterialModel::init_state_at`, obalanskorrektion F0·(1 − t) i första steget via `StaticSystem::unbalance`); `inp/keywords/geomaterials.rs`; fall i `tests/cases/geomaterials/`, tester i `tests/geomaterials.rs` |
| Utökad Drucker–Prager, Cap, krypning och viskoelasticitet (materials3) | `materials/invariant_return.rs` (gemensam bakåt Euler-retur i (p, q, t) med K, 8 obekanta och Newton med differens-Jacobian; `newton_fd` för små system), `materials/plasticity_dp2.rs` (*DRUCKER PRAGER, SHEAR CRITERION=HYPERBOLIC/EXPONENT FORM och LINEAR med K ≠ 1; `plasticity_dp2::claims` före `plasticity_dp.rs`), `materials/cap.rs` (*CAP PLASTICITY, *CAP HARDENING: kappa, övergångsyta, skjuvyta, regionlåst reserv och hörnretur), `materials/creep_ext.rs` (*CREEP LAW=ANAND/DARVEAUX/DOUBLE POWER, *SWELLING, *POTENTIAL för krypning; `creep_ext::claims` före `creep.rs`, som delar `CreepLaw`), `materials/viscoelastic_freq.rs` (*VISCOELASTIC, FREQUENCY=PRONY/TABULAR; `MaterialModel::frequency_moduli` och `frequency_matrices`/`correct_stress`, anropade från `steps/steady_state.rs` i DIRECT), `materials/viscoelastic_trs.rs` (*TRS WLF/ARRHENIUS/TABULAR; omsluter `viscoelastic.rs` med reducerad tid). `time_dependence.rs`: krypning/viskoelasticitet aktiva i *COUPLED TEMPERATURE-DISPLACEMENT (explicit krypning med temperaturen vid inkrementets början), gaskonstanten från *PHYSICAL CONSTANTS. Nyckelord i `inp/keywords/materials3.rs`; fall i `tests/cases/materials3/`, tester i `tests/materials3.rs` |
| Material, del 5 (materials5) | `materials/crushable_foam.rs` (*CRUSHABLE FOAM VOLUMETRIC/ISOTROPIC; pc i hårdnandetabellens punkter), `materials/porous_metal.rs` (*POROUS METAL PLASTICITY, *VOID NUCLEATION, *POROUS FAILURE CRITERIA; `apply_relative_density` från `steps::run` och `MaterialModel::init_relative_density` för *INITIAL CONDITIONS, TYPE=RELATIVE DENSITY), `materials/deformation_plasticity.rs` (Ramberg–Osgood), `materials/hashin.rs` (CRITERION=HASHIN med LAMINA i plan spänning), `materials/forming_limit.rs` (FLD/FLSD/MSFLD som omslag kring plasticiteten; `ductile_damage.rs` lämnar Hashin- och formbarhetsmaterial), `materials/damage_tables.rs` (hastighets-/temperaturberoende DUCTILE/SHEAR-tabeller i `ductile_damage.rs`), `inp/keywords/materials5.rs`. Fall i `tests/cases/materials5/`, tester i `tests/materials5.rs` |
| Kontakt och villkor | `interactions/contact.rs`, `constraints.rs`, `connector.rs`, `geom.rs`, `fem/mpc.rs`, `inp/keywords/interactions.rs` |
| Kopplingselement | `interactions/connector.rs` (element, komponenter g = c·u, villkor via `fem/mpc.rs`, villkorskrafter ur den okondenserade residualen med `Interaction::observe_residual`, utdata via `Interaction::elem_output`), `connector_kin.rs` (kopplingstypernas kinematik), `connector_beh.rs` (elasticitet, plasticitet, skada, friktion, stopp, lås, dämpning) – båda inkluderas med `#[path]`; `inp/keywords/connectors.rs`. Fall i `tests/cases/connectors/`, kontaktbeteenden i `tests/cases/contact2/`, tester i `tests/connectors.rs` |
| Kopplingselement med NLGEOM/dynamik, kontakt del 3 (contact3) | `interactions/connector_nl.rs` (ändliga mått per komponent, gradienter och geometrisk styvhet; modeller med NLGEOM- eller explicita steg upprätthåller låsta komponenter med straff i implicita steg), `steps/explicit_constraints.rs` (kinematiska villkor i explicit dynamik via `Interaction::kinematic_constraints`/`set_constraint_forces`), villkorskrafter i implicit dynamik i `steps/dynamic.rs` (HHT-viktade); `interactions/contact_ext.rs` (självkontakt, *FRICTION, ANISOTROPIC, töjningsfri justering ADJUST=/*CONTACT INITIALIZATION DATA via `contact::adjust_slaves` när filen är läst), `inp/keywords/contact3.rs`; kontakt och spaltkonduktans i *COUPLED TEMPERATURE-DISPLACEMENT i `steps/heat_coupled_contact.rs` (blockvis Gauss–Seidel per inkrement) och `steps/heat_stress.rs` (interaktioner i jämvikten). Fall i `tests/cases/contact3/`, tester i `tests/contact3.rs` |
| Dynamik | `steps/dynamic.rs`, `dynamic_amplitude.rs`, `explicit.rs`, `inp/keywords/dynamics.rs` |
| Explicit, del 2 | `steps/explicit_moduli.rs` (effektiva moduler för volymviskositet och Δt), `explicit_dt.rs::improved_plane_stress` (IMPROVED DT METHOD för skal/membran), `kinematics/continuum.rs::set_explicit_end_config` (explicit spänningsuppdatering vid NLGEOM), `explicit_energy.rs::elastic_energy` (ALLSE/ALLPD), `materials/eos.rs` + `inp/keywords/eos.rs` (*EOS, TYPE=USUP), VUMAT (se ovan). Fall i `tests/cases/explicit2/`, tester i `tests/explicit2.rs` |
| Explicit, del 3 | `materials/eos.rs` (*EOS, TYPE=USUP och IDEAL GAS, *TENSILE FAILURE, `apply_initial` för *INITIAL CONDITIONS, TYPE=SPECIFIC ENERGY/TEMPERATURE; `MaterialModel::explicit_full_integration` – trycket i varje integrationspunkt utan B-bar – och `init_energy`), `inp/keywords/eos.rs`, `steps/explicit_bulk.rs::quadratic_factor` (1/8 för C3D8), `steps/explicit_contact.rs` (självkontakt, skal/membran med kontakttjocklek, *CONTACT DAMPING; `only_explicit` gör att `interactions/contact.rs` inte bygger implicit kontakt), `steps/model_change.rs::Activity::set_explicit_rule` (elementet tas bort när alla integrationspunkter nått Dmax). Fall i `tests/cases/explicit3/`, tester i `tests/explicit3.rs` |
| Modala steg | `steps/modal.rs` (*MODAL DYNAMIC), `modal_basis.rs` (moder från *FREQUENCY via `frequency::extract`, *MODAL DAMPING, *SELECT EIGENMODES, *BASE MOTION), `steady_state.rs`, `modal_spectrum.rs`, `inp/keywords/modal.rs` |
| Värme | `elements/heat.rs`, `heat_coupled.rs`, `steps/heat.rs`, `heat_coupled.rs`, `heat_stress.rs` (bara `*TEMPERATURE, FILE=`), `inp/keywords/heat.rs` |
| Lösare | `solver/direct.rs`, `indefinite.rs`, `iterative.rs` + `iterative/`, `sparse.rs` + `sparse/` |
| Rör, infinita element, modifierad tetraeder | `elements/pipe.rs` (PIPE21/22/31/32: Timoshenkobalkarna med ringtöjning på interna noder, *DLOAD PI/PE), `elements/infinite.rs` (CINPE4/CINPS4/CINAX4/CIN3D8; `frequency_stiffness` tar bort styvheten i *FREQUENCY, anropas från `steps/frequency.rs`), `elements/solid_modified.rs` (C3D10M: fyra hexaedrar med likformig töjning och intern nod). Fall i `tests/cases/elements2/`, tester i `tests/elements2.rs` |
| Hybridelement, del 2 (hybrid2) | `elements/hybrid_tet.rs` (C3D4H: kontinuerligt nodvis tryck på trycknoder −(3·10⁹ + 64·nod + material) som `setup` lägger sist i elementets nodlista när .inp-filen är läst, anropas från `inp/mod.rs`; stabilisering α = 0,005 med inkrementell uppdatering vid små töjningar; indefinit lösare via `auto_solver` i `steps::run`), `elements/hybrid2.rs` (C3D8RH, CPE4RH, CAX4RH, CPEG3H, CPEG4H, CPEG4RH, CPEG6H, C3D10MH som grundelementet; spänningsstyvhet för hybridelementen), `materials/incompressible_elastic.rs` (*ELASTIC med ν = 0,5 → 0,4999999, `claim` först i `materials::build`). Fall i `tests/cases/hybrid2/`, tester i `tests/hybrid2.rs` |
| Stegprocedurer | `steps/static_stabilize.rs` (*STATIC, STABILIZE: viskösa krafter c M* v, VF, ALLSD, med och utan NLGEOM), `steps/model_change.rs` (*MODEL CHANGE för element och kontaktpar, elementborttagning vid skada och STATUS; tillståndet ligger i `Prepared::activity` och frågas från elementslingorna i `fem/assembly.rs`, `steps/nlgeom.rs` och `steps/explicit.rs`; `steps::run` anropar `begin_analysis`/`begin_step`), `inp/keywords/procedures.rs`. Fall i `tests/cases/procedures/`, tester i `tests/procedures.rs` |
| Stegprocedurer, del 2 | `steps/complex_frequency.rs` (*COMPLEX FREQUENCY: projicerad dämpning Cᵣ/Sᵣ från dämpare och *DAMPING på modbasen, första ordningens form; `complex_dense.rs` med tät komplex QR-egenvärdeslösare och LU, inkluderas med `#[path]`), `steps/random_response.rs` (*RANDOM RESPONSE, *PSD-DEFINITION, *CORRELATION; `*CLOAD, LOAD CASE=` sparas som `_LOAD CASE CLOAD` i stegets `extra` av `inp/keywords/loads.rs`), `steps/subspace.rs` (*STEADY STATE DYNAMICS, SUBSPACE PROJECTION och *DYNAMIC, SUBSPACE), `steps/direct_cyclic.rs` (*DIRECT CYCLIC med Fourierserie), `steps/anneal.rs` (*ANNEAL via `MaterialModel::anneal`, implementerad för von Mises), `steps/explicit_coupled.rs` (*DYNAMIC TEMPERATURE-DISPLACEMENT, EXPLICIT med CPE3T/CPS3T/C3D4T från `elements/heat_coupled.rs`), `inp/keywords/procedures2.rs`. Fall i `tests/cases/procedures2/`, tester i `tests/procedures2.rs` |
| Kopplade analyser, del 2 (coupled2) | `steps/explicit_coupled_full.rs` (*DYNAMIC TEMPERATURE-DISPLACEMENT, EXPLICIT i den allmänna explicita lösaren, inkluderas med `#[path]` från `explicit.rs` och tar över från `explicit_coupled.rs`: temperatur med framåt Euler, termisk och kontaktvis stabilitetsgräns, volymviskositetens längd c_d·Δtₑ via `BulkViscosity::le_scale`, plastisk värme), `elements/heat_coupled_reduced.rs` (CPE4RT, CPS4RT, CAX4RT, C3D8RT), `materials/inelastic_heat.rs` (*INELASTIC HEAT FRACTION: omslaget `Adiabatic` med [T, T_före, aktiv] sist i tillståndet, `nodal_heat` för de kopplade stegen), `steps/heat_coupled_contact.rs::solve_coupled_with` (plastisk värme i *COUPLED TEMPERATURE-DISPLACEMENT), `steps/adiabatic.rs` (*STATIC/*DYNAMIC, ADIABATIC: slår på läget och lämnar över till den vanliga steg-typen), `elements/thermal_electric_structural.rs` + `steps/thermal_electric_structural.rs` (Q3D4/Q3D8/Q3D20, *COUPLED TEMPERATURE-DISPLACEMENT, ELECTRICAL; inkluderas från `thermal_electric.rs`), `elements/beam_timoshenko_damping.rs` + `steps/complex_beam_shear.rs` (balkarnas skjuvning utesluten ur BETA/STRUCTURAL i de projicerade stegen), `inp/keywords/coupled2.rs`. PE efter *ANNEAL: förskjutningen sist i von Mises-tillståndet (`plasticity.rs`). Fall i `tests/cases/coupled2/`, tester i `tests/coupled2.rs` |
| Multifysik | `elements/thermal_electric.rs` (DC2D3E, DC2D4E, DC2D8E, DCAX4E, DC3D4E, DC3D8E, DC3D20E), `elements/piezo.rs` (C3D8E, C3D20E, CPE4E, CPS4E), `steps/thermal_electric.rs` (*COUPLED THERMAL-ELECTRICAL), `steps/piezo.rs` (*STATIC/*FREQUENCY med prioritet 1, lämnar annars över till `static_linear.rs`/`frequency.rs`), `steps/mass_diffusion.rs` (*MASS DIFFUSION via `HeatSystem::new_with`), `interactions/thermal_contact.rs` (*GAP CONDUCTANCE, *GAP RADIATION; `contact.rs` hoppar över par på rena värmeledningsytor), `materials/multiphysics.rs`, `inp/keywords/multiphysics.rs`; värmeutökningarna (*LATENT HEAT, *FILM PROPERTY, AMPLITUDE= på flöden och film, *CONTROLS FIELD) i `steps/heat.rs`, `inp/keywords/heat.rs`, `materials/heat.rs`. Fall i `tests/cases/multiphysics/`, tester i `tests/multiphysics.rs` |
| Akustik | `elements/acoustic.rs` (AC2D3/4/8, ACAX3/4/8, AC3D4/6/8/10/20, frihetsgrad 8 = POR i slot 7), `materials/acoustic.rs` (*ACOUSTIC MEDIUM), `interactions/acoustic.rs` (akustisk–strukturell *TIE, yta mot yta och nod mot yta), `steps/acoustic.rs` + `acoustic_solve.rs` (*FREQUENCY med osymmetrisk subspace-iteration, *STEADY STATE DYNAMICS, DIRECT, *DYNAMIC; impedans och strålningsränder), `inp/keywords/acoustic.rs`. Stegtypen tar över via `StepType::claim_model` när modellen har akustiska element. Fall i `tests/cases/acoustics/`, tester i `tests/acoustics.rs` |
| Analysinfrastruktur | `steps/restart.rs` (jobbkontexten `Job`/`scope`, *RESTART, WRITE/READ med binär omstartfil `jobb.rfres`, `read_job` för omstarts- och *IMPORT-filer, krokarna `before_run`/`after_step` i `steps::run`), `steps/submodel.rs` (*SUBMODEL, *BOUNDARY, SUBMODEL; nätet som `#X`-rader i `--out`-filen, interpolation; `prepare` anropas först i `steps::run`), `steps/substructure.rs` (*SUBSTRUCTURE GENERATE, register och `jobb_Zn.rfsub`), `steps/matrix_generate.rs` (*MATRIX GENERATE/*MATRIX OUTPUT, `.mtx`), `elements/matrix_element.rs` (TYPE=Zn och *MATRIX ASSEMBLE som matriselement), `inp/keywords/infrastructure.rs`. CLI: `--oldjob`, `--global`. Fall i `tests/cases/infrastructure/`, tester i `tests/infrastructure.rs` |
| Balkar, del 2 | `elements/beams2.rs` (B21H, B22H, B31H, B32H, B23H, B33H), `beams2_open.rs` (B31OS, B32OS: välvning, frihetsgrad 7), `beams2_sections.rs` (HEX, TRAPEZOID, THICK PIPE, ARBITRARY, J för RECT, sektoriella storheter, sektionspunkter för L och ARBITRARY), `beams2_eb_output.rs` (SF/SM för B23/B33). Fall i `tests/cases/beams2/`, tester i `tests/beams2.rs` |
| Fler användarrutiner, fältvariabler, *PARAMETER, .dat | `user/context.rs`, `routines.rs`, `samples2.rs`, `dynlib2.rs`, `loads.rs`; `materials/field_vars.rs`, `field_dependent.rs`, `user_routines.rs`; `inp/parameters.rs`, `inp/keywords/user2.rs`; `fem/dat.rs`. Fall i `tests/cases/user2/`, tester i `tests/user2.rs` |
| Nya element (elements3) | `elements/elements3.rs` (register), `elements3_stri65.rs` (STRI65: T6-geometri, membran och böjning i tre punkter, Kirchhoffstraff i kanternas gausspunkter och i den inre triangelns kantmittpunkter, konsistent massa med sexpunktsregel, fem frihetsgrader per nod som i `shell3.rs`), `elements3_cgax.rs` (CGAX3, CGAX4, CGAX6, CGAX8, CGAX8R: CAX-elementens formfunktioner, regler och B-bar plus vridningen φ, frihetsgrad 5, med γrθ = −r φ,r och γzθ = −r φ,z), `elements3_saxa.rs` (SAXA11–SAXA24: jämna komponenter som planvärden i θ = kπ/N på noderna nod + 100000·k, udda som sin-koefficienter på noderna nod + 10⁷·m; plannoderna skapas av `inp/keywords/element.rs` via `create_plane_nodes` och läggs i elementen av `saxa::setup`, anropad från `inp/mod.rs` efter `finalize`; tvärskjuvningen i diskreta θ-punkter).fall i `tests/cases/elements3/`, tester i `tests/elements3.rs` |
| Balkar, del 3 (beams3) | `elements/elbow.rs` + `elbow_kin.rs` (ELBOW31, ELBOW32, ELBOW31B, ELBOW31C med `*BEAM SECTION, SECTION=ELBOW`: Fourierovalisering, välvning och normalens lutning per nod på genererade noder −(6·10¹² + 16·nod + k), per element −(7·10¹² + 16·element + k) för 31B/31C, skapade av `elbow::setup` från `inp/mod.rs`; NOWARP/NOOVAL/NODEFORM som markörer i `inp/keywords/boundary.rs`; plan spänning genom väggen, även plasticitet), `elements/beam_eb_curved.rs` (B23/B33 som krökta Hermitebalkar när nodnormalerna lutar, inre sträckning λ kondenserad, i egenvärdessteg via `beam_mass::internal_dofs`), `elements/beam_twist.rs` (B31 med vridna nodnormaler: medelvinkel, koppling axialtöjning–vridning, rotationströghet i nodernas riktningar); B23, B33 och B31 är med i `normals::group`. Fall i `tests/cases/beams3/`, tester i `tests/beams3.rs` |
| Nodnormaler | `elements/normals.rs` (medelvärdesbildade nodnormaler inom 20° i slutna grupper, `*NORMAL` med prioritet; räknas när filen är läst, `inp::dispatch` → `normals::apply`, och lagras i ett register som tvärsnittens parameter `_NORMALS` pekar ut; elementen hämtar dem med `normals::of(ctx)`; används av S8R), `inp/keywords/normals.rs` (*NORMAL, även i parter). Fall i `tests/cases/normals/`, tester i `tests/normals.rs` |
| Stela kroppar och användarrutiner | `elements/rigid.rs`, `interactions/rigid_body.rs`, `inp/keywords/rigid.rs`; `user/` (UMAT, UEL, bibliotek, exempel), `materials/user_material.rs`, `elements/user_element.rs`, `interactions/user_element.rs`, `inp/keywords/user.rs`. Fall i `tests/cases/rigid_user/`, tester i `tests/rigid_user.rs` |
| Brottmekanik och packningar | `steps/fracture.rs` (*CONTOUR INTEGRAL i *STATIC: J med domänintegralmetoden, K1/K2/K3 och T-spänning via interaktionsintegral, history `J at <SPRICKA>_<NODMÄNGD>_Contour_<k>`), `steps/fracture_aux.rs` (hjälpfält), `elements/gasket.rs` (GK3D8, GK3D6, GKPE4, GKAX4, GK2D2, GKAX2, GK3D2 och N-varianterna), `materials/gasket.rs`, `inp/keywords/fracture.rs`, `inp/keywords/gasket.rs`. Fall i `tests/cases/fracture/`, tester i `tests/fracture.rs` |
| XFEM (*ENRICHMENT) | `elements/xfem.rs` (läckta XFEM-element per element som delegerar till grundelementet; fantom- och spetsnoder med nummer −(10⁹ + 8·nod + k); `setup` anropas från `inp/mod.rs` när filen är läst; sprickväxt i `Crack::propagate`), `elements/xfem_geom.rs` (referenselement, delsimplex längs φ/ψ, kollapsade gaussregler, spetsfunktioner; inkluderas med `#[path]`), `interactions/xfem.rs` (återställning, sprickväxt i `commit`, PHILSM/PSILSM), `materials/xfem_damage.rs` (grundmaterial + *DAMAGE INITIATION MAXPS/MAXPE/MAXS/QUADS/MAXE/QUADE, *DAMAGE EVOLUTION, *DAMAGE STABILIZATION; kohesiv lag via `materials/cohesive.rs`), `steps/xfem_contour.rs` (*CONTOUR INTEGRAL, XFEM), `inp/keywords/xfem.rs` (även *INITIAL CONDITIONS, TYPE=ENRICHMENT från `temperature.rs`). Fall i `tests/cases/xfem/`, tester i `tests/xfem.rs` |
| Kända luckor, del 1 (gaps1) | Akustik: `materials/acoustic_porous.rs` (COMPLEX BULK MODULUS, COMPLEX DENSITY, POROUS MODEL=DELANY BAZLEY/MIKI, CAVITATION LIMIT läses; `AcousticProps::complex_at`), `steps/acoustic_nonrefl.rs` (impedans- och strålningsränder, ELLIPTICAL/PROLATE SPHEROIDAL som CIRCULAR/SPHERICAL med r₁ = a), `steps/acoustic_improved.rs` (NONREFLECTING=IMPROVED med cos θ per inkrement), automatisk inkrementering i akustisk *DYNAMIC via `dynamic_incr.rs`. EOS: `materials/eos_forms.rs` (TYPE=TABULAR med regularisering, TYPE=JWL med *DETONATION POINT; `MaterialModel::init_at_point`). Balkar: `elements/beam_sections_gaps1.rs` (CHANNEL, HAT), `elements/beam_nonlinear_general.rs` (SECTION=NONLINEAR GENERAL med *AXIAL/*M1/*M2/*TORQUE), `interactions/gaps1.rs` (Newton för olinjära tvärsnittssamband). *IMPORT från explicit analys: `steps/import_explicit.rs` (obalanskrafter via `initial_state::UNBALANCE`, även i `steps/nlgeom.rs`; UPDATE=YES flyttar noderna). Nyckelord i `inp/keywords/gaps1.rs`. Fall i `tests/cases/gaps1/`, tester i `tests/gaps1.rs` |
| Akustik och explicit, del 2 (acoustics2) | Akustiska infinita element: `elements/acoustic_infinite.rs` (ACIN2D2/2D3/3D3/3D4/3D6/3D8/AX2/AX3, Astleys konjugerade element med tio radiella funktioner; delade interna noder −(3·10⁹ + 16·nod + k) skapas i `FeModel::finalize` via `add_nodes`, referenspunkten och q_α ligger i de två första interna nodernas koordinater), `steps/acoustic_infinite.rs` (matriserna per material i *FREQUENCY, *STEADY STATE DYNAMICS och *DYNAMIC; inkluderas från `steps/acoustic.rs`). Akustik i explicit dynamik: `steps/acoustic_explicit.rs` (centrala differenser för det kopplade systemet, kavitation med pseudotryck, volymviskositet b₁ Lₑ/c · K_e, U i bundna fluidnoder), `elements/acoustic_explicit.rs` (AC2D4R, AC3D8R, ACAX4R), `inp/keywords/acoustics2.rs` (*INITIAL CONDITIONS, TYPE=ACOUSTIC STATIC PRESSURE). *CONTACT DAMPING: `interactions/contact_damping.rs` (tolkning, μ(c) i Standard; inkluderas från `contact.rs`), explicit i `steps/explicit_contact.rs`. *EOS COMPACTION: `materials/eos_compaction.rs` (P-α; densiteten ρ_s/α₀ sätts i `FeModel::finalize`), volymviskositetens arbete i EOS-energin `materials/eos.rs::add_bulk_work` (anropas från `steps/explicit.rs`). Porös modell: kompressionsmodulens belopp K_f i `materials/acoustic_porous.rs`. Fall i `tests/cases/acoustics2/`, tester i `tests/acoustics2.rs` |
| Avancerade explicita metoder (explicit_adv) | `steps/explicit_adv.rs` (inkluderas med `#[path]` från `explicit.rs`; krokarna i `explicit.rs` är märkta "område explicit_adv": bygg, diagonal massa, krafter, `after_forces` för Euler-/ALE-nodernas lägen och hastigheter, stabilitetsgräns, utdata och energier), `explicit_adv_sph.rs` (SPH, PC3D: kärnor, standardutjämningslängd, hastighetsgradient, Jaumann, artificiell viskositet), `explicit_adv_conversion.rs` (C3D8R/C3D8 → PC3D i analysens början, anropas från `inp/mod.rs`; föräldrarna blir vilande `SPH_PARENT`), `explicit_adv_dem.rs` (PD3D, partikelkontakt mot partiklar och facetter, *DISCRETE ELASTICITY; `explicit_contact.rs` hoppar över par med partikelytor), `explicit_adv_euler.rs` (EC3D8R: material per element, Lagrangesteg, ytflöden med givare–mottagare-gränsytor, elementmittprojektion av rörelsemängden; `eos::set_jacobian`), `explicit_adv_cel.rs` (Euler–Lagrange-kontakt med straffpunkter), `explicit_adv_ale.rs` (*ADAPTIVE MESH: utjämning, advektion, massor; elementen utan NLGEOM räknar med materialets förskjutning via `Advanced::hourglass_u`); elementen i `elements/explicit_adv.rs` (familjen `Other`, inga egna krafter), nyckelord i `inp/keywords/explicit_adv.rs` (även *INITIAL CONDITIONS, TYPE=VOLUME FRACTION från `temperature.rs`; partikelytor utan ytbeteckning i `surface.rs`). Fall i `tests/cases/explicit_adv/`, tester i `tests/explicit_adv.rs` |
| Kavitetsstrålning, påtvingad konvektion, elektromagnetism (physics3) | `steps/heat_cavity.rs` (fasetter, synfaktorer, radiositet; krokar i `steps/heat.rs`: fältet `cavities`, `start_increment`/`contribute`), `interactions/radiation_cavity.rs` (matrisstrukturen), `elements/heat_convection.rs` + `steps/heat_convection.rs` (DCC-element och *MASS FLOW RATE; fältet `convection` i `HeatSystem`), `elements/electromagnetic.rs` + `steps/electromagnetic.rs` + `materials/electromagnetic.rs` (EMC3D4/EMC2D3, kantfrihetsgrader i eget system), `inp/keywords/physics3.rs`; `RUSTYFEM_VF_DUMP=fil` skriver synfaktorerna. Fall i `tests/cases/physics3/`, tester i `tests/physics3.rs` |
| Avancerade explicita metoder (explicit_adv) | `steps/explicit_adv.rs` (inkluderas med `#[path]` från `explicit.rs`; krokarna i `explicit.rs` är märkta "område explicit_adv": bygg, diagonal massa, krafter, `after_forces` för Euler-/ALE-nodernas lägen och hastigheter, stabilitetsgräns, utdata och energier), `explicit_adv_sph.rs` (SPH, PC3D: kärnor, standardutjämningslängd, hastighetsgradient, Jaumann, artificiell viskositet), `explicit_adv_conversion.rs` (C3D8R/C3D8 → PC3D i analysens början, anropas från `inp/mod.rs`; föräldrarna blir vilande `SPH_PARENT`), `explicit_adv_dem.rs` (PD3D, partikelkontakt mot partiklar och facetter, *DISCRETE ELASTICITY; `explicit_contact.rs` hoppar över par med partikelytor), `explicit_adv_euler.rs` (EC3D8R: material per element, Lagrangesteg, ytflöden med givare–mottagare-gränsytor, elementmittprojektion av rörelsemängden; `eos::set_jacobian`), `explicit_adv_cel.rs` (Euler–Lagrange-kontakt med straffpunkter), `explicit_adv_ale.rs` (*ADAPTIVE MESH: utjämning, advektion, massor; elementen utan NLGEOM räknar med materialets förskjutning via `Advanced::hourglass_u`); elementen i `elements/explicit_adv.rs` (familjen `Other`, inga egna krafter), nyckelord i `inp/keywords/explicit_adv.rs` (även *INITIAL CONDITIONS, TYPE=VOLUME FRACTION från `temperature.rs`; partikelytor utan ytbeteckning i `surface.rs`). C3D8R/CPE4R/CPS4R med NLGEOM i explicit dynamik: `explicit_solid_nlgeom.rs` (inkrementell hourglass i aktuell geometri, vriden med elementets spinn; Standard-hourglassen i `elements/solid_higher_nlgeom.rs` stängs av med `skip_standard`), NLGEOM=YES som standard i `explicit.rs` (inte i modeller med *ANNEAL), höger derivata av amplituder vid t = 0 (`dynamic_amplitude.rs::rates`). Fall i `tests/cases/explicit_adv/`, tester i `tests/explicit_adv.rs` |
| Analytiska stela ytor, stela kroppar i parter och i dynamik | `interactions/analytical.rs`, `inp/keywords/analytical.rs`, `interactions/rigid_body.rs` (parter, POSITION=CENTER OF MASS, DENSITY=); nollflöde i `steps/newton.rs`, prediktor för föreskrivna förskjutningar i `steps/nlgeom.rs`, villkor och begynnelseacceleration i `steps/dynamic.rs`. Fall i `tests/cases/rigid2/`, tester i `tests/rigid2.rs` |
| Kaviteter, kohesiv kontakt och materials4 | `interactions/cavity.rs` (*FLUID CAVITY: tryckfrihetsgrad 8 i referensnoden i Standard, tryck ur volymen i Explicit; `auto_solver` och `step_history` anropas från `steps::run`), `interactions/cohesive_contact.rs` (*COHESIVE BEHAVIOR för *CONTACT PAIR med lagen i `materials/cohesive.rs`; `contact.rs` hoppar över sådana par), `inp/keywords/cavity.rs`, `inp/keywords/cohesive_contact.rs` (skadenyckelorden efter *COHESIVE BEHAVIOR via `inp::dispatch`), `materials/plasticity_rate_tables.rs` (*PLASTIC, RATE= och temperaturberoende *POTENTIAL för `plasticity_hill.rs`), `materials/hyperelastic_fit.rs` (ARRUDA-BOYCE, TEST DATA INPUT), `inp/keywords/materials4.rs`. Fall i `tests/cases/cavity/` och `tests/cases/materials4/`, tester i `tests/cavity.rs` |

Nya nyckelord för ett område läggs i `src/inp/keywords/<område>.rs` (ny fil,
EN rad i `keyword_modules!`). Varje nyckelord får registreras **en** gång
(dubbelregistrering ger panik vid start); `*ORIENTATION`, `*EXPANSION`,
`*INITIAL CONDITIONS`, `*TEMPERATURE` och `*FREQUENCY` har vid
sammanslagningarna flyttats till en enda ägare.

Kvarvarande skuld efter sammanslagningen:

- (Åtgärdat i feat3/materials_soil: elementmatriser assembleras med
  `SystemMatrix::add_unsym` för osymmetriska lösare, se "Lägga till ett
  material".)

## Status

Se docs/REFERENS.md: "Funktioner per område" för vad som stöds, "Tester" för
regressionsfallen och "Begränsningar" för vad som saknas.
