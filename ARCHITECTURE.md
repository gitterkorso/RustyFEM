# RustyFEM architecture

This document describes how RustyFEM is structured and **exactly** how to
add elements, materials, step types, .inp keywords, interactions,
solvers and regression cases. The structure is designed so that several people (or
agents) can work in parallel on their own git branches with as few
merge conflicts as possible.

## Data flow

```text
file.inp ──inp::read_file──► FeModel ──steps::run──► AnalysisOutput ──to_result_set──► ResultSet
                               ▲                                                         │
file.rusty ──script──► Model ──legacy::to_fe                    regression::compare ◄──────┘
                                                                        ▲
case.ref (baseline, rustyfem-res 1) ──ResultSet::parse──────────────────┘
```

- `FeModel` (`src/fem/model.rs`) is the common format: nodes and elements
  with numbers from the input file per instance, sets, surfaces, materials (as raw behaviors),
  cross-sections, boundary conditions and steps.
- `steps::run` (`src/steps/mod.rs`) runs the steps in sequence with inherited state
  (`SolutionState`), selects the step type via the registry and collects the output.
- Results are stored per step and frame (`src/fem/output.rs`) and written as
  text in the `rustyfem-res 1` format.

## Directories

| Directory | Contents |
|---|---|
| `src/fem/` | **The core** (shared): model, degrees of freedom, assembly, state, output, equilibrium system |
| `src/elements/` | Element library, **one file per element family**. `api.rs` = the interface, `common.rs` = continuum core |
| `src/materials/` | Material models. `api.rs` = the interface |
| `src/steps/` | Step types, one file per type. `newton.rs` = the Newton–Raphson driver, `api.rs` = the interface |
| `src/inp/` | The .inp reader. `reader.rs` = tokenization and *INCLUDE, `keywords/` = handlers, one file per area |
| `src/interactions/` | Contact, tie, coupling, MPC, equations, springs, masses |
| `src/solver/` | Linear equation solvers (`direct.rs` = sparse LDLᵀ) |
| `src/regression/` | Regression tests: cases, sidecar files, running and comparison against the baseline |
| `src/model.rs`, `script.rs`, `legacy.rs`, `results.rs` | The `.rusty` language and its model, translation to `FeModel` |
| `src/sparse.rs`, `par.rs` | Sparse multifrontal LDLᵀ and thread helpers (used by `solver/direct.rs` and the FFI) |
| `ffi/` | The C interface: `rustyfem_factor/rustyfem_solve/rustyfem_info/rustyfem_order/rustyfem_free` (the signatures must not change; used by OpenFEM) plus `rustyfem_factor_indef/rustyfem_factor_unsym/rustyfem_inertia` |
| `../RustyFEM tests/tests/cases/<area>/` | Regression cases: `<case>.inp` + `<case>.toml` + `<case>.ref` (separate folder, see below) |
| `src/user/` | User routines: UMAT, UEL, VUMAT, DLOAD, DISP, UAMP, USDFLD, UFIELD, UHARD, SIGINI, UEXPAN, CREEP, HETVAL, FILM, UMATHT; run context, registry, shared libraries, examples |
| `../RustyFEM tests/tests/` | Separate test crate (`rustyfem-tests`, the engine via a path): `analytic.rs` (.rusty), `core.rs` (the core), one test program per area (`solids.rs`, `shells_beams.rs`, `eigen.rs`, `nlgeom.rs`, `materials.rs`, `contact.rs`, `dynamics.rs`, `heat_analytic.rs`, `solver.rs`, `rigid_user.rs`) and `regression.rs` (all cases against `.ref`) |

## Rules for parallel work

1. **New things in new files.** Each area has its own file (or several) in
   the right directory. The placeholders below already exist and compile.
2. **Shared registration lists are changed with ONE line per registration, in
   alphabetical order**, without moving or reformatting other lines:

   | List | File | Line |
   |---|---|---|
   | Element families | `src/elements/mod.rs` | `    my_family,` in `element_modules!` |
   | Material modules | `src/materials/mod.rs` | `    my_material,` in `material_modules!` |
   | Step types | `src/steps/mod.rs` | `    my_step_type,` in `step_modules!` |
   | Keyword modules | `src/inp/keywords/mod.rs` | `    my_keywords,` in `keyword_modules!` |
   | Interactions | `src/interactions/mod.rs` | `    my_interaction,` in `interaction_modules!` |
   | Solvers | `src/solver/mod.rs` | `    my_solver,` in `solver_modules!` |

   The macro creates both the `pub mod` and the registration, so one line is enough.
   Within a module the types are listed in the module's own `TYPES`/`FACTORIES`/
   `STEP_TYPES`/`KEYWORDS`/`SOLVERS` list (which only that module owns).
3. **Core files** (`src/fem/*`, all `api.rs`, `elements/common.rs`,
   `steps/newton.rs`, `inp/reader.rs`, `src/regression/*`) are only changed when an
   interface really has to be extended. Extend *additively*: new trait methods
   always get a default implementation (normally `not_implemented(...)`), new
   struct fields get `Default`. Never change existing signatures.
4. **Existing elements** (`continuum2d.rs`, `continuum3d.rs`, `truss.rs`,
   `beam_eb.rs`) already have stubs for capabilities that other areas are to fill in,
   marked with a comment `// area <name>: ...`. Replace only that
   stub's line(s), and put the implementation itself in a separate file (e.g.
   `elements/large_strain.rs` for NLGEOM kinematics or `elements/geometric.rs`
   for geometric stiffness) that the stub calls.
5. **Tests** in a separate file per area: `tests/<area>.rs` in the
   `RustyFEM tests` folder. Regression cases in `tests/cases/<area>/` there. Run
   `cargo test --release -j 2` in that folder before committing.
6. No external dependencies. No reformatting of other people's code.

## Conventions

- **Degrees of freedom** follow the numbering of the `.inp` format (`src/fem/dof.rs`): 1–3 displacement,
  4–6 rotation, 7 warping, 11 temperature NT11. Each node has 8 slots
  (slots 0–7) in all nodal vectors (`NodeVec = [f64; 8]`). The pore pressure
  POR (degree of freedom 8, *SOILS) shares slot 7 with NT11. Degree of freedom 9
  (electric potential EPOT) shares slot 6 with warping; the normalized
  concentration of mass diffusion (NNC11) is degree of freedom 11.
- **Stress/strain** in the component order 11, 22, 33, 12, 13, 23 with
  engineering shear strain (see `src/materials/api.rs`). Output uses the
  usual names (S11, S22, S33, S12, ...). Plane stress is written with S33 = 0
  and E33 = 0.
- **Names** (sets, surfaces, materials, instances) are stored in upper case. Nodes and elements
  are identified by (instance, number). A model without *PART has the instance
  `MODEL`, nodes directly in *ASSEMBLY the instance `ASSEMBLY`.
  Sets in instances are named `INSTANCE.SET`.
- **Matrices** from elements are dense, row-major n×n `Vec<f64>`; the element vector is
  node by node (`node_dofs(0)`, then `node_dofs(1)`, ...).
- **Time**: step types work with the step time fraction t ∈ [0, 1]; `Controls::period`
  gives the step length. `SolutionState::time` is the total time.
- **Loads and boundary conditions** are inherited between steps according to the rules of the `.inp` format (OP=MOD/NEW,
  ramp from the previous value, released boundary conditions are replaced by a ramped reaction,
  perturbation steps start from the base state). See `src/fem/state.rs`.
- **Errors** are `FemError` with a plain-language message in English that gives the file/line or element.

## Adding an element

1. Create or open your family's file, e.g. `src/elements/solid_higher.rs`.
2. Define a struct and implement `ElementType` (`src/elements/api.rs`):

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
       fn faces(&self) -> &'static [&'static [usize]] { &[/* faces S1..S6 */] }
       fn init_state(&self, ctx: &ElemCtx) -> Result<Vec<f64>> {
           Ok(common::init_state(27, StressMode::ThreeD, ctx.material(self.name())?))
       }
       fn stiffness(&self, ctx: &ElemCtx) -> Result<Vec<f64>> { /* ips → common::stiffness */ }
       fn internal_force(&self, ctx: &ElemCtx, u: &[f64], du: &[f64], so: &[f64], sn: &mut Vec<f64>, tangent: bool)
           -> Result<ElemResponse> { /* ips → common::response */ }
       fn output(&self, ctx: &ElemCtx, u: &[f64], state: &[f64]) -> Result<Vec<IpField>> { /* common::output */ }
       // mass, geometric_stiffness, distributed_load, volume_weights as needed
   }
   ```

   Methods that are not implemented automatically give the error
   "... is not implemented for C3D20". The default version of
   `internal_force` uses `stiffness`, but only for linear materials without
   NLGEOM.
3. The integration points must be numbered according to the convention of the `.inp` format and
   the faces (`faces`) must have the standard numbering S1..S6, otherwise S and *DLOAD Pn
   will be wrong.
4. If the file is new: add ONE line to `element_modules!` in `src/elements/mod.rs`.
   (All existing families, see the table "Areas and their files", are
   already registered.)
5. The element can now be used in `*ELEMENT, TYPE=C3D20`. Add a
   regression case (see below).

The element manages its own state (`state`). The continuum core
stores, per integration point, the stress (ncomp values) followed by the material's
state variables (`common::ip_stride`).

## Adding a material

1. Write the model in your file, e.g. `src/materials/plasticity.rs`, and
   implement `MaterialModel` (`src/materials/api.rs`): `update` receives the total
   strain, the strain increment, and the stress and state at the start of the increment
   and returns the stress and the consistent tangent. `n_state`/`init_state` specify
   the state variables, `output_names`/`output_values` e.g. `PEEQ`.
2. Register a factory in the module's `FACTORIES`:

   ```rust
   pub static FACTORIES: &[MaterialFactory] = &[MaterialFactory {
       name: "von Mises plasticity",
       consumes: &["ELASTIC", "PLASTIC"],
       priority: 10,               // higher than isotropic elasticity (0)
       build: Some(build),         // fn(&MaterialDef) -> Result<Option<Arc<dyn MaterialModel>>>
   }];
   ```

   `build` returns `Ok(None)` if the material definition does not fit.
   All behaviors in a material must be consumed (`consumes`) by the chosen
   factory, otherwise it is an error. Behaviors that are only read by other parts (e.g.
   `CONDUCTIVITY` for heat conduction) are declared with a passive entry
   `build: None, consumes: &["CONDUCTIVITY", ...]`.
3. Register the keyword for the behavior in a separate keyword file (see below) with
   `Kind::Behavior` and the handler `crate::inp::keywords::material::behavior`.
4. New module file? ONE line in `material_modules!` in `src/materials/mod.rs`.
5. Time-dependent materials (creep, viscoelasticity) return
   `Some(&TimeDependence)` from `time_dependence()` (`materials/time_dependence.rs`):
   `steps::run` sets the mode per step there (active in *VISCO/*DYNAMIC, off in
   *STATIC) and *VISCO reads the error measure for CETOL and the stability limit.
   Materials that wrap others (e.g. `thermal.rs`) forward the method.

The continuum elements already call `update` at every integration point, so a
new material works directly with CPS/CPE/C3D8/T2D2/T3D2. *STATIC automatically
chooses Newton–Raphson when any material is not linear (`is_linear`).

Softening materials and unsymmetric tangents:

- `MaterialModel::update_with_length(input, L, ...)` receives the element's
  characteristic length L (`elements::common::char_length`: the cube root of the
  volume, the square root of the area in the plane/r–z plane, half the length for
  second-order elements). `common::response` calls it; the default
  is `update`. Used for fracture energy/crack width (Hillerborg) in
  `concrete_damaged.rs` and `ductile_damage.rs`.
- The material tangent may be unsymmetric. The element matrices are assembled with
  `SystemMatrix::add_unsym` (the full matrix) when the solver is unsymmetric and
  as the symmetric part ½(K + Kᵀ) otherwise (`fem/assembly.rs`, `steps/nlgeom.rs`).
- `MaterialModel::requires_unsymmetric()` (default `false`) makes
  `steps::run` choose `unsymmetric` for the steps when the user has not chosen a
  solver or given `UNSYMM=` (`materials::principal::auto_unsymmetric`),
  as for *CONCRETE DAMAGED PLASTICITY.
- `materials/principal.rs` has common building blocks for return algorithms in
  principal stress space (invariants p, q, θ, scalar root finding, numerical
  tangent with forward or central differences).

## Adding a step type

1. Write the type in your file, e.g. `src/steps/frequency.rs`, and implement
   `StepType` (`src/steps/api.rs`):
   - `claim(step, nlgeom)` returns a priority if the type can run the step
     (e.g. `step.procedure_name() == "FREQUENCY"`). The highest priority wins,
     so `nlgeom.rs` can take *STATIC with NLGEOM=YES without touching `static_linear.rs`.
   - `run(ctx)` runs the step. `ctx.loads` contains boundary conditions and loads with
     values at the start and end of the step, `ctx.state` the state before the step
     (write the state after the step; with PERTURBATION it is restored
     automatically), `ctx.out.frames` receives `FrameData`.
2. Register in the module's `STEP_TYPES`. New module file? ONE line in `step_modules!`.
3. The procedure keyword (*FREQUENCY, *BUCKLE, *DYNAMIC, *HEAT TRANSFER, ...)
   is registered in a separate keyword file with `Kind::Step` and the handler
   `crate::inp::procedure`.
4. Incremental steps use the Newton driver: build a `fem::system::StaticSystem`
   (or your own `NonlinearSystem`) and call
   `newton::run(&mut sys, &Controls::from_procedure(..), &mut |sys, t| { ... })`.
   The callback is called after every accepted increment – save frames there.
   The driver handles automatic incrementation with cutbacks, increases and
   convergence criteria per field (displacement, rotation, temperature).
5. Eigenvalue steps write one frame per mode with `value` = eigenvalue/frequency and
   `mode` = mode number, and frame numbers in sequence, so that
   `FRAMEVALUE` and mode shapes can be compared.

## Adding an .inp keyword

1. Create a file in `src/inp/keywords/`, e.g. `contact.rs`:

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

   `RawKeyword` has parameters (`param`, `param_upper`, `flag`, `has`) and
   data lines (`data`, `records()` which joins lines that end with a comma,
   `floats`). `InpCtx` provides `mesh()` (the part being defined or the model),
   `step_mut`, `material_mut`, `node_ref`/`elem_ref` (number, `INST.number`
   or set) and `warn`.
2. Ready-made handlers: `store_model` (store in `FeModel::extra`), `store_step`
   (store in the step's `extra`), `procedure` (the step's procedure),
   `keywords::material::behavior` (material behavior), `ignore`.
3. `Kind` specifies where the keyword may appear: `Model`, `Step`, `Both` or
   `Behavior` (directly after *MATERIAL).
4. New file: ONE line in `keyword_modules!` in `src/inp/keywords/mod.rs`. A
   keyword may only be registered once (tested in `tests/core.rs`).
5. Unknown keywords give a warning. Keywords in `src/inp/unsupported.rs`
   affect the result and give an error until someone registers a handler – a
   registered handler always takes precedence over the list, so the list does not need to be changed.

## Adding an interaction or a constraint

Store the keyword with `store_model` (see above) and write an
`InteractionFactory` in `src/interactions/<area>.rs` that builds
`Box<dyn Interaction>` from `FeModel::extra`. An interaction can:

- add active degrees of freedom (`active_dofs`) and couplings for
  the matrix structure (`connectivity`),
- contribute internal force and tangent (`contribute`, via `TangentSink`) – penalty,
  contact, springs,
- provide linear constraints (`linear_constraints`) – these are eliminated in
  `src/fem/mpc.rs` with a master–slave transformation,
- provide constraints that depend on the configuration (`constraints_depend_on_u` = true and
  `linear_constraints_at(u, nlgeom)`): the NLGEOM step then relinearizes them
  before every evaluation (`MpcMap::update_at`), with the same terms and slaves
  as `linear_constraints`, the coefficients as derivatives (spatial spins for
  rotations) and the rhs chosen so that the slave's position is exact (used by
  `*RIGID BODY` for finite rotations),
- provide mass contributions (`mass`).

The NLGEOM step (`steps/nlgeom.rs`) sets the constraint slaves and condenses
the forces (`MpcMap::update_at`, `condense`) just like the linear steps;
implicit `*DYNAMIC` with NLGEOM also relinearizes the constraints in every
iteration (`steps/dynamic.rs`), explicit dynamics eliminates no constraints.

Surfaces from *SURFACE are in `FeModel::surfaces` (element faces with
label S1..S6/SPOS/SNEG, or nodes). SPOS/SNEG apply to elements with a
single face (rigid elements, shells, membranes): the normal follows the node ordering
(right-hand rule in 3D, node 1 → node 2 rotated 90° counterclockwise in 2D) or
the opposite (`interactions/geom.rs::facets`).

## Rigid bodies

Rigid elements (R3D4, R3D3, R2D2, RAX2 in `elements/rigid.rs`) have nodes and
faces but no stiffness or mass. `*RIGID BODY, REF NODE=, ELSET=,
PIN NSET=, TIE NSET=` (`interactions/rigid_body.rs`) gives linear constraints that
tie the nodes' translations (and rotations for ELSET/TIE NSET) to the
reference node, with NLGEOM with finite rotations via `linear_constraints_at`.
A contact surface on rigid elements (`*SURFACE ... SPOS/SNEG`) becomes a rigid
master surface: `interactions/contact.rs` takes the penalty stiffness from the slave side
(`elements::rigid::is_rigid`, `HARD_FACTOR_RIGID`).

Analytical rigid surfaces (`*SURFACE, TYPE=SEGMENTS/CYLINDER/REVOLUTION`):
`inp/keywords/analytical.rs` (called from `surface.rs`) registers the surface's name
in `FeModel::surfaces` (the type in `kind`) and stores the keyword in
`FeModel::extra`; `interactions/analytical.rs` builds the node-to-surface contact for
`*CONTACT PAIR` with an analytical master surface (`contact.rs` skips such
pairs) and finds the reference node via `*RIGID BODY, ANALYTICAL SURFACE=`.
The contact properties are read with `contact::PairProps`. Keywords that appear in a
`*PART` (`*RIGID BODY`, analytical surfaces) are marked with `_PART`; `*INSTANCE`
adds `_INSTANCE` (the name) and `_PLACE` (the placement,
`analytical::encode_placement`), and the names are interpreted with instance prefixes
(`analytical::scoped`). `rigid_body::reposition` (called when the file has been read)
moves the reference node for POSITION=CENTER OF MASS.

## User routines (UMAT and UEL)

`src/user/` has the established interfaces for user materials and user elements:

| File | Contents |
|---|---|
| `user/umat.rs` | `trait Umat { fn umat(&self, a: &mut UmatArgs) -> Result<()> }`, `UmatArgs` with the standard arguments (STRESS, STATEV, DDSDDE column-major, SSE/SPD/SCD, STRAN, DSTRAN, TIME, DTIME, TEMP, DTEMP, CMNAME, NDI, NSHR, NTENS, NSTATV, PROPS, DROT, PNEWDT, DFGRD0, DFGRD1, JSTEP, ...), registry per material name prefix (`register`, `find`), `UmatFn`/`DllUmat` for libraries |
| `user/uel.rs` | `trait Uel { fn uel(&self, a: &mut UelArgs) -> Result<()> }`, `UelArgs` (RHS, AMATRX column-major, SVARS, ENERGY, PROPS, JPROPS, COORDS, U, DU, V, A, JTYPE, TIME, LFLAGS, PNEWDT, ...), registry per type number (`register(Some(n))`, `None` = all), `UelFn`/`DllUel` |
| `user/dynlib.rs` | LoadLibraryW/GetProcAddress and dlopen/dlsym respectively, directly via FFI |
| `user/mod.rs` | `load_library(path)` (the symbols `umat_`, `umat`, `UMAT`, `uel_`, ...), `RUSTYFEM_USER` |
| `user/samples.rs` | examples: UMAT `EXAMPLE_ELASTIC` and `EXAMPLE_MISES`, UEL U9001 (truss) and U9002 (CPS4 quadrilateral) |
| `materials/user_material.rs` | *USER MATERIAL + *DEPVAR → `MaterialModel` that calls the UMAT |
| `user/vumat.rs`, `materials/user_vumat.rs` | VUMAT for *DYNAMIC, EXPLICIT: `trait Vumat { fn vumat(&self, a: &mut VumatArgs) -> Result<()>; fn elastic_moduli(..) }`, the standard arguments for explicit dynamics (STRAININC with tensor shear, STRESSOLD/NEW, STATEOLD/NEW, ENERINTERN/INELAS, DEFGRAD, STRETCH, RELSPININC, DENSITY), one point per call, registry per material name prefix; a registered VUMAT takes precedence over the UMAT. Examples `EXAMPLE_VELASTIC` and `EXAMPLE_VMISES` |
| `elements/user_element.rs` | *USER ELEMENT + *UEL PROPERTY → `ElementType` that calls the UEL (created when *ELEMENT, TYPE=Un is read, see `inp/keywords/element.rs`) |
| `interactions/user_element.rs` | makes models with user elements nonlinear so that *STATIC uses Newton |
| `inp/keywords/user.rs` | the keywords |

The semantics are the established ones: the UMAT receives the stress and state at the start of the
increment, the strain at the start and the strain increment (engineering shear strain,
component order 11, 22, 33, 12, 13, 23) and returns the stress, the state
and the Jacobian ∂Δσ/∂Δε; the UEL returns RHS = −F_int and AMATRX = ∂F_int/∂u
with the degrees of freedom node by node in ascending order. Both matrices are symmetrized.
A shared library is called with the Fortran convention (all arguments as
pointers, `int32` for integers, CMNAME as 80 characters with the hidden length last) and
one call at a time. The element interface (UEL) lacks the step time (TIME(1) =
total time, PERIOD = 0) and distributed loads.

### Run context (`user/context.rs`)

The material and element interfaces do not pass the integration point's position,
element and step numbers or field variables. Instead, these are available in
`user::context`:

- `Run` (one per analysis, created by `steps::run` with `begin_analysis`, passed
  on to the worker threads in `par::map`): KSTEP, KINC (`newton::run`),
  the step's start time and length, procedure code, the field history
  (`materials::field_vars::FieldHistory`) and the requested PNEWDT (the smallest value in
  the increment; `newton::run` redoes the increment with PNEWDT · Δt).
- `Point` (per thread): the element (`Prepared::ctx` → `set_element`, with
  the nodes' field values) and the integration point (`set_ip` from
  `elements::common::response` and `kinematics::continuum`: NPT, COORDS,
  CELENT, weight). The fields at the point are the element's nodal average (like
  the temperature).
- `record_ips(f)` records the integration points (position, weight) while an
  element computes its internal force (SIGINI, the energies in `.dat`).

New elements that want to give user routines the correct NPT/COORDS call `set_ip`
per integration point (otherwise NPT counts as the call number and COORDS is
the element's center).

### More user routines (`user/routines.rs`)

| Routine | Input | Hooked into |
|---|---|---|
| DLOAD | `*DLOAD` PnNU (load type `PnNU@SET`) | `user/loads.rs::dload`, called from `fem/assembly.rs::external` and `kinematics/pressure.rs` |
| DISP | `*BOUNDARY, USER` (`_USER BOUNDARY` in the step's `extra`) | `StepLoads::user_bc`, `bc_at` (`fem/state.rs`), the value at the end of the step is inherited (`steps::run`) |
| UAMP | `*AMPLITUDE, DEFINITION=USER` | `Def::User` in `steps/dynamic_amplitude.rs` |
| USDFLD | `*USER DEFINED FIELD` | `materials/field_dependent.rs` |
| UFIELD | `*FIELD, USER` | `materials/field_vars.rs` |
| UHARD, UEXPAN | `*PLASTIC, HARDENING=USER`, `*EXPANSION, USER` | `materials/user_routines.rs` (`claim` before the registry) |
| CREEP | `*CREEP, LAW=USER` | `Law::User` in `materials/creep.rs` |
| SIGINI | `*INITIAL CONDITIONS, TYPE=STRESS, USER` | `materials/initial_state.rs` |
| HETVAL, FILM, UMATHT | `*HEAT GENERATION`, `*FILM` FnNU, `*USER MATERIAL, TYPE=THERMAL` | `steps/heat.rs` via `user/loads.rs` |

Each routine is a trait with the standard arguments (`DloadArgs`, ...) and a
registry per name prefix (`register_dload(prefix, ...)`; the key is the name of
the element set, node set, amplitude or material). Examples in
`user/samples2.rs` (names `EXAMPLE_*`), shared libraries in `user/dynlib2.rs`
(`dload_`, `disp_`, `uhard_`, `sigini_`, `ufield_`, `uexpan_`, `hetval_`,
`film_`, `creep_`, `umatht_`). Field-dependent material data (DEPENDENCIES=) and
USDFLD are resolved in `materials/field_dependent.rs`, which `materials::build` asks
first. `*PARAMETER` is expanded in `inp/parameters.rs` before the keywords are interpreted
(`inp::dispatch`). `*NODE PRINT`, `*EL PRINT` and `*ENERGY PRINT` are written to
`job.dat` by `fem/dat.rs` (`rustyfem run`). Cases in `tests/cases/user2/`,
tests in `tests/user2.rs`.

Your own UMAT in Rust:

```rust
use rustyfem::user::umat::{self, Umat, UmatArgs};
struct MyUmat;
impl Umat for MyUmat {
    fn umat(&self, a: &mut UmatArgs) -> rustyfem::Result<()> { /* as in Fortran */ Ok(()) }
}
umat::register("MY_", std::sync::Arc::new(MyUmat)); // material MY_*
```

## Adding a solver

Implement `LinearSolver` and `SystemMatrix` (`src/solver/api.rs`) in
`src/solver/<name>.rs`, add it to `SOLVERS` and (for a new file) ONE line in
`solver_modules!`. Choose it with `RUSTYFEM_SOLVER=<name>` or `RunOptions::solver`.
The solver chooses the equation ordering (`first_dof` per block) and receives the lower
triangle via `add`.

### Registered solvers

| Name | File | Matrix | Method |
|---|---|---|---|
| `direct` (default) | `solver/direct.rs`, `sparse.rs` | symmetric, positive definite | multifrontal LDLᵀ, error on a non-positive pivot (mechanism) |
| `indefinite` | `solver/indefinite.rs`, `sparse/indef.rs` | symmetric, indefinite | multifrontal LDLᵀ with 1×1/2×2 pivots (Bunch–Kaufman threshold u = 0.01 within the front), pivots delayed to the parent, inertia |
| `unsymmetric` | `solver/indefinite.rs`, `sparse/lu.rs` | unsymmetric, symmetric pattern | multifrontal LU, diagonal threshold pivoting with delay, partial pivoting at the root |
| `iterative` | `solver/iterative.rs`, `iterative/amg.rs` | symmetric, positive definite | PCG with algebraic multigrid (smoothed aggregation, rigid-body modes as the null space, symmetric Gauss–Seidel) |
| `iterative-ic0`, `iterative-jacobi` | `solver/iterative.rs`, `iterative/csr.rs` | symmetric, positive definite | PCG with IC(0) (diagonal shift on breakdown) and Jacobi respectively |

Choice per step (`solver::iterative::for_step`, called in `steps::run`):
`*STEP, SOLVER=ITERATIVE` (`*STATIC, SOLVER=ITERATIVE` is also
accepted) gives `iterative`, `*STEP, UNSYMM=YES` gives `unsymmetric`. A solver
chosen with `--solver`/`RUSTYFEM_SOLVER`/`RunOptions::solver` always takes precedence.
`tests/solver.rs` runs all cases in `tests/cases/solver/` with every solver against
the baseline.

Unsymmetric contributions: `SystemMatrix::add_unsym(r, c, v)` (arbitrary r, c).
`add(r, c, v)` still means a symmetric contribution to both (r, c) and
(c, r). For symmetric solvers the default implementation of
`add_unsym` keeps only the lower triangle; `is_symmetric()` tells whether the solver stores
the full matrix. Contact with friction and follower loads should therefore choose
`unsymmetric` (e.g. via `*STEP, UNSYMM=YES`) and assemble with `add_unsym`.

Inertia: `SystemMatrix::inertia()` gives the (negative, zero, positive) pivots after
`factor` for `indefinite` (otherwise `None`). If the matrix has negative pivots,
`indefinite` prints a warning that the system matrix has N negative eigenvalues;
`RUSTYFEM_QUIET=1` turns it off.

Environment variables: `RUSTYFEM_ITER_TOL` (relative residual for PCG, default 1e-10),
`RUSTYFEM_TIMING` (times, PCG iterations, AMG levels, delayed pivots).
If PCG does not converge, the system is solved directly (with a warning).

### The C interface (ffi/)

The original functions `rustyfem_factor`, `rustyfem_solve`, `rustyfem_info`,
`rustyfem_order` and `rustyfem_free` are unchanged. New functions (same
input: n degrees of freedom, nn nodes, `loc`, `coords`, CSC `indptr`/`indices`/`data`
in the caller's numbering):

```c
/* Symmetric indefinite LDLᵀ. inertia_out[3] = {negative, zero, positive} (may be NULL).
   NULL on error: *err_dof = singular degree of freedom, or −1 for invalid input. */
Handle *rustyfem_factor_indef(size_t n, size_t nn, const int64_t *loc, const double *coords,
                             const int64_t *indptr, const int64_t *indices, const double *data,
                             int64_t *inertia_out, int64_t *err_dof);
/* Unsymmetric LU; CSC with both triangles. */
Handle *rustyfem_factor_unsym(size_t n, size_t nn, const int64_t *loc, const double *coords,
                             const int64_t *indptr, const int64_t *indices, const double *data,
                             int64_t *err_dof);
/* Inertia for a handle; 0 = ok, −1 = not available (LU). rustyfem_factor gives {0, 0, n}. */
int32_t rustyfem_inertia(const Handle *h, int64_t *out3);
```

The handles are solved with `rustyfem_solve` and freed with `rustyfem_free` as before,
so a caller (e.g. OpenFEM) can try `rustyfem_factor` and on failure
(non-positive pivot) switch to `rustyfem_factor_indef` instead of an
external solver, and read the number of negative eigenvalues with `rustyfem_inertia`.

## Adding a regression case

The regression module (`src/regression/`) runs a case in RustyFEM and compares
the result with a baseline that RustyFEM itself has written:

| File | Contents |
|---|---|
| `regression/mod.rs` | `run_rustyfem`/`run_rustyfem_spec` (runs the case, also with previous jobs for submodeling and restart), `check_case` (compares or writes the baseline) |
| `regression/case.rs` | `CaseSpec` and the sidecar file's TOML subset, `discover` (all cases under a directory), `spec_path`/`ref_path` |
| `regression/compare.rs` | `compare` per step, frame and quantity, `CaseReport` with text and the largest error |

1. Write `tests/cases/<area>/<case>.inp`. Included files may have any
   extension but **not** `.toml` (only an .inp with a sidecar file counts as a
   case).
2. Write the sidecar file `tests/cases/<area>/<case>.toml`:

   ```toml
   description = "What the case tests"
   tol = 1e-4                  # largest relative error, normalized by the largest |value| in the field
   fields = ["U", "RF", "S"]   # nodal fields (U, UR, RF, RM, NT11, ...), element fields (S, E, PEEQ, ...),
                               # FRAMEVALUE (step time/eigenvalue), H:NAME (history, last value)
   frames = "last"             # or "all" (frames are then compared with the same frame number)
   tol.S = 1e-3                # tolerance per field (optional)
   atol.RF = 1e-8              # absolute tolerance near zero (optional)
   sign_free = ["U"]           # the sign may flip per frame (mode shapes)
   steps = [2, 3]              # compare only these steps (optional)
   global_inp = "global.inp"   # submodel: the global model is run first (--global)
   old_inp = "old.inp"         # restart/*IMPORT: the first analysis is run first (--oldjob)
   floor_elem = 1e-3           # element fields are normalized at least by floor_elem · the largest
                               # |value| in the field in the whole case (e.g. damage
                               # and plastic strains ≈ 0 before initiation)
   ```

3. Run `rustyfem baseline tests/cases/<area>/<case>.inp`. The command runs
   the case and writes the result to `<case>.ref` (the text format
   `rustyfem-res 1`). Review the result before checking in the baseline.
4. Check in `.inp`, `.toml` and `.ref`. `cargo test` (tests/regression.rs)
   then compares all cases against their baselines.
5. `rustyfem check <case.inp ...>` compares individual cases and writes a
   report per quantity; `rustyfem check --all [directory]` runs all cases and
   writes a table. If a change intentionally alters the results, the
   baselines are rewritten with `rustyfem baseline <case.inp ...|--all [directory]>`.

Comparison rules (`src/regression/compare.rs`): nodal fields are driven by
the baseline's nodes (if a node is missing in the result, the value counts as 0); element fields
are driven by the result's points (every point must exist in the baseline;
baseline points without a counterpart are reported but do not give an error); a field that
is missing in both is reported as a note. Complex results (*STEADY STATE DYNAMICS)
have the imaginary part in the field `FIELD_IM` (U_IM, S_IM, ...). Request output in
the .inp file (`*OUTPUT, FIELD` / `*NODE OUTPUT` / `*ELEMENT OUTPUT`) if
the default output is not enough. Cases with previous jobs are run in a temporary
working directory that is removed afterwards.

## Areas and their files

All areas are merged into `main` (2026-10-23). The files below are the
actual implementations; no placeholders remain.

| Area | Files |
|---|---|
| Continuum, first order | `elements/continuum2d.rs`, `continuum3d.rs`, `common.rs` (core, B-bar/F-bar, NLGEOM) |
| Higher-order/reduced/incompatible solids | `elements/solid_higher.rs`, `solid_higher_shapes.rs`, `solid_higher_tech.rs` (R/I technique), `solid_higher_gps.rs` (CPEG), `solid_higher_nlgeom.rs` (NLGEOM for the engine, including CAX), `solid_higher_incompat.rs` (NLGEOM for incompatible modes in incremental form; single-element tests in `tests/cases/accuracy2/`, tests in `tests/accuracy2.rs`), `solid_higher_gps_nlgeom.rs` (NLGEOM for CPEG3/6/8/8R, included with `#[path]` from `solid_higher_gps.rs`); the volumetric term in the B-bar tangent for CPE4/C3D8 with NLGEOM in `kinematics/bbar_tangent.rs` |
| Axisymmetry | `elements/axisym.rs` |
| Shells and membranes | `elements/shell.rs` (S4/S4R, M3D4/M3D3), `shell_tri.rs` (S3/S3R), `inp/keywords/shells_beams.rs` |
| Beams | `elements/beam_eb.rs` (B23/B33), `beam_mass.rs`, `beam_timoshenko.rs`, `beam_timoshenko_section.rs` (B21/B22/B31/B32, cross-sections) |
| Trusses | `elements/truss.rs` |
| Springs, masses, dashpots | `elements/point.rs`, `interactions/springs.rs` |
| Geometric stiffness (K_σ) | `elements/geometric.rs` |
| Natural frequencies/buckling | `steps/frequency.rs`, `steps/buckle.rs`, `solver/eigen.rs`, `inp/keywords/eigen.rs` |
| NLGEOM, Riks | `steps/nlgeom.rs`, `nlgeom_riks.rs`, `nlgeom_solver.rs` (the main solver with the indefinite solver as a fallback), `inp/keywords/nlgeom.rs` |
| NLGEOM for shells, membranes and Timoshenko beams | `elements/shell_nlgeom.rs` (corotational, logarithmic membrane strain, S4R stress stiffness), `elements/beam_timoshenko_nlgeom.rs` (incremental beam, geometric stiffness from all section forces); included with `#[path]` from `shell.rs` and `beam_timoshenko.rs`. Follower pressure P on shells in `kinematics/pressure.rs`, consistent shell mass in `steps/frequency.rs::mass_kind`. Cases in `tests/cases/nlgeom_struct/`, tests in `tests/nlgeom_struct.rs` |
| Shells and Timoshenko beams beyond linear statics | `elements/shell_section.rs` (through-thickness integration: plasticity, temperature with gradient, *ORIENTATION, *SHELL SECTION, COMPOSITE; including NLGEOM), `elements/beam_timoshenko_plastic.rs` (section points for RECT/CIRC/PIPE/BOX/I, including NLGEOM); the mass type in implicit dynamics in `steps/dynamic.rs::implicit_mass_kind`. Temperature gradients: for shells and beams `ElemCtx::temp` has 2·nn values (nodal temperatures, then gradients) when the model has gradients (`TemperatureField::gradient_at`). Cases in `tests/cases/struct_nonlin/`, tests in `tests/struct_nonlin.rs` |
| Thin and axisymmetric shells | `elements/shell3.rs` (shared: cross-section, the nodal basis and tangent projection of the five-DOF shells, output), `shell3_axi.rs` (SAX1, SAX2, MAX1, MAX2), `shell3_tri.rs` (STRI3), `shell3_q4.rs` (S4R5), `shell3_q9.rs` (S9R5, S8R5 with internal mid-node). Cases in `tests/cases/shells3/`, tests in `tests/shells3.rs` |
| Materials | `materials/elastic.rs`, `anisotropic.rs`, `plasticity.rs`, `plasticity_dp.rs`, `hyperelastic.rs`, `thermal.rs`, `thermal_elastic.rs`, `damping.rs`, `heat.rs`; `inp/keywords/material_behaviors.rs`, `temperature.rs` |
| Metal plasticity (Hill, rate-dependent, Johnson–Cook) | `materials/plasticity_hill.rs` (*POTENTIAL, von Mises with anisotropic elasticity; chosen via `plasticity_hill::claims` before `plasticity.rs`), `materials/plasticity_rate.rs` (*RATE DEPENDENT, HARDENING=JOHNSON COOK, `JcFailure` for *DAMAGE INITIATION, CRITERION=JOHNSON COOK in `ductile_damage.rs`); `inp/keywords/metal_plasticity.rs`; cases in `tests/cases/plasticity2/`, tests in `tests/plasticity2.rs` |
| Pressure-dependent plasticity and damage | `materials/mohr_coulomb.rs`, `concrete_damaged.rs`, `ductile_damage.rs`, `principal.rs`; `inp/keywords/materials_soil.rs`; cases in `tests/cases/materials_soil/`, tests in `tests/materials_soil.rs` |
| Elastomers and foams | `materials/hyperfoam.rs` (*HYPERFOAM), `hyperelastic_vdw.rs` (VAN DER WAALS), `mullins.rs` (*MULLINS EFFECT, SENER/DMENER), `hyperelastic_testdata.rs` (TEST DATA INPUT for the polynomial forms); `HyperModel` in `hyperelastic_split.rs` has `kirchhoff_full` (non-split models), `energy` and history variables (`n_hist`, `kirchhoff_hist`, `tau_dev_hist`) that F-bar (`kinematics/fbar_hyper.rs::eval_hist`), the hybrid elements (`hybrid::nl_eval_hist`) and CAX4 (`hybrid::cax4_hyper`) use; `inp/keywords/foam.rs`. Cases in `tests/cases/foam/`, tests in `tests/foam.rs` |
| Pore pressure–displacement (*SOILS, *GEOSTATIC) | `elements/soils.rs` (CPE4P, CPE8P, CPE8RP, CAX4P, CAX8P, CAX8RP, C3D8P, C3D20P, C3D20RP), `steps/soils.rs`, `materials/porous.rs` (*PERMEABILITY, *POROUS BULK MODULI), `inp/keywords/soils.rs` (including *INITIAL CONDITIONS, TYPE=PORE PRESSURE/RATIO/STRESS, *CFLOW, *DFLOW, *FLOW); cases in `tests/cases/soils/`, tests in `tests/soils.rs` |
| Geomaterials and smeared concrete | `materials/clay.rs` (*CLAY PLASTICITY, *CLAY HARDENING, *POROUS ELASTIC), `materials/mohr_coulomb_cutoff.rs` (*TENSION CUTOFF, included with `#[path]` from `mohr_coulomb.rs`), `materials/concrete_smeared.rs` (*CONCRETE, *TENSION STIFFENING, *SHEAR RETENTION, *FAILURE RATIOS), `materials/initial_state.rs` (*INITIAL CONDITIONS, TYPE=STRESS/RATIO: `MaterialModel::init_state_at`, unbalance correction F0·(1 − t) in the first step via `StaticSystem::unbalance`); `inp/keywords/geomaterials.rs`; cases in `tests/cases/geomaterials/`, tests in `tests/geomaterials.rs` |
| Extended Drucker–Prager, Cap, creep and viscoelasticity (materials3) | `materials/invariant_return.rs` (shared backward Euler return in (p, q, t) with K, 8 unknowns and Newton with a finite-difference Jacobian; `newton_fd` for small systems), `materials/plasticity_dp2.rs` (*DRUCKER PRAGER, SHEAR CRITERION=HYPERBOLIC/EXPONENT FORM and LINEAR with K ≠ 1; `plasticity_dp2::claims` before `plasticity_dp.rs`), `materials/cap.rs` (*CAP PLASTICITY, *CAP HARDENING: kappa, transition surface, shear surface, region-locked fallback and corner return), `materials/creep_ext.rs` (*CREEP LAW=ANAND/DARVEAUX/DOUBLE POWER, *SWELLING, *POTENTIAL for creep; `creep_ext::claims` before `creep.rs`, which shares `CreepLaw`), `materials/viscoelastic_freq.rs` (*VISCOELASTIC, FREQUENCY=PRONY/TABULAR; `MaterialModel::frequency_moduli` and `frequency_matrices`/`correct_stress`, called from `steps/steady_state.rs` in DIRECT), `materials/viscoelastic_trs.rs` (*TRS WLF/ARRHENIUS/TABULAR; wraps `viscoelastic.rs` with reduced time). `time_dependence.rs`: creep/viscoelasticity active in *COUPLED TEMPERATURE-DISPLACEMENT (explicit creep with the temperature at the start of the increment), the gas constant from *PHYSICAL CONSTANTS. Keywords in `inp/keywords/materials3.rs`; cases in `tests/cases/materials3/`, tests in `tests/materials3.rs` |
| Materials, part 5 (materials5) | `materials/crushable_foam.rs` (*CRUSHABLE FOAM VOLUMETRIC/ISOTROPIC; pc at the points of the hardening table), `materials/porous_metal.rs` (*POROUS METAL PLASTICITY, *VOID NUCLEATION, *POROUS FAILURE CRITERIA; `apply_relative_density` from `steps::run` and `MaterialModel::init_relative_density` for *INITIAL CONDITIONS, TYPE=RELATIVE DENSITY), `materials/deformation_plasticity.rs` (Ramberg–Osgood), `materials/hashin.rs` (CRITERION=HASHIN with LAMINA in plane stress), `materials/forming_limit.rs` (FLD/FLSD/MSFLD as a wrapper around the plasticity; `ductile_damage.rs` leaves Hashin and formability materials alone), `materials/damage_tables.rs` (rate-/temperature-dependent DUCTILE/SHEAR tables in `ductile_damage.rs`), `inp/keywords/materials5.rs`. Cases in `tests/cases/materials5/`, tests in `tests/materials5.rs` |
| Contact and constraints | `interactions/contact.rs`, `constraints.rs`, `connector.rs`, `geom.rs`, `fem/mpc.rs`, `inp/keywords/interactions.rs` |
| Connector elements | `interactions/connector.rs` (elements, components g = c·u, constraints via `fem/mpc.rs`, constraint forces from the uncondensed residual with `Interaction::observe_residual`, output via `Interaction::elem_output`), `connector_kin.rs` (kinematics of the connection types), `connector_beh.rs` (elasticity, plasticity, damage, friction, stops, locks, damping) – both included with `#[path]`; `inp/keywords/connectors.rs`. Cases in `tests/cases/connectors/`, contact behaviors in `tests/cases/contact2/`, tests in `tests/connectors.rs` |
| Connector elements with NLGEOM/dynamics, contact part 3 (contact3) | `interactions/connector_nl.rs` (finite measures per component, gradients and geometric stiffness; models with NLGEOM or explicit steps enforce locked components with a penalty in implicit steps), `steps/explicit_constraints.rs` (kinematic constraints in explicit dynamics via `Interaction::kinematic_constraints`/`set_constraint_forces`), constraint forces in implicit dynamics in `steps/dynamic.rs` (HHT-weighted); `interactions/contact_ext.rs` (self-contact, *FRICTION, ANISOTROPIC, strain-free adjustment ADJUST=/*CONTACT INITIALIZATION DATA via `contact::adjust_slaves` when the file has been read), `inp/keywords/contact3.rs`; contact and gap conductance in *COUPLED TEMPERATURE-DISPLACEMENT in `steps/heat_coupled_contact.rs` (blockwise Gauss–Seidel per increment) and `steps/heat_stress.rs` (interactions in the equilibrium). Cases in `tests/cases/contact3/`, tests in `tests/contact3.rs` |
| Dynamics | `steps/dynamic.rs`, `dynamic_amplitude.rs`, `explicit.rs`, `inp/keywords/dynamics.rs` |
| Explicit, part 2 | `steps/explicit_moduli.rs` (effective moduli for bulk viscosity and Δt), `explicit_dt.rs::improved_plane_stress` (IMPROVED DT METHOD for shells/membranes), `kinematics/continuum.rs::set_explicit_end_config` (explicit stress update with NLGEOM), `explicit_energy.rs::elastic_energy` (ALLSE/ALLPD), `materials/eos.rs` + `inp/keywords/eos.rs` (*EOS, TYPE=USUP), VUMAT (see above). Cases in `tests/cases/explicit2/`, tests in `tests/explicit2.rs` |
| Explicit, part 3 | `materials/eos.rs` (*EOS, TYPE=USUP and IDEAL GAS, *TENSILE FAILURE, `apply_initial` for *INITIAL CONDITIONS, TYPE=SPECIFIC ENERGY/TEMPERATURE; `MaterialModel::explicit_full_integration` – the pressure at every integration point without B-bar – and `init_energy`), `inp/keywords/eos.rs`, `steps/explicit_bulk.rs::quadratic_factor` (1/8 for C3D8), `steps/explicit_contact.rs` (self-contact, shells/membranes with contact thickness, *CONTACT DAMPING; `only_explicit` makes `interactions/contact.rs` not build implicit contact), `steps/model_change.rs::Activity::set_explicit_rule` (the element is removed when all integration points have reached Dmax). Cases in `tests/cases/explicit3/`, tests in `tests/explicit3.rs` |
| Modal steps | `steps/modal.rs` (*MODAL DYNAMIC), `modal_basis.rs` (modes from *FREQUENCY via `frequency::extract`, *MODAL DAMPING, *SELECT EIGENMODES, *BASE MOTION), `steady_state.rs`, `modal_spectrum.rs`, `inp/keywords/modal.rs` |
| Heat | `elements/heat.rs`, `heat_coupled.rs`, `steps/heat.rs`, `heat_coupled.rs`, `heat_stress.rs` (only `*TEMPERATURE, FILE=`), `inp/keywords/heat.rs` |
| Solvers | `solver/direct.rs`, `indefinite.rs`, `iterative.rs` + `iterative/`, `sparse.rs` + `sparse/` |
| Pipes, infinite elements, modified tetrahedron | `elements/pipe.rs` (PIPE21/22/31/32: the Timoshenko beams with hoop strain on internal nodes, *DLOAD PI/PE), `elements/infinite.rs` (CINPE4/CINPS4/CINAX4/CIN3D8; `frequency_stiffness` removes the stiffness in *FREQUENCY, called from `steps/frequency.rs`), `elements/solid_modified.rs` (C3D10M: four uniform-strain hexahedra and an internal node). Cases in `tests/cases/elements2/`, tests in `tests/elements2.rs` |
| Hybrid elements, part 2 (hybrid2) | `elements/hybrid_tet.rs` (C3D4H: continuous nodal pressure on pressure nodes −(3·10⁹ + 64·node + material) that `setup` appends to the element's node list when the .inp file has been read, called from `inp/mod.rs`; stabilization α = 0.005 with incremental update at small strains; indefinite solver via `auto_solver` in `steps::run`), `elements/hybrid2.rs` (C3D8RH, CPE4RH, CAX4RH, CPEG3H, CPEG4H, CPEG4RH, CPEG6H, C3D10MH as the base element; stress stiffness for the hybrid elements), `materials/incompressible_elastic.rs` (*ELASTIC with ν = 0.5 → 0.4999999, `claim` first in `materials::build`). Cases in `tests/cases/hybrid2/`, tests in `tests/hybrid2.rs` |
| Step procedures | `steps/static_stabilize.rs` (*STATIC, STABILIZE: viscous forces c M* v, VF, ALLSD, with and without NLGEOM), `steps/model_change.rs` (*MODEL CHANGE for elements and contact pairs, element deletion on damage and STATUS; the state lives in `Prepared::activity` and is queried from the element loops in `fem/assembly.rs`, `steps/nlgeom.rs` and `steps/explicit.rs`; `steps::run` calls `begin_analysis`/`begin_step`), `inp/keywords/procedures.rs`. Cases in `tests/cases/procedures/`, tests in `tests/procedures.rs` |
| Step procedures, part 2 | `steps/complex_frequency.rs` (*COMPLEX FREQUENCY: projected damping Cᵣ/Sᵣ from dashpots and *DAMPING on the modal basis, first-order form; `complex_dense.rs` with a dense complex QR eigenvalue solver and LU, included with `#[path]`), `steps/random_response.rs` (*RANDOM RESPONSE, *PSD-DEFINITION, *CORRELATION; `*CLOAD, LOAD CASE=` is stored as `_LOAD CASE CLOAD` in the step's `extra` by `inp/keywords/loads.rs`), `steps/subspace.rs` (*STEADY STATE DYNAMICS, SUBSPACE PROJECTION and *DYNAMIC, SUBSPACE), `steps/direct_cyclic.rs` (*DIRECT CYCLIC with Fourier series), `steps/anneal.rs` (*ANNEAL via `MaterialModel::anneal`, implemented for von Mises), `steps/explicit_coupled.rs` (*DYNAMIC TEMPERATURE-DISPLACEMENT, EXPLICIT with CPE3T/CPS3T/C3D4T from `elements/heat_coupled.rs`), `inp/keywords/procedures2.rs`. Cases in `tests/cases/procedures2/`, tests in `tests/procedures2.rs` |
| Coupled analyses, part 2 (coupled2) | `steps/explicit_coupled_full.rs` (*DYNAMIC TEMPERATURE-DISPLACEMENT, EXPLICIT in the general explicit solver, included with `#[path]` from `explicit.rs` and takes over from `explicit_coupled.rs`: temperature with forward Euler, thermal and contact-based stability limit, the bulk viscosity length c_d·Δtₑ via `BulkViscosity::le_scale`, plastic heating), `elements/heat_coupled_reduced.rs` (CPE4RT, CPS4RT, CAX4RT, C3D8RT), `materials/inelastic_heat.rs` (*INELASTIC HEAT FRACTION: the `Adiabatic` wrapper with [T, T_prev, active] last in the state, `nodal_heat` for the coupled steps), `steps/heat_coupled_contact.rs::solve_coupled_with` (plastic heating in *COUPLED TEMPERATURE-DISPLACEMENT), `steps/adiabatic.rs` (*STATIC/*DYNAMIC, ADIABATIC: switches the mode on and hands over to the ordinary step type), `elements/thermal_electric_structural.rs` + `steps/thermal_electric_structural.rs` (Q3D4/Q3D8/Q3D20, *COUPLED TEMPERATURE-DISPLACEMENT, ELECTRICAL; included from `thermal_electric.rs`), `elements/beam_timoshenko_damping.rs` + `steps/complex_beam_shear.rs` (the beams' shear excluded from BETA/STRUCTURAL in the projected steps), `inp/keywords/coupled2.rs`. PE after *ANNEAL: the shift last in the von Mises state (`plasticity.rs`). Cases in `tests/cases/coupled2/`, tests in `tests/coupled2.rs` |
| Multiphysics | `elements/thermal_electric.rs` (DC2D3E, DC2D4E, DC2D8E, DCAX4E, DC3D4E, DC3D8E, DC3D20E), `elements/piezo.rs` (C3D8E, C3D20E, CPE4E, CPS4E), `steps/thermal_electric.rs` (*COUPLED THERMAL-ELECTRICAL), `steps/piezo.rs` (*STATIC/*FREQUENCY with priority 1, otherwise hands over to `static_linear.rs`/`frequency.rs`), `steps/mass_diffusion.rs` (*MASS DIFFUSION via `HeatSystem::new_with`), `interactions/thermal_contact.rs` (*GAP CONDUCTANCE, *GAP RADIATION; `contact.rs` skips pairs on pure heat conduction surfaces), `materials/multiphysics.rs`, `inp/keywords/multiphysics.rs`; the heat transfer extensions (*LATENT HEAT, *FILM PROPERTY, AMPLITUDE= on fluxes and film, *CONTROLS FIELD) in `steps/heat.rs`, `inp/keywords/heat.rs`, `materials/heat.rs`. Cases in `tests/cases/multiphysics/`, tests in `tests/multiphysics.rs` |
| Acoustics | `elements/acoustic.rs` (AC2D3/4/8, ACAX3/4/8, AC3D4/6/8/10/20, degree of freedom 8 = POR in slot 7), `materials/acoustic.rs` (*ACOUSTIC MEDIUM), `interactions/acoustic.rs` (acoustic–structural *TIE, surface to surface and node to surface), `steps/acoustic.rs` + `acoustic_solve.rs` (*FREQUENCY with unsymmetric subspace iteration, *STEADY STATE DYNAMICS, DIRECT, *DYNAMIC; impedance and radiation boundaries), `inp/keywords/acoustic.rs`. The step type takes over via `StepType::claim_model` when the model has acoustic elements. Cases in `tests/cases/acoustics/`, tests in `tests/acoustics.rs` |
| Analysis infrastructure | `steps/restart.rs` (the job context `Job`/`scope`, *RESTART, WRITE/READ with the binary restart file `job.rfres`, `read_job` for restart and *IMPORT files, the hooks `before_run`/`after_step` in `steps::run`), `steps/submodel.rs` (*SUBMODEL, *BOUNDARY, SUBMODEL; the mesh as `#X` lines in the `--out` file, interpolation; `prepare` is called first in `steps::run`), `steps/substructure.rs` (*SUBSTRUCTURE GENERATE, registry and `job_Zn.rfsub`), `steps/matrix_generate.rs` (*MATRIX GENERATE/*MATRIX OUTPUT, `.mtx`), `elements/matrix_element.rs` (TYPE=Zn and *MATRIX ASSEMBLE as matrix elements), `inp/keywords/infrastructure.rs`. CLI: `--oldjob`, `--global`. Cases in `tests/cases/infrastructure/`, tests in `tests/infrastructure.rs` |
| Beams, part 2 | `elements/beams2.rs` (B21H, B22H, B31H, B32H, B23H, B33H), `beams2_open.rs` (B31OS, B32OS: warping, degree of freedom 7), `beams2_sections.rs` (HEX, TRAPEZOID, THICK PIPE, ARBITRARY, J for RECT, sectorial properties, section points for L and ARBITRARY), `beams2_eb_output.rs` (SF/SM for B23/B33). Cases in `tests/cases/beams2/`, tests in `tests/beams2.rs` |
| More user routines, field variables, *PARAMETER, .dat | `user/context.rs`, `routines.rs`, `samples2.rs`, `dynlib2.rs`, `loads.rs`; `materials/field_vars.rs`, `field_dependent.rs`, `user_routines.rs`; `inp/parameters.rs`, `inp/keywords/user2.rs`; `fem/dat.rs`. Cases in `tests/cases/user2/`, tests in `tests/user2.rs` |
| New elements (elements3) | `elements/elements3.rs` (registry), `elements3_stri65.rs` (STRI65: T6 geometry, membrane and bending at three points, Kirchhoff penalty at the edges' Gauss points and at the edge midpoints of the inner triangle, consistent mass with a six-point rule, five degrees of freedom per node as in `shell3.rs`), `elements3_cgax.rs` (CGAX3, CGAX4, CGAX6, CGAX8, CGAX8R: the shape functions, rules and B-bar of the CAX elements plus the twist φ, degree of freedom 5, with γrθ = −r φ,r and γzθ = −r φ,z), `elements3_saxa.rs` (SAXA11–SAXA24: even components as plane values at θ = kπ/N on the nodes node + 100000·k, odd ones as sine coefficients on the nodes node + 10⁷·m; the plane nodes are created by `inp/keywords/element.rs` via `create_plane_nodes` and added to the elements by `saxa::setup`, called from `inp/mod.rs` after `finalize`; the transverse shear at discrete θ points). Cases in `tests/cases/elements3/`, tests in `tests/elements3.rs` |
| Beams, part 3 (beams3) | `elements/elbow.rs` + `elbow_kin.rs` (ELBOW31, ELBOW32, ELBOW31B, ELBOW31C with `*BEAM SECTION, SECTION=ELBOW`: Fourier ovalization, warping and the normal's slope per node on generated nodes −(6·10¹² + 16·node + k), per element −(7·10¹² + 16·element + k) for 31B/31C, created by `elbow::setup` from `inp/mod.rs`; NOWARP/NOOVAL/NODEFORM as markers in `inp/keywords/boundary.rs`; plane stress through the wall, including plasticity), `elements/beam_eb_curved.rs` (B23/B33 as curved Hermite beams when the nodal normals are inclined, the internal stretch λ condensed, in eigenvalue steps via `beam_mass::internal_dofs`), `elements/beam_twist.rs` (B31 with twisted nodal normals: mean angle, coupling of axial strain and twist, rotary inertia in the nodes' directions); B23, B33 and B31 are included in `normals::group`. Cases in `tests/cases/beams3/`, tests in `tests/beams3.rs` |
| Nodal normals | `elements/normals.rs` (averaged nodal normals within 20° in closed groups, `*NORMAL` with priority; computed when the file has been read, `inp::dispatch` → `normals::apply`, and stored in a registry that the cross-section parameter `_NORMALS` points to; the elements fetch them with `normals::of(ctx)`; used by S8R), `inp/keywords/normals.rs` (*NORMAL, including in parts). Cases in `tests/cases/normals/`, tests in `tests/normals.rs` |
| Rigid bodies and user routines | `elements/rigid.rs`, `interactions/rigid_body.rs`, `inp/keywords/rigid.rs`; `user/` (UMAT, UEL, libraries, examples), `materials/user_material.rs`, `elements/user_element.rs`, `interactions/user_element.rs`, `inp/keywords/user.rs`. Cases in `tests/cases/rigid_user/`, tests in `tests/rigid_user.rs` |
| Fracture mechanics and gaskets | `steps/fracture.rs` (*CONTOUR INTEGRAL in *STATIC: J with the domain integral method, K1/K2/K3 and T-stress via the interaction integral, history `J at <CRACK>_<NODESET>_Contour_<k>`), `steps/fracture_aux.rs` (auxiliary fields), `elements/gasket.rs` (GK3D8, GK3D6, GKPE4, GKAX4, GK2D2, GKAX2, GK3D2 and the N variants), `materials/gasket.rs`, `inp/keywords/fracture.rs`, `inp/keywords/gasket.rs`. Cases in `tests/cases/fracture/`, tests in `tests/fracture.rs` |
| XFEM (*ENRICHMENT) | `elements/xfem.rs` (enriched XFEM elements per element that delegate to the base element; phantom and tip nodes numbered −(10⁹ + 8·node + k); `setup` is called from `inp/mod.rs` when the file has been read; crack growth in `Crack::propagate`), `elements/xfem_geom.rs` (reference element, sub-simplices along φ/ψ, collapsed Gauss rules, tip functions; included with `#[path]`), `interactions/xfem.rs` (recovery, crack growth in `commit`, PHILSM/PSILSM), `materials/xfem_damage.rs` (base material + *DAMAGE INITIATION MAXPS/MAXPE/MAXS/QUADS/MAXE/QUADE, *DAMAGE EVOLUTION, *DAMAGE STABILIZATION; cohesive law via `materials/cohesive.rs`), `steps/xfem_contour.rs` (*CONTOUR INTEGRAL, XFEM), `inp/keywords/xfem.rs` (including *INITIAL CONDITIONS, TYPE=ENRICHMENT from `temperature.rs`). Cases in `tests/cases/xfem/`, tests in `tests/xfem.rs` |
| Known gaps, part 1 (gaps1) | Acoustics: `materials/acoustic_porous.rs` (COMPLEX BULK MODULUS, COMPLEX DENSITY, POROUS MODEL=DELANY BAZLEY/MIKI, CAVITATION LIMIT are read; `AcousticProps::complex_at`), `steps/acoustic_nonrefl.rs` (impedance and radiation boundaries, ELLIPTICAL/PROLATE SPHEROIDAL as CIRCULAR/SPHERICAL with r₁ = a), `steps/acoustic_improved.rs` (NONREFLECTING=IMPROVED with cos θ per increment), automatic incrementation in acoustic *DYNAMIC via `dynamic_incr.rs`. EOS: `materials/eos_forms.rs` (TYPE=TABULAR with regularization, TYPE=JWL with *DETONATION POINT; `MaterialModel::init_at_point`). Beams: `elements/beam_sections_gaps1.rs` (CHANNEL, HAT), `elements/beam_nonlinear_general.rs` (SECTION=NONLINEAR GENERAL with *AXIAL/*M1/*M2/*TORQUE), `interactions/gaps1.rs` (Newton for nonlinear section relations). *IMPORT from an explicit analysis: `steps/import_explicit.rs` (unbalance forces via `initial_state::UNBALANCE`, also in `steps/nlgeom.rs`; UPDATE=YES moves the nodes). Keywords in `inp/keywords/gaps1.rs`. Cases in `tests/cases/gaps1/`, tests in `tests/gaps1.rs` |
| Acoustics and explicit, part 2 (acoustics2) | Acoustic infinite elements: `elements/acoustic_infinite.rs` (ACIN2D2/2D3/3D3/3D4/3D6/3D8/AX2/AX3, Astley's conjugated elements with ten radial functions; shared internal nodes −(3·10⁹ + 16·node + k) are created in `FeModel::finalize` via `add_nodes`, the reference point and q_α are stored in the coordinates of the first two internal nodes), `steps/acoustic_infinite.rs` (the matrices per material in *FREQUENCY, *STEADY STATE DYNAMICS and *DYNAMIC; included from `steps/acoustic.rs`). Acoustics in explicit dynamics: `steps/acoustic_explicit.rs` (central differences for the coupled system, cavitation with pseudo-pressure, bulk viscosity b₁ Lₑ/c · K_e, U at tied fluid nodes), `elements/acoustic_explicit.rs` (AC2D4R, AC3D8R, ACAX4R), `inp/keywords/acoustics2.rs` (*INITIAL CONDITIONS, TYPE=ACOUSTIC STATIC PRESSURE). *CONTACT DAMPING: `interactions/contact_damping.rs` (parsing, μ(c) in implicit steps; included from `contact.rs`), explicit in `steps/explicit_contact.rs`. *EOS COMPACTION: `materials/eos_compaction.rs` (P-α; the density ρ_s/α₀ is set in `FeModel::finalize`), the work of the bulk viscosity in the EOS energy `materials/eos.rs::add_bulk_work` (called from `steps/explicit.rs`). Porous model: the magnitude of the bulk modulus K_f in `materials/acoustic_porous.rs`. Cases in `tests/cases/acoustics2/`, tests in `tests/acoustics2.rs` |
| Cavity radiation, forced convection, electromagnetics (physics3) | `steps/heat_cavity.rs` (facets, view factors, radiosity; hooks in `steps/heat.rs`: the field `cavities`, `start_increment`/`contribute`), `interactions/radiation_cavity.rs` (the matrix structure), `elements/heat_convection.rs` + `steps/heat_convection.rs` (DCC elements and *MASS FLOW RATE; the field `convection` in `HeatSystem`), `elements/electromagnetic.rs` + `steps/electromagnetic.rs` + `materials/electromagnetic.rs` (EMC3D4/EMC2D3, edge degrees of freedom in a separate system), `inp/keywords/physics3.rs`; `RUSTYFEM_VF_DUMP=file` writes the view factors. Cases in `tests/cases/physics3/`, tests in `tests/physics3.rs` |
| Advanced explicit methods (explicit_adv) | `steps/explicit_adv.rs` (included with `#[path]` from `explicit.rs`; the hooks in `explicit.rs` are marked "area explicit_adv": build, lumped mass, forces, `after_forces` for the positions and velocities of the Euler/ALE nodes, stability limit, output and energies), `explicit_adv_sph.rs` (SPH, PC3D: kernels, default smoothing length, velocity gradient, Jaumann, artificial viscosity), `explicit_adv_conversion.rs` (C3D8R/C3D8 → PC3D at the start of the analysis, called from `inp/mod.rs`; the parents become dormant `SPH_PARENT`), `explicit_adv_dem.rs` (PD3D, particle contact against particles and facets, *DISCRETE ELASTICITY; `explicit_contact.rs` skips pairs with particle surfaces), `explicit_adv_euler.rs` (EC3D8R: material per element, Lagrangian step, face fluxes with donor–acceptor interfaces, element-center projection of the momentum; `eos::set_jacobian`), `explicit_adv_cel.rs` (Eulerian–Lagrangian contact with penalty points), `explicit_adv_ale.rs` (*ADAPTIVE MESH: smoothing, advection, masses; the elements without NLGEOM compute with the material displacement via `Advanced::hourglass_u`); the elements in `elements/explicit_adv.rs` (family `Other`, no forces of their own), keywords in `inp/keywords/explicit_adv.rs` (including *INITIAL CONDITIONS, TYPE=VOLUME FRACTION from `temperature.rs`; particle surfaces without a face label in `surface.rs`). C3D8R/CPE4R/CPS4R with NLGEOM in explicit dynamics: `explicit_solid_nlgeom.rs` (incremental hourglass control in the current geometry, rotated with the element's spin; the implicit hourglass control in `elements/solid_higher_nlgeom.rs` is turned off with `skip_standard`), NLGEOM=YES by default in `explicit.rs` (not in models with *ANNEAL), right derivative of amplitudes at t = 0 (`dynamic_amplitude.rs::rates`). Cases in `tests/cases/explicit_adv/`, tests in `tests/explicit_adv.rs` |
| Analytical rigid surfaces, rigid bodies in parts and in dynamics | `interactions/analytical.rs`, `inp/keywords/analytical.rs`, `interactions/rigid_body.rs` (parts, POSITION=CENTER OF MASS, DENSITY=); zero flux in `steps/newton.rs`, predictor for prescribed displacements in `steps/nlgeom.rs`, constraints and initial acceleration in `steps/dynamic.rs`. Cases in `tests/cases/rigid2/`, tests in `tests/rigid2.rs` |
| Cavities, cohesive contact and materials4 | `interactions/cavity.rs` (*FLUID CAVITY: pressure degree of freedom 8 at the reference node in implicit steps, pressure from the volume in explicit dynamics; `auto_solver` and `step_history` are called from `steps::run`), `interactions/cohesive_contact.rs` (*COHESIVE BEHAVIOR for *CONTACT PAIR with the law in `materials/cohesive.rs`; `contact.rs` skips such pairs), `inp/keywords/cavity.rs`, `inp/keywords/cohesive_contact.rs` (the damage keywords after *COHESIVE BEHAVIOR via `inp::dispatch`), `materials/plasticity_rate_tables.rs` (*PLASTIC, RATE= and temperature-dependent *POTENTIAL for `plasticity_hill.rs`), `materials/hyperelastic_fit.rs` (ARRUDA-BOYCE, TEST DATA INPUT), `inp/keywords/materials4.rs`. Cases in `tests/cases/cavity/` and `tests/cases/materials4/`, tests in `tests/cavity.rs` |

New keywords for an area are placed in `src/inp/keywords/<area>.rs` (new file,
ONE line in `keyword_modules!`). Each keyword may be registered **once**
(double registration panics at startup); `*ORIENTATION`, `*EXPANSION`,
`*INITIAL CONDITIONS`, `*TEMPERATURE` and `*FREQUENCY` have been moved
to a single owner during the merges.

Remaining debt after the merge:

- (Fixed in feat3/materials_soil: element matrices are assembled with
  `SystemMatrix::add_unsym` for unsymmetric solvers, see "Adding a
  material".)

## Status

See [docs/REFERENCE.md](docs/REFERENCE.md): "Features by area" for what is supported, "Tests" for
the regression cases and "Limitations" for what is missing.
