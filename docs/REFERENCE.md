# RustyFEM – technical reference

![RustyFEM](../web/public/brand/rustyfem-logo.png)

A FEM solver written in Rust; the engine has no external dependencies. It
reads both its own simple scripting language (`.rusty`) and input files in
`.inp` format, and is built from open FEM theory. The core is designed to be
extended area by area (elements, materials, step types, keywords,
interactions, solvers) – see [ARCHITECTURE.md](../ARCHITECTURE.md).

## Quick start

```bash
cargo build --release
./target/release/rustyfem examples/cantilever_plate.rusty
```

```text
# Cantilever in plane stress
material steel E=210e9 nu=0.3
plate x=0..2 y=0..0.2 t=0.01 size=0.025
fix x=0
load x=2 fy=-1000
```

Five lines make a complete analysis. The output shows the maximum
displacement, the highest von Mises stress, the sum of the reaction forces and
an equilibrium check. The full field is written to `cantilever_plate.vtk`,
which can be opened in ParaView.

## The `.rusty` workflow

- `plate`, `box`, `beam`, `truss`: one line creates, meshes and places the geometry.
- `size=` on the geometry controls the mesh; the element type is chosen from the geometry and model type.
- Selections are made directly by geometry: `x=0`, `y=0..0.1`, `at=(2,0)`, `all`.
- Nodes at the same location are merged automatically (a beam on a plate works directly).
- `load` gives the **total force**, which is distributed consistently over the selected edges and faces.
- One linear static step with displacement, reactions and stress; other
  analysis types are run through input files in `.inp` format.
- Errors are reported in plain language, e.g. *"node 5 at (1, 0, 0) can move
  freely in ux – add a support"*.

## Commands (`.rusty`)

One command per line. `#` starts a comment. You choose the units yourself, but
they must be consistent: SI (m, N, Pa) or mm, N, MPa.

| Command | Example |
|---|---|
| `model 2d\|3d` | Optional; otherwise determined by the geometry (`box` and 3D points give 3D) |
| `plane strain\|stress` | Plane strain or plane stress in 2D (default: stress) |
| `elements quad\|tri` | Element shape for 2D surfaces (default: quad) |
| `material NAME E= nu= rho=` | `material steel` and `material aluminium` have predefined values |
| `plate x=a..b y=c..d t= size=` | Rectangular plate |
| `quad (x,y) (x,y) (x,y) (x,y) t= size=` | Arbitrary quadrilateral with a structured mesh |
| `box x= y= z= size=` | Rectangular block with C3D8 elements |
| `beam P1 P2 [P3 ...] section= [n=10] [up=(0,0,1)]` | Beam or frame through the points |
| `truss P1 P2 [P3 ...] area=` | Truss members between the points |
| `fix SELECTION` | Clamped: all degrees of freedom fixed |
| `pin SELECTION` | Pinned: translations fixed |
| `support SELECTION ux uy=0.002 rz` | Selected degrees of freedom, with a prescribed value if given |
| `load SELECTION fx= fy= fz= mx= my= mz=` | Total force and total moment |
| `pressure SELECTION p=` | Normal pressure on boundary edges/faces (positive pushes into the material) |
| `gravity [g=9.81]` or `gravity (gx,gy,gz)` | Self-weight; requires `rho` on the material |

**Selections:** `x=0`, `x=0..1 y=0.5` (conditions are combined), `at=(2,0)` (a point)
and `all`.

**Cross-sections:** `rect(b,h)`, `circle(r)`, `pipe(r,t)`, `ibeam(h,b,tf,tw)` and
`general(A,I11,I22,J)`. The height `h` lies along y in 2D and along z in 3D. For
vertical beams in 3D it lies along x, unless `up=` is given.

The geometry uses the most recently defined material, or
`material=NAME`. Without any material command it is steel.

## The command line

```bash
rustyfem model.rusty                        # analysis + model.vtk
rustyfem run model.rusty --vtk out.vtk      # write the VTK file elsewhere
rustyfem run model.rusty --csv nodes.csv    # also nodal results as CSV
rustyfem model.inp                          # run an input file in .inp format
rustyfem run model.inp --out res.txt        # all steps and frames as text (rustyfem-res 1)
rustyfem run model.inp --all-frames -v      # keep all frames, verbose output
rustyfem run model.inp --solver iterative   # choose the equation solver
rustyfem run sub.inp --global global.res    # submodel driven by the global model's results
rustyfem run new.inp --oldjob old           # restart from old.rfres
rustyfem check tests/cases/static/case.inp  # run a regression case and compare with case.ref
rustyfem check --all                        # all regression cases, with a table
rustyfem baseline tests/cases/static/case.inp  # rewrite case.ref from RustyFEM
rustyfem list                               # registered elements, keywords, steps, solvers
```

`RUSTYFEM_USER=path` loads a shared library with user routines
(`umat_`, `uel_`, `dload_`, ...). With `*NODE PRINT`,
`*EL PRINT` or `*ENERGY PRINT`, `rustyfem run` also writes `job.dat`.

## Features by area

Input files in `.inp` format are read directly. Unknown keywords give a
warning; keywords that affect the result but are not supported give a clear error.
`rustyfem list` prints everything that is registered (elements, keywords,
step types, materials and solvers).

### Model and input

| Area | Support |
|---|---|
| Geometry | `*NODE`, `*ELEMENT` (including continuation lines, TYPE=Un for user elements, TYPE=Zn for substructures), `*NSET`/`*ELSET` (GENERATE, INSTANCE=, sets in the data lines), `*SURFACE` (element faces S1..S6/SPOS/SNEG – SPOS/SNEG on rigid elements, shells and membranes with the normal from the node ordering –, TYPE=NODE with nodal areas, analytical rigid surfaces TYPE=SEGMENTS/CYLINDER/REVOLUTION) |
| Normals | Shell and beam normals are averaged at nodes where the elements differ by < 20° (used by S8R); `*NORMAL` takes precedence over the average |
| Parts | `*PART`, `*ASSEMBLY`, `*INSTANCE` with translation and rotation, references `Instance.node` and `Instance.set`; `*RIGID BODY`, analytical surfaces and `*NORMAL` also inside a `*PART` |
| Cross-sections | `*SOLID SECTION` (including ORIENTATION=, REF NODE= for CPEG and infinite acoustic elements), `*SHELL SECTION` (including ORIENTATION= and COMPOSITE with the same material in all layers), `*SHELL GENERAL SECTION`, `*MEMBRANE SECTION`, `*BEAM SECTION` (RECT, CIRC, PIPE, THICK PIPE, BOX, I, L, HEX, TRAPEZOID, ARBITRARY), `*BEAM GENERAL SECTION` (GENERAL, RECT, CHANNEL, HAT, NONLINEAR GENERAL with `*AXIAL`/`*M1`/`*M2`/`*TORQUE` etc., DENSITY=), `*TRANSVERSE SHEAR STIFFNESS`, `*COHESIVE SECTION`, `*GASKET SECTION` |
| Boundary conditions | `*BOUNDARY` in model data and steps, all type names (ENCASTRE, PINNED, XSYMM, ...), prescribed values, OP=NEW/MOD, degree of freedom 8 (pore pressure, acoustic pressure), 9 (electric potential) and 11 (temperature, concentration); USER (DISP) and SUBMODEL |
| Loads | `*CLOAD`; `*DLOAD` P/P1–P6, BX/BY/BZ/BR, GRAV, CENTRIF, PX/PY/PZ/P1/P2 on beams, PI/PE on pipes, PnNU (DLOAD); `*DSLOAD` P, TRVEC and TRSHR (traction as a dead load on 3D faces and shells); follower pressure with NLGEOM; `*AMPLITUDE` (TABULAR, EQUALLY SPACED, PERIODIC, SMOOTH STEP, DECAY, USER) |
| Initial conditions | `*INITIAL CONDITIONS` TYPE=TEMPERATURE, VELOCITY, STRESS (including GEOSTATIC and USER), RATIO, PORE PRESSURE, FIELD, CONCENTRATION, SPECIFIC ENERGY, ENRICHMENT, ACOUSTIC STATIC PRESSURE; field variables with `*FIELD` (including USER) |
| Parameters | `*PARAMETER` with Python expressions (arithmetic, `math` functions) and `<name>` in keyword lines and data lines |
| Steps | Several `*STEP`s in sequence with rules for inherited loads and boundary conditions, PERTURBATION, `*CONTROLS` (PARAMETERS=FIELD, TIME INCREMENTATION), `SOLVER=`/`UNSYMM=` |
| Restart and import | `*RESTART, WRITE/READ` (`--oldjob`, binary restart file `job.rfres`), `*IMPORT` from a previous analysis, including from explicit to implicit analysis (UPDATE=) |
| Submodeling | `*SUBMODEL` and `*BOUNDARY, SUBMODEL` (node-based; the global model with `--global`, from an `--out` file or a restart file) |
| Substructures and matrices | `*SUBSTRUCTURE GENERATE` (Guyan, `*RETAINED NODAL DOFS`) and use as TYPE=Zn with `*SUBSTRUCTURE PROPERTY`; `*MATRIX GENERATE`/`*MATRIX OUTPUT` (`.mtx`), `*MATRIX INPUT`/`*MATRIX ASSEMBLE` |
| Output | `--out` (all steps and frames as text), VTK, CSV; `*NODE PRINT`, `*EL PRINT` and `*ENERGY PRINT` to `job.dat`; other output keywords are accepted (RustyFEM always writes U, RF, S, E, ... and the fields of the area) |
| Other | `*INCLUDE`, `*HEADING` |

### Elements

The element names follow the naming convention of the `.inp` format:

| Element | Formulation |
|---|---|
| CPS4 | Bilinear quadrilateral, full 2×2 integration |
| CPE4 | Full integration with B-bar, where the in-plane volumetric strain ε11+ε22 is averaged and ε33 = 0 is kept (selectively reduced integration) |
| C3D8 | Full 2×2×2 integration with B-bar, where the volumetric strain is averaged over the element |
| CPS3, CPE3 | Constant strain |
| C3D4, C3D10, C3D6, C3D15, C3D20, C3D20R | Isoparametric tetrahedra, wedges and 20-node hexahedra with the elements' standard integration rules (`elements/solid_higher.rs`) |
| C3D10M | Modified tetrahedron (`elements/solid_modified.rs`): four uniform-strain hexahedra, internal node, lumped mass and enhanced stabilization |
| C3D8R, CPS4R, CPE4R, CAX4R | One point (uniform strain) with hourglass stiffness |
| C3D8I, CPS4I, CPE4I, CAX4I | Incompatible modes, condensed at element level; with NLGEOM in incremental form (`elements/solid_higher_incompat.rs`) |
| CPS6, CPE6, CPS8, CPE8, CPS8R, CPE8R | Second-order plane elements |
| CAX3, CAX4, CAX6, CAX8, CAX8R | Axisymmetric elements (`elements/axisym.rs`) |
| CGAX3, CGAX4, CGAX6, CGAX8, CGAX8R | Generalized axisymmetric elements with twist (degree of freedom 5, `elements/elements3_cgax.rs`) |
| C3D8H, C3D6H, C3D10H, C3D15H, C3D20H, C3D20RH, CPE3H, CPE4H, CPE6H, CPE8H, CPE8RH, CAX3H, CAX4H, CAX6H, CAX8H, CAX8RH | Hybrid elements (`elements/hybrid.rs`): three-field formulation (u, p, θ) with constant (C3D8H, C3D6H, C3D10H, CPE3H, CPE4H, CAX3H, CAX4H) or linear pressure per element, condensed at element level; fully incompressible materials (D1 = 0, ν = 0.5) with NLGEOM, with the pressure as Lagrange multipliers on internal nodes. C3D8H/CPE4H/CAX4H have F-bar and volumetric hourglass stiffness like C3D8/CPE4/CAX4 and are computed as C3D8/CPE4/CAX4 for materials other than hyperelastic; geometric stiffness in `*BUCKLE` |
| C3D4H | Hybrid tetrahedron with a continuous nodal pressure field (pressure nodes, indefinite solver) and stabilization (`elements/hybrid_tet.rs`) |
| C3D8RH, CPE4RH, CAX4RH, CPEG3H, CPEG4H, CPEG4RH, CPEG6H, C3D10MH | Hybrid variants computed as the base element (condensed stiffness, `elements/hybrid2.rs`); ν = 0.5 as 0.4999999 |
| CPEG3, CPEG4, CPEG4R, CPEG4I, CPEG6, CPEG8, CPEG8R | Generalized plane strain with a reference node (`elements/solid_higher_gps.rs`); NLGEOM for CPEG3/6/8/8R |
| S4R | Flat four-node shell (6 degrees of freedom per node): one-point integration with hourglass stabilization, assumed transverse shear (MITC4 type), drilling stiffness. With a nonlinear material, *ORIENTATION or a layered section, the section is integrated at section points (Simpson, 5 points by default; `elements/shell_section.rs`) |
| S4 | Like S4R but with full 2×2 integration of bending and assumed membrane strain; through-thickness integration as for S4R |
| S3, S3R | Three-node shell: CST membrane, constant curvature, edge-based constant transverse shear with stabilization, drilling; through-thickness integration as for S4R |
| S8R | Eight-node doubly curved thick shell (6 degrees of freedom per node, `elements/shell2_s8r.rs`): serendipity interpolation with nodal normals, 2×2 integration, transverse shear with slenderness compensation per integration point, drilling per node and curvature terms; consistent mass (3×3) with rotary inertia for all three rotations |
| S4R5, S8R5, S9R5, STRI3, STRI65 | Thin shells (`elements/shell3*.rs`, `elements3_stri65.rs`): S4R5, S8R5 (internal mid-node), S9R5 and STRI65 with five degrees of freedom per node and the Kirchhoff constraint as a penalty; STRI3 as a discrete Kirchhoff triangle (DKT) with drilling. |
| SC8R, SC6R | Continuum shells (displacements only, thickness from the nodes, `elements/shell2_sc.rs`): mid-surface and director from the bottom/top nodes, S4R and S3R formulation respectively without drilling, thickness strain with the thickness modulus 2G (or `THICKNESS MODULUS=`) and the Poisson strain for POISSON = 0.5 |
| SAX1, SAX2, MAX1, MAX2 | Axisymmetric shells and membranes (`elements/shell3_axi.rs`) |
| SAXA11–SAXA14, SAXA21–SAXA24 | Axisymmetric shells with asymmetric deformation (Fourier modes, linear elastic, `elements/elements3_saxa.rs`) |
| M3D4, M3D3 | Membranes (`*MEMBRANE SECTION`) |
| B23, B33 (B23H, B33H) | Euler–Bernoulli beam (cubic, two internal axial degrees of freedom per element), SF/SM at three integration points. |
| B21, B22, B31, B32 (B21H, B22H, B31H, B32H) | Timoshenko beam (linear/quadratic, reduced integration, slenderness compensation); output SF/SM. With a nonlinear material (*PLASTIC, *EXPANSION) and *BEAM SECTION, the section is integrated at section points (RECT 5×5, CIRC 17, PIPE 8, BOX 16, I 13; L and ARBITRARY; plane beams 5 through the height; `elements/beam_timoshenko_plastic.rs`) with output S11 and PEEQ. The hybrid beams have a shear stiffness (1 + 10⁻⁶) times larger |
| B31OS, B32OS | Open thin-walled beams with warping (degree of freedom 7), BIMOM (`elements/beams2_open.rs`) |
| PIPE21, PIPE22, PIPE31, PIPE32 | Pipe elements: Timoshenko beam with hoop strain on internal nodes, internal/external pressure PI/PE with end-cap force (`elements/pipe.rs`) |
| T2D2, T3D2 | Truss |
| COH2D4, COHAX4, COH3D8, COH3D6 | Cohesive elements with traction–separation (`elements/cohesive.rs`), including NLGEOM; output S, E (logarithmic strain with NLGEOM), SDEG, initiation criteria, STATUS |
| GK3D8, GK3D6, GKPE4, GKAX4, GK2D2, GKAX2, GK3D2 (and the N variants) | Gasket elements with thickness behavior, membrane and transverse shear (`elements/gasket.rs`) |
| CINPE4, CINPS4, CINAX4, CIN3D8 | Infinite elements (statics; no stiffness or mass in *FREQUENCY; `elements/infinite.rs`) |
| CPE4P, CPE8P, CPE8RP, CAX4P, CAX8P, CAX8RP, C3D8P, C3D20P, C3D20RP | Pore pressure–displacement (*SOILS), pore pressure at the corner nodes (`elements/soils.rs`) |
| DC2D3, DC2D4, DC2D8, DCAX3, DCAX4, DCAX8, DC3D4, DC3D6, DC3D8, DC3D10, DC3D20 | Heat conduction and mass diffusion; first-order elements have lumped capacity |
| DC2D3E, DC2D4E, DC2D8E, DCAX4E, DC3D4E, DC3D8E, DC3D20E | Coupled thermal-electrical conduction (degrees of freedom 9 and 11) |
| C3D8E, C3D20E, CPE4E, CPS4E | Piezoelectric elements (displacements and potential; output S, E, EFLX, EPG) |
| C3D8T, CPE4T, CPS4T, CAX4T, CPE3T, CPS3T, C3D4T | Coupled temperature–displacement: mechanically like the base elements, thermally like the DC elements; CPE3T/CPS3T/C3D4T in explicit coupled analysis |
| AC2D3, AC2D4, AC2D8, ACAX3, ACAX4, ACAX8, AC3D4, AC3D6, AC3D8, AC3D10, AC3D20 | Acoustic elements (pressure POR, degree of freedom 8, `elements/acoustic.rs`) |
| AC2D4R, AC3D8R, ACAX4R | Acoustic elements for explicit dynamics with lumped mass |
| PC3D | SPH particles (explicit dynamics), mass ρ(2L)³ from `*SOLID SECTION` (`steps/explicit_adv_sph.rs`) |
| PD3D | Discrete particles (DEM, rigid spheres with rotation) from `*DISCRETE SECTION` (`steps/explicit_adv_dem.rs`) |
| EC3D8R | Eulerian elements with multiple materials and void, `*EULERIAN SECTION` (`steps/explicit_adv_euler.rs`) |
| ACIN2D2, ACIN2D3, ACIN3D3, ACIN3D4, ACIN3D6, ACIN3D8, ACINAX2, ACINAX3 | Acoustic infinite elements (Astley's conjugated elements with ten radial functions, `elements/acoustic_infinite.rs`) |
| XFEM (`*ENRICHMENT`) | Enriched C3D8, C3D8R, C3D4, CPE4 and CPS4: Heaviside with phantom nodes, tip functions for stationary cracks, crack growth one element layer per increment with a cohesive law (`elements/xfem.rs`) |
| SPRING1, SPRING2, SPRINGA, DASHPOT1, DASHPOT2, DASHPOTA, MASS, ROTARYI, CONN2D2, CONN3D2 | Springs, dashpots, point masses and connector elements (`elements/point.rs`, `interactions/`) |
| R3D4, R3D3, R2D2, RAX2 | Rigid elements (`elements/rigid.rs`): no stiffness or mass, the nodes follow the reference node in `*RIGID BODY`; surfaces with SPOS/SNEG as rigid master surfaces in contact |
| Un (`*USER ELEMENT`) | User elements with the UEL interface (`elements/user_element.rs`, `user/uel.rs`): NODES, COORDINATES, PROPERTIES, I PROPERTIES, VARIABLES, UNSYMM, active degrees of freedom per node, `*UEL PROPERTY` |
| Zn | Matrix elements: substructures from `*SUBSTRUCTURE GENERATE` and matrices from `*MATRIX INPUT` (`elements/matrix_element.rs`) |

### Materials

| Area | Support |
|---|---|
| Elasticity | `*ELASTIC` isotropic (including temperature and field dependence, DEPENDENCIES=; ν = 0.5 in hybrid elements), TYPE=ENGINEERING CONSTANTS/ORTHOTROPIC/ANISOTROPIC/LAMINA/TRACTION; `*ORIENTATION` (RECTANGULAR, CYLINDRICAL, additional rotation) via `ORIENTATION=` on `*SOLID SECTION` and `*SHELL SECTION` (shells: the in-plane material axes are chosen from the axis of the additional rotation); S and E are reported in local axes; `*DENSITY` |
| Plasticity | `*PLASTIC` von Mises: HARDENING=ISOTROPIC (table, including temperature-dependent), KINEMATIC (linear), COMBINED with DATATYPE=PARAMETERS and NUMBER BACKSTRESSES (Chaboche), JOHNSON COOK, USER (UHARD), `*CYCLIC HARDENING`; radial return with consistent tangent; output PEEQ and PE; in shells (plane stress per section point) and in Timoshenko beams (uniaxial per section point), with and without NLGEOM |
| Metal plasticity | Hill's anisotropic yield criterion (`*POTENTIAL`, including temperature-dependent), yield curves per strain rate (`*PLASTIC, RATE=`), `*RATE DEPENDENT` (POWER LAW, JOHNSON COOK, YIELD RATIO), von Mises with anisotropic elasticity; also with kinematic hardening, *ORIENTATION and NLGEOM |
| Drucker–Prager | `*DRUCKER PRAGER` SHEAR CRITERION=LINEAR (including K ≠ 1), HYPERBOLIC and EXPONENT FORM with `*DRUCKER PRAGER HARDENING` TYPE=COMPRESSION/TENSION/SHEAR; the Cap model `*CAP PLASTICITY` with `*CAP HARDENING`; the unsymmetric tangent is assembled in full with `*STEP, UNSYMM=YES` |
| Mohr–Coulomb | `*MOHR COULOMB` (φ, ψ, DEVIATORIC ECCENTRICITY=) with `*MOHR COULOMB HARDENING` and `*TENSION CUTOFF` (Rankine surface); Menétrey–Willam smooth flow potential, return in principal stress space including corners and apex; numerical unsymmetric tangent; output PEEQ and PE |
| Cam-clay | `*CLAY PLASTICITY` (including TABULAR) with `*CLAY HARDENING` and porous elasticity `*POROUS ELASTIC` |
| Concrete | Damaged plasticity: `*CONCRETE DAMAGED PLASTICITY` with `*CONCRETE COMPRESSION HARDENING`, `*CONCRETE TENSION STIFFENING` (STRAIN/DISPLACEMENT/GFI), `*CONCRETE COMPRESSION DAMAGE` and `*CONCRETE TENSION DAMAGE`; Lee–Fenves with stiffness recovery and viscous regularization; output PEEQ, PEEQT, PE, DAMAGEC, DAMAGET, SDEG. Smeared cracking: `*CONCRETE` with `*TENSION STIFFENING`, `*SHEAR RETENTION` and `*FAILURE RATIOS` |
| Ductile damage | `*DAMAGE INITIATION, CRITERION=DUCTILE`, `SHEAR` and `JOHNSON COOK` with `*DAMAGE EVOLUTION` TYPE=DISPLACEMENT (LINEAR/EXPONENTIAL/TABULAR) or ENERGY (LINEAR/EXPONENTIAL), DEGRADATION=MAXIMUM/MULTIPLICATIVE, on top of `*PLASTIC`; DUCTILE/SHEAR with strain-rate- and temperature-dependent tables; Hillerborg with the element's characteristic length; element deletion (`*SECTION CONTROLS, ELEMENT DELETION=, MAX DEGRADATION=`) and STATUS; output DUCTCRT, SHRCRT, SDEG |
| Cohesive | `*ELASTIC, TYPE=TRACTION` with `*DAMAGE INITIATION` MAXS/QUADS/MAXE/QUADE and `*DAMAGE EVOLUTION` TYPE=DISPLACEMENT/ENERGY (including mixed mode), viscous regularization; the same law on XFEM crack surfaces (also MAXPS, MAXPE, `*DAMAGE STABILIZATION`) |
| Gaskets | `*GASKET BEHAVIOR` with `*GASKET THICKNESS BEHAVIOR` (loading curve) and `*GASKET ELASTICITY` |
| Thermal strain | `*EXPANSION` (ISO/ORTHO/ANISO, ZERO=, α(T) table, USER); `*INITIAL CONDITIONS, TYPE=TEMPERATURE`, `*TEMPERATURE` (data, OP=NEW/MOD, ramped over the step, or FILE=job where the heat transfer analysis is run with RustyFEM); for shells also the temperature gradient through the thickness; mean temperature per element |
| Hyperelasticity | `*HYPERELASTIC` NEO HOOKE, MOONEY-RIVLIN, YEOH, POLYNOMIAL, REDUCED POLYNOMIAL, OGDEN (N ≤ 6), ARRUDA-BOYCE, VAN DER WAALS; TEST DATA INPUT with `*UNIAXIAL`/`*BIAXIAL`/`*PLANAR`/`*VOLUMETRIC TEST DATA` for the polynomial forms and ARRUDA-BOYCE (μ, λm fitted by least squares); compressible (D1 > 0) in all continuum elements, fully incompressible (D1 = 0) in the hybrid elements; requires NLGEOM=YES |
| Crushable foam | `*CRUSHABLE FOAM`, HARDENING=VOLUMETRIC (k, kt) and ISOTROPIC (k, νp) with `*CRUSHABLE FOAM HARDENING` and `*ELASTIC`; output PEEQ, PE |
| Porous metals | `*POROUS METAL PLASTICITY` (Gurson–Tvergaard–Needleman, q1–q3, RELATIVE DENSITY=) on top of `*PLASTIC`, `*VOID NUCLEATION`, `*POROUS FAILURE CRITERIA`, `*INITIAL CONDITIONS, TYPE=RELATIVE DENSITY`; output PEEQ, PE, VVF, VVFG, VVFN |
| Deformation plasticity | `*DEFORMATION PLASTICITY` (Ramberg–Osgood, total formulation); output PE, PEEQ |
| Composite damage | `*ELASTIC, TYPE=LAMINA` with `*DAMAGE INITIATION, CRITERION=HASHIN` (ALPHA=), `*DAMAGE EVOLUTION, TYPE=ENERGY` and `*DAMAGE STABILIZATION` in CPS elements and shells; output DAMAGEFT/FC/MT/MC, DAMAGESHR, HSNFTCRT/FCCRT/MTCRT/MCCRT |
| Formability | `*DAMAGE INITIATION, CRITERION=FLD`, `FLSD` and `MSFLD` on top of `*PLASTIC` in CPS elements and shells; output FLDCRT, FLSDCRT, MSFLDCRT |
| Foam and Mullins | `*HYPERFOAM` (N ≤ 6, POISSON=) and `*MULLINS EFFECT` (Ogden–Roxburgh, on top of all hyperelastic forms and foam; output SENER, DMENER) |
| Creep | `*CREEP` LAW=TIME, STRAIN, TIME POWER LAW, POWER LAW, HYPERB, ANAND, DARVEAUX, DOUBLE POWER and USER; `*SWELLING`, anisotropic creep with `*POTENTIAL`; temperature-dependent parameters; together with `*PLASTIC`; explicit and implicit integration; active in `*VISCO` and `*COUPLED TEMPERATURE-DISPLACEMENT`; output CEEQ, CE |
| Viscoelasticity | `*VISCOELASTIC, TIME=PRONY` with isotropic `*ELASTIC` or `*HYPERELASTIC` (large strains); MODULI=LONG TERM or INSTANTANEOUS; active in `*VISCO` and `*DYNAMIC`, instantaneous response in `*STATIC`, fully relaxed with `*STATIC, LONG TERM`; `*TRS` (WLF, ARRHENIUS, TABULAR); FREQUENCY=PRONY/TABULAR in `*STEADY STATE DYNAMICS, DIRECT` |
| Equations of state | `*EOS` TYPE=USUP, IDEAL GAS, TABULAR and JWL (`*DETONATION POINT`), `*EOS COMPACTION` (P-α), `*TENSILE FAILURE`, `*INITIAL CONDITIONS, TYPE=SPECIFIC ENERGY` (explicit dynamics) |
| Pore fluid | `*PERMEABILITY` (void-ratio dependent), `*POROUS BULK MODULI` |
| Acoustic medium | `*ACOUSTIC MEDIUM`: bulk modulus, VOLUMETRIC DRAG, COMPLEX BULK MODULUS/COMPLEX DENSITY, POROUS MODEL=DELANY BAZLEY/MIKI, CAVITATION LIMIT |
| Multiphysics | `*ELECTRICAL CONDUCTIVITY`, `*JOULE HEAT FRACTION`, `*PIEZOELECTRIC`, `*DIELECTRIC`, `*DIFFUSIVITY`, `*SOLUBILITY` |
| Damping | `*DAMPING` (Rayleigh ALPHA/BETA, STRUCTURAL) in dynamic steps |
| Heat | `*CONDUCTIVITY`, `*SPECIFIC HEAT` (temperature-dependent), `*LATENT HEAT`, `*HEAT GENERATION` (HETVAL) |
| Field variables | Material data with DEPENDENCIES= on field variables from `*INITIAL CONDITIONS, TYPE=FIELD`, `*FIELD` (UFIELD) or `*USER DEFINED FIELD` (USDFLD) |
| User materials | `*USER MATERIAL` with UMAT (TYPE=MECHANICAL, including UNSYMM), UMATHT (TYPE=THERMAL) and VUMAT (explicit dynamics), see User routines |

### Analysis types (steps)

| Step | Support |
|---|---|
| Linear statics | `*STATIC` (including DIRECT, PERTURBATION, increment data); Newton–Raphson with automatic incrementation and standard convergence criteria for nonlinear materials, contact and NLGEOM; creep is ignored and viscoelasticity gives the instantaneous response (LONG TERM: fully relaxed); unbalance correction for initial stresses |
| Stabilization | `*STATIC, STABILIZE` (FACTOR=, dissipated energy fraction, ALLSDTOL=, CONTINUE=) with and without NLGEOM and contact; output VF and ALLSD |
| Removed elements | `*MODEL CHANGE, REMOVE/ADD` (STRAIN FREE, WITH STRAIN) for elements and contact pairs |
| Quasi-statics with time-dependent materials | `*VISCO` with or without NLGEOM: fixed increments or automatic incrementation with CETOL, stability limit and switch to implicit creep integration, CREEP=EXPLICIT |
| Geometric nonlinearity | `*STEP, NLGEOM=YES` with `*STATIC` (automatic incrementation or DIRECT) and `*STATIC, RIKS` (including DIRECT, termination at λ or a nodal displacement); output logarithmic strain and Cauchy stress S. Elements: the continuum elements (including CPEG3/6/8/8R and incompatible modes; not C3D10M and CPEG4/4R), trusses, beams, shells (S4/S4R/S3/S3R corotational with logarithmic membrane strain, S8R/SC8R/SC6R corotational with Green membrane strain), membranes and cohesive elements; pressure on shells and membranes follows the surface, plasticity with rotated plastic strain and backstress. Tangents that are not positive definite are solved with the indefinite LDLᵀ solver |
| Natural frequencies | `*FREQUENCY` (LANCZOS/SUBSPACE/AMS data lines: number, frequency range, shift; NORMALIZATION, SIM=NO); output per mode U, RF, S, EIGVAL, EIGFREQ, GM, PF1–PF6, EM1–EM6 |
| Complex eigenvalues | `*COMPLEX FREQUENCY` on the modes from the preceding `*FREQUENCY`, with dashpots and the material's *DAMPING |
| Buckling | `*BUCKLE` (SUBSPACE/LANCZOS) with preload from the preceding step, K_σ with Jaumann correction, load stiffness from follower pressure |
| Implicit dynamics | `*DYNAMIC` (HHT-α with standard parameters for APPLICATION=TRANSIENT FIDELITY/MODERATE DISSIPATION, ALPHA/BETA/GAMMA; QUASI-STATIC and BWE), fixed or automatic increments (`steps/dynamic_incr.rs`: HAFTOL/HALFINC SCALE FACTOR, INCREMENTATION=CONSERVATIVE/AGGRESSIVE, increments hit output times); `*TIE`/MPC/`*EQUATION`, rigid bodies (including NLGEOM), connector elements and contact; `*INITIAL CONDITIONS, TYPE=VELOCITY`; shells and Timoshenko beams with lumped mass; `*MASS` and `*ROTARY INERTIA` |
| Explicit dynamics | `*DYNAMIC, EXPLICIT` (NLGEOM=YES by default; C3D8R/CPE4R/CPS4R with NLGEOM have incremental hourglass control): central difference with lumped mass, fixed Δt (`DIRECT USER CONTROL`) or element-by-element stability limit (`IMPROVED DT METHOD` for shells and membranes), `*BULK VISCOSITY`, hourglass control (`RELAX STIFFNESS`, `STIFFNESS`), `*FIXED MASS SCALING`, `*VARIABLE MASS SCALING`, `*AMPLITUDE`, energies (`*ENERGY OUTPUT`: ALLKE, ALLIE, ALLSE, ALLPD, ALLAE, ALLVD, ALLFD, ALLPW, ALLWK, ETOTAL), `*CONTACT` (general contact, including self-contact and shells with contact thickness) and `*CONTACT PAIR` with mass-based penalty force, Coulomb friction and `*CONTACT DAMPING`, NLGEOM with plasticity, `*EOS`, VUMAT, element deletion on damage, locked components of connector elements |
| Coupled explicit thermomechanics | `*DYNAMIC TEMPERATURE-DISPLACEMENT, EXPLICIT` with CPE3T, CPS3T and C3D4T |
| SPH, DEM, Euler and ALE (explicit dynamics) | SPH with PC3D: cubic, quadratic and quintic kernel (`*SECTION CONTROLS, KERNEL=`), default smoothing length, user-specified length and scale factor, constant or variable h, artificial viscosity (UNIFORM, VORTICITY WEIGHTED), all materials (including *EOS); conversion of C3D8R/C3D8 to particles at the start of the analysis (`ELEMENT CONVERSION=YES`, PER ELEMENT with TIME 0 and BACKGROUND GRID). DEM with PD3D: `*DISCRETE SECTION` (DENSITY, ALPHA), particle contact in general contact against particles and element faces (LINEAR, HERTZ with `*DISCRETE ELASTICITY`, hard penalty contact, default damping, `*CONTACT DAMPING`, Coulomb friction with rolling), gravity. Euler (CEL) with EC3D8R: `*EULERIAN SECTION` with multiple materials and void, `*INITIAL CONDITIONS, TYPE=VOLUME FRACTION`, Lagrangian step and advection (ADVECTION=FIRST/SECOND ORDER) with preserved interfaces, *EOS and other materials, gravity, bulk viscosity, output EVF_*, EVF_VOID, SVAVG, PEEQVAVG; coupled Eulerian–Lagrangian contact in general contact (ALL EXTERIOR). ALE: `*ADAPTIVE MESH` (FREQUENCY, MESH SWEEPS, INITIAL MESH SWEEPS) with `*ADAPTIVE MESH CONTROLS` (volumetric and Laplacian smoothing) for C3D8R, advection of state, mass and momentum |
| Modal methods | `*MODAL DYNAMIC`, `*STEADY STATE DYNAMICS` (modal, DIRECT and SUBSPACE PROJECTION), `*RESPONSE SPECTRUM` (`*SPECTRUM`, SRSS/ABS/CQC/NRL/TENP), `*RANDOM RESPONSE` (`*PSD-DEFINITION`, `*CORRELATION`, load cases) and `*DYNAMIC, SUBSPACE` with the modes from the preceding `*FREQUENCY`; `*MODAL DAMPING` (VISCOUS, RAYLEIGH, STRUCTURAL), `*SELECT EIGENMODES`, `*BASE MOTION`, frequency-dependent amplitudes; SSD writes real and imaginary parts |
| Cyclic response | `*DIRECT CYCLIC` (Fourier series, stabilized cycle) |
| Annealing | `*ANNEAL` (stress-free state in the deformed geometry, von Mises) |
| Heat conduction | `*HEAT TRANSFER` (STEADY STATE, transient with fixed increments or DELTMX=), `*CFLUX`, `*DFLUX`/`*DSFLUX` (BF, Sn, AMPLITUDE=), `*FILM`/`*SFILM` (including `*FILM PROPERTY` and the FILM routine), `*RADIATE`/`*SRADIATE`, `*PHYSICAL CONSTANTS`, latent heat; output NT11, RFL11, HFL |
| Coupled thermomechanics | `*COUPLED TEMPERATURE-DISPLACEMENT` (STEADY STATE, transient with fixed increments or DELTMX=), including with contact and `*GAP CONDUCTANCE`/`*GAP RADIATION`; thermal stress with `*STATIC` + `*TEMPERATURE` |
| Thermal-electrical | `*COUPLED THERMAL-ELECTRICAL` (steady state, transient) with Joule heating, `*CECURRENT`, `*DECURRENT`/`*DSECURRENT` |
| Piezoelectricity | `*STATIC` and `*FREQUENCY` with piezoelectric elements, `*CECHARGE` |
| Mass diffusion | `*MASS DIFFUSION` (steady state, transient with fixed increments or DCMAX=), `*INITIAL CONDITIONS, TYPE=CONCENTRATION` |
| Pore pressure | `*SOILS` (steady state and CONSOLIDATION) and `*GEOSTATIC` with `*CFLOW`, `*DFLOW`/`*DSFLOW`, `*FLOW`/`*SFLOW`; output POR, VOIDR, RVF; initial stress, void ratio and pore pressure |
| Acoustics | `*FREQUENCY`, `*STEADY STATE DYNAMICS, DIRECT` and `*DYNAMIC` (including automatic incrementation) with acoustic elements, acoustic–structural `*TIE`, `*IMPEDANCE`/`*SIMPEDANCE` and nonreflecting boundaries (PLANAR, IMPROVED, CIRCULAR, SPHERICAL, ELLIPTICAL, PROLATE SPHEROIDAL); acoustics in `*DYNAMIC, EXPLICIT` with cavitation |
| Fracture mechanics | `*CONTOUR INTEGRAL` in `*STATIC`: J, K FACTORS (K1, K2, K3, direction) and T-STRESS with the domain integral method; also for XFEM cracks |
| Matrices and substructures | `*MATRIX GENERATE` and `*SUBSTRUCTURE GENERATE` as separate steps |
| Cavity radiation | `*CAVITY DEFINITION` (closed/open), `*SURFACE PROPERTY`/`*EMISSIVITY` (ε(T) explicit), `*RADIATION VIEW FACTOR` (BLOCKING, REFLECTION, VTOL, LUMPED AREA, INFINITESIMAL, INTEGRATION), `*RADIATION SYMMETRY` + `*REFLECTION`; view factors in 2D (crossed strings) and 3D (contour/Hottel/lumped), steady-state and transient `*HEAT TRANSFER` |
| Forced convection | DCC1D2, DCCAX2, DCC2D4, DCCAX4, DCC3D8 with `*MASS FLOW RATE` and Yu–Heinrich upwind weighting, steady state and transient (trapezoidal rule, consistent capacity) |
| Low-frequency electromagnetics | `*ELECTROMAGNETIC, LOW FREQUENCY, TIME HARMONIC` with EMC3D4 and EMC2D3 (Nédélec edge elements), `*MAGNETIC PERMEABILITY`, `*ELECTRICAL CONDUCTIVITY`, `*DECURRENT` CJ; output EMB, EMH, EMCD, EMJH |

### Constraints, contact and interactions

`*EQUATION`, `*MPC` (BEAM, PIN, TIE), `*COUPLING` (KINEMATIC, DISTRIBUTING),
`*KINEMATIC COUPLING`, `*TIE` (NODE TO SURFACE and SURFACE TO SURFACE/mortar;
also acoustic–structural), `*RIGID BODY` (REF NODE, ELSET, PIN NSET, TIE
NSET, POSITION=CENTER OF MASS, DENSITY=, ANALYTICAL SURFACE=; with NLGEOM
with finite rotations, also in `*DYNAMIC`), `*SPRING`, `*MASS`, `*ROTARY
INERTIA`, `*DASHPOT`.

Connector elements (`*CONNECTOR SECTION`, `*CONNECTOR BEHAVIOR`): all types in
the connection library except ACCELEROMETER, FLOW-CONVERTER, RETRACTOR and
SLIPRING, with `*CONNECTOR ELASTICITY` (linear, NONLINEAR, RIGID),
`*CONNECTOR DAMPING`, `*CONNECTOR PLASTICITY`/`*CONNECTOR HARDENING`,
`*CONNECTOR DAMAGE INITIATION`/`EVOLUTION`, `*CONNECTOR STOP`,
`*CONNECTOR LOCK`, `*CONNECTOR FRICTION`, `*CONNECTOR MOTION` and
`*CONNECTOR LOAD`; finite measures with NLGEOM, in implicit and explicit dynamics.

Contact: `*CONTACT PAIR` and `*CONTACT` (general contact with
`*CONTACT INCLUSIONS`/`EXCLUSIONS` and `*CONTACT PROPERTY ASSIGNMENT`) with
node to surface and surface to surface, small/finite sliding, self-contact,
deformable or rigid master surfaces (R3D4, R3D3, R2D2 and analytical surfaces),
HARD (including NO SEPARATION), LINEAR, PENALTY and TABULAR (`*SURFACE BEHAVIOR`),
`*FRICTION` (Coulomb, ELASTIC SLIP, ROUGH, LAGRANGE, EXPONENTIAL DECAY,
ANISOTROPIC), `*CLEARANCE`, strain-free adjustment (ADJUST=,
`*CONTACT INITIALIZATION DATA`), `*CONTACT DAMPING`, thermal contact
(`*GAP CONDUCTANCE`, `*GAP RADIATION`); output CPRESS, COPEN, CSHEAR, CSLIP.
Cohesive contact: `*COHESIVE BEHAVIOR` (TYPE=UNCOUPLED, ELIGIBILITY=ORIGINAL CONTACTS) under `*SURFACE INTERACTION` with `*DAMAGE INITIATION` (MAXS, QUADS, MAXU, QUADU), `*DAMAGE EVOLUTION` and `*DAMAGE STABILIZATION` for `*CONTACT PAIR` node to surface; output CSDMG, CSMAXSCRT, CSQUADSCRT, CSMAXUCRT, CSQUADUCRT.
Fluid- and gas-filled cavities: `*FLUID CAVITY` (SURFACE=, REF NODE=, ADDED VOLUME=, AMBIENT PRESSURE=) with `*FLUID BEHAVIOR` (`*FLUID DENSITY`, `*FLUID BULK MODULUS`, `*FLUID EXPANSION` or ideal gas with `*MOLECULAR WEIGHT`, `*CAPACITY`), pressure degree of freedom 8 at the reference node (also prescribed), `*FLUID FLUX`, `*INITIAL CONDITIONS, TYPE=FLUID PRESSURE`, the surfaces as F2D2/FAX2/F3D3/F3D4 facets (second-order faces are subdivided); implicit and explicit; output PCAV and CVOL (nodal and history).
Linear constraints are eliminated with a master–slave transformation (`src/fem/mpc.rs`),
also with NLGEOM; rigid bodies are then relinearized in every iteration so that
the rigid-body motion is exact.

### Equation solvers

In-house sparse solvers without external dependencies (see "Solvers and performance" below):
`direct` (multifrontal LDLᵀ, default), `indefinite` (1×1/2×2 pivoting,
inertia), `unsymmetric` (multifrontal LU, `*STEP, UNSYMM=YES`; element matrices with
an unsymmetric material tangent are then also assembled in full with `add_unsym`), `iterative`
(PCG with algebraic multigrid, `*STEP, SOLVER=ITERATIVE`), `iterative-ic0`
and `iterative-jacobi`. The eigenvalue solver is blocked subspace iteration with
shift-invert (`src/solver/eigen.rs`); the acoustic steps have an unsymmetric
variant and the projected steps a dense complex QR solver.

## As a Rust library

```rust
use rustyfem::{Material, Model, Sel};

let mut m = Model::new_2d();
let steel = m.material(Material::steel());
m.plate([0.0, 0.0], [2.0, 0.2], 0.01, 0.025, steel)?;
m.fix(Sel::x(0.0));
m.force(Sel::x(2.0), [0.0, -1000.0, 0.0]);
let r = m.solve()?;
println!("{}", r.summary(&m));
```

See `examples/api_cantilever.rs` (`cargo run --release --example api_cantilever`).

## User routines (UMAT and UEL)

RustyFEM has the established interfaces for user materials (UMAT,
`*USER MATERIAL` + `*DEPVAR`) and user elements (UEL, `*USER ELEMENT` +
`*UEL PROPERTY`), with the standard argument lists and their semantics (see
`src/user/umat.rs` and `src/user/uel.rs`). Since Rust does not compile Fortran, there are two routes:

- **In Rust:** implement `rustyfem::user::umat::Umat` (or `uel::Uel`) and
  register it before the run. The UMAT is selected by material name (longest prefix),
  the UEL by the type number n in TYPE=Un:

  ```rust
  use rustyfem::user::umat::{self, Umat, UmatArgs};
  struct MyUmat;
  impl Umat for MyUmat {
      fn umat(&self, a: &mut UmatArgs) -> rustyfem::Result<()> {
          // a.stress, a.statev, a.ddsdde (column-major), a.dstran, a.props, ... as in Fortran
          Ok(())
      }
  }
  umat::register("MY_", std::sync::Arc::new(MyUmat)); // *MATERIAL, NAME=MY_STEEL
  ```

- **Shared library:** a compiled UMAT/UEL (Fortran or C) that
  exports `umat_`/`uel_` (`umat`, `UMAT`, `uel`, `UEL` are also tried) with
  the standard argument list in the Fortran convention (all arguments as pointers,
  integers as `int32`, CMNAME as 80 characters with the hidden length last). Specify
  the library with `RUSTYFEM_USER=path/umat.dll` or
  load it with `rustyfem::user::load_library` and register the routines.
  The library is loaded with the operating system's loader (LoadLibraryW/dlopen) without
  external packages, and the calls are made one at a time.

Bundled examples (`src/user/samples.rs`) that give the same results
as the built-in models up to rounding error (≤ 5e-14, `tests/rigid_user.rs`):
UMAT `EXAMPLE_ELASTIC` (PROPS = E, ν) and `EXAMPLE_MISES` (PROPS = E, ν,
σy₁, ε̄p₁, σy₂, ε̄p₂, ...; STATEV(1) = PEEQ, STATEV(2..) = plastic strain),
UEL U9001 (truss, PROPS = E, A[, ρ]) and U9002 (plane stress quadrilateral
like CPS4, PROPS = E, ν, t).

**More routines.** VUMAT (`*USER MATERIAL` in `*DYNAMIC, EXPLICIT`,
`user/vumat.rs`, examples `EXAMPLE_VELASTIC` and `EXAMPLE_VMISES`) and
the routines in `user/routines.rs`: DLOAD (`*DLOAD` PnNU), DISP
(`*BOUNDARY, USER`), UAMP (`*AMPLITUDE, DEFINITION=USER`), USDFLD
(`*USER DEFINED FIELD`), UFIELD (`*FIELD, USER`), UHARD
(`*PLASTIC, HARDENING=USER`), SIGINI (`*INITIAL CONDITIONS, TYPE=STRESS,
USER`), UEXPAN (`*EXPANSION, USER`), CREEP (`*CREEP, LAW=USER`), HETVAL
(`*HEAT GENERATION`), FILM (`*FILM` FnNU) and UMATHT (`*USER MATERIAL,
TYPE=THERMAL`). They are selected by the longest prefix of the material, set or
amplitude name and are written as Rust traits or loaded from the library
(`dload_`, `disp_`, `uhard_`, `sigini_`, `ufield_`, `uexpan_`, `hetval_`,
`film_`, `creep_`, `umatht_`; USDFLD and UAMP only in Rust). The run context
(`user/context.rs`) gives the routines KSTEP, KINC, TIME, NOEL, NPT, COORDS,
CELENT and the field variables, and PNEWDT < 1 redoes the increment. The examples
`EXAMPLE_*` (`user/samples2.rs`) are tested in `tests/cases/user2/`.

## C interface (`ffi/`)

`cargo build --release -p rustyfem-ffi` builds `rustyfem_ffi.dll` (Windows) /
`librustyfem_ffi.so`/`.dylib`. The original functions `rustyfem_factor`,
`rustyfem_solve`, `rustyfem_info`, `rustyfem_order` and `rustyfem_free` have
unchanged signatures; `rustyfem_factor_indef`, `rustyfem_factor_unsym` and
`rustyfem_inertia` are new (see ARCHITECTURE.md). OpenFEM
(`../openFEManalysis`, `openfem/fem/plugin_rustyfem.py`) uses the library
via ctypes; `tests/fem_core/test_plugin_rustyfem.py` there passes (6 tests)
against the merged code.

## Solvers and performance

The system of equations is solved with an in-house sparse direct solver without external dependencies:

- **Ordering:** geometric nested dissection. The nodes are split recursively along the
  longest axis, and the separators are numbered last.
- **Factorization:** multifrontal LDLᵀ. Each separator becomes a dense block
  that is factorized blockwise (right-looking) with a GEMM microkernel.
- **Parallelism:** independent subtrees, large fronts and element computations
  run on all cores. AVX2/FMA is used if the processor supports it (this
  is determined at run time).

Measured on the development machine (16 threads), solution time including assembly
and stress recovery (before the restructuring into the step framework; the 2D case
with 100,000 CPS4 now takes about 1.0 s instead of 0.7 s because of the more
general core, the 3D cases are unchanged):

| Model | Unknowns | Before (skyline) | Now |
|---|---|---|---|
| 2D plate, 100,000 CPS4 | 202,000 | 5.5 s | 0.7 s |
| 3D block, 4,000 C3D8 | 14,520 | 1.9 s | 0.3 s |
| 3D block, 24,000 C3D8 | 79,380 | 140 s | 2.7 s |
| 3D block, 72,000 C3D8 | 230,640 | – | 13 s (about 3 GB of memory) |

Set `RUSTYFEM_TIMING=1` to see time and memory use per step.

### Indefinite, unsymmetric and iterative solvers

Choose with `--solver <name>` or `RUSTYFEM_SOLVER=<name>` (see ARCHITECTURE.md):

- `indefinite` – multifrontal LDLᵀ with 1×1/2×2 pivoting (Bunch–Kaufman
  threshold within the fronts, delayed pivots). Computes the inertia and warns
  about negative eigenvalues.
- `unsymmetric` – multifrontal LU for unsymmetric matrices; chosen automatically
  by `*STEP, UNSYMM=YES`.
- `iterative` – PCG with algebraic multigrid (smoothed aggregation); chosen
  automatically by `*STEP, SOLVER=ITERATIVE`. `iterative-ic0` and
  `iterative-jacobi` use IC(0) and Jacobi respectively.

Benchmark (`cargo run --release --example solver_bench -- 20 30 40`, C3D8 cube
fixed at the bottom, the full analysis; 16 threads shared with other jobs, so the times
are approximate):

| Model | Unknowns | direct | indefinite | unsymmetric | iterative (AMG) | iterative-ic0 | iterative-jacobi |
|---|---|---|---|---|---|---|---|
| C3D8 20³ | 26,460 | 0.8 s | 1.0 s | 1.0 s | 1.0 s | 0.8 s | 1.2 s |
| C3D8 30³ | 86,490 | 3.3 s | 4.5 s | 5.6 s | 1.9 s | 2.3 s | 3.1 s |
| C3D8 40³ | 201,720 | 15 s | 28 s | 35 s | 8.4 s | 7.8 s | – |

All solvers give the same displacements as `direct` (largest relative difference
4·10⁻¹² for PCG with tolerance 10⁻¹⁰, 6·10⁻¹⁵ for the direct ones). The indefinite
factorization of a shifted 3D Laplace operator (512,000 unknowns, about
10 % negative eigenvalues) takes 44 s and gives exactly the analytical number of
negative eigenvalues (51,205).

## Tests

The tests live in a separate folder, `RustyFEM tests`, next to this one (a separate
crate that uses the engine via `path = "../RustyFEM Repository"`). The paths
below are relative to that folder. The regression cases live in
`tests/cases/<area>/`. A case consists of three files:

- `<case>.inp` – the model;
- `<case>.toml` – the sidecar file with a description, the quantities compared (`fields`)
  and the tolerance (`tol`, per field `tol.<field>`, absolute limit near zero
  `atol.<field>`, normalization floor `floor`/`floor_elem`, `frames`, `steps`,
  `sign_free`); cases that require a previous analysis specify it with
  `global_inp` (submodel) or `old_inp` (restart, `*IMPORT`);
- `<case>.ref` – the baseline, RustyFEM's own results in the text format
  `rustyfem-res 1`.

The error is normalized by the largest magnitude of each quantity (per step and
frame) and compared with the case's tolerance.

```bash
cargo test --release                          # in RustyFEM tests: all tests and cases (tests/regression.rs)
rustyfem check tests/cases/static/case.inp    # one or more cases, with a report per quantity
rustyfem check --all [directory]              # all cases under tests/cases/ (or directory), with a table
rustyfem baseline tests/cases/static/case.inp # rewrite case.ref from RustyFEM
rustyfem baseline --all [directory]
```

To add a case: write `<case>.inp` and `<case>.toml`, run
`rustyfem baseline` on the case, review the result and check in all three
files. When a change intentionally alters the results, the affected
baselines are rewritten with `rustyfem baseline`; the diff in the `.ref` files shows what
has changed.

In addition to the regression cases there are analytical tests (truss, cantilever beams,
patch test, heat conduction in `tests/heat_analytic.rs`, solvers in
`tests/solver.rs`, dynamics, contact, NLGEOM, XFEM) and one test program per
area in `tests/` in `RustyFEM tests`.

## Limitations

This is what RustyFEM does not support, or supports only partially. The list is not
complete but covers what is closest to the existing features.

**Elements**

- No ELBOW31/ELBOW32, no quadratic infinite elements (CINPE5R, CINPS5R,
  CINAX5R, CIN3D12R), no convection/advection elements (DCC*), no
  hybrid variants of CGAX and no axisymmetric beams.
- DCC elements: dispersion control (DCC…D), loads on the DCC elements and AMPLITUDE=/USER on `*MASS FLOW RATE` are missing.
- Cavity radiation: no axisymmetric cavities, `*PERIODIC`, `*CYCLIC`, `*MOTION`, BLOCKING=PARTIAL, RANGE= or PROPERTY= on `*SURFACE`; the increment sequence after MXDEM cutbacks is simplified.
- Electromagnetics: no EMC3D6/EMC3D8, no `*ELECTROMAGNETIC, TRANSIENT`, no `*MAGNETOSTATIC`, no STABILIZATION (nonconducting regions), no frequency ranges or anisotropic/tabulated properties.
- Fracture mechanics in contact is missing: VCCT (`*DEBOND`, `*FRACTURE CRITERION`), bonded contact via `*INITIAL CONDITIONS, TYPE=CONTACT` and output BDSTAT/ENRRTXFN/CSDMG.
- Infinite elements: linear elastic, without mass; in direct dynamic steps
  and `*STEADY STATE DYNAMICS` they give their static stiffness (no quiet
  boundaries). C3D10M and C3D10MH lack NLGEOM.
- Hybrid elements: C3D4H supports linear elasticity, small strains with other
  materials and hyperelasticity with NLGEOM, but not *ELASTIC/plasticity
  with NLGEOM; C3D8RH, CPE4RH, CAX4RH, CPEG*H and C3D10MH do not support
  hyperelastic materials; *HYPERFOAM not in hybrid elements.
- CPEG4 and CPEG4R lack NLGEOM (only CPEG3/6/8/8R).
- The SAXA elements: only linear elasticity without NLGEOM, no element output.
  B31OS/B32OS: no NLGEOM or nonlinear materials, no warping inertia,
  no bimoment term in the geometric stiffness.
- Rigid bodies: analytical surfaces only as master surface node to surface (no
  SURFACE TO SURFACE, FILLET RADIUS, TYPE=USER or *CLEARANCE). The normal of
  a rigid contact surface made of elements is determined in the reference configuration. Do not prescribe
  boundary conditions on the nodes of rigid bodies (only on the reference node).
- User elements: no output per integration point, no
  distributed loads (`*DLOAD`) or FILE=/LINEAR; perturbation steps are not
  supported; PERIOD = 0, TIME(1) = total time. AMATRX is symmetrized.
- NLGEOM for shells (S4, S4R, S3, S3R) is corotational with the linear
  element in a co-rotating frame: small strains in bending/shear
  (error of the order of the strain). `*SHELL SECTION, POISSON=` (other than 0.5) is not supported with
  NLGEOM. Moment loads in 3D give an
  unsymmetric tangent (symmetrized). Prescribed rotations in several
  components in later steps are interpreted as total rotation vectors.
- NLGEOM for S8R/SC8R/SC6R is corotational with Green membrane strain: small
  strains, large rotations; no finite membrane strains in SC8R/SC6R.
- S8R, SC8R and SC6R: only homogeneous `*SHELL SECTION` with linear elastic
  material; `*BUCKLE` only with the membrane force term, no temperature,
  no `OFFSET=`, no layered sections, `POISSON=` only 0.5 for SC8R/SC6R.
- Layered shells (`*SHELL SECTION, COMPOSITE`): all layers must have the same
  material; layer orientation only as an angle. For LAMINA,
  G13 = G23 = G12 is assumed in transverse shear. TEMPERATURE= on
  *SHELL SECTION is not supported; *BUCKLE with through-thickness integration is missing.
- `*SHELL SECTION, OFFSET=` (SPOS, SNEG or a fraction of the thickness) for S4, S4R,
  S3 and S3R: the section is integrated through the thickness about the reference
  surface, and the mass includes the offset. Not with *BUCKLE or explicit dynamics.
- Timoshenko beams with nonlinear material: torsion and transverse shear are
  elastic; *BEAM GENERAL SECTION (except NONLINEAR GENERAL) is not supported;
  temperature gradients in beams are not supported. NONLINEAR GENERAL only
  geometrically linear. Prescribed displacements with plasticity in beams
  converge poorly (force control works).
- `*DYNAMIC` with NLGEOM and large rotations about several axes:
  the rotary inertia is a constant mass matrix, without a gyroscopic term.
- Thermal stress in second-order elements
  uses the element's mean temperature.
- Cohesive elements: MIXED MODE BEHAVIOR only BK and POWER LAW with
  MODE MIX RATIO=ENERGY, SOFTENING only LINEAR/EXPONENTIAL, no
  DEGRADATION=. Gaskets: no unloading curves (DIRECTION=UNLOADING),
  no TYPE=DAMAGE/ELASTIC-PLASTIC.

**Materials**

- No Marlow hyperelasticity, `*HYPERFOAM` with test data, `*CONCRETE
  FAILURE` or anisotropic viscoelasticity (TYPE=ORTHOTROPIC/TRACTION,
  test data).
- TEST DATA INPUT only for the polynomial forms and ARRUDA-BOYCE (not OGDEN or
  VAN DER WAALS).
- `*PLASTIC, RATE=` only with HARDENING=ISOTROPIC without temperature-dependent curves.
- VUMAT is called with one point at a time (NBLOCK = 1), with the stress in
  RustyFEM's Jaumann-rotated basis (not the Green–Naghdi basis), only 3D and
  plane strain.
- Hill and rate-dependent plasticity: no temperature-dependent
  rate parameters, not with LAMINA elasticity; *POTENTIAL can depend on
  temperature but not on field variables.
- Cam-clay, Cap and smeared concrete: not temperature- or field-dependent;
  `*CONCRETE` at most one crack per integration point, numerical tangent.
- Creep: ANAND, DARVEAUX, DOUBLE POWER and *SWELLING not together with
  *PLASTIC; no field-dependent swelling or *POTENTIAL. With NLGEOM in
  CPE4/C3D8 the material gets no temperature.
- Viscoelasticity: large strains not in plane stress, trusses or
  hybrid elements and not with a fully incompressible material (D1 = 0).
  `*STATIC, LONG TERM` gives the relaxed response in every increment (only
  the end of the step is correct).
- CPE6H and C3D15H with D1 = 0 and high pressures lock and do not converge.
- UMAT: DDSDDE is symmetrized without UNSYMM; no `*INITIAL CONDITIONS,
  TYPE=SOLUTION` for STATEV. STATEV is not rotated. The example UMAT
  EXAMPLE_MISES does not support plane stress. USDFLD and UAMP are only available in Rust
  (not in shared libraries).
- Mohr–Coulomb: temperature- and rate-dependent parameters are not supported;
  the meridional eccentricity of the flow potential is always 0.1; numerical
  tangent.
- Concrete damaged plasticity: rate-/temperature-dependent tables are not
  supported; the damage is interpolated at the points of the hardening table; numerical tangent.
- Ductile damage: only von Mises plasticity as the underlying model is
  tested; MIXED MODE BEHAVIOR is not supported. The characteristic length (Hillerborg) is only passed by
  the continuum elements; trusses, beams and shells get L = 1.
- Crushable foam: no `*RATE DEPENDENT`, temperature-dependent data or
  `*LOW DENSITY FOAM`.
- Porous metals: no temperature-dependent data; element deletion at fF
  is missing.
- Deformation plasticity: no temperature-dependent data.
- Hashin: only plane stress; shells get the characteristic length 1; no
  element deletion at full damage.
- FLD/FLSD/MSFLD: only the initiation criterion (no *DAMAGE EVOLUTION), no
  rate-/temperature-dependent limits, no OMEGA filter; no MK.
- Missing entirely: `*CAST IRON PLASTICITY`, `*PARALLEL RHEOLOGICAL FRAMEWORK`,
  `*VISCOELASTIC, NONLINEAR`/`*HYSTERESIS`, `*JOINTED MATERIAL`,
  superelasticity and `*SOFT ROCK PLASTICITY`.
- EOS: no IGNITION AND GROWTH or user-defined form, no
  temperature output, only 3D and plane strain.
- Unsymmetric element matrices are assembled as their symmetric part with the
  symmetric solvers; the unsymmetric solver is chosen automatically for concrete,
  ductile damage with evolution and UMAT with UNSYMM (Mohr–Coulomb and
  Drucker–Prager require `*STEP, UNSYMM=YES`).

**Steps and procedures**

- `*FREQUENCY`/`*BUCKLE` after an NLGEOM step is tested only for
  B31/B32 and S4R. Load stiffness from pressure in the base state is not included in
  `*BUCKLE`, nor is load stiffness for pressure on shells.
- `*DYNAMIC` (implicit) uses lumped mass for all elements except
  B22/B32 (consistent). `*DASHPOT` is not included in the damping in
  implicit `*DYNAMIC` (but it is in explicit dynamics, `*COMPLEX FREQUENCY` and
  the subspace steps).
- Restart only at the end of a step; contact status, friction and
  element deletion are not saved in the restart file. Submodeling only
  node-based (not TYPE=SURFACE). Substructures without load cases, retained
  eigenmodes and recovery of internal results. `*MATRIX GENERATE` without LOAD
  and damping matrices.
- `*STATIC, STABILIZE`: the damping factor is determined in the first increment; no
  adaptive stabilization (element-wise factors).
- `*DIRECT CYCLIC` without low-cycle fatigue (FATIGUE) and NLGEOM;
  `*ANNEAL` only for von Mises materials.
- `*CONTOUR INTEGRAL`: only linear isotropic elastic materials without thermal
  strain or pressure on the crack faces; K and T at 3D front nodes on a free surface
  deviate (J is correct); no TYPE=C(t), CRACK TIP NODES or
  DIRECTION=MERR/KII0. XFEM: the tip enrichment is somewhat soft
  near the crack tip; stresses are reported only for unenriched elements.
- `*DYNAMIC TEMPERATURE-DISPLACEMENT, EXPLICIT`: no bulk viscosity,
  NLGEOM, contact or heat generation from plastic dissipation; the
  element-by-element stability limit is estimated (use a fixed Δt).
- `*COUPLED THERMAL-ELECTRICAL` only fully coupled (not ELECTRICAL).
  Acoustics in explicit dynamics: linear structure, no infinite elements.
- Creep is not active in `*SOILS`.
- SPH, DEM, Euler and ALE (explicit dynamics): SPH FORMULATION=NSPH/XSPH,
  TENSILE INSTABILITY CONTROL, particle generators and outlets (`*PARTICLE
  GENERATOR`, `*PARTICLE OUTLET`), `*SPH SURFACE BEHAVIOR` and conversion
  based on strain/stress criteria or at t > 0 are not supported;
  the stability limit for particles and Eulerian elements is estimated (use a
  fixed Δt). DEM: no clusters, distributions or swelling.
  Euler: no hourglass control, no `*EULERIAN BOUNDARY`/mesh motion,
  the nodal velocities are advected to first order; the CEL contact is a
  penalty method on the solids' surfaces (no shells, no friction). ALE: only
  interior nodes are smoothed (the boundary nodes follow the material), no `*ADAPTIVE MESH
  CONSTRAINT`, materials with a total deformation gradient (*EOS,
  hyperelasticity) are not supported in the ALE domain.
- `*DYNAMIC, EXPLICIT`, simplifications and what is missing:
  - Stability limit: the improved method (`IMPROVED DT METHOD`,
    default) exists for shells and membranes but not for 3D solids, and there
    is no global estimate.
    The beams' limit is taken from the element's highest natural frequency.
  - The hourglass relaxation (`RELAX STIFFNESS`) is calibrated on monotonic
    ramps; for oscillating hourglass modes it is approximate.
    `*SECTION CONTROLS` only supports `HOURGLASS=RELAX STIFFNESS` and
    `STIFFNESS`.
  - Contact uses a mass-based penalty force F = m/Δt² (0.5436 p +
    0.0422 Δp); kinematic contact and element-based penalty stiffness
    use the same penalty force.
  - Linear constraints (*TIE, *COUPLING, *MPC, *EQUATION, *RIGID BODY)
    are not eliminated (only the locked components of connector elements and
    acoustic–structural *TIE).
- `SOLVER=` only accepts DIRECT and ITERATIVE.

**Implicit dynamics**

- The half-increment tolerance for moments is HAFTOL × characteristic element length;
  not verified for cases where the moment governs. INCREMENTATION=CONSERVATIVE
  in nonlinear problems is partly assumed.
- TRANSIENT FIDELITY with contact lacks IMPACT=AVERAGE TIME and
  is used as IMPACT=NO. Constraints (*TIE, *MPC, *EQUATION) are eliminated with a
  fixed transformation; rigid bodies and connector elements are relinearized
  with NLGEOM.

**Contact and constraints**

- `*TIE` surface to surface uses dual mortar. The contact segment weighting
  applies to quadrilaterals (C3D8 faces) and two-node edges; triangles and
  quadratic faces use dual mortar.
- The contact tangent with finite sliding is not consistent; with
  the standard convergence criteria a small residual error may remain
  (tighten the criterion with `*CONTROLS`).
- The sticking state in friction (including ROUGH) is penalty-regularized, not based on
  Lagrange multipliers.
- CPRESS is only output at the slave nodes (not at the master nodes).
- No `*CONTACT CLEARANCE` (general contact), `*CHANGE FRICTION`,
  `*PRESSURE PENETRATION` or wear.
- Cohesive contact: only `*CONTACT PAIR` node to surface in implicit steps (not
  general contact or explicit dynamics), an explicit stiffness is required (no
  default stiffness), not TYPE=COUPLED, ELIGIBILITY/COHERE only ORIGINAL
  CONTACTS, no friction; in compression a penalty stiffness 1e3·Knn instead of
  hard contact.
- Cavities: no `*FLUID EXCHANGE`, `*FLUID INFLATOR`, gas mixtures
  (MIXTURE=), ADIABATIC, UFLUID or field-dependent fluid data; `*FLUID FLUX`
  only in implicit steps; perturbation steps do not see the cavity.

**Heat**

- No cavity radiation and no heat generation from plastic work.

**Input, output and other**

- The results are written as text (`--out`, format `rustyfem-res 1`), VTK, CSV and `.dat` (`*NODE PRINT`, `*EL PRINT`,
  `*ENERGY PRINT`). Other output keywords are accepted but do not control what is
  written.
- `*PARAMETER TABLE`, `*TABLE COLLECTION`, `*PARAMETER DEPENDENCE` and
  `INPUT=` on `*PARAMETER` are not supported.
- The `.rusty` language only has structured meshes of plates, quadrilaterals and
  blocks; no CAD import and no in-house mesher for arbitrary geometry.
- The direct solver needs a lot of memory for large 3D models: about 3 GB at
  230,000 unknowns, and the requirement grows faster than the model size. For
  millions of unknowns the iterative solver is needed, which only handles
  symmetric, positive definite systems.
