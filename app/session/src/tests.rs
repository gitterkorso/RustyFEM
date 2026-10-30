use super::*;
use rustyfem::fem::output::AnalysisOutput;
use std::sync::Mutex;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn max_u(o: &AnalysisOutput) -> f64 {
    let fr = o.steps.last().unwrap().frames.last().unwrap();
    fr.u.iter().map(|u| (u[0] * u[0] + u[1] * u[1] + u[2] * u[2]).sqrt()).fold(0.0, f64::max)
}

#[test]
fn session_runs_like_engine_with_progress() {
    let path = root().join("app/session/tests/data/cpe4_ett_element_inhomogen.inp");
    let s = Session::open(&path).unwrap();
    let direct = rustyfem::steps::run(&rustyfem::inp::read_file(&path).unwrap().model, &RunOptions::default()).unwrap().0;
    let events = Arc::new(Mutex::new(Vec::new()));
    let ev = events.clone();
    let out = run_analysis(s.model(), None, false, Arc::new(AtomicBool::new(false)), move |e| ev.lock().unwrap().push(e)).unwrap();
    assert_eq!(max_u(&out), max_u(&direct));
    let ev = events.lock().unwrap();
    assert!(ev.iter().any(|e| matches!(e, RunEvent::Step { .. })));
    assert!(ev.iter().any(|e| matches!(e, RunEvent::Increment { .. })));
}

#[test]
fn cancel_stops_run() {
    let s = Session::open(&root().join("app/session/tests/data/cpe4_ett_element_inhomogen.inp")).unwrap();
    let err = run_analysis(s.model(), None, false, Arc::new(AtomicBool::new(true)), |_| {}).unwrap_err();
    assert!(err.contains(rustyfem::progress::CANCELLED), "{err}");
}

#[test]
fn fem_edit_undo_redo_and_mesh() {
    let mut s = Session::new_fem(NEW_MODEL).unwrap();
    let n0 = s.model().nodes.len();
    assert!(!s.mesh().tris.is_empty());
    let text = NEW_MODEL.replace("size=0.05", "size=0.1");
    assert!(s.apply(Edit::FemText { text }).unwrap());
    let n1 = s.model().nodes.len();
    assert!(n1 < n0);
    s.undo().unwrap();
    assert_eq!(s.model().nodes.len(), n0);
    s.redo().unwrap();
    assert_eq!(s.model().nodes.len(), n1);
    assert!(s.apply(Edit::FemText { text: "plate x=0..1".into() }).is_err());
    assert_eq!(s.model().nodes.len(), n1);
}

#[test]
fn inp_edit_and_project_roundtrip() {
    let path = root().join("app/session/tests/data/cpe4_ett_element_inhomogen.inp");
    let mut s = Session::open(&path).unwrap();
    let old = s.model().steps[0].max_increments;
    s.apply(Edit::MaxIncrements { step: 0, value: 77 }).unwrap();
    assert_eq!(s.model().steps[0].max_increments, 77);
    let out = run_analysis(s.model(), None, true, Arc::new(AtomicBool::new(false)), |_| {}).unwrap();
    s.set_results(out);
    let bytes = s.to_project();
    let p = std::env::temp_dir().join("rustyfem_session_test.rustyproj");
    std::fs::write(&p, &bytes).unwrap();
    let t = Session::open(&p).unwrap();
    assert_eq!(t.model().steps[0].max_increments, 77);
    assert_eq!(max_u(t.results.as_ref().unwrap()), max_u(s.results.as_ref().unwrap()));
    assert_eq!(t.field_values(0, 0, "U", "Magnitude").unwrap(), s.field_values(0, 0, "U", "Magnitude").unwrap());
    let mut t = t;
    t.undo().unwrap();
    assert_eq!(t.model().steps[0].max_increments, old);
}

#[test]
fn surface_of_solid_is_exterior_only() {
    let text = "model 3d\nmaterial steel E=210e9 nu=0.3\nbox x=0..1 y=0..1 z=0..1 size=0.5\nfix x=0\nload x=1 fx=1\n";
    let s = Session::new_fem(text).unwrap();
    // 2×2×2 hexaedrar: 24 yttre ytor, 2 trianglar var
    assert_eq!(s.mesh().tri_elem.len(), 48);
    let est = memory::estimate(s.model(), 1e12, None, 1);
    assert!(est.fits && est.direct_bytes > 0.0);
}

#[test]
fn project_include_resolved_from_file_contents() {
    let dir = std::env::temp_dir().join(format!("rustyfem_inc_{}", std::process::id()));
    std::fs::create_dir_all(dir.join("del")).unwrap();
    let case = std::fs::read_to_string(root().join("app/session/tests/data/cpe4_ett_element_inhomogen.inp")).unwrap();
    // flytta allt före första *STEP till en inkluderad fil
    let k = case.find("*STEP").or_else(|| case.find("*Step")).unwrap();
    std::fs::write(dir.join("del/modell.inp"), &case[..k]).unwrap();
    std::fs::write(dir.join("huvud.inp"), format!("*INCLUDE, INPUT=del/modell.inp\n{}", &case[k..])).unwrap();
    let s = Session::open(&dir.join("huvud.inp")).unwrap();
    let bytes = s.to_project();
    std::fs::remove_dir_all(&dir).unwrap();
    // inkluderade filen finns inte längre på disk: läses ur projektfilen
    let p = std::env::temp_dir().join("rustyfem_inc_test.rustyproj");
    std::fs::write(&p, bytes).unwrap();
    let t = Session::open(&p).unwrap();
    assert_eq!(t.model().nodes.len(), s.model().nodes.len());
    assert_eq!(t.model().elements.len(), s.model().elements.len());
}

#[test]
fn units_default_guess_and_roundtrip() {
    let mut s = Session::new_fem(NEW_MODEL).unwrap();
    assert_eq!(s.units, UnitSystem::Si);
    assert_eq!(s.info().unit_symbols.stress, "Pa");
    s.set_units(UnitSystem::SiMm);
    assert!(s.dirty);
    let p = Project::from_bytes(&s.to_project()).unwrap();
    assert_eq!(p.units, "si_mm");
    assert_eq!(Session::from_project(p, None).unwrap().units, UnitSystem::SiMm);

    // äldre projektfil: etiketten ger standardvalet för källan
    let mut p = Project::from_bytes(&s.to_project()).unwrap();
    p.units = units::LEGACY_LABEL.into();
    assert_eq!(Session::from_project(p, None).unwrap().units, UnitSystem::Si);

    let mut m = FeModel::default();
    m.materials.push(rustyfem::fem::model::MaterialDef::elastic("STEEL", 210000.0, 0.3, 7.85e-9));
    assert_eq!(UnitSystem::guess(&m), UnitSystem::SiMm);
    m.materials[0] = rustyfem::fem::model::MaterialDef::elastic("STEEL", 29e6, 0.3, 7.3e-4);
    assert_eq!(UnitSystem::guess(&m), UnitSystem::Us);
    assert_eq!(UnitSystem::guess(&FeModel::default()), UnitSystem::None);
}

#[test]
fn incomplete_script_is_built_step_by_step() {
    // tom modell: går att öppna men inte att lösa
    let mut s = Session::new_fem(EMPTY_3D).unwrap();
    assert_eq!(s.info().script_dim, Some(3));
    assert!(s.problem().unwrap().contains("inga element"), "{:?}", s.problem());
    assert!(s.model().steps.is_empty());
    // stöd före geometri
    let t = format!("{EMPTY_2D}fix x=0\n");
    s.apply(Edit::FemText { text: t.clone() }).unwrap();
    assert_eq!(s.info().script_dim, Some(2));
    assert!(s.problem().is_some());
    // geometri som stödet träffar: komplett
    let t = format!("{t}plate x=0..1 y=0..0.1 t=0.01 size=0.05\nload x=1 fy=-100\n");
    s.apply(Edit::FemText { text: t.clone() }).unwrap();
    assert_eq!(s.problem(), None);
    assert_eq!(s.model().steps.len(), 1);
    // stöd som inte träffar några noder: byggs, men med ett besked
    s.apply(Edit::FemText { text: t.replace("fix x=0", "fix x=5") }).unwrap();
    assert!(s.problem().unwrap().contains("träffade inga noder"), "{:?}", s.problem());
    // fel i skriptet avvisas och modellen lämnas orörd
    let n = s.model().nodes.len();
    assert!(s.apply(Edit::FemText { text: "model 2d\nplatta x=0..1\n".into() }).unwrap_err().contains("rad 2"));
    assert_eq!(s.model().nodes.len(), n);
    s.undo().unwrap();
    assert_eq!(s.problem(), None);
}
