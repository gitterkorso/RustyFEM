//! Samma konsol som konsol_platta.rusty, men byggd direkt i Rust.
//! Kör: cargo run --release --example api_konsol

use rustyfem::{Material, Model, Sel};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut m = Model::new_2d();
    let steel = m.material(Material::steel());
    m.plate([0.0, 0.0], [2.0, 0.2], 0.01, 0.025, steel)?;
    m.fix(Sel::x(0.0));
    m.force(Sel::x(2.0), [0.0, -1000.0, 0.0]);

    let r = m.solve()?;
    print!("{}", r.summary(&m));
    r.write_vtk(&m, std::path::Path::new("api_konsol.vtk"))?;
    Ok(())
}
