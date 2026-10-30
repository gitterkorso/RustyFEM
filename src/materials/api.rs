//! Materialgränssnittet: spänningsuppdatering med tillstånd per integrationspunkt.
//!
//! En materialmodell byggs från en `MaterialDef` (beteendena *ELASTIC,
//! *PLASTIC, ... som de står i .inp-filen) av en registrerad `MaterialFactory`.
//! Elementet anropar `update` i varje integrationspunkt med töjningen i
//! integrationspunktens spänningstillstånd (`StressMode`).
//!
//! Komponentordning och skjuvtöjningar som ingenjörstöjningar (γ):
//! - `ThreeD`:      [11, 22, 33, 12, 13, 23]
//! - `PlaneStrain`: [11, 22, 33, 12]   (ε33 = 0)
//! - `Axisymmetric`:[11, 22, 33, 12]   (33 = ringled)
//! - `PlaneStress`: [11, 22, 12]       (σ33 = 0)
//! - `Uniaxial`:    [11]               (stänger)

use crate::fem::model::MaterialDef;
use crate::model::{fail, Result};
use std::sync::Arc;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StressMode {
    ThreeD,
    PlaneStrain,
    Axisymmetric,
    PlaneStress,
    Uniaxial,
}

impl StressMode {
    pub fn ncomp(self) -> usize {
        match self {
            StressMode::ThreeD => 6,
            StressMode::PlaneStrain | StressMode::Axisymmetric => 4,
            StressMode::PlaneStress => 3,
            StressMode::Uniaxial => 1,
        }
    }
}

/// Indata till en spänningsuppdatering.
pub struct MatInput<'a> {
    pub mode: StressMode,
    /// Total (liten) töjning vid inkrementets slut.
    pub strain: &'a [f64],
    /// Töjningsinkrement sedan inkrementets början.
    pub dstrain: &'a [f64],
    /// Spänning vid inkrementets början.
    pub stress_old: &'a [f64],
    pub time: f64,
    pub dt: f64,
    /// Temperatur vid inkrementets slut och dess ändring (0 om ingen temperatur).
    pub temperature: f64,
    pub dtemp: f64,
    /// Deformationsgradient vid NLGEOM (radvis), annars `None`.
    pub defgrad: Option<[[f64; 3]; 3]>,
}

/// Resultat av en spänningsuppdatering.
#[derive(Debug, Clone, Default)]
pub struct MatOutput {
    pub stress: Vec<f64>,
    /// Konsistent tangent dσ/dε, ncomp × ncomp radvis.
    pub tangent: Vec<f64>,
}

pub trait MaterialModel: Send + Sync {
    /// Modellens namn för meddelanden, t.ex. "ELASTIC".
    fn name(&self) -> &str;

    /// Linjärt material: spänningen är `D ε` med konstant `D` och inget tillstånd.
    fn is_linear(&self) -> bool {
        false
    }

    /// Linjärt svar när temperaturen är given (t.ex. termisk töjning ovanpå
    /// ett linjärt material): spänningen är affin i töjningen inom ett
    /// inkrement. Används av den kopplade analysen (`steps/heat_coupled.rs`)
    /// för att lösa den mekaniska delen med en enda lösning.
    fn is_linear_at_given_temperature(&self) -> bool {
        self.is_linear()
    }

    /// Antal tillståndsvariabler per integrationspunkt.
    fn n_state(&self) -> usize {
        0
    }

    fn init_state(&self, _state: &mut [f64]) {}

    /// Starttillstånd med begynnelsevillkor i integrationspunkten
    /// (*INITIAL CONDITIONS, TYPE=STRESS och TYPE=RATIO, se
    /// `initial_state.rs`): spänningen `stress` (komponenterna för `mode`)
    /// och portalet e0 om det är givet. Anropas efter `init_state`; elementets
    /// spänning är redan satt. Standard: fel om spänningen inte är noll, eftersom
    /// materialet då inte räknar med en begynnelsespänning.
    fn init_state_at(&self, _mode: StressMode, stress: &[f64], _void_ratio: Option<f64>, _state: &mut [f64]) -> Result<()> {
        if stress.iter().any(|&s| s != 0.0) {
            return fail(format!("materialmodellen {} stöder inte begynnelsespänning (*INITIAL CONDITIONS, TYPE=STRESS)", self.name()));
        }
        Ok(())
    }

    /// (E, ν) för element som behöver elasticitetsmoduler direkt (balkar, stänger).
    fn elastic_moduli(&self) -> Option<(f64, f64)> {
        None
    }

    /// Spänningsuppdatering: ny spänning, tangent och nytt tillstånd.
    fn update(&self, input: &MatInput, state_old: &[f64], state_new: &mut [f64]) -> Result<MatOutput>;

    /// Spänningsuppdatering med elementets karakteristiska längd `length`
    /// (Hillerborg: mjuknande material vars brottenergi eller brottförskjutning
    /// ska regulariseras med elementstorleken). Kontinuumelementen
    /// (`elements::common::response`) anropar den här. Standard: `update`.
    fn update_with_length(&self, input: &MatInput, _length: f64, state_old: &[f64], state_new: &mut [f64]) -> Result<MatOutput> {
        self.update(input, state_old, state_new)
    }

    /// Materialet har en osymmetrisk tangent som kräver den osymmetriska
    /// lösaren (väljs automatiskt, t.ex. för *CONCRETE DAMAGED PLASTICITY
    /// även utan *STEP, UNSYMM=YES). Standard: nej.
    fn requires_unsymmetric(&self) -> bool {
        false
    }

    /// Roterar tensoriella tillståndsvariabler (plastisk töjning, mottensorer,
    /// ...) med den inkrementella rotationen `r` vid NLGEOM (x' = R x Rᵀ), så att
    /// de följer materialet som spänningen gör (Hughes–Winget). Standard: inget
    /// tillstånd att rotera.
    fn rotate_state(&self, _r: &[[f64; 3]; 3], _state: &mut [f64]) {}

    /// *ANNEAL (`steps/anneal.rs`): gör materialpunkten spänningsfri i den
    /// aktuella totala töjningen `strain` (11, 22, 33, 12, 13, 23 med
    /// ingenjörsskjuvning) och nollställer hårdnandet. Returnerar `false` om
    /// materialet inte stöder glödgning. Standard: nej.
    fn anneal(&self, _strain: &[f64; 6], _state: &mut [f64]) -> bool {
        false
    }

    /// Aktuell (isentropisk) kompressionsmodul ρc² ur materialpunktens
    /// tillstånd för stabilitetsgränsen i explicit dynamik (t.ex. *EOS).
    /// Standard: `None` (den elastiska kompressionsmodulen används).
    fn current_bulk_modulus(&self, _state: &[f64]) -> Option<f64> {
        None
    }

    /// Explicit analys integrerar materialet i varje integrationspunkt utan
    /// B-bar/F-bar (full integration även av volymdelen) för C3D8 med NLGEOM
    /// (*EOS: trycket varierar mellan punkterna). Standard: `false`.
    fn explicit_full_integration(&self) -> bool {
        false
    }

    /// Begynnelsetillstånd för tillståndsekvationer (*EOS): den specifika
    /// energin `em` (*INITIAL CONDITIONS, TYPE=SPECIFIC ENERGY) eller
    /// begynnelsetemperaturen med absoluta nollpunkten. Returnerar
    /// (begynnelsetrycket p₀ (spänningen −p₀ I), specifik inre energi för
    /// ALLIE), `None` = inget att göra. Standard: `None`.
    fn init_energy(&self, _state: &mut [f64], _em: Option<f64>, _temperature: Option<f64>, _abs_zero: f64) -> Result<Option<(f64, f64)>> {
        Ok(None)
    }

    /// Område gaps1: integrationspunktens läge `x` och elementets
    /// karakteristiska längd i analysens början (*EOS, TYPE=JWL: detonationsvågens
    /// ankomsttid). Anropas av `eos::apply_initial` efter `init_energy`.
    /// Standard: inget.
    fn init_at_point(&self, _state: &mut [f64], _x: [f64; 3], _length: f64) {}

    /// Område materials5: begynnelsevärdet r av den relativa densiteten
    /// (*INITIAL CONDITIONS, TYPE=RELATIVE DENSITY) för porösa material
    /// (*POROUS METAL PLASTICITY). Anropas av
    /// `porous_metal::apply_relative_density` efter `init_state`. Returnerar
    /// `false` om materialet inte har någon relativ densitet. Standard: nej.
    fn init_relative_density(&self, _r: f64, _state: &mut [f64]) -> bool {
        false
    }

    /// Elastisk (linjär) tangent för linjär styvhet och perturbationssteg.
    fn tangent_linear(&self, mode: StressMode) -> Result<Vec<f64>> {
        let n = mode.ncomp();
        let z = vec![0.0; n];
        let mut st = vec![0.0; self.n_state()];
        self.init_state(&mut st);
        let mut st2 = st.clone();
        let inp = MatInput { mode, strain: &z, dstrain: &z, stress_old: &z, time: 0.0, dt: 0.0, temperature: 0.0, dtemp: 0.0, defgrad: None };
        Ok(self.update(&inp, &st, &mut st2)?.tangent)
    }

    /// Namn på extra utdata per integrationspunkt, t.ex. ["PEEQ"].
    fn output_names(&self) -> &[&'static str] {
        &[]
    }

    /// Värden för `output_names` ur tillståndet.
    fn output_values(&self, _state: &[f64]) -> Vec<f64> {
        Vec::new()
    }

    /// Utdatafält med komponentnamn för spänningstillståndet `mode`, t.ex.
    /// `[("PEEQ", ["PEEQ"]), ("PE", ["PE11", "PE22", ...])]`.
    /// Standard: `output_names` som skalära fält.
    fn output_fields(&self, _mode: StressMode) -> Vec<(&'static str, Vec<&'static str>)> {
        self.output_names().iter().map(|n| (*n, vec![*n])).collect()
    }

    /// Värden för `output_fields`: alla komponenter i följd.
    fn output_field_values(&self, _mode: StressMode, state: &[f64]) -> Vec<f64> {
        self.output_values(state)
    }

    /// Hyperelastiska material: uppdelningen i deviatorisk och volymetrisk del
    /// för element som behandlar volymdelen själva (F-bar vid NLGEOM, plan
    /// spänning, hybridelement). Se `hyperelastic_split.rs`.
    fn hyperelastic(&self) -> Option<&dyn super::hyperelastic_split::HyperModel> {
        None
    }

    /// Tidsberoende material (krypning, viskoelasticitet): läge per steg och
    /// övervakning av tidsintegrationen för *VISCO (CETOL). Material som
    /// omsluter andra (t.ex. termisk expansion) ska vidarebefordra anropet.
    /// Se `time_dependence.rs`.
    fn time_dependence(&self) -> Option<&super::time_dependence::TimeDependence> {
        None
    }

    /// Viskoelasticitet i frekvensplanet (*VISCOELASTIC, FREQUENCY=...) för
    /// *STEADY STATE DYNAMICS, DIRECT: (ΔGs, ΔKs, Gℓ, Kℓ) vid vinkelfrekvensen
    /// `omega` och temperaturen `temperature`, där ΔGs och ΔKs är
    /// lagringsmodulernas tillskott till modulerna i `tangent_linear` och Gℓ,
    /// Kℓ förlustmodulerna (se `viscoelastic_freq.rs`). Standard: inget.
    fn frequency_moduli(&self, _omega: f64, _temperature: f64) -> Option<[f64; 4]> {
        None
    }

    /// Traction–separationslag för kohesiva element (*ELASTIC, TYPE=TRACTION,
    /// se `cohesive.rs`). Standard: inget.
    fn cohesive(&self) -> Option<&super::cohesive::TractionSeparation> {
        None
    }
}

/// Bygger en materialmodell från en materialdefinition.
///
/// `build` returnerar `Ok(None)` om fabriken inte känner igen materialet (t.ex.
/// fel TYPE= på *ELASTIC). Bland fabriker som svarar väljs den med högst
/// `priority`. Alla beteenden i materialet måste förbrukas av den valda
/// fabriken (`consumes`) eller av en passiv post (`build: None`), annars blir
/// det fel – inget beteende ignoreras tyst.
pub struct MaterialFactory {
    pub name: &'static str,
    /// Beteenden (nyckelord utan '*') som fabriken använder.
    pub consumes: &'static [&'static str],
    pub priority: i32,
    pub build: Option<fn(&MaterialDef) -> Result<Option<Arc<dyn MaterialModel>>>>,
}

/// Hjälp för materialmodeller: radvis matris gånger vektor.
pub fn matvec(d: &[f64], x: &[f64]) -> Vec<f64> {
    let n = x.len();
    (0..n).map(|i| (0..n).map(|j| d[i * n + j] * x[j]).sum()).collect()
}

pub fn unsupported<T>(def: &MaterialDef, what: &str) -> Result<T> {
    fail(format!("material {}: {what}", def.name))
}
