//! Elementgränssnittet: allt ett element kan bidra med till en analys.
//!
//! Ett element implementerar `ElementType` för ett eller flera elementnamn och
//! registreras i sin familjs `TYPES`-lista (se `elements/mod.rs`). Alla metoder
//! utom de grundläggande har standardimplementationer som returnerar felet
//! "inte implementerat för <element>", så en ny elementtyp kan börja med bara
//! linjär styvhet och växa metod för metod.
//!
//! Konventioner:
//! - Elementets frihetsgradsvektor är nodvis: för nod 0 alla `node_dofs(0)`,
//!   sedan nod 1 osv. Matriser är täta, radvisa n×n (`Vec<f64>`).
//! - Koordinater (`ElemCtx::x`) är referenskonfigurationen. `u` är total
//!   förskjutning från referensläget, `du` ändringen sedan inkrementets början.
//! - Elementets tillstånd (`state`) är en fri `Vec<f64>` som elementet själv
//!   delar upp, typiskt per integrationspunkt: spänning + materialets tillstånd.
//!   Kärnan sparar ett godkänt (`state_old`) och ett prövat (`state_new`) exemplar.

use crate::fem::dof::Dof;
use crate::fem::model::{DistLoad, SectionDef};
use crate::materials::MaterialModel;
use crate::model::{fail, Result};

/// Elementfamilj (för utdata, ytor och kontroller).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Family {
    /// Plana kontinuumelement (plan spänning/töjning).
    Continuum2D,
    /// Axisymmetriska kontinuumelement.
    Axisymmetric,
    Continuum3D,
    Truss,
    Beam,
    Shell,
    Membrane,
    /// Fjädrar, dämpare, massor, kopplingselement.
    Connector,
    /// Värmeledning (endast NT11).
    Heat,
    Other,
}

/// Massmatris: konsistent eller diagonal (lumped).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MassKind {
    Consistent,
    Lumped,
}

/// Allt elementet behöver veta om sin omgivning.
#[derive(Clone)]
pub struct ElemCtx<'a> {
    /// Elementets index i modellen och elementnummer.
    pub index: usize,
    pub label: i64,
    /// Nodkoordinater i referensläget.
    pub x: Vec<[f64; 3]>,
    pub section: Option<&'a SectionDef>,
    pub material: Option<&'a dyn MaterialModel>,
    /// Densitet från *DENSITY (0 om den saknas).
    pub density: f64,
    /// Geometriskt olinjär analys (NLGEOM=YES).
    pub nlgeom: bool,
    /// Total tid vid inkrementets slut och inkrementets längd.
    pub time: f64,
    pub dt: f64,
    /// Nodtemperaturer (fördefinierat temperaturfält, *TEMPERATURE och
    /// *INITIAL CONDITIONS, TYPE=TEMPERATURE) vid inkrementets slut respektive
    /// början, i elementets nodordning. Tomma om modellen saknar temperaturfält.
    /// Har modellen temperaturgradienter genom tjockleken (tredje fältet på
    /// *TEMPERATURE) får skal och balkar 2·nn värden: nodtemperaturerna följda
    /// av gradienterna (`fem::assembly::Prepared::ctx`).
    pub temp: Vec<f64>,
    pub temp_old: Vec<f64>,
    /// Lokal materialorientering från tvärsnittets ORIENTATION= (*ORIENTATION).
    pub orientation: Option<&'a crate::materials::anisotropic::Orientation>,
}

impl<'a> ElemCtx<'a> {
    /// Temperatur och temperaturändring i en integrationspunkt med
    /// formfunktionerna `n` ((0, 0) om modellen saknar temperaturfält).
    ///
    /// Första ordningens element använder medelvärdet av elementets
    /// nodtemperaturer för termisk töjning (tests/cases/materials: C3D8
    /// med sned mesh, CPS4 och T2D2).
    /// Andra ordningens och inkompatibla kontinuumelement använder i stället
    /// elementets temperaturinterpolation (`Solid::temperatures`).
    pub fn temperature_at(&self, n: &[f64]) -> (f64, f64) {
        if self.temp.len() != n.len() || self.temp_old.len() != n.len() {
            return (0.0, 0.0);
        }
        let m = n.len() as f64;
        let t: f64 = self.temp.iter().sum::<f64>() / m;
        let t0: f64 = self.temp_old.iter().sum::<f64>() / m;
        (t, t - t0)
    }

    /// Temperatur och temperaturändring interpolerade med formfunktionerna `n`.
    pub fn temperature_interpolated(&self, n: &[f64]) -> (f64, f64) {
        if self.temp.len() != n.len() || self.temp_old.len() != n.len() {
            return (0.0, 0.0);
        }
        let t: f64 = n.iter().zip(&self.temp).map(|(a, b)| a * b).sum();
        let t0: f64 = n.iter().zip(&self.temp_old).map(|(a, b)| a * b).sum();
        (t, t - t0)
    }

    pub fn section(&self, elem: &str) -> Result<&'a SectionDef> {
        match self.section {
            Some(s) => Ok(s),
            None => fail(format!("element {} ({elem}) saknar tvärsnitt (*SOLID SECTION, *BEAM SECTION, ...)", self.label)),
        }
    }

    pub fn material(&self, elem: &str) -> Result<&'a dyn MaterialModel> {
        match self.material {
            Some(m) => Ok(m),
            None => fail(format!("element {} ({elem}) saknar material", self.label)),
        }
    }
}

/// Inre kraft och (valfritt) tangentstyvhet.
#[derive(Debug, Clone, Default)]
pub struct ElemResponse {
    pub fint: Vec<f64>,
    pub k: Option<Vec<f64>>,
}

/// Ett utdatafält per integrationspunkt, t.ex. S med komponenterna S11, S22, S33, S12.
#[derive(Debug, Clone, Default)]
pub struct IpField {
    /// Fältnamn: "S", "E", "PEEQ", "SF", "SM", ...
    pub name: String,
    /// Komponentnamn (för skalärer samma som fältnamnet).
    pub labels: Vec<String>,
    /// (integrationspunkt 1.., sektionspunkt 0 = ingen, värden)
    pub points: Vec<(usize, usize, Vec<f64>)>,
}

impl IpField {
    pub fn new(name: &str, labels: &[&str]) -> Self {
        IpField { name: name.into(), labels: labels.iter().map(|s| s.to_string()).collect(), points: Vec::new() }
    }
}

/// Fel för metoder som en elementtyp ännu inte har.
pub fn not_implemented<T>(what: &str, elem: &str) -> Result<T> {
    fail(format!("{what} är inte implementerat för {elem}"))
}

/// En elementtyp, namngiven enligt .inp-formatet.
pub trait ElementType: Send + Sync {
    /// Elementnamnet, t.ex. "CPS4".
    fn name(&self) -> &'static str;

    fn family(&self) -> Family;

    /// Rumsdimension (2 för plana element, 3 annars).
    fn dim(&self) -> usize;

    fn n_nodes(&self) -> usize;

    /// Antal noder på *ELEMENT-raden. Standard: alla noder. Element vars sista
    /// noder anges i tvärsnittet (referensnoden för generaliserad plan töjning,
    /// *SOLID SECTION, REF NODE=) har färre; de läggs till av `FeModel::finalize`
    /// från `section_nodes`.
    fn n_connectivity(&self) -> usize {
        self.n_nodes()
    }

    /// Noder (nodnummer i elementets instans) som läggs sist i elementets
    /// nodlista utifrån tvärsnittet. Standard: inga.
    fn section_nodes(&self, _section: &SectionDef) -> Result<Vec<i64>> {
        Ok(Vec::new())
    }

    /// Antal interna noder som `FeModel::finalize` skapar för elementet (i
    /// elementets tyngdpunkt, sist i nodlistan),
    /// t.ex. för hybridelementens tryckvariabler. Standard: inga. Elementet ska då
    /// ha `n_connectivity()` = antalet noder på *ELEMENT-raden.
    fn internal_nodes(&self) -> usize {
        0
    }

    /// Frihetsgrader för en nod (frihetsgradsnummer, se `fem::dof`). 1–7 och 11 stöds.
    fn node_dofs(&self, local_node: usize) -> &'static [Dof];

    /// Antal frihetsgrader i elementet.
    fn n_dofs(&self) -> usize {
        (0..self.n_nodes()).map(|i| self.node_dofs(i).len()).sum()
    }

    /// Antal integrationspunkter (för utdata och tillstånd).
    fn n_ips(&self) -> usize {
        0
    }

    /// VTK-celltyp för ParaView (0 = ingen).
    fn vtk_type(&self) -> u8 {
        0
    }

    /// Elementets ytor i .inp-formatets numrering (yta 1 = index 0). Används för
    /// *DLOAD P1..P6, randytor och kontakt.
    fn faces(&self) -> &'static [&'static [usize]] {
        &[]
    }

    /// Startvärde för elementets tillstånd (t.ex. noll spänning och materialets
    /// starttillstånd i varje integrationspunkt).
    fn init_state(&self, _ctx: &ElemCtx) -> Result<Vec<f64>> {
        Ok(Vec::new())
    }

    /// Linjär styvhetsmatris i globala koordinater.
    fn stiffness(&self, _ctx: &ElemCtx) -> Result<Vec<f64>> {
        not_implemented("linjär styvhet", self.name())
    }

    /// Inre kraft och tangent givet förskjutning och materialtillstånd.
    ///
    /// Standard: `K u` med den linjära styvheten, men bara om materialet är
    /// linjärt och analysen är geometriskt linjär. Element som ska användas
    /// med olinjära material eller NLGEOM måste implementera metoden.
    fn internal_force(
        &self,
        ctx: &ElemCtx,
        u: &[f64],
        _du: &[f64],
        _state_old: &[f64],
        _state_new: &mut Vec<f64>,
        tangent: bool,
    ) -> Result<ElemResponse> {
        if ctx.nlgeom {
            return not_implemented("geometrisk olinjäritet (NLGEOM)", self.name());
        }
        if let Some(m) = ctx.material {
            if !m.is_linear() {
                return not_implemented(&format!("olinjärt material ({})", m.name()), self.name());
            }
        }
        let k = self.stiffness(ctx)?;
        let n = u.len();
        let fint = (0..n).map(|i| (0..n).map(|j| k[i * n + j] * u[j]).sum()).collect();
        Ok(ElemResponse { fint, k: tangent.then_some(k) })
    }

    /// Massmatris.
    fn mass(&self, _ctx: &ElemCtx, _kind: MassKind) -> Result<Vec<f64>> {
        not_implemented("massmatris", self.name())
    }

    /// Geometrisk styvhet (spänningsstyvhet) för knäckning, givet aktuellt tillstånd.
    fn geometric_stiffness(&self, _ctx: &ElemCtx, _u: &[f64], _state: &[f64]) -> Result<Vec<f64>> {
        not_implemented("geometrisk styvhet", self.name())
    }

    /// Ekvivalenta nodkrafter för en fördelad last (*DLOAD). `u` är aktuell
    /// förskjutning (för följande laster vid NLGEOM).
    fn distributed_load(&self, _ctx: &ElemCtx, load: &DistLoad, _u: &[f64]) -> Result<Vec<f64>> {
        not_implemented(&format!("lasttypen {}", load.kind), self.name())
    }

    /// ∫Nᵢ dV per nod (inklusive tjocklek eller tvärsnittsarea). Används för
    /// egenvikt i .rusty-modeller och diagonal massa.
    fn volume_weights(&self, _ctx: &ElemCtx) -> Result<Vec<f64>> {
        not_implemented("volymvikter", self.name())
    }

    /// Utdata per integrationspunkt (S, E, PEEQ, SF, SM, ...).
    fn output(&self, _ctx: &ElemCtx, _u: &[f64], _state: &[f64]) -> Result<Vec<IpField>> {
        Ok(Vec::new())
    }
}
