// Unit labels based on the model's unit system. The engine is unitless;
// no values are converted here, the labels only describe what the numbers mean.

import type { UnitSymbols } from "../engine/Engine";

type Quantity = keyof UnitSymbols;

/** Quantity per column for known keywords (null = dimensionless or unknown). */
const timeColumns: (Quantity | null)[] = ["time", "time", "time", "time"];
const columnQuantities: Record<string, (Quantity | null)[]> = {
  ELASTIC: ["stress", null, null],
  PLASTIC: ["stress", null, null],
  DENSITY: ["density"],
  STATIC: timeColumns,
  "HEAT TRANSFER": timeColumns,
  DYNAMIC: timeColumns,
  VISCO: timeColumns,
  "COUPLED TEMPERATURE-DISPLACEMENT": timeColumns,
};

export function columnUnit(keyword: string, column: number, u: UnitSymbols | undefined): string {
  const q = columnQuantities[keyword]?.[column];
  return q && u ? u[q] : "";
}

/** "Name [unit]", or just the name when there is no unit. */
export function withUnit(label: string, unit: string | undefined): string {
  return unit ? `${label} [${unit}]` : label;
}

/** The unit of a boundary condition (displacement) or a point load (force) on a degree of freedom. */
export function dofUnit(dof: string, load: boolean, u: UnitSymbols | undefined): string {
  if (!u) return "";
  if (/^UR\d$/.test(dof)) return load ? u.moment : u.angle;
  if (/^U\d$/.test(dof)) return load ? u.force : u.length;
  return "";
}

/** The unit of a distributed load by type (P, GRAV, BX …). */
export function dloadUnit(kind: string, u: UnitSymbols | undefined): string {
  if (!u || !u.length) return "";
  const k = kind.toUpperCase();
  if (/^P\d*$|^PN?$|^TRVEC|^TRSHR/.test(k)) return u.stress;
  if (k === "GRAV") return u.acceleration;
  if (/^B[XYZ]$/.test(k)) return `${u.force}/${u.length}³`;
  if (/^P[XYZ]$/.test(k)) return `${u.force}/${u.length}`;
  return "";
}

// Conversion on input: the tools let the user type in any unit
// and write the number in the model's unit system in the script.

export type ConvQ = "length" | "area" | "length4" | "force" | "moment" | "stress" | "density" | "acceleration" | "angle";

/** Units per quantity with factor to SI. */
const convUnits: Record<ConvQ, [string, number][]> = {
  length: [["mm", 1e-3], ["cm", 1e-2], ["m", 1], ["in", 0.0254], ["ft", 0.3048]],
  area: [["mm²", 1e-6], ["cm²", 1e-4], ["m²", 1], ["in²", 0.0254 ** 2]],
  length4: [["mm⁴", 1e-12], ["cm⁴", 1e-8], ["m⁴", 1], ["in⁴", 0.0254 ** 4]],
  force: [["N", 1], ["kN", 1e3], ["MN", 1e6], ["lbf", 4.4482216152605], ["kip", 4448.2216152605]],
  moment: [["N·mm", 1e-3], ["N·m", 1], ["kN·m", 1e3], ["MN·m", 1e6], ["lbf·in", 4.4482216152605 * 0.0254], ["kip·ft", 4448.2216152605 * 0.3048]],
  stress: [["Pa", 1], ["kPa", 1e3], ["MPa", 1e6], ["GPa", 1e9], ["psi", 6894.757293168], ["ksi", 6894757.293168]],
  density: [["kg/m³", 1], ["g/cm³", 1e3], ["t/mm³", 1e12], ["lb/in³", 0.45359237 / 0.0254 ** 3], ["lbf·s²/in⁴", 4.4482216152605 / 0.0254 ** 4]],
  acceleration: [["m/s²", 1], ["mm/s²", 1e-3], ["in/s²", 0.0254], ["ft/s²", 0.3048]],
  angle: [["rad", 1], ["mrad", 1e-3], ["°", Math.PI / 180]],
};

/** The number `v` in the unit `from` expressed in `to` (unchanged if either is missing). */
export function convertValue(v: number, q: ConvQ, from: string | null, to: string | null): number {
  if (!from || !to || from === to) return v;
  const f = (u: string) => convUnits[q].find(([x]) => x === u)?.[1] ?? 1;
  return (v * f(from)) / f(to);
}

/** Force units to show loads in, with the corresponding moment unit. */
export const loadUnits: [force: string, moment: string][] = [
  ["N", "N·m"],
  ["kN", "kN·m"],
  ["MN", "MN·m"],
  ["lbf", "lbf·in"],
  ["kip", "kip·ft"],
];

/** The model's unit per quantity (corresponds to the symbols in `units.rs`). */
const systemUnits: Record<string, Record<ConvQ, string>> = {
  si: { length: "m", area: "m²", length4: "m⁴", force: "N", moment: "N·m", stress: "Pa", density: "kg/m³", acceleration: "m/s²", angle: "rad" },
  si_mm: { length: "mm", area: "mm²", length4: "mm⁴", force: "N", moment: "N·mm", stress: "MPa", density: "t/mm³", acceleration: "mm/s²", angle: "rad" },
  us: { length: "in", area: "in²", length4: "in⁴", force: "lbf", moment: "lbf·in", stress: "psi", density: "lbf·s²/in⁴", acceleration: "in/s²", angle: "rad" },
};

/** The quantity a unit symbol belongs to (null = not convertible, e.g. "" or a composite unit). */
export function quantityOf(unit: string | undefined): ConvQ | null {
  if (!unit) return null;
  for (const q of Object.keys(convUnits) as ConvQ[]) if (convUnits[q].some(([u]) => u === unit)) return q;
  return null;
}

/** Display unit for a result value in the model's unit `from`, per the chosen result units. */
export function resultUnit(from: string | undefined, chosen: Record<string, string>): { q: ConvQ | null; to: string } {
  const q = quantityOf(from);
  const c = q ? chosen[q] : undefined;
  return { q, to: q && c && convUnits[q].some(([u]) => u === c) ? c : (from ?? "") };
}

export function unitOptions(q: ConvQ): string[] {
  return convUnits[q].map(([u]) => u);
}

/** The model's unit for the quantity, or null if the unit system is not specified. */
export function modelUnit(q: ConvQ, system: string | undefined): string | null {
  return (system && systemUnits[system]?.[q]) || null;
}

const prefKey = "rustyfem.units.";

/** The most recently chosen unit for the quantity (otherwise the model's). */
export function preferredUnit(q: ConvQ, system: string | undefined): string | null {
  const m = modelUnit(q, system);
  if (!m) return null;
  try {
    const u = localStorage.getItem(prefKey + q);
    if (u && convUnits[q].some(([x]) => x === u)) return u;
  } catch {
    /* no storage */
  }
  return m;
}

export function rememberUnit(q: ConvQ, u: string) {
  try {
    localStorage.setItem(prefKey + q, u);
  } catch {
    /* no storage */
  }
}

/** Short number for the script: exponent for very large and small numbers. */
export function scriptNum(v: number): string {
  if (v === 0) return "0";
  const r = Number(v.toPrecision(10));
  const a = Math.abs(r);
  return a >= 1e6 || a < 1e-3 ? r.toExponential().replace("e+", "e") : String(r);
}

/**
 * Converts all numbers in the text (e.g. "210", "0..2" or "(0,0) (2,0)") from
 * the unit `from` to the model's unit. Text that is not a number is left untouched.
 */
export function convertText(s: string, q: ConvQ, from: string | null, system: string | undefined): string {
  return convert(s, q, from, modelUnit(q, system));
}

/** Like `convertText` but between two arbitrary units ("SI" = the quantity's SI unit). */
export function convert(s: string, q: ConvQ, from: string | null, to: string | null): string {
  if (!from || !to || from === to) return s;
  const f = (convUnits[q].find(([u]) => u === from)?.[1] ?? 1) / (convUnits[q].find(([u]) => u === to)?.[1] ?? 1);
  return s.replace(/[^\s(),]+/g, (tok) =>
    tok
      .split("..")
      .map((t) => {
        const v = Number(t);
        return t.trim() !== "" && Number.isFinite(v) ? scriptNum(v * f) : t;
      })
      .join(".."),
  );
}
