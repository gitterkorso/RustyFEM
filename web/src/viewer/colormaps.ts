// Färgskalor för konturer, som 256 RGB-värden.

export type ColormapName = "regnbåge" | "turbo" | "viridis" | "kall–varm" | "grå";

const stops: Record<ColormapName, [number, number, number][]> = {
  // regnbågsskala: blått → cyan → grönt → gult → rött
  regnbåge: [
    [0, 0, 255],
    [0, 93, 255],
    [0, 185, 255],
    [0, 255, 232],
    [0, 255, 139],
    [0, 255, 46],
    [46, 255, 0],
    [139, 255, 0],
    [232, 255, 0],
    [255, 185, 0],
    [255, 93, 0],
    [255, 0, 0],
  ],
  turbo: [
    [48, 18, 59],
    [70, 107, 227],
    [40, 187, 236],
    [49, 242, 153],
    [162, 252, 60],
    [237, 208, 58],
    [251, 128, 34],
    [208, 47, 5],
    [122, 4, 3],
  ],
  viridis: [
    [68, 1, 84],
    [72, 40, 120],
    [62, 74, 137],
    [49, 104, 142],
    [38, 130, 142],
    [31, 158, 137],
    [53, 183, 121],
    [109, 205, 89],
    [180, 222, 44],
    [253, 231, 37],
  ],
  "kall–varm": [
    [59, 76, 192],
    [98, 130, 234],
    [141, 176, 254],
    [184, 208, 249],
    [221, 221, 221],
    [245, 196, 173],
    [244, 154, 123],
    [222, 96, 77],
    [180, 4, 38],
  ],
  grå: [
    [30, 30, 30],
    [235, 235, 235],
  ],
};

export const colormapNames = Object.keys(stops) as ColormapName[];

/** 256 färger (RGBA, 0–255) för en färgskala. */
export function colormap(name: ColormapName): Uint8Array {
  const s = stops[name];
  const out = new Uint8Array(256 * 4);
  for (let i = 0; i < 256; i++) {
    const t = (i / 255) * (s.length - 1);
    const k = Math.min(Math.floor(t), s.length - 2);
    const f = t - k;
    for (let c = 0; c < 3; c++) out[i * 4 + c] = Math.round(s[k][c] * (1 - f) + s[k + 1][c] * f);
    out[i * 4 + 3] = 255;
  }
  return out;
}

/** CSS-gradient för färgskalan (för teckenförklaringen). */
export function colormapCss(name: ColormapName, levels: number, dir = "to top"): string {
  const s = stops[name];
  if (levels > 0) {
    const c = colormap(name);
    const parts: string[] = [];
    for (let i = 0; i < levels; i++) {
      const k = Math.round(((i + 0.5) / levels) * 255);
      const col = `rgb(${c[k * 4]},${c[k * 4 + 1]},${c[k * 4 + 2]})`;
      parts.push(`${col} ${(i / levels) * 100}%`, `${col} ${((i + 1) / levels) * 100}%`);
    }
    return `linear-gradient(${dir}, ${parts.join(", ")})`;
  }
  return `linear-gradient(${dir}, ${s.map((c, i) => `rgb(${c.join(",")}) ${(i / (s.length - 1)) * 100}%`).join(", ")})`;
}
