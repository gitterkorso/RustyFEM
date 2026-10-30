// Disclaimer: shown on first start until the user has acknowledged it, and at
// any time from the command palette.

import { persist, set, useStore } from "../store";
import { Icon } from "./ui";

export function DisclaimerText() {
  return (
    <>
      <p>
        <b>
          RustyFEM is a tool for education, research and development. It is not intended for, and must not be used as the sole basis
          for, the design, verification or approval of real structures, products or installations.
        </b>
      </p>
      <ul>
        <li>Results may be wrong, because of errors in the program, the model or the input data.</li>
        <li>
          No guarantee is given that the results are correct, complete or fit for any particular purpose. The software is provided
          "as is", without warranty of any kind.
        </li>
        <li>
          You are solely responsible for reviewing and verifying the results and for complying with applicable codes. Load-bearing
          structures must be designed and checked by a qualified engineer.
        </li>
        <li>
          To the extent permitted by law, the author and contributors are not liable for any damage, loss or cost arising from the
          use of the software or its results.
        </li>
      </ul>
    </>
  );
}

export function Disclaimer() {
  const open = useStore((s) => s.disclaimer);
  if (!open) return null;
  const accept = () => {
    persist("disclaimerAccepted", true);
    set({ disclaimer: false });
  };
  return (
    <div className="overlay" onMouseDown={accept}>
      <div className="dialog disclaimer" onMouseDown={(e) => e.stopPropagation()} role="dialog" aria-label="Disclaimer">
        <header className="dialog-head">
          <div>
            <h2>Disclaimer</h2>
            <p className="muted">Please read this before using the results.</p>
          </div>
          <button type="button" className="ibtn" aria-label="Close" onClick={accept}>
            <Icon name="close" />
          </button>
        </header>
        <div className="disclaimer-body">
          <DisclaimerText />
        </div>
        <footer className="disclaimer-foot">
          <button type="button" className="btn primary" onClick={accept}>
            I understand
          </button>
        </footer>
      </div>
    </div>
  );
}
