# Project Summary & Task Breakdown

## Context & Constraints
* **Package Manager:** Use `uv`.
* **External Dependencies:** Inspect and parse the GitHub-hosted `taylor_swft` package early to determine usable APIs.
* **Testing Strategy:** Unit test external standalone functions instead of complex internal class methods.

---

## Technical Refactoring & Core Architecture
* **RT Profile Parameter:** Refactor the system to directly optimize the RT profile parameter per `center_freqs` rather than computing `alpha` per `center_freqs` and mapping back.
* **Frequency Distribution:** Implement and evaluate both logarithmic and linear distributions for `center_freqs`.
* **Baseline Integration:** Implement baseline RIR estimation algorithms (referencing [LouisLalay/VPR-RIR](https://github.com/LouisLalay/VPR-RIR)). This repository uses DEC and CBF.
* **Dataset Selection:** Integrate suitable datasets (e.g., BRAS benchmark for RIRs, LibriSpeech for sources, WHAMR! for noise).

---

## Mixing Engine & Experiment Metadata
* **Mixing Model:** Build $y = h * s + \text{noise}$ supporting:
  * No noise
  * Additive Gaussian noise across multiple SNRs
  * WHAMR! noise across multiple SNRs
* **Metadata Persistence (`mixes.json`):** Store complete audio generation parameters to ensure exact reproducibility across baseline comparisons:
  * Reference RIR filename
  * Source speech file
  * Noise file/properties and generation seed for exact replication
  * Target SNR

---

## Execution & Evaluation Pipeline
1. **Mix Generation:** Pre-generate audio mixes, save them, and output `mixes.json`.
2. **Estimation Phase:** Run the refactored vpr-rir estimation algorithm of this repo and baselines using identical source/noise inputs.
3. **Decoupled Metric Computation:** Save raw estimated RIRs directly to disk during fitting so metrics can be computed or re-evaluated asynchronously without re-running fits.
4. **Experiment Structure:** Standardize each experiment folder with:
   * Algorithm/baseline parameters
   * `mixes.json`
   * Output estimated RIRs
   * `metrics.json`
5. **Metrics Logging:** Ensure `metrics.json` captures key metadata (`sr`, reference RIR path, parameter lengths, `center_freqs`, and a parameter hash to skip duplicate runs).