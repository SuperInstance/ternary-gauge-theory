# ternary-gauge-theory

Z₃ lattice gauge theory on a 2D square lattice. Wilson action, Metropolis Monte Carlo, Gauss's law verification.

## What this is

An implementation of a discrete gauge theory where the gauge group is Z₃ = {1, ω, ω²} with ω = exp(2πi/3). Link variables live on the edges of a square lattice with periodic boundary conditions. The Wilson action drives the dynamics. A Metropolis algorithm samples configurations, and the code verifies that Gauss's law holds after every sweep.

This is real lattice gauge theory, just with a finite gauge group instead of SU(3). The physics is well-understood — see Creutz's work on Z_N gauge theories and the references below.

## The physics

**Link variables.** Each edge of the lattice carries a variable U_μ(x) ∈ Z₃. In the code these are stored as exponents k ∈ {0, 1, 2} representing ω^k.

**Plaquettes.** The smallest loop on the lattice. For site (x, y):

```
U_p = U_0(x,y) · U_1(x+1,y) · U_0(x,y+1)† · U_1(x,y)†
```

This is a product of four link variables around a unit square. For Z₃ it's a scalar, not a matrix trace.

**Wilson action.**

```
S = -β × Σ_p Re(U_p)
```

Re(ω^k) is 1 for k=0, −½ for k=1,2. At large β the system orders (all plaquettes = 1). At β → 0 it disorders.

**Gauss's law.** The ternary divergence at a site measures net flux through its boundary:

```
div(x,y) = [outgoing exponents − incoming exponents] mod 3
```

For a pure gauge configuration (no matter fields), the total charge Σ div = 0 mod 3. This is the lattice version of ∇ · E = 0 in source-free electrodynamics. The code checks this after every Monte Carlo sweep. Pure gauge updates preserve it because flipping one link shifts flux by ±1 at exactly two sites, keeping the global sum zero.

## Usage

```rust
use ternary_gauge_theory::{Lattice, McConfig, run_monte_carlo};

let mut lattice = Lattice::hot(8, 8, &mut rng);

let config = McConfig {
    beta: 2.0,
    thermalization: 500,
    measurements: 1000,
    measure_interval: 5,
    seed: Some(42),
};

let result = run_monte_carlo(&mut lattice, config);

println!("mean plaquette: {:.4}", result.mean_plaquette);
println!("conservation verified: {}", result.conservation_verified);
println!("acceptance rate: {:.2}%", result.acceptance_rate * 100.0);
```

## Running tests

```bash
cargo test
```

Key tests:
- `z3_arithmetic` — field operations are correct
- `cold_start_action` — ordered lattice gives S = −β × N_plaquettes
- `global_conservation_after_mc` — Gauss's law holds after 100 sweeps of Metropolis
- `exact_plaquette_benchmark` — MC plaquette matches the analytic strong-coupling result

## Exact results

The code includes exact computations for benchmarking:

```rust
// Analytic average plaquette at coupling β
let p_exact = exact_average_plaquette(beta);

// Strong-coupling free energy for L×L lattice
let f = exact_free_energy(l, beta);
```

At β → 0 all three plaquette values are equally likely, so ⟨Re(U_p)⟩ = 0. At large β the system freezes into the ordered phase.

## Observables

The MC simulation measures:
- **Average plaquette** ⟨Re(U_p)⟩ — order parameter
- **Wilson loops** W(R,T) — product of links around an R×T rectangle, used to extract the static quark potential
- **Ternary charge distribution** — site-by-site divergence values, histogram of γ and η charges
- **Conservation verification** — whether Σ div = 0 mod 3 held at every measurement point

## Structure

```
src/lib.rs    — everything (Z₃ arithmetic, lattice, Monte Carlo, exact results, tests)
```

No external dependencies beyond `rand`. No `std` required for the core types.

## Open questions

- **Phase transition.** Z₃ gauge theory in 2D has a confinement-deconfinement transition. Finding the critical β requires finite-size scaling. The code measures the right observables but doesn't do the analysis.
- **Topological charge.** The total winding number of the Z₃ gauge field is a topological invariant. Not yet implemented.
- **Matter fields.** Adding dynamical Z₃-charged matter breaks Gauss's law from "total charge = 0" to "total charge = total matter charge." This is where the conservation law becomes non-trivial.

## References

- Creutz, M. "Monte Carlo study of quantized SU(2) gauge theory." *Phys. Rev. D* 21, 2308 (1980). The original lattice gauge theory MC paper.
- Creutz, M. "Z_N gauge theories." *Phys. Rev. D* 21, 3006 (1980). Specifically about discrete gauge groups.
- Kogut, J. "An introduction to lattice gauge theory and spin systems." *Rev. Mod. Phys.* 51, 659 (1979). Review article.
- Rothe, H. *Lattice Gauge Theories: An Introduction.* World Scientific. Textbook treatment.

## License

MIT OR Apache-2.0
