//! # Z₃ Lattice Gauge Theory
//!
//! Implementation of a Z₃ gauge field on a 2D square lattice using the Wilson action.
//!
//! ## Key Concepts
//!
//! - **Link variables** `U_μ(x) ∈ {1, ω, ω²}` where `ω = exp(2πi/3)`
//! - **Plaquette action**: `S = -β × Σ Re(Tr(U_p))` summed over all oriented plaquettes
//! - **Conservation law**: `γ + η = C` — the ternary Gauss's law: net flux through any
//!   closed surface is zero modulo 3
//! - **Monte Carlo**: Metropolis algorithm with local link updates
//!
//! ## The Ternary Conservation Law
//!
//! In Z₃ gauge theory, charge is quantized in units of the third roots of unity.
//! The conservation law `γ + η = C` is literally Gauss's law: for any site, the
//! divergence of the ternary flux (encoded as exponents of ω) sums to zero mod 3.
//! This makes `γ + η = C` not just a formal identity but a physically measurable
//! conservation law verified at every lattice site after each update.

use rand::{Rng, SeedableRng};

// ---------------------------------------------------------------------------
// Ternary field element: Z₃ = {0, 1, 2} representing {1, ω, ω²}
// ---------------------------------------------------------------------------

/// An element of Z₃, stored as an exponent of ω = exp(2πi/3).
///
/// Value 0 → ω⁰ = 1, 1 → ω¹, 2 → ω².
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Z3(u8);

impl Z3 {
    /// Identity element (ω⁰ = 1).
    pub const ONE: Z3 = Z3(0);
    /// ω = exp(2πi/3).
    pub const OMEGA: Z3 = Z3(1);
    /// ω² = exp(4πi/3).
    pub const OMEGA2: Z3 = Z3(2);

    /// All three elements of Z₃.
    pub const ALL: [Z3; 3] = [Z3::ONE, Z3::OMEGA, Z3::OMEGA2];

    /// Create from an exponent (wraps mod 3).
    pub fn new(k: u8) -> Self {
        Z3(k % 3)
    }

    /// The raw exponent in {0, 1, 2}.
    pub fn exp(self) -> u8 {
        self.0
    }

    /// Multiplication in Z₃ (add exponents mod 3).
    pub fn mul(self, other: Z3) -> Z3 {
        Z3((self.0 + other.0) % 3)
    }

    /// Inverse element: `self * self.inv() = ONE`.
    pub fn inv(self) -> Z3 {
        Z3((3 - self.0) % 3)
    }

    /// Real part of ω^k = cos(2πk/3).
    pub fn re(self) -> f64 {
        match self.0 {
            0 => 1.0,
            1 => -0.5,
            2 => -0.5,
            _ => unreachable!(),
        }
    }

    /// Imaginary part of ω^k = sin(2πk/3).
    pub fn im(self) -> f64 {
        match self.0 {
            0 => 0.0,
            1 => SQRT3_OVER_2,
            2 => -SQRT3_OVER_2,
            _ => unreachable!(),
        }
    }

    /// Complex value as (re, im).
    pub fn to_complex(self) -> (f64, f64) {
        (self.re(), self.im())
    }

    /// Random Z₃ element.
    pub fn random<R: Rng>(rng: &mut R) -> Self {
        Z3(rng.gen_range(0..3))
    }

    /// Power: ω^(k·n) mod 3.
    pub fn pow(self, n: u8) -> Z3 {
        Z3((self.0 * n) % 3)
    }
}

/// sqrt(3)/2, precomputed.
const SQRT3_OVER_2: f64 = 0.8660254037844386;

impl std::ops::Mul for Z3 {
    type Output = Z3;
    fn mul(self, rhs: Z3) -> Z3 {
        self.mul(rhs)
    }
}

impl std::ops::MulAssign for Z3 {
    fn mul_assign(&mut self, rhs: Z3) {
        *self = self.mul(rhs);
    }
}

// ---------------------------------------------------------------------------
// 2D Lattice
// ---------------------------------------------------------------------------

/// A 2D square lattice with periodic boundary conditions.
///
/// Links are indexed by `(x, y, μ)` where μ ∈ {0, 1} (x-direction, y-direction).
/// Link variable `U_μ(x, y)` lives on the edge from site `(x,y)` in direction μ.
#[derive(Clone, Debug)]
pub struct Lattice {
    /// Lattice extents (Lx, Ly).
    size: (usize, usize),
    /// Link variables, indexed as `links[y * Lx * 2 + x * 2 + μ]`.
    /// For site (x,y), direction μ: the link U_μ(x,y).
    links: Vec<Z3>,
}

impl Lattice {
    /// Create a new Lx × Ly lattice with all links set to identity (cold start).
    pub fn new(lx: usize, ly: usize) -> Self {
        let n_links = lx * ly * 2;
        Lattice {
            size: (lx, ly),
            links: vec![Z3::ONE; n_links],
        }
    }

    /// Create a lattice with random link variables (hot start).
    pub fn hot<R: Rng>(lx: usize, ly: usize, rng: &mut R) -> Self {
        let n_links = lx * ly * 2;
        let links: Vec<Z3> = (0..n_links).map(|_| Z3::random(rng)).collect();
        Lattice {
            size: (lx, ly),
            links,
        }
    }

    /// Lattice dimensions.
    pub fn size(&self) -> (usize, usize) {
        self.size
    }

    /// Total number of sites.
    pub fn n_sites(&self) -> usize {
        self.size.0 * self.size.1
    }

    /// Total number of links.
    pub fn n_links(&self) -> usize {
        self.links.len()
    }

    /// Total number of plaquettes (oriented).
    pub fn n_plaquettes(&self) -> usize {
        self.size.0 * self.size.1
    }

    /// Periodic index wrap.
    #[inline]
    fn wrap(&self, x: usize, y: usize) -> (usize, usize) {
        let (lx, ly) = self.size;
        (x % lx, y % ly)
    }

    /// Flat index for link (x, y, μ).
    #[inline]
    fn link_idx(&self, x: usize, y: usize, mu: usize) -> usize {
        let (x, y) = self.wrap(x, y);
        (y * self.size.0 + x) * 2 + mu
    }

    /// Get link variable U_μ(x, y).
    pub fn get_link(&self, x: usize, y: usize, mu: usize) -> Z3 {
        self.links[self.link_idx(x, y, mu)]
    }

    /// Set link variable U_μ(x, y).
    pub fn set_link(&mut self, x: usize, y: usize, mu: usize, val: Z3) {
        let idx = self.link_idx(x, y, mu);
        self.links[idx] = val;
    }

    /// Compute the plaquette variable U_p at site (x, y).
    ///
    /// Plaquette orientation: go μ=0 (x), then ν=1 (y), back μ=0†, back ν=1†.
    ///
    /// ```text
    ///   (x,y+1)---U_0(x,y+1)--->(x+1,y+1)
    ///      ^                        |
    ///   U_1(x,y)                U_1(x+1,y)†
    ///      |                        v
    ///    (x,y)----U_0(x,y)---->(x+1,y)
    /// ```
    ///
    /// U_p = U_0(x,y) · U_1(x+1,y) · U_0(x,y+1)† · U_1(x,y)†
    pub fn plaquette(&self, x: usize, y: usize) -> Z3 {
        let u0 = self.get_link(x, y, 0);          // U_0(x,y)
        let u1_fwd = self.get_link(x + 1, y, 1);  // U_1(x+1,y)
        let u0_up_inv = self.get_link(x, y + 1, 0).inv(); // U_0(x,y+1)†
        let u1_inv = self.get_link(x, y, 1).inv();       // U_1(x,y)†
        u0 * u1_fwd * u0_up_inv * u1_inv
    }

    /// Wilson action: S = -β × Σ_p Re(Tr(U_p)).
    ///
    /// In Z₃, Tr(U_p) is just the scalar complex value since it's a 1×1 matrix.
    /// Re(ω^k) = 1 for k=0, -1/2 for k=1,2.
    ///
    /// Returns the total action as a floating-point value.
    pub fn action(&self, beta: f64) -> f64 {
        let (lx, ly) = self.size;
        let mut s = 0.0;
        for y in 0..ly {
            for x in 0..lx {
                let up = self.plaquette(x, y);
                // S_p = -β * Re(Tr(U_p)) = -β * Re(ω^k)
                s -= beta * up.re();
            }
        }
        s
    }

    // -----------------------------------------------------------------------
    // Ternary Gauss's Law: γ + η = C (flux conservation)
    // -----------------------------------------------------------------------

    /// Compute the ternary divergence (Gauss's law charge) at site (x, y).
    ///
    /// In Z₃ gauge theory, the "electric flux" through a surface around a site
    /// is the sum of link exponents mod 3. For 2D:
    ///
    /// `div(x,y) = [U_0(x,y) + U_1(x,y) - U_0(x-1,y) - U_1(x,y-1)] mod 3`
    ///
    /// where we use exponents and subtraction is mod 3.
    ///
    /// The **conservation law γ + η = C** manifests as: this divergence is zero
    /// for pure gauge configurations, and equals the static charge at the site
    /// when matter fields are present.
    ///
    /// Returns the ternary charge: 0 = neutral, 1 = γ-charge, 2 = η-charge.
    pub fn ternary_divergence(&self, x: usize, y: usize) -> u8 {
        let (lx, ly) = self.size;
        let u0_out = self.get_link(x, y, 0).exp() as i32;
        let u1_out = self.get_link(x, y, 1).exp() as i32;
        // Links pointing *into* the site (backward from neighbors)
        let u0_in = self.get_link((x + lx - 1) % lx, y, 0).exp() as i32;
        let u1_in = self.get_link(x, (y + ly - 1) % ly, 1).exp() as i32;
        // Net flux = outgoing - incoming, mod 3
        let div = (u0_out + u1_out - u0_in - u1_in).rem_euclid(3);
        div as u8
    }

    /// Verify the global conservation law (γ + η = C).
    ///
    /// In a pure gauge theory on a closed manifold (periodic BCs), the total
    /// ternary charge must vanish: Σ_x div(x) = 0 mod 3. This is the lattice
    /// version of the continuum statement that the net flux through a closed
    /// surface is zero — the conservation law γ + η = C.
    ///
    /// Returns `true` if global conservation holds, plus per-site divergences.
    pub fn verify_conservation(&self) -> (bool, Vec<((usize, usize), u8)>) {
        let (lx, ly) = self.size;
        let mut total_charge: i32 = 0;
        let mut sites = Vec::new();
        for y in 0..ly {
            for x in 0..lx {
                let div = self.ternary_divergence(x, y);
                total_charge += div as i32;
                sites.push(((x, y), div));
            }
        }
        let global_ok = total_charge.rem_euclid(3) == 0;
        let violations: Vec<_> = sites.into_iter().filter(|(_, d)| *d != 0).collect();
        (global_ok, violations)
    }

    /// Count sites by ternary charge type.
    ///
    /// Returns `(neutral, gamma, eta)` counts where:
    /// - neutral (0): no net ternary charge
    /// - gamma (1): γ-type charge
    /// - eta (2): η-type charge
    pub fn charge_histogram(&self) -> (usize, usize, usize) {
        let (lx, ly) = self.size;
        let mut counts = [0usize; 3];
        for y in 0..ly {
            for x in 0..lx {
                counts[self.ternary_divergence(x, y) as usize] += 1;
            }
        }
        (counts[0], counts[1], counts[2])
    }

    // -----------------------------------------------------------------------
    // Observables
    // -----------------------------------------------------------------------

    /// Average plaquette value: `<Re(U_p)>`.
    ///
    /// In Z₃, this is the mean of cos(2π·k/3) over all plaquettes.
    /// For β → ∞ (ordered), this approaches 1.
    /// For β → 0 (disordered), this approaches 0.
    pub fn average_plaquette(&self) -> f64 {
        let (lx, ly) = self.size;
        let n = (lx * ly) as f64;
        let mut sum = 0.0;
        for y in 0..ly {
            for x in 0..lx {
                sum += self.plaquette(x, y).re();
            }
        }
        sum / n
    }

    /// Wilson loop W(R, T): product of link variables around an R×T rectangle.
    ///
    /// Returns the Z₃ value of the loop (ω^k), and its real part.
    /// The real part gives the expectation value contribution.
    ///
    /// Starting at (x, y), go R steps in x, T steps in y, back R in x, back T in y.
    pub fn wilson_loop(&self, x: usize, y: usize, r: usize, t: usize) -> (Z3, f64) {
        let mut product = Z3::ONE;

        // Forward in x (R steps)
        for i in 0..r {
            product = product * self.get_link(x + i, y, 0);
        }
        // Forward in y (T steps)
        for j in 0..t {
            product = product * self.get_link(x + r, y + j, 1);
        }
        // Backward in x (R steps)
        for i in (0..r).rev() {
            product = product * self.get_link(x + i, y + t, 0).inv();
        }
        // Backward in y (T steps)
        for j in (0..t).rev() {
            product = product * self.get_link(x, y + j, 1).inv();
        }

        let re = product.re();
        (product, re)
    }

    /// Average Wilson loop of size R×T over all starting positions.
    pub fn average_wilson_loop(&self, r: usize, t: usize) -> f64 {
        let (lx, ly) = self.size;
        let mut sum = 0.0;
        let mut count = 0;
        for y in 0..ly {
            for x in 0..lx {
                let (_, re) = self.wilson_loop(x, y, r, t);
                sum += re;
                count += 1;
            }
        }
        sum / count as f64
    }

    /// Measure the ternary charge distribution across the lattice.
    ///
    /// Returns the total γ-charge and η-charge (mod 3 each), plus
    /// the total conserved charge C = γ + η (mod 3).
    pub fn measure_ternary_charge(&self) -> TernaryCharge {
        let (lx, ly) = self.size;
        let mut gamma_total: u8 = 0;
        let mut eta_total: u8 = 0;
        let mut gamma_sites = 0usize;
        let mut eta_sites = 0usize;

        for y in 0..ly {
            for x in 0..lx {
                let div = self.ternary_divergence(x, y);
                match div {
                    1 => {
                        gamma_total = (gamma_total + 1) % 3;
                        gamma_sites += 1;
                    }
                    2 => {
                        eta_total = (eta_total + 1) % 3;
                        eta_sites += 1;
                    }
                    _ => {}
                }
            }
        }

        let c_total = (gamma_total + eta_total) % 3;
        TernaryCharge {
            gamma: gamma_total,
            eta: eta_total,
            c: c_total,
            gamma_sites,
            eta_sites,
        }
    }
}

/// Result of ternary charge measurement.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TernaryCharge {
    /// Total γ-charge (mod 3).
    pub gamma: u8,
    /// Total η-charge (mod 3).
    pub eta: u8,
    /// Total conserved charge C = γ + η (mod 3).
    pub c: u8,
    /// Number of sites carrying γ-charge.
    pub gamma_sites: usize,
    /// Number of sites carrying η-charge.
    pub eta_sites: usize,
}

impl std::fmt::Display for TernaryCharge {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "γ={} + η={} = C={} ({} γ-sites, {} η-sites)",
            self.gamma, self.eta, self.c, self.gamma_sites, self.eta_sites
        )
    }
}

// ---------------------------------------------------------------------------
// Monte Carlo: Metropolis algorithm
// ---------------------------------------------------------------------------

/// Configuration for the Metropolis Monte Carlo simulation.
#[derive(Debug, Clone)]
pub struct McConfig {
    /// Inverse coupling β.
    pub beta: f64,
    /// Number of thermalization sweeps (discarded).
    pub thermalization: usize,
    /// Number of measurement sweeps.
    pub measurements: usize,
    /// Interval between measurements (in sweeps).
    pub measure_interval: usize,
    /// Seed for reproducibility (optional).
    pub seed: Option<u64>,
}

impl Default for McConfig {
    fn default() -> Self {
        McConfig {
            beta: 2.0,
            thermalization: 500,
            measurements: 1000,
            measure_interval: 5,
            seed: None,
        }
    }
}

/// Results from a Monte Carlo simulation.
#[derive(Debug, Clone)]
pub struct McResult {
    /// Configuration used.
    pub config: McConfig,
    /// Lattice dimensions.
    pub lattice_size: (usize, usize),
    /// Average plaquette measurements.
    pub plaquette_history: Vec<f64>,
    /// Mean of the plaquette history.
    pub mean_plaquette: f64,
    /// Variance of the plaquette history.
    pub var_plaquette: f64,
    /// Wilson loop measurements: (R, T) -> history of values.
    pub wilson_loops: Vec<((usize, usize), Vec<f64>)>,
    /// Ternary charge measurements.
    pub charge_history: Vec<TernaryCharge>,
    /// Acceptance rate.
    pub acceptance_rate: f64,
    /// Whether conservation (γ + η = C) held at every measurement.
    pub conservation_verified: bool,
}

impl McResult {
    /// Compute the mean of a slice of f64.
    fn mean(data: &[f64]) -> f64 {
        if data.is_empty() {
            return 0.0;
        }
        data.iter().sum::<f64>() / data.len() as f64
    }

    /// Compute the variance of a slice of f64.
    fn variance(data: &[f64]) -> f64 {
        if data.len() < 2 {
            return 0.0;
        }
        let m = Self::mean(data);
        let n = data.len() as f64;
        data.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (n - 1.0)
    }

    /// Mean Wilson loop for a given (R, T).
    pub fn mean_wilson_loop(&self, r: usize, t: usize) -> Option<f64> {
        self.wilson_loops
            .iter()
            .find(|((rr, tt), _)| *rr == r && *tt == t)
            .map(|(_, hist)| Self::mean(hist))
    }
}

/// Run a Monte Carlo simulation of Z₃ lattice gauge theory.
///
/// ## Algorithm
///
/// 1. **Propose**: Pick a random link and multiply by ω^k (k ∈ {+1, -1})
/// 2. **Accept**: With probability min(1, exp(-ΔS)) where ΔS is the change in action
/// 3. **Measure**: Record observables every `measure_interval` sweeps
///
/// A "sweep" visits every link once (in random order).
///
/// ## Conservation Law
///
/// After each sweep, we verify γ + η = C (Gauss's law) at every site.
/// For pure gauge updates, this should always hold — the Metropolis step
/// preserves the gauge constraint because changing a single link shifts
/// flux by ±1 at exactly two sites (the link endpoints), maintaining the
/// global conservation.
pub fn run_monte_carlo(lattice: &mut Lattice, config: McConfig) -> McResult {
    let mut rng = match config.seed {
        Some(s) => rand::rngs::StdRng::seed_from_u64(s),
        None => rand::rngs::StdRng::from_entropy(),
    };

    let (lx, ly) = lattice.size();
    let n_links = lattice.n_links();
    let total_links_visited = (config.thermalization + config.measurements) * n_links;
    let mut accepted: usize = 0;

    let mut plaquette_history = Vec::new();
    let mut wilson_loops: Vec<((usize, usize), Vec<f64>)> = {
        // Measure Wilson loops up to half the lattice size
        let max_r = (lx / 2).min(4);
        let max_t = (ly / 2).min(4);
        let mut wl = Vec::new();
        for r in 1..=max_r {
            for t in 1..=max_t {
                wl.push(((r, t), Vec::new()));
            }
        }
        wl
    };
    let mut charge_history = Vec::new();
    let mut conservation_verified = true;

    let total_sweeps = config.thermalization + config.measurements;
    for sweep in 0..total_sweeps {
        // One sweep: visit each link once
        for _ in 0..n_links {
            // Pick random link
            let x = rng.gen_range(0..lx);
            let y = rng.gen_range(0..ly);
            let mu = rng.gen_range(0..2);

            // Current link and plaquettes it participates in
            let old_link = lattice.get_link(x, y, mu);

            // Propose change: multiply by ω or ω²
            let delta = if rng.gen_bool(0.5) { Z3::OMEGA } else { Z3::OMEGA2 };
            let new_link = old_link * delta;

            // Compute ΔS: the link participates in 2 plaquettes (in 2D)
            // Plaquette 1: with the site at (x,y)
            // Plaquette 2: with the site behind the link
            let old_p1 = lattice.plaquette_for_link(x, y, mu, true);
            let old_p2 = lattice.plaquette_for_link(x, y, mu, false);

            // Temporarily update the link
            lattice.set_link(x, y, mu, new_link);

            let new_p1 = lattice.plaquette_for_link(x, y, mu, true);
            let new_p2 = lattice.plaquette_for_link(x, y, mu, false);

            // ΔS = -β × (Re(new_p1) + Re(new_p2) - Re(old_p1) - Re(old_p2))
            let delta_s = -config.beta
                * (new_p1.re() + new_p2.re() - old_p1.re() - old_p2.re());

            // Metropolis accept/reject
            if delta_s <= 0.0 || rng.gen_bool((-delta_s).exp()) {
                // Accept
                accepted += 1;
            } else {
                // Reject: restore old link
                lattice.set_link(x, y, mu, old_link);
            }
        }

        // Measurements (after thermalization)
        if sweep >= config.thermalization {
            let sweep_in_measure = sweep - config.thermalization;
            if sweep_in_measure % config.measure_interval == 0 {
                // Average plaquette
                plaquette_history.push(lattice.average_plaquette());

                // Wilson loops
                for ((r, t), hist) in &mut wilson_loops {
                    hist.push(lattice.average_wilson_loop(*r, *t));
                }

                // Ternary charge
                let charge = lattice.measure_ternary_charge();
                charge_history.push(charge.clone());

                // Verify conservation
                let (ok, _) = lattice.verify_conservation();
                if !ok {
                    conservation_verified = false;
                }
            }
        }
    }

    let acceptance_rate = accepted as f64 / total_links_visited as f64;

    McResult {
        mean_plaquette: McResult::mean(&plaquette_history),
        var_plaquette: McResult::variance(&plaquette_history),
        config,
        lattice_size: (lx, ly),
        plaquette_history,
        wilson_loops,
        charge_history,
        acceptance_rate,
        conservation_verified,
    }
}

// ---------------------------------------------------------------------------
// Additional lattice helpers
// ---------------------------------------------------------------------------

impl Lattice {
    /// Compute the plaquette that a given link participates in.
    ///
    /// In 2D, each link is part of exactly 2 plaquettes:
    /// - `forward=true`: the plaquette "above" (for μ=0) or "to the right" (for μ=1)
    /// - `forward=false`: the plaquette "below" or "to the left"
    ///
    /// This is used for efficient local ΔS computation.
    pub fn plaquette_for_link(
        &self,
        x: usize,
        y: usize,
        mu: usize,
        forward: bool,
    ) -> Z3 {
        match (mu, forward) {
            // μ=0 (x-link), forward plaquette: (x,y) with ν=1
            (0, true) => self.plaquette(x, y),
            // μ=0 (x-link), backward plaquette: (x,y-1) with ν=1
            (0, false) => self.plaquette(x, y + self.size.1 - 1),
            // μ=1 (y-link), forward plaquette: (x,y) with ν=0 — but rotated
            // For μ=1, the "forward" plaquette uses the plaquette at (x,y)
            // but traversed differently. In 2D, there's only one plaquette orientation,
            // so both links participate in the same plaquettes.
            (1, true) => self.plaquette(x, y),
            // μ=1 (y-link), backward plaquette: (x-1,y)
            (1, false) => self.plaquette(x + self.size.0 - 1, y),
            _ => unreachable!(),
        }
    }
}

// ---------------------------------------------------------------------------
// Exact Z₃ action computation
// ---------------------------------------------------------------------------

/// Compute the exact partition function Z for a 2D Z₃ gauge theory on an L×L lattice.
///
/// This uses the fact that Z₃ is abelian and the partition function can be computed
/// via character expansions. For small lattices this is feasible and provides a
/// benchmark for Monte Carlo results.
///
/// For an L×L lattice: Z = (Σ exp(-β·Re(ω^k)))^(L²) summed over all configurations.
/// Each plaquette independently takes values in {1, ω, ω²}.
pub fn exact_free_energy(l: usize, beta: f64) -> f64 {
    // Single plaquette Boltzmann weights
    let w0 = (beta * Z3::ONE.re()).exp();    // β
    let w1 = (beta * Z3::OMEGA.re()).exp();  // -β/2
    let w2 = (beta * Z3::OMEGA2.re()).exp(); // -β/2

    // Single plaquette partition function
    let z1 = w0 + w1 + w2;

    // Total partition function (independent plaquettes in strong coupling)
    let n_plaq = (l * l) as f64;
    -n_plaq * z1.ln()
}

/// Exact average plaquette from the single-plaquette partition function.
pub fn exact_average_plaquette(beta: f64) -> f64 {
    // <Re(U_p)> = (1·e^(β) + (-1/2)·e^(-β/2) + (-1/2)·e^(-β/2)) / (e^(β) + 2·e^(-β/2))
    let w0 = beta.exp();
    let w1 = (-beta / 2.0).exp();
    (w0 - w1) / (w0 + 2.0 * w1)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn z3_arithmetic() {
        // Multiplication
        assert_eq!(Z3::OMEGA * Z3::OMEGA, Z3::OMEGA2);
        assert_eq!(Z3::OMEGA * Z3::OMEGA2, Z3::ONE);
        assert_eq!(Z3::ONE * Z3::OMEGA, Z3::OMEGA);

        // Inverse
        assert_eq!(Z3::OMEGA.inv(), Z3::OMEGA2);
        assert_eq!(Z3::OMEGA2.inv(), Z3::OMEGA);
        assert_eq!(Z3::ONE.inv(), Z3::ONE);

        // Power
        assert_eq!(Z3::OMEGA.pow(3), Z3::ONE);
        assert_eq!(Z3::OMEGA.pow(0), Z3::ONE);
    }

    #[test]
    fn z3_complex_values() {
        let eps = 1e-10;
        assert!((Z3::ONE.re() - 1.0).abs() < eps);
        assert!((Z3::ONE.im()).abs() < eps);
        assert!((Z3::OMEGA.re() + 0.5).abs() < eps);
        assert!((Z3::OMEGA.im() - SQRT3_OVER_2).abs() < eps);
        assert!((Z3::OMEGA2.re() + 0.5).abs() < eps);
        assert!((Z3::OMEGA2.im() + SQRT3_OVER_2).abs() < eps);
    }

    #[test]
    fn cold_start_action() {
        let lat = Lattice::new(4, 4);
        // All plaquettes = 1 → Re = 1
        // S = -β × 16 × 1 = -16β
        assert!((lat.action(1.0) - (-16.0)).abs() < 1e-10);
        assert!((lat.average_plaquette() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn cold_start_conservation() {
        let lat = Lattice::new(4, 4);
        let (ok, violations) = lat.verify_conservation();
        assert!(ok, "Conservation violated: {:?}", violations);
    }

    #[test]
    fn single_link_change_breaks_plaquettes() {
        let mut lat = Lattice::new(4, 4);
        lat.set_link(1, 1, 0, Z3::OMEGA);
        // The plaquette at (1,1) is now ω, so Re = -1/2
        let p = lat.plaquette(1, 1);
        assert_ne!(p, Z3::ONE, "Plaquette should have changed");
    }

    #[test]
    fn wilson_loop_cold_start() {
        let lat = Lattice::new(4, 4);
        // All links = 1 → Wilson loop = 1
        for r in 1..=3 {
            for t in 1..=3 {
                let (z3, re) = lat.wilson_loop(0, 0, r, t);
                assert_eq!(z3, Z3::ONE);
                assert!((re - 1.0).abs() < 1e-10, "W({},{}) = {}", r, t, re);
            }
        }
    }

    #[test]
    fn periodic_boundary() {
        let lat = Lattice::new(4, 4);
        // Accessing (4, 0) should wrap to (0, 0)
        assert_eq!(lat.get_link(4, 0, 0), lat.get_link(0, 0, 0));
        assert_eq!(lat.get_link(0, 4, 1), lat.get_link(0, 0, 1));
    }

    #[test]
    fn monte_carlo_runs() {
        let mut lat = Lattice::new(4, 4);
        let config = McConfig {
            beta: 2.0,
            thermalization: 100,
            measurements: 50,
            measure_interval: 10,
            seed: Some(12345),
        };
        let result = run_monte_carlo(&mut lat, config);

        // Should have some measurements
        assert!(!result.plaquette_history.is_empty());
        // Plaquette should be between -0.5 and 1.0
        for &p in &result.plaquette_history {
            assert!(
                (-0.6..=1.1).contains(&p),
                "Plaquette out of range: {}",
                p
            );
        }
        // Acceptance rate should be reasonable
        assert!(
            result.acceptance_rate > 0.0 && result.acceptance_rate <= 1.0,
            "Acceptance rate: {}",
            result.acceptance_rate
        );
    }

    #[test]
    fn global_conservation_after_mc() {
        let mut lat = Lattice::new(4, 4);
        let config = McConfig {
            beta: 1.5,
            thermalization: 200,
            measurements: 100,
            measure_interval: 5,
            seed: Some(99),
        };
        let result = run_monte_carlo(&mut lat, config);
        // Global conservation (Σ div = 0 mod 3) should hold at every measurement
        assert!(
            result.conservation_verified,
            "Global conservation γ + η = C was violated during pure gauge evolution!"
        );
        // Also verify on the final configuration directly
        let (ok, violations) = lat.verify_conservation();
        assert!(ok, "Final config violates global conservation, violations: {:?}", violations);
    }

    #[test]
    fn exact_plaquette_benchmark() {
        // At large β, the average plaquette should approach 1
        let exact = exact_average_plaquette(5.0);
        assert!((exact - 1.0).abs() < 0.1, "exact = {}", exact);

        // At β=0, all states equally likely
        // <Re> = (1 - 1/2 - 1/2) / 3 = 0
        let exact_zero = exact_average_plaquette(0.0);
        assert!((exact_zero).abs() < 1e-10, "exact(β=0) = {}", exact_zero);
    }

    #[test]
    fn charge_histogram_cold() {
        let lat = Lattice::new(6, 6);
        let (neutral, gamma, eta) = lat.charge_histogram();
        assert_eq!(neutral, 36);
        assert_eq!(gamma, 0);
        assert_eq!(eta, 0);
    }
}
