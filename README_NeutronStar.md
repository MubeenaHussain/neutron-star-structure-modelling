# ⭐ Neutron Star Structure Modelling
### Classical & Relativistic Simulation · TOV Equations · Runge-Kutta 4th Order · 

![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat-square&logo=python)
![NumPy](https://img.shields.io/badge/NumPy-Scientific%20Computing-orange?style=flat-square&logo=numpy)
![Method](https://img.shields.io/badge/Method-Runge--Kutta%20RK4-green?style=flat-square)
![Physics](https://img.shields.io/badge/Physics-TOV%20%2B%20Hydrostatic%20Equilibrium-blueviolet?style=flat-square)
![Units](https://img.shields.io/badge/Units-Planck%20Natural%20Units-red?style=flat-square)

---

## 🌌 What Is a Neutron Star?

When a massive star exhausts its nuclear fuel, gravity wins — the core collapses in a  
**supernova explosion**, compressing protons and electrons into neutrons.  
What remains is a **neutron star**: an object with more mass than our Sun,  
compressed into a sphere roughly the size of a city (~10–15 km radius).

Their density is so extreme that a teaspoon of neutron star material  
would weigh **a billion tonnes** on Earth.

> **Goal:** Numerically simulate the internal pressure and mass profiles  
> of a neutron star using both classical and relativistic physics —  
> and quantify exactly how much the classical model diverges from reality.

---

## ⚙️ The Physics — Two Models, One Question

The structure of a neutron star is governed by the balance between  
**gravitational collapse** and **internal pressure**.  
Two equations describe this balance:

### Classical Model — Hydrostatic Equilibrium
$$\frac{dP}{dr} = -\frac{m\rho}{r^2}$$

Assumes non-relativistic particles, no spacetime curvature.  
Valid for ordinary stars. **Not valid for neutron stars.**

### Relativistic Model — TOV Equation (Tolman-Oppenheimer-Volkoff)
$$\frac{dP}{dr} = -\frac{(\rho + P)(m + Pr^3)}{r^2 - 2mr}$$

Derived from **General Relativity** — accounts for spacetime curvature,  
relativistic pressure contributions, and the full stress-energy tensor.  
**The correct equation for neutron stars.**

Both equations are solved simultaneously with the **mass gradient:**
$$\frac{dm}{dr} = r^2\rho$$

> All equations use **Planck natural units** where G = c = 4π = 1,  
> reducing constants and simplifying the coupled ODE system.

---

## 🧠 Where the Real Thinking Happened

### 1. Newton-Raphson for Initial Neutron Density

Before the simulation can start, the **initial neutron number density n₀** at r = 0  
must be found — but the governing equation is nonlinear and has no closed-form solution:

$$f(n) = 236 \cdot n^{2.54} + n \cdot m_n - \rho_s = 0$$

**Solution: Newton-Raphson iterative method**

```python
def initial_n():
    n = 1                    # starting guess
    err = 1
    tol = 1e-15              # convergence tolerance — 15 decimal places

    while err > tol:
        fn  = 236 * n**2.54 + n * mn - rho_s      # f(n)
        dfn = 236 * 2.54 * n**1.54 + mn            # f'(n)
        temp = n - fn / dfn                         # Newton-Raphson update
        err = np.abs(n - temp)
        n = temp

    return n
```

**Why tolerance 1e-15?** The initial pressure is derived directly from n₀.  
Any error at this step propagates through all 1,500 RK4 iterations —  
a loose tolerance compounds into physically wrong mass and radius predictions.

---

### 2. RK4 Solver — Solving a Coupled ODE System

The pressure and mass gradients are **coupled** — P depends on m, and m depends on P.  
They cannot be solved independently. The Runge-Kutta 4th order method handles this  
by computing **four intermediate gradient estimates** per step:

```python
def RK4Solver(r, m, p, h, flag):
    # 4 gradient estimates for mass (k1_) and pressure (k2_)
    k11 = dm_dr(r, m, p);              k21 = dp_dr(r, m, p, flag)
    k12 = dm_dr(r+0.5*h, m+0.5*k11*h, p+0.5*k21*h)
    k22 = dp_dr(r+0.5*h, m+0.5*k11*h, p+0.5*k21*h, flag)
    k13 = dm_dr(r+0.5*h, m+0.5*k12*h, p+0.5*k22*h)
    k23 = dp_dr(r+0.5*h, m+0.5*k12*h, p+0.5*k22*h, flag)
    k14 = dm_dr(r+h, m+k13*h, p+k23*h)
    k24 = dp_dr(r+h, m+k13*h, p+k23*h, flag)

    # Weighted average update (RK4 formula)
    y[0] = m + h * (k11 + 2*k12 + 2*k13 + k14) / 6   # new mass
    y[1] = p + h * (k21 + 2*k22 + 2*k23 + k24) / 6   # new pressure
    return y
```

**Simulation parameters:**
- Radial steps: **N = 1,501** points from r = 0 to r = 15 (Planck units)
- Step size: h = r[1] - r[0]
- Termination: when pressure drops below tolerance (P → 0 = stellar surface)

---

### 3. Energy Density from Pressure — The Equation of State

At each radial step, the **energy density ρ(P)** must be recalculated  
from the current pressure using the nuclear equation of state:

$$n = \left(\frac{P \cdot \rho_s}{363.44}\right)^{1/2.54}$$
$$\rho(P) = \frac{236 \cdot n^{2.54} + n \cdot m_n}{\rho_s}$$

```python
def rho(p):
    n = (p * rho_s / 363.44) ** (1 / 2.54)
    return (236. * n**2.54 + n * mn) / rho_s
```

This is called at **every RK4 sub-step** — meaning it runs ~6,000 times per simulation.

---

### 4. Unit Conversion — Back to Physical Reality

Results are computed in Planck units and must be converted to observable quantities:

```python
hc    = 197.327                      # MeV·fm  (ℏc)
G     = hc * 6.67259e-45             # gravitational constant in MeV⁻¹·fm³·kg⁻¹
Ms    = 1.1157467e60                 # mass of Sun in MeV
rho_s = 1665.3                       # nuclear central density (MeV/fm³)
mn    = 938.926                      # neutron mass (MeV/c²)
M0    = (4 * π * G**3 * rho_s)**(-0.5)
R0    = G * M0

# Convert radius: Planck units → km
r_km = r * R0 * 1e-18

# Convert mass: Planck units → solar masses
m_solar = m * M0 / Ms
```

---

## 📊 Results — Classical vs. Relativistic

| Property | Classical Model | Relativistic (TOV) Model |
|---|---|---|
| **Predicted Mass** | Physically inconsistent | ~1–2 M☉ ✅ |
| **Predicted Radius** | Diverges | ~10–15 km ✅ |
| **Spacetime curvature** | Ignored | Fully included |
| **Verdict** | ❌ Wrong for neutron stars | ✅ Physically consistent |

**Key finding:** The classical model diverges because it ignores that  
at neutron star densities, **pressure itself contributes to gravity** —  
a purely relativistic effect captured only by the TOV equation.  
This is not a numerical artefact; it is a fundamental physics result.

---

## 📈 Outputs — Two Profile Plots

**Mass Profile:** M/M☉ vs. radius (km) — shows how mass accumulates from core to surface  
**Pressure Profile:** P (MeV/fm³) vs. radius (km) — shows pressure decay from core to zero at surface

Both plotted for classical and relativistic models simultaneously,  
with surface defined as the radius where P drops below tolerance threshold.

---

## 🚀 Real-World Relevance for Space Applications

| This Project | Space Industry Application |
|---|---|
| Coupled ODE solving (RK4) | Orbital mechanics integration, trajectory propagation |
| Newton-Raphson at 1e-15 tolerance | Precision convergence in mission-critical solvers |
| Physical unit conversion chains | Telemetry engineering unit translation |
| Classical vs. relativistic model validation | Model fidelity testing for spacecraft dynamics |
| Equation of state at each timestep | State estimation in real-time onboard systems |

---

## 📁 Repository Structure

```
neutron-star-structure-modelling/
│
├── MNS_Spartificial.ipynb    ← Full notebook
├── README.md                  ← You are here
└── requirements.txt           ← Dependencies
```

---

## ⚙️ How to Run

```bash
# 1. Clone the repo
git clone https://github.com/YOUR-USERNAME/neutron-star-structure-modelling
cd neutron-star-structure-modelling

# 2. Install dependencies
pip install numpy matplotlib

# 3. Open the notebook
jupyter notebook MNS_Spartificial.ipynb
```

No external dataset needed — all physics constants are defined inline.

---

## 🔭 Future Work

- Extend to **rotating neutron stars** using Hartle-Thorne perturbation equations
- Model **quark star** interiors using a different equation of state
- Compare multiple nuclear equations of state (APR, SLy, BSk) for mass-radius predictions
- Validate against observed **LIGO/NICER neutron star mass-radius measurements**

---

## 📚 References

- Tolman (1939), Oppenheimer & Volkoff (1939): Original TOV derivation
- Creighton & Anderson — Gravitational-Wave Physics (ICTP-SAIFR lecture notes)
- Planck Units: https://en.wikipedia.org/wiki/Planck_units
- NICER Mission neutron star observations: https://www.nasa.gov/nicer

---

## 👩‍💻 Author

**Mubeena Hussain**
MSc Statistics — University of Kerala
📧 mubeenahussain1205@gmail.com
🔗 [LinkedIn](www.linkedin.com/in/mubeena-hussain-a357b920b)


---

*"To understand a neutron star is to hold the universe's most extreme physics in an equation."*
