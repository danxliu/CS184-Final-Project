# Final Video Voiceover Script

Target length: about 2 minutes.  
Speakers: Atharv, Michael, Daniel.

## Atharv, slides 1-3, about 45 seconds

Slide 1:
Hi, we are Limitless: Infinity. Our project explores repulsive surface optimization: moving triangle meshes while avoiding self-intersection.

Slide 2:
Repulsive Shells asks whether collision avoidance can be built into the optimization problem itself, instead of handled afterward by collision detection. For the final version, we implemented the Repulsive Surfaces machinery underneath that idea, then used the shell and path layer for one focused interpolation demo.

Slide 3:
The core object is tangent-point energy. We sum a repulsive kernel over triangle pairs. When distant patches approach contact, the denominator goes to zero and the energy blows up, so minimization pushes the mesh toward embedded, non-self-intersecting states.

## Michael, slides 4-6, about 45 seconds

Slide 4:
To make this usable, we built a CPU optimization pipeline around the energy. The code computes geometry and analytical gradients, accelerates interactions with a BVH and block-cluster tree, solves for Sobolev-preconditioned descent directions, runs line search, and remeshes as triangle quality changes.

Slide 5:
Our first final result is the canonical embedding gallery. Starting from tangled unknotted surfaces, TPE descent untangles genus zero through genus five into embedded shapes. These runs also stress remeshing, because the geometry changes substantially.

Slide 6:
Quantitatively, normalized TPE drops for every genus over 500 iterations. Genus zero drops by about seven thousand four hundred times, and genus five by about one hundred times. We also cross-checked all-pairs TPE values against Repulsor.

## Daniel, slides 7-9, about 50 seconds

Slide 7:
Our second result is a 30-frame ball-through-tube path. The endpoints are fixed, and the optimizer solves for the intermediate frames. The objective combines shell energy, self TPE, a tube-surface TPE barrier, and a small signed-distance guard, while keeping positive tube clearance.

Slide 8:
The hardest part was numerical discipline. Many bugs did not crash; they just converged to the wrong thing. Finite-difference checks caught gradient sign errors, and Repulsor parity caught kernel mistakes internal tests missed. Deterministic reductions and remeshing budgets made outputs reproducible.

Slide 9:
We did not finish the full GPU Repulsive Shells system, but we built the CPU core it would need: fast TPE, preconditioned descent, remeshing, validation, and a shape-path demo. The result is a reproducible system for embedded surfaces and collision-aware interpolation.
