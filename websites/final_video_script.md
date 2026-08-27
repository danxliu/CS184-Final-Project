# Final Video Voiceover Script

Target length: ~2:00. Three speakers: Michael, Daniel, Atharv.
Atharv handles the live demos, so live cuts and the slides that follow them are his.
Live cuts are marked **[LIVE CUT]**.

---

## Michael — slides 1-2 (~0:00-0:22)

**Slide 1 — Title (0:00-0:06)**
We're Limitless: Infinity. Our project is repulsive-surface optimization — deforming 3D meshes without ever letting them pass through themselves.

**Slide 2 — The goal (0:06-0:22)**
This is a reimplementation of the Repulsive Surfaces and Repulsive Shells papers from the Crane group. Both papers encode collision avoidance directly in the mesh energy, so from an embedded start the optimizer can't cross into a self-intersection. We descoped GPU acceleration after the milestone, so everything in this video runs on CPU.

---

## Daniel — slides 3-4 (~0:22-0:58)

**Slide 3 — The energy (0:22-0:38)**
The core object is the discrete tangent-point energy. It sums a repulsive kernel over every pair of triangles. When two distant patches approach contact, the denominator goes to zero and the kernel blows up — so any descent step has to push the mesh away from itself.

**Slide 4 — The pipeline (0:38-0:58)**
To make minimization tractable, we built a four-step CPU pipeline that runs every iteration. We compute per-triangle geometry, accelerate the energy from quadratic to n-log-n with a hierarchy of triangle clusters, take a Sobolev-preconditioned step that's resolution-independent, and remesh on the fly so triangle quality stays under control.

---

## Atharv — demos + slides 5-7 (~0:58-2:00)

**[LIVE CUT — Polyscope playback of genus 3 descent.] (0:58-1:18)**
The first result is the canonical embedding gallery. What you're looking at is a procedural genus-three surface — three handles tangled into the starting mesh — flowing under the tangent-point energy alone. No external forces, just self-repulsion. As it relaxes, the three holes spread out and the body smooths into a roughly tetrahedral arrangement, which is the symmetric local minimum the paper predicts for genus three. We ran this for genus zero through six; I'm only showing genus three because it's the most striking.

**Slide 5 — Canonical embeddings graph (1:18-1:30)**
Here's the energy plot for the full sweep. Each curve is normalized by its starting energy, over five hundred iterations. Genus zero collapses fastest — a sphere is the global minimum — and the higher genera plateau into local minima with their handles intact.

**[LIVE CUT — ball-through-tube sequence.] (1:30-1:50)**
The second result lifts the same energy onto a path. The two endpoints are fixed: a sphere outside the tube on one side, and the same sphere on the other side. The thirty interior frames are optimized jointly under a trust-region solver. The objective adds shell elasticity and a repulsive barrier from the tube surface, so as the sphere stretches through the tube it deforms cleanly and never touches the wall.

**Slide 6 — Sphere through a tube graph (1:50-1:58)**
The path's self-repulsion energy drops by two orders of magnitude in the first five iterations, and the sphere stays clear of the wall for the full sequence.

**Slide 7 — Takeaway (1:58-2:10)**
We built the CPU core the full Repulsive Shells system depends on — fast TPE, preconditioned descent, remeshing, and external cross-validation — plus a collision-free path demo on top. The whole pipeline reproduces from a saved frame sequence. Thanks for watching.
