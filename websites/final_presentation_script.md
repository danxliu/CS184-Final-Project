# Final Presentation Script

Target: 5 minutes sharp, 7 minutes hard ceiling. Three speakers: Michael, Daniel, Atharv. Same speaker rotation pattern as the video script. Michael handles the early intro, Daniel handles the middle technical content, Atharv handles the demos and the closing.

Speaking rotation (matches `final_presentation_slides.tex` slide comments):
- Michael: slides 1, 2, 3 (title, motivation, energy)
- Daniel: slides 4, 5, 6, 7 (pipeline 1 through 4)
- Atharv: slides 8, 9, 10, 11, 12 (canonical embeddings, sphere through tube, limitations, future, summary)

Live cuts marked **[LIVE]** are Atharv's, shown when we get to the demos.

---

## Michael, Slide 1: Title (~0:00-0:15)

We're Limitless: Infinity. We built a repulsive-surface optimizer, a system for deforming 3D meshes that physically refuses to let them pass through themselves. I'm Michael, this is Daniel, and Atharv.

---

## Michael, Slide 2: Motivation (~0:15-0:55)

Real solid objects can't pass through themselves, but most representations in geometric computing impose no such constraint, so standard mesh operations like interpolation, averaging, and packing routinely produce self-intersecting outputs. The two papers we reimplement, Repulsive Surfaces and Repulsive Shells from the Crane group at CMU, both add a tangent-point energy that diverges to infinity at any self-contact, so optimization can never cross into a self-intersecting state. Mathematically the construction extends classical knot energies, originally developed in differential topology by O'Hara and others, from curves to surfaces; Repulsive Shells then frames intersection-free interpolation, extrapolation, and averaging as ordinary geodesic problems on a Riemannian shape space. The application areas the papers cite range from computational anatomy and motion planning for soft-body robotics to biological membrane simulation and digital manufacturing.

---

## Michael, Slide 3: The energy (~0:55-1:15)

The thing making this work is the discrete tangent-point energy. It sums a kernel over every pair of triangles in the mesh. The numerator measures how much one triangle's normal aligns with the segment to the other; the denominator is the distance between them, raised to the twelfth power. As any two patches approach contact, that denominator goes to zero and the energy blows up. So from any embedded starting mesh, descending this energy can never cross into self-intersection: the barrier is infinite.

---

## Daniel, Slide 4: Pipeline, step 1 of 4 (~1:15-1:30)

Each descent iteration runs four stages. First: per-triangle geometry. We cache the centroid, unit normal, and area of every face, plus the Jacobians of each with respect to the three corner vertex positions. Every later stage reads from this same buffer, so it gets rebuilt once per outer iteration and reused everywhere downstream.

---

## Daniel, Slide 5: Pipeline, step 2 of 4 (~1:30-2:00)

Second: fast energy and gradient. The energy has $n^2$ pair terms, which kills you at thirty thousand triangles. We build a bounding-volume hierarchy over the triangles, then a paired hierarchy over clusters. A cluster pair is admissible if it's far enough away relative to its size, the inequality on screen, and admissible blocks contribute one rank-one cluster-aggregate term instead of all the triangle pairs they cover. Everything that fails admissibility gets evaluated exactly. The total cost ends up $\mathcal{O}(n \log n)$. The hierarchical version agrees with the brute-force version to machine precision when we set the admissibility threshold to zero, and the error decays at the expected order-$\theta^2$ rate as we open it up, so the approximation is doing what the theory says it should.

---

## Daniel, Slide 6: Pipeline, step 3 of 4 (~2:00-2:35)

Third: a preconditioned step that doesn't slow down as the mesh refines. Plain gradient descent stalls because you're stepping in Euclidean space but the gradient lives in a higher-order Sobolev space, so the metrics are mismatched. The fix from Repulsive Surfaces is the sandwich operator on screen: two cotan-Laplacian inverses around one fractional-order matvec, at exponent two minus s, with s equal to five-thirds. We solve this system with left-preconditioned GMRES every outer iteration. The payoff is what you'd want: GMRES iteration count stays flat as the mesh refines, instead of blowing up the way it does without the preconditioner.

---

## Daniel, Slide 7: Pipeline, step 4 of 4 (~2:35-2:55)

Fourth: dynamic remeshing. As the surface deforms, triangle quality collapses without intervention. Each outer iteration we split edges longer than four-thirds of the target length and collapse edges shorter than four-fifths. A Delaunay flip pass restores good opposite-angle structure, then tangential smoothing toward the area-weighted circumcenter polishes the result, projected back onto the local tangent plane so we don't drift normal to the surface. A no-foldover guard rejects any collapse that would flip a face normal.

---

## Atharv, Slide 8: Canonical embeddings (~2:55-3:30)

Now the results. First, the canonical embedding gallery.

**[LIVE] Switch to Polyscope and play the genus-three descent.**

A procedural genus-three surface flowing under the tangent-point energy alone, no external forces. The three handles spread out and the body settles into the roughly tetrahedral shape the paper predicts. The plot shows energy curves for genus zero through five over five hundred iterations: genus zero collapses to a sphere, higher genera plateau into local minima with their handles intact.

---

## Atharv, Slide 9: Sphere through a tube (~3:30-4:05)

Same energy, lifted onto a path. The Repulsive Shells idea is that if you embed each shape as a pair, the shape itself plus its tangent-point energy value, the resulting graph manifold has a metric that diverges at self-contact, so a discrete geodesic cannot cross self-intersection. We minimize the path energy with trust-region Steihaug CG.

**[LIVE] Switch to ball-through-tube playback, multiple angles.**

Endpoints fixed; thirty interior frames optimized jointly. The sphere stretches through the tube without ever touching the wall, and self-repulsion drops two orders of magnitude in the first five iterations.

---

## Atharv, Slide 10: Limitations (~4:05-4:20)

Two real limitations. Everything runs CPU only; we descoped the GPU port after the milestone to land the path solver. And the second hero demo from Repulsive Shells, cloth eversion, isn't shipping; the missing piece is the mid-surface initialization, which the paper itself flags as needed for that example.

---

## Atharv, Slide 11: Future directions (~4:20-4:45)

Four natural extensions. Karcher means: weighted multi-mesh averaging via the same path solver. The packing demo from Repulsive Shells Figure 21, mostly a driver on the obstacle barrier we already have. The GPU port: cluster-tree traversal parallelizes well. And volumetric coupling, embedding thin repulsive shells inside elastic FEM solids, extends the framework beyond surfaces.

---

## Atharv, Slide 12: Summary (~4:45-5:00)

So we have a working repulsive-surface optimizer: fast tangent-point energy, Sobolev-preconditioned descent, dynamic remeshing, and a graph-manifold path solver, with two end-to-end demos. Thanks, happy to take questions.

---

## Q&A prep, likely questions

**Why TPE specifically and not IPC or some other repulsive potential?**
TPE has the right invariance: it's mesh-resolution-aware in the right way and continuous to all orders. IPC is great for contact but is fundamentally a barrier method, not a metric. The Crane group's whole pitch is that TPE turns the no-intersection condition into an intrinsic geometric quantity.

**Did you write the BVH from scratch?**
Yes, a 16-bin SAH BVH with cluster aggregates. Same module also builds the paired hierarchy.

**Why CPU and not GPU like the proposal?**
The paper bottlenecks on CPU at four to eight threads. We descoped GPU after the milestone to land the path solver instead: cleaner story, same scope of "replicate the paper."

**Does the energy ever blow up numerically?**
The signed power form `α · s · |s|^(α−2)` we use is well-defined at $s = 0$, and the BVH ensures that any pair close enough to numerically explode falls into the near-field exact path, not the cluster approximation.

**What's the biggest surprise in the project?**
That the path solver converged on the ball-through-tube case starting from a piecewise-constant initialization, without needing the manual mid-surface initialization we had braced for and which the Repulsive Shells paper itself flags as necessary for the harder cases.
