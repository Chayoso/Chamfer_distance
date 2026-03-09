% ==============================================================
% SUPPLEMENTARY: FORMAL PROOFS
% ==============================================================

\documentclass[runningheads]{llncs}
\usepackage[review,year=2026,ID=11274]{eccv}
\usepackage{eccvabbrv}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{amsmath}
\usepackage{amssymb}

\begin{document}

\title{On the Structural Failure of Chamfer Distance\\
in 3D Shape Optimization -- Supplementary Material}

\author{Anonymous}
\authorrunning{Anonymous}
\institute{}

\maketitle

\section{Formal Proofs}
\label{sec:proofs}

We provide complete proofs of Propositions~1--3 and Corollary~1 stated in the main paper.

\paragraph{Setup.}
Let $\mathcal{S} = \{p_1, \ldots, p_N\} \subset \mathbb{R}^3$ denote source points and $\mathcal{T} = \{q_1, \ldots, q_M\} \subset \mathbb{R}^3$ denote fixed target points.
The forward (s$\to$t) Chamfer loss is
\begin{equation}
\mathcal{L}_{\text{fwd}} = \frac{1}{N}\sum_{i=1}^{N}\|p_i - \mathrm{NN}_{\mathcal{T}}(p_i)\|^2,
\label{eq:fwd_loss}
\end{equation}
where $\mathrm{NN}_{\mathcal{T}}(p) = \arg\min_{q \in \mathcal{T}} \|p - q\|$.
The reverse (t$\to$s) Chamfer loss is
\begin{equation}
\mathcal{L}_{\text{rev}} = \frac{1}{M}\sum_{j=1}^{M}\|q_j - \mathrm{NN}_{\mathcal{S}}(q_j)\|^2.
\label{eq:rev_loss}
\end{equation}
All analysis is conducted within a fixed region of nearest-neighbor assignment (\ie, within a single Voronoi cell where $\mathrm{NN}_{\mathcal{T}}(p_i)$ does not change).

% --------------------------------------------------
\subsection{Proposition 1: Many-to-One Collapse Is the Unique Attractor}
\label{sec:proof_prop1}

\begin{proposition}
Consider a cluster $C = \{p_1, \ldots, p_k\}$ of source points sharing the same nearest target $q^* = \mathrm{NN}_{\mathcal{T}}(p_i)$ for all $i$.
Under the forward Chamfer gradient:
\begin{enumerate}
    \item[(a)] The state $p_i = q^*$ for all $i$ is the unique equilibrium of $C$.
    \item[(b)] The Hessian at this equilibrium is positive definite, making it a stable attractor.
\end{enumerate}
\end{proposition}

\begin{proof}
\textbf{Part (a).}
Within the Voronoi cell where $\mathrm{NN}_{\mathcal{T}}(p_i) = q^*$, the gradient of the forward loss with respect to $p_i$ is
\begin{equation}
\frac{\partial \mathcal{L}_{\text{fwd}}}{\partial p_i} = \frac{2}{N}(p_i - q^*).
\end{equation}
Setting this to zero yields $p_i = q^*$.
Since $q^*$ is fixed, this solution is unique for each $i \in \{1, \ldots, k\}$.
Hence the unique equilibrium is the many-to-one collapse state $p_1 = p_2 = \cdots = p_k = q^*$.

\textbf{Part (b).}
The Hessian of the forward loss with respect to $p_i$ is
\begin{equation}
\frac{\partial^2 \mathcal{L}_{\text{fwd}}}{\partial p_i^2} = \frac{2}{N}\, I_{3 \times 3},
\end{equation}
which is positive definite.
For the full cluster, the joint Hessian is $\frac{2}{N}\, I_{3k \times 3k}$ (block-diagonal with identical blocks), which is positive definite.
Therefore the collapse equilibrium is a stable local minimum and the unique attractor within this Voronoi cell.
\end{proof}

\begin{remark}
Globally, source space is partitioned into Voronoi cells of the target points.
Within each cell, Proposition~1 guarantees that all source points collapse onto the corresponding target point.
The global attractor is thus the many-to-one collapse state where each source point sits exactly on its nearest target.
Note that during gradient descent a source point may cross a Voronoi boundary and be reassigned to a different target; however, this merely redirects the point toward a \emph{new} attractor of the same kind, so the many-to-one collapse structure is preserved globally.
\end{remark}


% --------------------------------------------------
\subsection{Proposition 2: The Reverse Term Cannot Separate Collapsed Points}
\label{sec:proof_prop2}

\begin{proposition}
Let $k$ source points be collapsed at a target point: $p_1 = p_2 = \cdots = p_k = q^*$.
The reverse Chamfer gradient $\partial \mathcal{L}_{\emph{rev}} / \partial p_i$ is nonzero for at most one of the $k$ points; the remaining $k{-}1$ receive zero gradient.
\end{proposition}

\begin{proof}
The reverse loss sums over target points:
\begin{equation}
\mathcal{L}_{\text{rev}} = \frac{1}{M}\sum_{j=1}^{M}\|q_j - \mathrm{NN}_{\mathcal{S}}(q_j)\|^2.
\end{equation}
Source point $p_i$ contributes to $\mathcal{L}_{\text{rev}}$ only if $\mathrm{NN}_{\mathcal{S}}(q_j) = p_i$ for some target point $q_j$.
When $p_1 = \cdots = p_k = q^*$, all $k$ points occupy the same spatial location.
For any target point $q_j$ whose nearest source is at $q^*$, the nearest-neighbor search returns exactly one index $i^*$ via tie-breaking (typically the one with the smallest index).
Therefore:
\begin{equation}
\frac{\partial \mathcal{L}_{\text{rev}}}{\partial p_i} = 
\begin{cases}
\displaystyle -\frac{2}{M}\sum_{\substack{j:\, \mathrm{NN}_{\mathcal{S}}(q_j) = p_{i^*}}} (q_j - q^*) & \text{if } i = i^*, \\[8pt]
0 & \text{if } i \neq i^*.
\end{cases}
\end{equation}
At most one point ($i^*$) receives a nonzero gradient from the reverse term; the remaining $k{-}1$ points have zero reverse gradient.

Furthermore, at the collapse point the forward gradient is also zero (by Proposition~1), so these $k{-}1$ points have zero total gradient under the bidirectional objective.
They are stuck at $q^*$ with no mechanism for separation.
\end{proof}

\begin{remark}
This explains why bidirectional Chamfer distance cannot resolve collapse: the forward term is at equilibrium (Proposition~1), and the reverse term provides a separation signal to at most one of the $k$ collapsed points.
The remaining $k{-}1$ points are in a deadlock state with zero gradient from both terms.
\end{remark}


% --------------------------------------------------
\subsection{Proposition 3: Local Regularizers Cannot Alter Cluster-Level Drift}
\label{sec:proof_prop3}

\begin{proposition}
Let $C = \{p_1, \ldots, p_k\}$ be a cluster of source points sharing the same nearest target $q^* = \mathrm{NN}_{\mathcal{T}}(p_i)$.
Let $R$ be any local regularizer that is translationally invariant, i.e., $R(\{p_i + \mathbf{v}\}) = R(\{p_i\})$ for any $\mathbf{v} \in \mathbb{R}^3$.
Then the dynamics of the cluster centroid $\bar{p} = \frac{1}{k}\sum_{i} p_i$ under the combined loss $\mathcal{L}_{\emph{fwd}} + \lambda R$ are independent of $\lambda$:
\begin{equation}
\frac{d\bar{p}}{dt} = -\frac{2\eta}{N}(\bar{p} - q^*),
\end{equation}
where $\eta$ is the learning rate.
\end{proposition}

\begin{proof}
Under gradient descent, the centroid evolves as
\begin{equation}
\frac{d\bar{p}}{dt} = -\frac{\eta}{k}\sum_{i=1}^{k}\left(\frac{\partial \mathcal{L}_{\text{fwd}}}{\partial p_i} + \lambda \frac{\partial R}{\partial p_i}\right).
\end{equation}
We evaluate each sum separately.

\textbf{Forward term.}
\begin{equation}
\sum_{i=1}^{k} \frac{\partial \mathcal{L}_{\text{fwd}}}{\partial p_i} = \sum_{i=1}^{k} \frac{2}{N}(p_i - q^*) = \frac{2}{N}\left(\sum_{i=1}^{k} p_i - k\, q^*\right) = \frac{2k}{N}(\bar{p} - q^*).
\end{equation}

\textbf{Regularizer term.}
Since $R$ is translationally invariant, differentiating with respect to the translation parameter $\mathbf{v}$ at $\mathbf{v}=0$ gives $\sum_{i=1}^{k} \partial R / \partial p_i = 0$ directly.

\textbf{Combined.}
Substituting:
\begin{equation}
\frac{d\bar{p}}{dt} = -\frac{\eta}{k}\left(\frac{2k}{N}(\bar{p} - q^*) + \lambda \cdot 0\right) = -\frac{2\eta}{N}(\bar{p} - q^*).
\end{equation}
The centroid moves toward $q^*$ at a rate independent of $\lambda$.
Local regularizers may rearrange points within the cluster (\eg, push them apart via repulsion), but they cannot alter the cluster's net drift toward the target.
\end{proof}

\begin{remark}
Translational invariance is satisfied by all standard local regularizers used in point cloud optimization; pairwise antisymmetry (Newton's third law) is one sufficient condition for it:
\begin{itemize}
    \item \textbf{Repulsion}: $f(p_i, p_j) = \lambda_r (p_i - p_j) / \|p_i - p_j\|^{2+\alpha}$, which satisfies $f(p_i,p_j) = -f(p_j,p_i)$.
    \item \textbf{Laplacian smoothness}: The gradient $p_i - \frac{1}{|\mathcal{N}(i)|}\sum_j p_j$ decomposes into pairwise terms that cancel under summation over symmetric neighborhoods.
    \item \textbf{Volume preservation}: Local volume penalties produce gradients from pairwise particle interactions that obey Newton's third law.
    \item \textbf{Density-aware re-weighting (DCD)}: DCD reweights each forward contribution by $w_i = 1/\hat{\rho}(p_i)$, yielding a primary gradient $\frac{2w_i}{N}(p_i - \mathrm{NN}_{\mathcal{T}}(p_i))$ that preserves the direction toward the nearest target, plus a secondary term $\propto \|p_i - \mathrm{NN}_{\mathcal{T}}(p_i)\|^2\,\partial w_i/\partial p_i$ that vanishes at the collapse equilibrium $\|p_i - q^*\| = 0$.  Near collapse, DCD therefore leaves the attraction structure invariant, consistent with the identical t$\to$s values between DCO and DCD in Table~2.
\end{itemize}
\end{remark}


% --------------------------------------------------
\subsection{Corollary 1: Collapse Suppression Requires Non-Local Coupling}
\label{sec:proof_cor1}

\begin{corollary}
Taken together, Propositions~1 and~3 imply that any coupling mechanism whose gradients depend only on particles within a local neighborhood $\mathcal{N}(i)$ cannot, in general, suppress many-to-one collapse while still reducing the Chamfer objective.
Collapse suppression therefore requires coupling that propagates beyond $\mathcal{N}(i)$.
\end{corollary}

\begin{proof}[Argument]
Suppose a coupling mechanism $G$ suppresses collapse, but its gradient is local:
\begin{equation}
\frac{\partial G}{\partial p_i} = g\bigl(\{p_j : j \in \mathcal{N}(i)\}\bigr).
\end{equation}

\textbf{Case 1 (pairwise local forces).}
If $G$ is translationally invariant, then by Proposition~3 its net contribution to the cluster centroid vanishes.
The centroid dynamics remain governed solely by the forward Chamfer term, and collapse proceeds as in the unregularized case.

\textbf{Case 2 (general local forces).}
Even if $G$ does not decompose into pairwise forces, a local mechanism faces a tension arising from the positive-definite Hessian of Proposition~1.
The forward Chamfer gradient defines a strictly convex basin around $q^*$; any perturbation away from $q^*$ is restored by the $\frac{2}{N}I$ curvature.
A local coupling term $G$ can oppose this basin only within its neighborhood radius.
If $\|{\partial G}/{\partial p_i}\| < \|{\partial \mathcal{L}_{\text{fwd}}}/{\partial p_i}\|$ near $q^*$, the Chamfer basin dominates and collapse proceeds.
If $\|{\partial G}/{\partial p_i}\| > \|{\partial \mathcal{L}_{\text{fwd}}}/{\partial p_i}\|$, the coupling overwhelms the Chamfer signal in that region, effectively replacing the Chamfer objective with the regularizer's own objective.
Table~2 of the main paper confirms this trade-off empirically: strong local repulsion ($\lambda = 0.1$) worsens two-sided CD from 0.286 to 0.309.

In neither case can purely local coupling suppress collapse while improving the Chamfer objective.
Non-local coupling circumvents this trade-off by providing a restoring force that does not originate from the immediate neighborhood of the collapsing cluster, and therefore does not oppose the per-point Chamfer gradient locally.
In MPM, this is realized through continuum stress propagation on the shared Eulerian grid: moving a single particle generates elastic stress that propagates throughout the domain, coupling distant particles through the continuum.
\end{proof}


% ==============================================================
% SUPPLEMENTARY: IMPLEMENTATION DETAILS
% ==============================================================

\section{Implementation Details}
\label{sec:implementation}

\subsection{MPM Simulation Parameters}
\label{sec:sim_params}

All experiments use a differentiable Material Point Method (MPM) simulation based on the framework of Xu~et~al.~\cite{xu2025differentiable}.
Table~\ref{tab:sim_params} summarizes the simulation parameters shared across all experiments.

\begin{table}[h]
\centering
\caption{MPM simulation parameters. All experiments share the same material model and grid configuration; only particle resolution varies.}
\label{tab:sim_params}
\begin{tabular}{lcc}
\toprule
Parameter & Symbol & Value \\
\midrule
\multicolumn{3}{l}{\textit{Grid}} \\
\quad Grid spacing & $\Delta x$ & 1.0 \\
\quad Domain & $[\mathbf{x}_{\min}, \mathbf{x}_{\max}]$ & $[-16, 16]^3$ \\
\quad Effective resolution & & $32^3$ nodes \\
\midrule
\multicolumn{3}{l}{\textit{Material (Neo-Hookean)}} \\
\quad First Lam\'e parameter & $\lambda$ & 38\,889 \\
\quad Shear modulus & $\mu$ & 58\,333 \\
\quad Density & $\rho$ & 75.0 \\
\midrule
\multicolumn{3}{l}{\textit{Time integration}} \\
\quad Timestep & $\Delta t$ & $8.33 \times 10^{-3}$ \\
\quad Steps per frame & & 10 \\
\quad Drag coefficient & & 0.5 \\
\quad Smoothing factor & $\alpha_s$ & 0.955 \\
\midrule
\multicolumn{3}{l}{\textit{Particles}} \\
\quad Points per cell (default) & PPC$^{1/3}$ & 3 (${\sim}$37K particles) \\
\quad Points per cell (dragon) & PPC$^{1/3}$ & 4 (${\sim}$89K particles) \\
\bottomrule
\end{tabular}
\end{table}

The Lam\'e parameters correspond to Young's modulus $E = 140\,000$ and Poisson's ratio $\nu = 0.20$, modeling a stiff elastic solid.
The drag coefficient provides velocity damping at each timestep to stabilize the simulation.
The smoothing factor $\alpha_s$ controls exponential smoothing of the deformation gradient field across frames, with $\alpha_s = 0.955$ for the pairwise evaluation and $\alpha_s = 0.95$ for the teaser experiments.

\subsection{Optimization Parameters}
\label{sec:opt_params}

Table~\ref{tab:opt_params} summarizes the optimization and Chamfer schedule parameters.

\begin{table}[h]
\centering
\caption{Optimization and Chamfer schedule parameters.}
\label{tab:opt_params}
\begin{tabular}{lcc}
\toprule
Parameter & Symbol & Value \\
\midrule
\multicolumn{3}{l}{\textit{Gradient descent}} \\
\quad Number of frames & $T$ & 40 \\
\quad Forward--backward passes & & 3 \\
\quad GD iterations per pass & & 1 \\
\quad Line search iterations & & 5 \\
\quad Initial step size & $\alpha_0$ & 0.01 \\
\quad Adaptive step target norm & & 2\,500 \\
\midrule
\multicolumn{3}{l}{\textit{Chamfer schedule}} \\
\quad Chamfer weight & $w_{\text{ch}}$ & 10.0 \\
\quad Chamfer start frame & $t_{\text{start}}$ & 5 \\
\quad Chamfer ramp frames & $t_{\text{ramp}}$ & 10 \\
\quad Reverse weight (base) & $w_{\text{rev,base}}$ & 1.0 \\
\quad Reverse mode & & coupled (Eq.~6) \\
\quad Reverse gradient clamp ratio & $\kappa$ & 3.0 \\
\midrule
\multicolumn{3}{l}{\textit{Physics decay (Phase 2)}} \\
\quad Decay start frame & & 15 \\
\quad Decay ramp frames & & 5 \\
\quad Final physics weight & $w_{\text{phys,final}}$ & 0.1 \\
\quad Smoothing (Phase 2) & & 0.7 (from frame 15) \\
\bottomrule
\end{tabular}
\end{table}

The Chamfer loss is linearly ramped from frame~5 to frame~15 ($w_{\text{ch}}(t) = w_{\text{ch}} \cdot \min(1, (t - t_{\text{start}}) / t_{\text{ramp}})$ for $t \geq t_{\text{start}}$, zero otherwise).
The physics weight decays linearly from 1.0 to $w_{\text{phys,final}} = 0.1$ over frames 15--20, transitioning from physics-dominated to Chamfer-guided refinement.
The coupled reverse schedule (Eq.~6 in the main paper) ties $w_{\text{rev}}(t) = w_{\text{rev,base}} \cdot w_{\text{phys}}(t)$, ensuring that reverse (t$\to$s) pressure never exceeds the elastic resistance provided by the physics prior.

\paragraph{Dragon case study.}
The Sphere$\to$dragon experiment (Section~5.4 of the main paper) uses 4 particles per cell (${\sim}$89K particles) with a C++ inline bidirectional Chamfer gradient implementation for computational efficiency.
All other parameters remain identical to the default configuration.

\paragraph{Pairwise evaluation.}
The 20-pair evaluation (Section~5.3) uses 3 particles per cell with the same schedule.
For pairs with non-sphere sources, the source mesh is sampled to match the particle count of the corresponding sphere initialization.


% ==============================================================
% SUPPLEMENTARY: ADDITIONAL QUANTITATIVE RESULTS
% ==============================================================

\section{Additional Quantitative Results}
\label{sec:additional_results}

\subsection{Pairwise Source-to-Target Chamfer Distance}
\label{sec:pairwise_s2t}

Table~\ref{tab:pairwise_s2t} reports the s$\to$t component of Chamfer distance for all 20 directed morphing pairs.
Our method improves s$\to$t on all 20 pairs, confirming that the forward Chamfer improvement is universal across source geometries.

\begin{table}[h]
\centering
\scriptsize
\caption{Pairwise morphing results (s$\to$t CD) in matrix format.  Each cell shows Physics\,/\,\textbf{Ours}.  \textbf{Bold}: improvement over physics-only.  Our method improves s$\to$t on all 20 directed pairs.}
\label{tab:pairwise_s2t}
\setlength{\tabcolsep}{3pt}
\begin{tabular}{l ccccc}
\toprule
Src\,$\downarrow$\,/\,Tgt\,$\rightarrow$ & Sphere & Bunny & Duck & Cow & Teapot \\
\midrule
Sphere & ---                      & 0.181\,/\,\textbf{0.157} & 0.184\,/\,\textbf{0.173} & 0.178\,/\,\textbf{0.167} & 0.180\,/\,\textbf{0.148} \\
Bunny  & 0.199\,/\,\textbf{0.159} & --- & 0.186\,/\,\textbf{0.167} & 0.178\,/\,\textbf{0.159} & 0.187\,/\,\textbf{0.150} \\
Duck    & 0.181\,/\,\textbf{0.168} & 0.183\,/\,\textbf{0.178} & --- & 0.184\,/\,\textbf{0.176} & 0.190\,/\,\textbf{0.172} \\
Cow   & 0.186\,/\,\textbf{0.167} & 0.188\,/\,\textbf{0.156} & 0.325\,/\,\textbf{0.168} & --- & 0.193\,/\,\textbf{0.161} \\
Teapot & 0.176\,/\,\textbf{0.146} & 0.178\,/\,\textbf{0.165} & 0.273\,/\,\textbf{0.176} & 0.178\,/\,\textbf{0.169} & --- \\
\bottomrule
\end{tabular}
\end{table}


% --------------------------------------------------
\subsection{2D Cross-Domain Collapse Experiment}
\label{sec:2d_collapse}

We provide full implementation details for the 2D collapse experiment presented in (Figure~3) of the main paper.

\paragraph{Setup.}
A source set of $N{=}600$ points uniformly sampled on a circle boundary ($r{=}0.8$, centered at the origin) is optimized to match a target star boundary ($M{=}200$ points, 5 arms, inner radius 0.25, outer radius 1.0).
We compare three optimization strategies, all running for 300 gradient steps with bidirectional Chamfer loss ($w_{\text{fwd}}{=}w_{\text{rev}}{=}1.0$) and 95th-percentile gradient clipping.

\begin{enumerate}
    \item \textbf{DCO} (per-point CD optimization, lr${=}0.015$): Each source point is independently moved along its Chamfer gradient. As predicted by Proposition~1, points collapse onto the nearest star vertices, leaving large target regions uncovered.
    \item \textbf{DCO + Repulsion} (lr${=}0.015$, $k$-NN with $k{=}6$, $\lambda_{\text{rep}}{=}2{\times}10^{-4}$): A local repulsive force ($1/d^2$) from the 6 nearest neighbors is added to the Chamfer gradient. Consistent with Proposition~3, the centroid drift is unchanged and collapse persists.
    \item \textbf{Shared-basis deformation} (lr${=}0.003$): The boundary is parameterized as $r(\theta) = a_0 + \sum_{k=1}^{K}[a_k \cos(k\theta) + b_k \sin(k\theta)]$ with $K{=}12$ Fourier modes. All points share the same coefficients $\{a_k, b_k\}$; the CD gradient backpropagates through the parameterization to update these shared coefficients. This provides global coupling analogous to how MPM couples particles through a shared Eulerian grid. $K{=}12$ is chosen to be the minimum number of modes that can represent a 5-arm star (5 concavities require at least $k{=}5$ harmonics; doubling to $K{=}12$ provides sufficient capacity while remaining low-dimensional).
\end{enumerate}

\paragraph{Metrics.}
We report three quantitative measures:
\begin{itemize}
    \item \textbf{Two-sided CD}: the standard bidirectional Chamfer distance between the optimized source and the target.
    \item \textbf{Coverage} (\%): the fraction of target points that have at least one source point within a distance threshold $\epsilon{=}0.05$.
    \item \textbf{Cluster fraction} (\%): the fraction of source points that are within $\epsilon{=}0.02$ of another source point, measuring the degree of many-to-one collapse.
\end{itemize}

\paragraph{Results.}
Table~\ref{tab:2d_results} summarizes the quantitative comparison.

\begin{table}[h]
\centering
\caption{Quantitative results of the 2D collapse experiment (circle $\to$ star).}
\label{tab:2d_results}
\begin{tabular}{lccc}
\toprule
Method & Two-sided CD $\downarrow$ & Coverage (\%) $\uparrow$ & Cluster Fraction (\%) $\downarrow$ \\
\midrule
DCO & 0.352 & 38 & 97 \\
DCO + Repulsion & 0.310 & 50 & 97 \\
Shared-basis (Fourier, $K{=}12$) & \textbf{0.123} & \textbf{81} & \textbf{52} \\
\bottomrule
\end{tabular}
\end{table}

DCO collapses nearly all source points onto the five star vertices (cluster fraction 97\%), covering only 38\% of the target.
Adding local repulsion marginally improves coverage (50\%) but does not reduce the cluster fraction, confirming Proposition~3: translational invariance ensures the regularizer gradient sums to zero, so the net drift toward the target is unchanged.
The shared-basis deformation reduces two-sided CD by 65\% (0.352 $\to$ 0.123) and cluster fraction from 97\% to 52\%, demonstrating that global coupling through shared parameters effectively suppresses collapse, consistent with Corollary~1.

The corresponding visualizations are shown in (Figure~3) of the main paper.


% ==============================================================
% SUPPLEMENTARY: ADDITIONAL FIGURES AND ABLATION
% ==============================================================

\section{Additional Figures and Ablation}
\label{sec:additional_figures}

\subsection{Frame-Wise Chamfer Distance Convergence}
\label{sec:cd_convergence}

\begin{figure}[ht]
\centering
\includegraphics[width=\linewidth]{figs/CD_LOSS.png}
\caption{Frame-wise source-to-target Chamfer distance across four target shapes. DCO and DCD converge to substantially worse plateaus on geometrically complex targets, while our method consistently refines the physics prior. On the nearly convex teapot, DCO achieves comparable s$\to$t but at the cost of structural collapse.}
\label{fig:CD_Loss_suppl}
\end{figure}

Figure~\ref{fig:CD_Loss_suppl} shows the s$\to$t Chamfer distance over 40 frames for all four target shapes. DCO and DCD plateau early at substantially worse values on geometrically complex targets, while our method continues to improve as the schedule transitions from physics-dominated to Chamfer-guided refinement.


\subsection{Trajectory and Internal Structure Comparison}
\label{sec:trajectory_suppl}

The trajectory and cross-section comparison between DCD and our method (Cow$\to$Duck) is presented in the main paper (Figure~5).
This comparison highlights the interior hollowing artifact unique to collapse-prone objectives, which global coupling prevents throughout the trajectory.


\subsection{Pairwise Scatter Plot}
\label{sec:pairwise_scatter_suppl}

\begin{figure}[ht]
\centering
\includegraphics[width=0.7\linewidth]{figs/pairwise_scatter.png}
\caption{\textbf{Pairwise two-sided CD: Physics-only vs.\ Ours.}
Each of the 20 directed pairs is one point; marker shape indicates the source.
Points below the diagonal (16/20) denote improvement by our method.
The four regressions all involve convex targets (Sphere or Cow) where the physics baseline already achieves near-optimal coverage.}
\label{fig:pairwise_scatter_suppl}
\end{figure}

Figure~\ref{fig:pairwise_scatter_suppl} visualizes the per-pair two-sided CD comparison across all 20 directed morphing pairs. Points below the diagonal indicate improvement by our method; the four regressions above the diagonal correspond to convex targets where the physics baseline already achieves near-optimal coverage.


\subsection{Ablation Study}
\label{sec:ablation_suppl}

We ablate the key design choices of our method on the Sphere$\to$bunny pair (Table~\ref{tab:ablation_suppl}).
Most configurations fall within $\pm 0.003$ of the full method in s$\to$t, indicating broad robustness.
The main sensitivity is to the clamp ratio~$\kappa$: aggressive clamping ($\kappa{=}1$) discards gradient signal, degrading two-sided CD by $+$0.019, while relaxing to $\kappa{=}5$ slightly improves s$\to$t at the cost of t$\to$s.
The coupled reverse schedule shows clear asymmetry: a fixed $w_{\text{rev}}{=}0.5$ underperforms ($+$0.015) because weakened reverse pressure cannot counteract many-to-one collapse.
Physics decay timing between frames~15 and~25 has negligible effect ($\leq$0.014).

\begin{table}[ht]
\centering
\scriptsize
\caption{Ablation study on Sphere$\to$bunny. Each row modifies one component from the full method.  Our default configuration achieves the best two-sided CD.}
\label{tab:ablation_suppl}
\begin{tabular}{lccc}
\toprule
Configuration & s$\to$t $\downarrow$ & t$\to$s $\downarrow$ & Two-sided $\downarrow$ \\
\midrule
Full method (Ours) & \textbf{0.157} & \textbf{0.163} & \textbf{0.226} \\
\midrule
\multicolumn{4}{l}{Coupled schedule} \\
\quad Fixed $w_{\text{rev}} = 1.0$ (no coupling) & 0.158 & 0.176 & 0.237 \\
\quad Fixed $w_{\text{rev}} = 0.5$ & 0.158 & 0.182 & 0.241 \\
\midrule
\multicolumn{4}{l}{Clamp ratio $\kappa$} \\
\quad $\kappa = 1$ & 0.162 & 0.184 & 0.245 \\
\quad $\kappa = 5$ & 0.156 & 0.179 & 0.237 \\
\midrule
\multicolumn{4}{l}{Physics decay timing} \\
\quad Decay at frame 15 & 0.158 & 0.179 & 0.239 \\
\quad Decay at frame 25 & 0.158 & 0.180 & 0.240 \\
\bottomrule
\end{tabular}
\end{table}

\end{document}
