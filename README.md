# Approximating images with text

Approximation   |  Targeted Image ([Friday Pilots Club](https://open.spotify.com/artist/3PgEvPJKRuil74QPX8wtXY?si=pXjOQP_ESNKRMYRthsO5uQ))
:-------------------------:|:-------------------------:
<img src="https://github.com/9Dread/stochastic-text-image-approximator/blob/main/example/ex_run.gif" alt="Algorithm example run gif" style="width:50%; height:auto;">  |  <img src="https://github.com/9Dread/stochastic-text-image-approximator/blob/main/example/fpc_in.png" alt="Targeted image of the band Friday Pilots Club" style="width:50%; height:auto;">

This repository contains a toy project that takes as input an arbitrary target
image, font file, and set of valid characters, and paints characters starting
from a blank image in order to approximate the target image to high precision.
The algorithm finds good characters to paint by constructing a distribution
over all possible paintable characters (and their characteristics i.e. size,
rotation, position) with probability density concentrated on characters which
are "good" according to a variety of heuristics. It then samples from this
distribution many times (running in parallel using a user-specified number of
threads) per iteration, and an iteration is completed by selecting the best
character to paint from all sampled characters.

A description of the algorithm and the CLI is given below.

# Algorithm

We first take as input a number of iterations to complete and optionally a
starting approximation image (which is useful if running the algorithm again to
continue after previous termination). If a starting image is not provided, we
start from an all-black image. Broadly speaking, one glyph is painted at each
iteration, although this may be skipped if no glyph which improves error is
found. 

Further details of certain topics with references are provided in the Further
Details section. Each iteration is completed as follows. 

## 1. Setup

Using the current approximation and the target image, we update a one-channel
image of absolute residuals and square residuals. We then construct a blurred version
of the residuals, and then construct an alias table on this image for sampling
center positions according to pixels with high error (blurring reduces the
effect of sharp edges). My alias table implementation can be found in [my other
repository](https://github.com/9Dread/odin-alias-sampling).

## 2. Sampling glyphs

The thread pool dispatches a signal telling the worker threads to wake up and
start sampling glyphs. Glyphs are sampled until a specified total number of
glyphs are created, the default being 64. Each step in the sampling procedure
and rationale is as follows.

### 2a. Sample a pixel for the glyph to be centered on.

This is done using the alias table constructed in the setup.

### 2b. Sample a rotation angle of the glyph

This is done uniformly, because honestly anything else would be too much a
bother. The suitability of an angle depends on the character itself. For
instance, rotating an 'o' would have no effect on the area it covers, but
rotating an 'l' has a huge effect on the area it covers. This can just as
easily be flipped though, and you could say that the effectiveness of a
specific character depends on the rotation and position. So one must decide
which to sacrifice if we are to avoid computing a huge number of combinations
to find the best configuration. Here, we sacrifice finding a good rotation.

### 2c. Sample the point size of the glyph

Point sizes are discretized on a logarithmic scale (base 2), starting from half
the height of the image and going down to 12 point. 

A certain scale is heuristically "good" if a character of that size has a lot
of high-error area to cover. To compute a metric for this, we take the
rectangular frame of the first character in the dictionary (without rotation,
although looking back it is probably worth incorporating rotation) and center
it on the position sampled in 2a. Then we simply sum the squared residuals
covered by this frame. To prevent bigger scales from always being better, we
normalize (divide) by the number of pixels in the frame, so we compute the
average square residual per pixel.

We then award probability mass to each size according to this metric and sample
a size on the constructed distribution, again using an alias table. This is
done using a softmax (see the softmax section under Further Details).

### 2d. Sample the character identity of the glyph

The goal is now to select a character from the dictionary to paint, e.g. to
decide whether painting 'A' or 'Z' is better (if those characters are both in
the dictionary). This was probably the most interesting step to think about. At
this stage, we have a position, a rotation angle, and a size. This means that,
for each character in the dictionary, we have enough information to create a
bitmap (a one-channel alpha mask of the character) and preview exactly how it
will be oriented on the image when it is painted. Just as for point size, we
construct a heuristic using this information and sample using a softmax.

We gather the bitmaps of each character in the dictionary and rotate and scale
them according to the samples from 2b and 2c. Let $m$ denote the bitmap mask
viewed as a vector of pixels and $r$ be the residual vector restricted to the
region covered by $m$. Then the score of the glyph is:

$$\text{score}(m,r) = \frac{\langle m,r \rangle^2}{\lVert m \rVert ^2}$$

where $\langle \cdot, \cdot \rangle$ is the inner product and $\lVert \cdot
\rVert$ is the vector magnitude. Here we are trying to determine whether the
character's spatial structure aligns with the residual vector, which is why the
inner product makes intuitive sense. More specifically, by
[Cauchy-Schwarz](https://en.wikipedia.org/wiki/Cauchy%E2%80%93Schwarz_inequality),
we have that $\langle m, r \rangle^2 \leq \lVert m \rVert ^2 \lVert r \rVert
^2$ so that $\frac{\langle m, r \rangle ^2}{\lVert m \rVert ^2} \leq \lVert r
\rVert ^2$ with equality exactly when $m$ is proportional to $r$. This means
that the score is maximized when a glyph mask is fully aligned with the
residual pattern in its region.

### 2e. Compute the optimal color of the glyph

We now know exactly what character will be painted and where. The optimal color
for the glyph can be computed analytically. Let $c_p \in \mathbb{R}^3$ be the
current image color at pixel $p$, $t_p \in \mathbb{R}^3$ be the target image
color at pixel $p$, $\alpha_p \in \[0,1\]$ be the glyph mask value at pixel
$p$, and $g \in \mathbb{R}^3$ be the glyph color we want to choose. After we
paint our glyph, the new color of pixel $p$ is $(1-\alpha_p)c_p + \alpha_p g$.
We thus want to minimize:

$$\sum_p \lVert t_p - ((1-\alpha_p)c_p+\alpha_p g) \rVert ^2$$

Because each RGB channel separates, the optimal value for each channel is the
same formula. With some calculus we can find this to be (with $C$ corresponding
to a specific channel):

$$g_C = \frac{\sum_p \alpha_p(t_{p,C}-(1-\alpha_p)c_{p,C})}{\sum_p \alpha_p^2}$$

$$ = \frac{\sum_p (\alpha_p t_{p,C}+(\alpha_p^2 - \alpha_p)c_{p,C})}{\sum_p \alpha_p^2}$$

## 3. Processing proposals

Once all proposals are computed, the worker threads signal completion, and once
all of them signal completion, the main thread wakes up. It then selects the
best glyph from the proposals and discards the rest. The best proposal is
painted if it leads to an improvement in error.

## Further details

### Softmax

Throughout the algorithm we often compute a heuristic of "good approximation"
over a number of possibilities for a characteristic of a proposal glyph (e.g.
size), and we want to sample a value of the characteristic according to how
well it scores on the heuristic. But using the raw heuristic score doesn't work
because the scale of a heuristic is not always good for sampling probabilities,
e.g. most values of the heuristic can be close together, giving us a very
uniform distribution, which isn't what we want. Instead we would rather sample
by emphasizing *differences* between the elements of the heuristic vector. This
is done done using a [softmax](https://en.wikipedia.org/wiki/Softmax_function).
The softmax transforms a real-number vector (in this case our heuristic vector)
into weights:

$$w_i = \exp(\lambda(a_i-a_{\max}))$$

where $a_i$ is the real score, $a_{\max}$ is the maximum score in the vector,
and $\lambda$ is a hyperparameter which controls how sharp the distribution is
(how much it favors higher scores over lower scores). Here $a_{\max}$ is subtracted simply
for numerical stabilization. These weights are normalized to one to give probabilities:

$$p_i = \frac{w_i}{\sum_j w_j}$$

Now the question is how to set $\lambda$. This is also not the same over all
heuristics because heuristics have different scales. To fix this we maintain an
odds ratio $R$ such that the ratio between the probabilities of the best
candidate and the $k$-th worst candidate can be controlled. Then we set

$$\lambda = \frac{\ln R}{a_{\max} - a_k}$$

To promote early exploration we start with a lower $R$ and gradually increase
it with more iterations to favor exploitation of the best candidates. The formula is 

$$R = 10 + \sqrt{\text{iterations completed}}$$

which is rather simple but works well enough. $R$ is also capped at $80$.

### Computing rotation and scaling of characters

Rotations and size scaling of the character bitmaps themselves are computed
using [signed distance
fields](https://en.wikipedia.org/wiki/Signed_distance_function) of the base
character from the ttf file. 

# Usage and command line interface

To install, download the repository and build the `src` directory with `odin
build`. The executable is run with a number of input flags described below:

`--out \[STRING\]` - (REQUIRED) path to a directory to save results.

`--in \[STRING\]` - (REQUIRED) path to the image to approximate, either a png
or a jpg. Sometimes odin's jpg parser may fail due to an unsupported format; if
this happens simply convert the image to a png. 

`--ttf \[STRING\]` - (REQUIRED) path to the ttf font file to use. This is where
we get the shape of each input character.

`--dict \[STRING\]` - (REQUIRED) dictionary of ASCII characters to use to
approximate the image. These should be comma-separated, for instance `--dict
A,a,B,b,C,c,1,2,3`.

`--iterations \[INTEGER\]` - (REQUIRED) number of iterations for the algorithm to run.

`--threads \[INTEGER\]` - (OPTIONAL) number of worker threads to spawn for
parallelism. Note that these are in addition to the main thread. As usual, you
should refrain from spawning more threads than cores available in your cpu. If
this option is not provided, it defaults to 1 (not parallel).

`--save_iters \[INTEGER\]` - (OPTIONAL) the algorithm saves an image at regular
intervals to the directory provided with `--out`. This controls the number of
iterations between saves and defaults to 20.

`--start \[STRING\]` - (OPTIONAL) path to a starting image if you would not
like the algorithm to start from scratch. This is useful for continuing an
approximation after an earlier termination of the algorithm; simply provide the
path to the last saved iteration.

`--iters_done \[INTEGER\]` - (OPTIONAL) a way for one to tell the algorithm how
many iterations were completed in a previous run *if the algorithm is
continuing after previous termination*. For instance if 2000 iterations were
previously ran before termination and one would like to resume, one can provide
the path using `--start` to the 2000th-iteration image (saved as `iter2000.png` in the `--out`
directory) and set `--iters_done 2000`. This ensures output images are saved
with correct names and that the odds ratio $R$ is computed correctly in the
algorithm.

`--proposal_count \[INTEGER\]` - (OPTIONAL) the number of characters to propose
at each iteration. Defaults to 64.

## Example calls

Starting from previous termination:

`./src --out /path/to/output/directory --in /path/to/target/image --start /path/to/output/directory/iter2000.png --iters_done 2000 --ttf .../LiberationMono/LiterationMonoNerdFont-Bold.ttf --threads 12 --dict F,P,C --iterations 5000 --save_iters 200`

Starting from scratch:

`./src --out /path/to/output/directory --in /path/to/target/image --ttf .../LiberationMono/LiterationMonoNerdFont-Bold.ttf --threads 12 --dict F,P,C --iterations 5000 --save_iters 50`


