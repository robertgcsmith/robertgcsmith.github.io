---
layout: post
title: "Research notes #1: Convergence is not counting - Some thoughts on shortcut learning and mathematical extrapolation in neural networks"
date: 2026-11-20
image: https://upload.wikimedia.org/wikipedia/commons/a/a3/Gradient_descent.gif?utm_source=commons.wikimedia.org&utm_campaign=index&utm_content=original
banner: https://upload.wikimedia.org/wikipedia/commons/a/a3/Gradient_descent.gif?utm_source=commons.wikimedia.org&utm_campaign=index&utm_content=original
banner_alt: "Gradient descent with 3 different initial conditions. "
banner_caption: "Gradient descent with 3 different initial conditions."
banner_credit: "Wikipedia. Creative Commons, courtesy of Jacopo Bertolotti."
category: Mathematical Reasoning
abstract: Some thoughts on shortcut learning, optimisation, and mathematical extrapolation in neural networks. I discuss why successful optimisation need not imply that a network has learned a reusable mathematical procedure.
tags: [Machine Learning, Shortcut Learning, Mathematical Reasoning, Out-of-Distribution Generalisation, Optimisation, Recurrent Neural Networks]
mathjax: true
permalink: /convergence-is-not-counting/
comments: true
---
One of the questions I have been thinking about in my research on mathematical reasoning and extrapolation is what it actually means for a neural network to have learned a mathematical computation.

It is a deceptively simple question. A network might achieve very low training loss, converge under a chosen optimisation procedure, and perform impressively on held-out examples; yet it may still fail when presented with inputs that differ systematically from those encountered during training.

There is a paper by [Geirhos et al.](https://www.nature.com/articles/s42256-020-00257-z)(2020), titled *Shortcut Learning in Deep Neural Networks*, that I find particularly interesting in this context. The authors discuss how neural networks can learn decision rules that are remarkably successful under familiar statistical conditions but fail when those conditions change. This is a rather important observation, because it raises questions not only about how well a neural network predicts, but also about what it has actually *learned*.

For mathematical reasoning, the distinction is especially interesting. Suppose we train a network to perform basic arithmetic operations including addition, subtraction, multiplication, division, and powers. Has it learned a reusable computational rule, or has it learned statistical relationships that happen to predict the right answers over the range of examples it has encountered? The same question arises for more complicated symbolic expressions, where a model might exploit sequence length, operator frequencies, or patterns of expression structure without implementing the intended mathematical interpretation. One of my current research projects investigates this problem using recurrent neural networks, particularly gated recurrent units (GRUs), as a controlled setting for studying mathematical generalisation.

There is a long history of using artificial neural networks to model aspects of human numerical cognition. For example, [Zorzi](https://escholarship.org/uc/item/00q3v9j1) (1999) developed a connectionist model of number comparison that reproduced phenomena such as the numerical distance effect, whereby comparing nearby numerical magnitudes tends to take longer than comparing more widely separated ones. This is interesting if our goal is to understand human cognition. But should we necessarily want a neural network performing symbolic arithmetic to exhibit similar limitations? If a machine has learned to add, surely we would like it to apply the same arithmetic rule to numbers outside the range on which it was trained? There are, of course, practical limits involving numerical representation, precision, and computational resources; nevertheless, the ambition is to develop computational representations and procedures whose correctness is not merely an accident of the training distribution.

If we don't want to build networks bounded by such biases for learning arithmetic, it then follows that, if a machine can add, it should be able to be able to perform the operation of addition regardless of the quantity and with the same confidence each time. This reliability is essential. This is really just another way of saying we want the representation and processing of the numbers to guarantee extrapolation on OOD data. While some of the most advanced models and networks are capable of producing the correct results on lengthy calculations; when it comes reasoning models, it is not perfect and can still fail on even the basic arithmetic operations such as multiplication. There exists reliance on how inputs are encoded and habits towards shortcutting. Furthermore, there due to the lack of transparency of the parameters of the architecture, there is no guarantee of the network’s ability to execute systematic solutions for mathematical tasks.

I find this particularly relevant to applications such as equation discovery and mathematical modelling in physics, where the ability to extrapolate beyond familiar observations is essential.

One thing that makes the problem fascinating is that successful optimisation does not resolve it. Indeed, a model might converge quite happily to a predictor that is less systematic than we would like.

## What does convergence actually tell us?
In supervised learning, we typically train a parameterised model $f_\theta$ by minimising an empirical objective of the form

$$
\mathcal L(\theta)=
\frac{1}{N}\sum_{i=1}^{N}
\ell(f_\theta(x_i),y_i),
$$

where $\theta$ denotes the model parameters, $\ell$ is a loss function, and $x_i,y_i$ are training examples. For ordinary gradient descent, we update the parameters according to the familiar equation

$$
\theta_{k+1} = \theta_k-\eta\nabla_\theta\mathcal L(\theta_k),
$$

which tells us how the parameters change (with $\eta>0$ the learning rate). Notice, however, that it does not tell us which mathematical procedure the network is learning.

In practical deep learning, convergence often refers to a training process in which the objective has stabilised, or further training produces little meaningful improvement. More formally, we might distinguish convergence of the parameters, convergence of objective values, and convergence towards stationary points. These are not identical properties. Moreover, a plateau in the loss function does not prove that the optimiser has reached a local minimum, let alone a global one. In stochastic optimisation, parameters may continue to fluctuate even when the overall learning process has become relatively stable.

There is also an interesting distinction between convergence and generalisation. A network can attain very low empirical loss while learning a predictor that fails under a distribution shift. Conversely, generalisation can sometimes improve long after the training data has been fitted successfully. One rather striking example is grokking, investigated by [Power et al.](https://arxiv.org/abs/2201.02177)(2022), where neural networks trained on small algorithmic datasets can exhibit a substantial delay between fitting the training examples and generalising to held out examples. This does not mean that further training inevitably produces a more systematic mathematical algorithm, nor that grokking necessarily implies extrapolation to greater input lengths or mathematical complexity. But it illustrates why the behaviour of training loss alone is insufficient to tell us what a network has learned.

Furthermore, there is a temptation to describe a network exploiting statistical correlations as a kind of "lazy interpolator". I rather like the intuition behind this phrase, although, as we shall see, lazy training already has a specific technical meaning in the literature. More importantly, shortcut learning need not be a property of an unconverged model. A network can converge while continuing to depend on a predictor that works well in one statistical environment but fails in another.

To see why this matters, I want to consider two rather simple examples. In the first, we deliberately make a statistical shortcut available to a recurrent neural network. In the second, we find that a useful shortcut emerges naturally from the probability distribution of a counting problem.

## An artificial shortcut: counting, or reading the answer?
Suppose we construct a binary classification task using strings containing 64 symbols, each either $A$ or $B$. For one class, the string contains exactly 31 occurrences of $A$. For the other, it contains exactly 33. The symbols are randomly arranged, and the correct classification depends only on their count. We can express this mathematically as

$$
y(x)=
\begin{cases}
0,&\text{if } N_A (x)=31,\\
1,&\text{if } N_A (x)=33.\\
\end{cases}
$$

There is no ambiguity about the mathematical rule. To classify every valid example correctly, the network must distinguish the two counts.

Now suppose we add one additional symbol at the end of every string. This new symbol is not part of the original counting task. It merely acts as a cue, indicating the correct class with probability $0.95$ during training. Suddenly, there are two predictive strategies available to the network: one is to process the original 64 symbols and distinguish the two counts, while the other is to look at the final cue and predict whichever class it indicates.

This is the basic experiment I want to consider. We can generate the dataset in PyTorch as follows:

```python
import torch
from torch import nn

torch.manual_seed(42)

def examples(n):
    y = torch.randint(0, 2, (n,))

    # Class 0: 31 As; class 1: 33 As.
    count_A = 31 + 2 * y

    ranks = torch.rand(n, 64).argsort(1)
    content = (ranks < count_A[:, None]).long()

    # A cue that agrees with the label 95% of the time.
    agrees = torch.rand(n) < 0.95
    cue_label = torch.where(agrees, y, 1 - y)

    # Content symbols: 0, 1; cue symbols: 2, 3.
    x = torch.cat(
        [content, (2 + cue_label)[:, None]], dim=1
    )
    return x, y
```

Here, the target label is determined exclusively by the first 64 tokens. The reason is because in the PyTorch training loop, examples() generates fresh online batches at every step, and the first 64 tokens determine $y$ noiselessly (i.e., $100\%$ Bayes optimal accuracy, $0$ cross-entropy loss). By contrast, a pure cue-only classifier has a strict population cross-entropy floor of

$$H_b(0.95) = -(0.95 \ln 0.95 + 0.05 \ln 0.05) \approx 0.1985 \text{ nats}.$$

Under online sampling, the cue only strategy should not be a global minimum of the objective function; in actuality, it should be a low-effort plateau approximately 0.20 nats above the counting solution. The main point is that, since we have intentionally positioned the cue at the final timestamp $t = 65$, there is an extreme asymmetric imbalance in backpropagation (BPTT) such that the error gradient should flow directly from the output layer into the final hidden state (i.e., $h_{65}$). And since we are working with a basic GRU, the weights for reading the current input $x_65$ are updated immediately. To learn the true counting solution, with the cue having path length equal to 1 and the count having path length from 1 to 64, the gradient must travel backward through up to 64 recurrent steps, surviving repeated Jacobian multiplications without vanishing or getting drowned out by noise, all to update the update/reset gates so they act as a lossless integrator. Alongside the statistical simplicity, this temporal asymmetry creates a plausible advantage for the cue.
It does not guarantee that the network will exploit it, nor does it mean that learning to distinguish the two counts requires a perfectly lossless counter. But it plausibly makes the cue attractive during optimisation.

Changing the cue does not change the correct answer.

With this experimental strategy in mind, next consider a conventional single layer GRU with an embedding layer and a linear classification head:

```python
class CountingGRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(4, 8)
        self.encoder = nn.GRU(8, 24, batch_first=True)
        self.head = nn.Linear(24, 2)

    def forward(self, x):
        _, hidden = self.encoder(self.embedding(x))
        return self.head(hidden[-1])


model = CountingGRU()
optimiser = torch.optim.Adam(model.parameters(), lr=2e-3)

for step in range(240):
    x, y = examples(128)
    loss = nn.functional.cross_entropy(model(x), y)

    optimiser.zero_grad()
    loss.backward()
    optimiser.step()
```

Nothing about the GRU explicitly instructs it to use the cue. We have simply arranged the training distribution so that the cue is predictive. This allows us to ask an interesting question: namely, whether the model actually takes advantage of the cue. One way to investigate this is to perform a paired intervention. To do so, we simply present the network with the original example and then with exactly the same example after flipping only the cue:

```python
model.eval()

with torch.no_grad():
    x, y = examples(4096)

    changed = x.clone()
    changed[:, -1] = 5 - changed[:, -1]

    original_pred = model(x).argmax(1)
    changed_pred = model(changed).argmax(1)

    print("Original accuracy:",
          (original_pred == y).float().mean().item())

    print("Cue-flipped accuracy:",
          (changed_pred == y).float().mean().item())

    print("Predictions changed:",
          (original_pred != changed_pred).float().mean().item())

    # A classifier that uses only the cue.
    cue_only = x[:, -1] - 2
    cue_only_flipped = changed[:, -1] - 2

    print("Cue-only accuracy:",
          (cue_only == y).float().mean().item())

    print("Cue-only flipped accuracy:",
          (cue_only_flipped == y).float().mean().item())
```

For a classifier that relies exclusively on the cue, we know the expected behaviour analytically. Its population accuracy under the training distribution is 95%. When those sampled cues are flipped, the accuracy becomes 5%. These are properties of the constructed distribution, not guaranteed measurements of the trained GRU. Moreover, if the GRU's predictions change systematically when we alter only the cue, while keeping the underlying task and correct label unchanged, we have evidence that the model depends on that cue.

It is important to be precise about what this establishes. Such a result demonstrates cue dependence on the tested examples. It does not prove that the network has learned nothing about the token counts, or that it relies exclusively on the cue. Nor does the fixed training budget establish that the network has converged. Different initialisations, optimisation settings, or training durations might produce different computational strategies. This is just part of the complexity of ML experiments. It is also precisely what makes the experiment useful: rather than merely observing poor out-of-distribution (OOD) performance, we have deliberately constructed a suspected shortcut and can intervene on it while holding the actual task fixed. The distinction between identifying a plausible shortcut and demonstrating that a model uses it is the goal.

[For the interested reader, I've uploaded the experiment to my [repo](https://github.com/robertgcsmith/gru_lab/tree/main/cue_shortcut_counting).]

## Counting without counting
There is another example I find particularly interesting, because this time we do not have to insert any artificial cue.

Let

$$
x=(x_1,\ldots,x_T),
\qquad x_t \in \{A,B \},
$$

and suppose the objective is to predict the number of occurrences of $A$, such that the target function has the form

$$
y(x)=\sum_{t=1}^{T}\mathbf 1[x_t=A],
$$

with $\mathbf 1[\cdot]$ is the indicator function. 

Now, assume that the symbols are sampled independently, with

$$
P(x_t=A)=P(x_t=B)=\frac12.
$$

Assume the sequence length $T$ is drawn first from some length distribution and, conditional on $T$, the tokens $x_1, \dots, x_T$ are i.i.d. Bernoulli($1/2$). Also let $\vert{}x\vert{} = T$. Conditional on the sequence length, the target count is thus binomially distributed

$$
Y\mid T\sim\operatorname{Binomial}(T,\tfrac12),
$$

which, consequently, means

$$
\mathbb E[Y\mid T]=\frac{T}{2}
$$

and 

$$
\operatorname{Var}(Y\mid T)=\frac T4.
$$

Notice that this gives us something rather interesting. A network trained to count $A$'s does not necessarily need to inspect every symbol to produce a statistically useful prediction. If the sequence lengths vary, it could estimate the length $T$ and simply predict
$$
\hat y_{\mathrm{shortcut}}(x)=\frac{|x|}{2}.
$$

There is a perfectly good statistical rationale for this strategy. Under squared error loss, the conditional expectation minimises expected prediction error. Among predictors that depend only on sequence length, $T/2$ is therefore optimal under our assumed distribution. Indeed,

$$
\mathbb E\left[
\left(Y-\frac T2\right)^2\mid T
\right]=\frac T4.
$$

Observe the distinction. The predictor is statistically optimal given the restricted information it uses, but it is not an exact counter. If the model had access to, and correctly processed, the full sequence, it could recover the target without error. One can see why the conditional MSE $T/4$ follows from the conditional expectation and variance. To shed more light on the matter, let's use the law of total variance to derive the overall variance and \(R^2\). When $T$ varies across training example,

$$\operatorname{Var}(Y) = \mathbb{E}[\operatorname{Var}(Y \mid T)] + \operatorname{Var}(\mathbb{E}[Y \mid T]) = \frac{\mathbb{E}[T] + \operatorname{Var}(T)}{4},$$

so the $R^2$ of the pure length shortcut is $\frac{\operatorname{Var}(T)}{\mathbb{E}[T] + \operatorname{Var}(T)}$. To ground this in our example, if $T \sim \text{Uniform}\{1, \dots, 100\}$, the length only shortcut explains approximately 94.3% of the variance in $Y$ without reading a single symbol. Think about it this way: it is actually remarkable that a length only predictor can explain approximately 94.3% of the target variance without inspecting a single symbol. This makes sequence length a potentially attractive statistical shortcut, although whether a neural network actually discovers it, and what the corresponding loss landscape looks like, are separate questions.

For a more concrete example, next consider two sequences of length $T=40$, one containing five $A$'s and the other containing thirty-five. The length-based heuristic predicts twenty for both.
An exact counting procedure, of course, returns five and thirty-five respectively. Supposing the probability of observing $A$ changes from $1/2$ to some $p\in(0,1)$, the conditional expectation becomes

$$
\mathbb E[Y\mid T]=pT,
$$

and the error of our original length-based predictor changes accordingly. In fact, we can calculate the new expected squared error

$$
\begin{aligned}
\mathbb E_p\left[
\left(Y-\frac T2\right)^2\mid T
\right] \\
&=
\operatorname{Var}_p(Y\mid T)
+
\left(\mathbb E_p[Y\mid T]-\frac T2\right)^2 \\
&=
Tp(1-p)+T^2\left(p-\frac12\right)^2.
\end{aligned}
$$

This is a simple decomposition into conditional target variance and squared bias. When $p\neq1/2$, the shortcut incurs a squared bias term that grows quadratically with sequence length. The statistical relationship on which the heuristic depended has changed. But the mathematical counting rule has *not* such that

$$
y(x)=\sum_{t=1}^{T}\mathbf 1[x_t=A].
$$

A model that genuinely implements this rule remains correct regardless of the probability distribution from which the input strings are sampled.

There is a useful qualification here. The shortcut's apparent success also depends on the evaluation metric. Under the original distribution, its squared error grows with $T$, while the expected squared error of the normalised count $Y/T$, predicted as $1/2$, decreases like $1/(4T)$. Thus, increasing sequence length alone does not expose the same failure equally clearly under every metric.

[There is also a subtle difference between this example and the preceding GRU experiment. In the artificial-cue experiment, every input has the same length, and the suspicious feature is the appended cue. In the second example, sequence length itself becomes the candidate shortcut. They illustrate different mechanisms and should not be confused.]

The more important scientific observation is that the statistical predictor and the counting procedure obey different invariances. One depends on assumptions about the data-generating distribution. The other is defined by the symbols actually present in the input. This is an example of a distinction I find interesting when thinking about mathematical reasoning and extrapolation. As comment at the outset, a network may produce useful predictions without implementing the computational rule we intended it to learn.

## A small excursion into physics: how might an optimiser find a shortcut?
This brings me to another question. How does a neural network arrive at a particular predictive strategy? Nothing in the training objective explicitly instructs the model to compute $T/2$, any more than it instructs the model to implement a correct counting algorithm. The optimiser does not compare these possibilities as a mathematician might. Rather, the parameters are updated according to the local behaviour of the objective function.

Coming from a physics background, I find the analogy of a particle moving through a potential landscape particularly illuminating. Imagine the model parameters $\theta$ as coordinates in a high-dimensional space, and the loss $\mathcal L(\theta)$ as a kind of potential energy. The lower the loss, the lower the corresponding point in the landscape. In the case of ordinary continuous time gradient flow, we write

$$
\frac{d\theta}{dt}=-\nabla_\theta\mathcal L(\theta).
$$

Along this trajectory, the chain rule gives

$$
\begin{aligned}
\frac{d\mathcal L}{dt}
&=
\nabla_\theta\mathcal L(\theta)^\top
\frac{d\theta}{dt} \\
&=
-\mid \mid \nabla_\theta\mathcal L(\theta) \mid \mid_2^2 \leq0.
\end{aligned}
$$

Easing the constraints of being completely rigorous for a moment, this has a rather appealing interpretation. In the continuous gradient flow idealisation, the loss decreases along the trajectory except at stationary points. For instance, imagine a marble descending into a valley. The marble does not possess a map of the landscape, nor does it know which valley is globally deepest. Its movement responds to the local slope.

Similarly, gradient flow responds to the local derivatives of the objective, not to any explicit assessment of whether the resulting model implements a valid mathematical algorithm.

But we can still push the intuition a little further. Now imagine that, for a particular training problem, a statistical shortcut corresponds to a comparatively accessible basin of low loss. Perhaps the model can reach that basin from many initial parameter configurations. A more systematic mathematical procedure might require a different configuration that is less readily discovered under the chosen optimisation dynamics. In this hypothetical picture, the shortcut behaves rather like a gravity well. The optimiser could descend into an accessible region of the landscape before discovering a different solution with better systematic generalisation.

I think this is a useful way to picture what it might mean for a neural network to behave like a "lazy interpolator".

But I want to emphasise that the imagined landscape is illustrative. We have not established that statistical shortcuts necessarily occupy broad valleys, or that genuine counting algorithms require traversing narrow, rugged regions of parameter space. Whether such geometric differences exist depends on the particular model, objective, and parameterisation.

There is a relevant [paper](https://proceedings.mlr.press/v70/dinh17b.html?utm_source=chatgpt.com) by Dinh et al. (2017), *Sharp Minima Can Generalize For Deep Nets*, which demonstrates that measures of sharpness and flatness can change under reparameterisation without changing the function represented by a network. This is an important warning against interpreting the geometry of a loss landscape too literally. Nevertheless, the underlying physical analogy has a more precise mathematical connection.

## Gradient flow as overdamped motion

Suppose a particle moves through a highly viscous medium under the influence of a conservative force derived from a potential $V(\theta)$. In the overdamped limit, inertial effects become negligible relative to friction. Under a simple isotropic damping model, the equation of motion becomes

$$ \gamma\frac{d\theta}{dt}=-\nabla_\theta V(\theta), $$

where $\gamma>0$ is the damping coefficient. If we identify $V$ with the loss $\mathcal L$, then

$$
\frac{d \theta}{dt} = -\frac{1}{\gamma}\nabla_\theta\mathcal L(\theta)
$$

which is gradient flow up to a rescaling of time. Ordinary gradient descent can be interpreted as a forward Euler discretisation of this system, with

$$
\eta=\frac{\Delta t}{\gamma},
$$

where $\Delta t$ is the discretisation time step. The correspondence is not merely metaphorical: the equations have the same mathematical form. Moreover, the rate of change of the potential is

$$
\frac{dV}{dt} = -\frac{1}{\gamma}|\nabla_\theta V(\theta)|_2^2.
$$

For me, this provides useful physical intuition about how an optimisation process can approach a solution without possessing an explicit representation of that solution's mathematical meaning. The particle responds to the local force. The optimiser responds to the local gradient. Of course, this analogy does not establish that the optimiser must enter the basin corresponding to a particular shortcut. What the equation describes is the instantaneous rate of decrease at the current point, not which basin will eventually be reached. It also needs to be said that actual neural network optimisation is typically more complicated than ordinary gradient flow. For example, Adam, introduced by [Kingma and Ba](https://arxiv.org/abs/1412.6980) (2015), uses adaptive estimates of the first and second moments of gradients, while minibatch training introduces stochasticity. Its parameter updates are not simply steps in the negative direction of the full batch gradient.

Still, as a pedagogical picture, I find the overdamped analogy interesting in a playful sort of way. It helps explain why successful optimisation need not entail discovering a mathematically systematic procedure.

## Why one feature rather than another?

Some closing thoughts for this reseach note.

If a training dataset admits multiple predictive strategies, what determines which features a network learns to exploit? There is a paper by [Shah et al.](https://proceedings.neurips.cc/paper/2020/hash/6cfe0e6127fa25df2a0ef2ae1067d915-Abstract.html) (2020), *The Pitfalls of Simplicity Bias in Neural Networks*, that is particularly interesting here. The authors investigate conditions under which neural networks favour relatively simple predictive features, sometimes at the expense of other informative features.

A related phenomenon is explored by [Pezeshki et al.](https://papers.nips.cc/paper/2021/hash/0987b8b338d6c90bbedd8631bc499221-Abstract.html) (2021) in *Gradient Starvation: A Learning Proclivity in Neural Networks*. Their analysis concerns situations in which learning some predictive features can suppress the learning of others.

These papers provide more specific reasons to investigate the interaction between optimisation and feature learning.

There is also the phenomenon of spectral bias, investigated by [Rahaman et al.](https://proceedings.mlr.press/v97/rahaman19a.html) (2019), concerning the tendency of certain neural networks to learn lower-frequency components of functions earlier in training. This is another interesting result, although we should be careful not to equate Fourier frequency with mathematical or algorithmic simplicity. Token counting, for example, is not automatically a high-frequency problem simply because it requires systematic computation.

Finally, there is a terminological point worth preserving. The phrase "lazy interpolator" is useful as an informal analogy, but lazy training has a specific meaning in the literature. [Chizat, Oyallon and Bach](https://papers.neurips.cc/paper/2019/hash/ae614c557843b1df326cb29c57225459-Abstract.html) (2019) analyse a training regime in which the network's behaviour is well approximated by linearisation around its initial parameters.
That is not synonymous with shortcut learning. A network can learn a shortcut while substantially changing its internal representations, and a network operating in a technically lazy regime need not rely on a spurious predictor.

All of this leaves me with an interesting collection of questions:

When a recurrent neural network learns to evaluate mathematical expressions, what information is actually being represented in its hidden states? Does the model learn a compact computational procedure, or does it exploit statistical regularities that happen to be predictive across the training distribution? And if we deliberately alter those regularities, which parts of the learned computation remain intact?

These questions cannot be settled merely by observing that the training loss has converged. For the GRU, there is one particularly interesting possibility that I want to explore next. Because the network repeatedly updates a fixed-dimensional hidden state, elapsed sequence length might itself become encoded in its recurrent dynamics. Under suitable conditions, this could give the model an internal clock that predicts something like $T/2$, without implementing an exact counting operation. Can a GRU learn to count, or does it sometimes learn to tell time instead?
