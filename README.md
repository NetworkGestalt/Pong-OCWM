## Pong-OCWM

<img src="assets/object_encoder.gif" height="200" />&nbsp;&nbsp;&nbsp;&nbsp;<img src="assets/dynamics_pred.gif" height="200" />

### Description

Object-Centric World Models (OCWMs) learn object-level latent representations of the environment and predict their future states. In this implementation, we train a Variational Autoencoder (VAE) on ground-truth Pong object masks (ball, left paddle, right paddle, score), tokenize the objects’ latent representations along with their positions, and train a transformer to predict next-step latent object states given previous frames and actions. In particular, the transformer uses spatial (inter-object) attention within each frame, and causal attention over a temporal window of spatially contextualized object states. To improve the accuracy of long autoregressive rollouts, we train the transformer with self-forcing; that is; we optimize a temporally discounted sum of per-step losses over an autoregressive rollout during training.

The code for our Pong environment is based on the CITRIS Interventional Pong dataset [1], released under the BSD 3-Clause Clear License, with slight changes to increase simulation speed (e.g., rendering frames by directly writing into preallocated NumPy arrays rather than rebuilding full Matplotlib figures). The overall approach of using ground-truth object mask and encoding the ball’s velocity by including its previous position in the observation is based on the Interventional Pong experiments in SPARTAN [2].

[1] Lippe, P., Magliacane, S., Löwe, S., Asano, Y. M., Cohen, T., & Gavves, S. (2022). CITRIS: Causal Identifiability from Temporal Intervened Sequences. Proceedings of the 39th International Conference on Machine Learning (Vol. 162, pp. 13557–13603). PMLR. https://proceedings.mlr.press/v162/lippe22a.html

[2] Lei, A., Schölkopf, B., & Posner, I. (2025). SPARTAN: A Sparse Transformer World Model Attending to What Matters. The Thirty-Ninth Annual Conference on Neural Information Processing Systems. NIPS. https://openreview.net/forum?id=uS5ch7GjZ4

*TODO: train the dynamics model for a larger number of steps, evaluate it using pixel MSE as a function of rollout length, and run an ablation on the context window size and self-forcing horizon; map keyboard inputs to paddle actions during rollout to make the generated Pong dynamics fully interactive.*

### Repository Contents (WIP)
- env/
  - `pong.py`:
- render/
  - `render.py`:
  - `animate.py`: 
- models/
  - `vae_model.py`:
  - `transformer_model.py`:
- train/
  - `vae_train.py`:
  - `transformer_train.py`: 
- eval/
- examples/
  - `sim_and_render.py`:
  - `run_vae_train.py`:
  - `run_transformer_train.py`:
- assets/

