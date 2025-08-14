import jax
import jax.numpy as jnp
from base_cox import BaseSA, ConfigParams
from utils import convert_to_jax_arrays, train_val_test_split, unroll, LazyTimesDataGenerator
from dataclasses import dataclass


@dataclass
class Config(ConfigParams):
    lambda_: float = 1.0
    num_steps: int = 5


@jax.jit
def bce_logits(targets, logits):
    """Compute the binary cross-entropy with logits.

    Letting `p = targets` and `q = sigmoid(logits)`, this function returns the
    binary cross-entropy `H(p, q) = -p * log(q) - (1 - p) * log(1 - q)`.
    """
    return -targets * logits + jax.nn.softplus(logits)


class LambdaSA(BaseSA):

    def __init__(self, config_kwargs, seed, name="TCSR", **kwargs):
        super().__init__(config_kwargs, seed, name, **kwargs)

        new_config = Config.from_dict(config_kwargs)
        self.lambda_ = new_config.lambda_
        self.num_steps = new_config.num_steps

    def get_train_val_test(self, val_size=.15, test_size=0.2, num_train_seqs=None):
        data_manager = LazyTimesDataGenerator

        subkey = self._next_rng_key()
        X_train, X_val, X_test, y_train, y_val, y_test, hws_train, hws_val, hws_test, \
        m_train, m_val, m_test, ts_train, ts_val, ts_test, cs_train, cs_val, cs_test, \
        rs_train, rs_val, rs_test, seqs_ts_train, seqs_ts_val, seqs_ts_test = train_val_test_split(self.data['seqs'],
                                                                    self.data['target'],
                                                                    self.data['h_ws'],
                                                                    self.data['mask'],
                                                                    self.data['ts'],
                                                                    self.data['cs'],
                                                                    self.data['rs'],
                                                                    self.data['seqs_ts'],
                                                                    seed=self.seed,
                                                                    val_size=val_size,
                                                                    test_size=test_size,
                                                                    num_train_seqs=num_train_seqs)

        subkey = self._next_rng_key()
        train_gen = data_manager(X=X_train, h_ws=hws_train,
                                 ts=ts_train, cs=cs_train,
                                 y=y_train, rs=rs_train,
                                 seqs_ts=seqs_ts_train, mask=m_train,
                                 batch_size=self.config.batch_size, rng=subkey)
        subkey = self._next_rng_key()
        val_gen = data_manager(X=X_val, h_ws=hws_val,
                                 ts=ts_val, cs=cs_val,
                                 y=y_val, rs=rs_val,
                                 seqs_ts=seqs_ts_val, mask=m_val,
                                 batch_size=self.config.batch_size, rng=subkey)
        subkey = self._next_rng_key()
        test_gen = data_manager(X=X_test, h_ws=hws_test,
                                ts=ts_test, cs=cs_test,
                                y=y_test, rs=rs_test,
                                seqs_ts=seqs_ts_test, mask=m_test,
                                batch_size=self.config.batch_size, rng=subkey)
        
        X_train, X_val, X_test, ts_train, ts_val, ts_test, cs_train, cs_val, cs_test = convert_to_jax_arrays(X_train, X_val, X_test, ts_train, ts_val, ts_test, cs_train, cs_val, cs_test)
        
        if self.config.landmark:
            X_train, ts_train, cs_train = unroll(X_train, ts_train, cs_train)
            X_val, ts_val, cs_val = unroll(X_val, ts_val, cs_val)
            X_test, ts_test, cs_test = unroll(X_test, ts_test, cs_test)

        return X_train, ts_train, cs_train, train_gen, val_gen, test_gen

    def _targets(self, m, seqs, ts, cs):
        """Compute pseudo-targets and weights for the m-step backup."""
        H = self.horizon
        ys = jnp.zeros((len(seqs), H))
        ws = jnp.zeros((len(seqs), H))
        # Observed outcomes within first m steps.
        # Seqs that reached terminal state within window.
        idx = (ts <= m) & ~cs
        ys = ys.at[idx, ts[idx] - 1].set(1.0)
        ws = ws.at[:, :m].set(
            (jnp.arange(m) < ts[:, jnp.newaxis]).astype(float))
        # Predicted outcomes after first m steps.
        if m < H:
            # Seqs that are still active after the window.
            idx = (ts > m) | ((ts == m) & cs)
            nxt = seqs[idx, m]
            logits = self.forward(self.state.params, nxt)
            log_hs = jax.nn.log_sigmoid(logits)
            ys = ys.at[idx, m:].set(jnp.exp(log_hs[:, :-m]))
            ws = ws.at[idx, m].set(1.0)
            ws = ws.at[idx, (m + 1):].set(jnp.exp(
                jnp.cumsum(
                    log_hs[:, : -(m + 1)] - logits[:, : -(m + 1)],
                    axis=1,
                )
            ))
        return (ys, ws)

    def _update_target(self, seqs, ts, cs, lambda_):

        H = self.horizon
        cs = cs.astype(bool)
        n = len(seqs)

        # Exponentially decreasing multipliers.
        multipliers = lambda_ ** jnp.arange(H)
        multipliers = multipliers.at[:-1].set(multipliers[:-1] * (1 - lambda_))

        ys = jnp.zeros((len(seqs), H))
        ws = jnp.zeros((len(seqs), H))
        # Compute backup targets and weights at all steps.
        for m, mult in enumerate(multipliers, start=1):
            if mult == 0.0:
                # Multiplier is zero, we can ignore this step. This leads to
                # a significant speedup for `lambda_ = 0` and `lambda_ = 1`
                continue
            ys_m, ws_m = self._targets(m, seqs, ts, cs)
            ys += mult * ys_m
            ws += mult * ws_m

        return ys, ws

    def _inner_loop(self, seqs, ys, ws):
        """Training loop"""
        epoch_loss = []
        for epoch in range(self.config.num_epochs):

            self.state, loss = self.update(
                self.state,
                seqs,
                ys,
                ws
            )

            # log
            if epoch % self.config.log_interval == 0 and epoch > 1:
                print(f"Epoch: {epoch+1}/{self.config.num_epochs}")
                print(f"Train classification loss: {loss.item():.3f} at epoch {epoch}")
                print()
            epoch_loss.append(loss.item())

        return epoch_loss

    def train(self, X_train, ts_train, cs_train):
        # Outer loop
        losses = []
        for i in range(self.num_steps):
            ys, ws = self._update_target(
                X_train, ts_train, cs_train, self.lambda_)
            loss = self._inner_loop(X_train[:, 0], ys, ws)
            losses.append(loss)

        return losses
