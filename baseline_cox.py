from dataclasses import dataclass
import inspect
import chex
import optax

from tqdm import tqdm
from utils import LazyTimesDataGenerator, TgtMskDataGenerator, train_val_test_split
#from utils import LazyTimesDataGenerator, TgtMskDataGenerator, train_test_split
from base_cox import BaseSA

Params = chex.ArrayTree
PRNGKey = chex.PRNGKey
State = chex.ArrayTree

# Config params


@dataclass
class ConfigParams:
    """A structure for configuration"""
    dataset_name: str
    batch_size: int
    learning_rate: float
    log_interval: int
    weight_decay: float
    num_epochs: int
    dataset_kwargs: dict
    landmark: bool = False
    output_file: str = None

    @classmethod
    def from_dict(cls, env):
        """To ignore args that are not in the class,
        see XXXX
        """
        return cls(**{
            k: v for k, v in env.items()
            if k in inspect.signature(cls).parameters
        })


# Model state
@chex.dataclass(frozen=True)
class ModelState:
    """A structure of the current model state"""
    params: Params
    opt_state: optax.OptState


class SA(BaseSA):

    def get_train_val_test(self, val_size=.15, test_size=.2):
        if self.config.calculate_tgt_and_mask:
            data_manager = TgtMskDataGenerator
        else:
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
                                                                    test_size=test_size)
        subkey = self._next_rng_key()
        train_gen = data_manager(X=X_train,
                                 ts=ts_train, cs=cs_train,
                                 y=y_train, mask=m_train,
                                 batch_size=self.config.batch_size, rng=subkey)
        subkey = self._next_rng_key()
        val_gen = data_manager(X=X_val,
                                 ts=ts_val, cs=cs_val,
                                 y=y_val, mask=m_val,
                                 batch_size=self.config.batch_size, rng=subkey)
        subkey = self._next_rng_key()
        test_gen = data_manager(X=X_test,
                                ts=ts_test, cs=cs_test,
                                y=y_test, mask=m_test,
                                batch_size=self.config.batch_size, rng=subkey)
        return train_gen, val_gen, test_gen

    def train(self, train_gen=None, test_gen=None):
        """Training loop"""
        train_loss = []
        test_loss = []

        if train_gen is None:
            train_gen, _ = self.get_train_val_test()

        iter_range = range(self.config.num_epochs)
        if self.config.verbose:
            iter_range = tqdm(iter_range)
        for epoch in iter_range:
            tr_loss = self.train_step(train_gen)
            train_loss.append(tr_loss)
            if test_gen is not None:
                te_loss = self.test_step(test_gen)
                test_loss.append(te_loss)

        return train_loss, test_loss
