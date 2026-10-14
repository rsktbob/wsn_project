"""CPU NumPy generators. MPS models amplitudes; probabilities obey Born's rule."""
import numpy as np


def unit(value):
    norm = np.linalg.norm(value)
    if not np.isfinite(norm) or norm <= 0:
        raise FloatingPointError('degenerate MPS contraction')
    return value / norm, np.log(norm)


class BornMPS:
    def __init__(self, domains, rank, rng):
        self.domains = np.asarray(domains, dtype=int)
        if self.domains.ndim != 1 or not len(self.domains) or np.any(self.domains < 1) or rank < 1:
            raise ValueError('positive domains and rank required')
        ranks = [1] + [int(rank)] * (len(domains) - 1) + [1]
        self.cores = [unit(rng.normal(size=(ranks[i], int(n), ranks[i+1])))[0]
                      for i, n in enumerate(domains)]
        self.m = [np.zeros_like(g) for g in self.cores]
        self.v = [np.zeros_like(g) for g in self.cores]
        self.step = 0

    def environments(self):
        """Scaled double-layer contractions avoid exponential under/overflow."""
        left = [np.ones((1, 1))]
        log_z = 0.
        for g in self.cores:
            env, scale = unit(np.einsum('ab,asr,bsq->rq', left[-1], g, g, optimize=True))
            left.append(env)
            log_z += scale
        right = [None] * (len(self.cores) + 1)
        right[-1] = np.ones((1, 1))
        for i in range(len(self.cores)-1, -1, -1):
            g = self.cores[i]
            right[i] = unit(np.einsum('asr,bsq,rq->ab', g, g, right[i+1], optimize=True))[0]
        return left, right, log_z

    def log_prob(self, samples):
        samples = np.atleast_2d(samples)
        _, _, log_z = self.environments()
        logs = []
        for x in samples:
            v, scale = np.ones(1), 0.
            for i, g in enumerate(self.cores):
                v = v @ g[:, x[i], :]
                norm = np.linalg.norm(v)
                if norm == 0:
                    scale = -np.inf
                    break
                v /= norm
                scale += np.log(norm)
            logs.append(2 * scale - log_z)
        return np.asarray(logs)

    def loss_grad(self, samples, weights):
        """Exact weighted NLL gradient, including derivative of log partition Z."""
        weights = np.asarray(weights, dtype=float)
        weights = weights / weights.sum()
        left, right, log_z = self.environments()
        grads = []
        for i, g in enumerate(self.cores):
            derivative = np.einsum('ab,bsq,rq->asr', left[i], g, right[i+1], optimize=True)
            denominator = np.sum(g * derivative)
            grads.append(2 * derivative / denominator)
        loss = log_z
        for x, w in zip(samples, weights):
            prefix = [np.ones(1)]
            log_amp = 0.
            for i, g in enumerate(self.cores):
                v, scale = unit(prefix[-1] @ g[:, x[i], :])
                prefix.append(v)
                log_amp += scale
            suffix = [None] * (len(self.cores) + 1)
            suffix[-1] = np.ones(1)
            for i in range(len(self.cores)-1, -1, -1):
                suffix[i] = unit(self.cores[i][:, x[i], :] @ suffix[i+1])[0]
            loss -= 2 * w * log_amp
            for i, g in enumerate(self.cores):
                a, b = prefix[i], suffix[i+1]
                amplitude = a @ g[:, x[i], :] @ b
                if abs(amplitude) < 1e-14:
                    raise FloatingPointError('training sample has near-zero amplitude')
                grads[i][:, x[i], :] -= 2 * w * np.outer(a, b) / amplitude
        return float(loss), grads

    def fit(self, samples, weights, steps=5, learning_rate=.02):
        for _ in range(steps):
            _, grads = self.loss_grad(samples, weights)
            norm = np.sqrt(sum(np.sum(g*g) for g in grads))
            self.step += 1
            for i, grad in enumerate(grads):
                grad = grad * min(1., 10. / max(norm, 1e-30))
                self.m[i] = .9*self.m[i] + .1*grad
                self.v[i] = .999*self.v[i] + .001*grad*grad
                mh = self.m[i] / (1 - .9**self.step)
                vh = self.v[i] / (1 - .999**self.step)
                self.cores[i] -= learning_rate * mh / (np.sqrt(vh) + 1e-8)
        return float(-np.dot(weights / np.sum(weights), self.log_prob(samples)))

    def sample(self, count, rng):
        _, right, _ = self.environments()
        result = np.empty((count, len(self.domains)), dtype=int)
        for j in range(count):
            prefix = np.ones(1)
            for i, g in enumerate(self.cores):
                vectors = np.einsum('a,asr->sr', prefix, g)
                p = np.einsum('sr,rq,sq->s', vectors, right[i+1], vectors)
                # Roundoff can create tiny negative entries in PSD contractions.
                p = np.maximum(p, 0.)
                if not np.isfinite(p).all() or p.sum() <= 0:
                    raise FloatingPointError('invalid conditional probability')
                value = rng.choice(len(p), p=p/p.sum())
                result[j, i] = value
                prefix = unit(vectors[value])[0]
        return result


class IndependentCategorical:
    """Weighted maximum-likelihood independent baseline (no TT correlations)."""
    def __init__(self, domains, rank, rng):
        self.domains = np.asarray(domains, dtype=int)
        self.probs = [np.ones(n)/n for n in self.domains]

    def fit(self, samples, weights, steps=5, learning_rate=.02):
        for i, n in enumerate(self.domains):
            counts = np.bincount(samples[:, i], weights=weights, minlength=n) + 1e-12
            self.probs[i] = counts/counts.sum()
        return float(-np.dot(weights / np.sum(weights), self.log_prob(samples)))

    def log_prob(self, samples):
        return sum(np.log(p[samples[:, i]]) for i, p in enumerate(self.probs))

    def sample(self, count, rng):
        return np.column_stack([rng.choice(len(p), count, p=p) for p in self.probs])
