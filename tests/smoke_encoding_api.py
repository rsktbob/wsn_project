"""Smoke tests for the Encoding hierarchy and code-only constructors."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from State.Encoding import Encoding
from State.SensorEncoding import SensorEncoding
from State.TargetEncoding import TargetEncoding


def test_class_hierarchy() -> None:
    assert issubclass(SensorEncoding, Encoding)
    assert issubclass(TargetEncoding, Encoding)


def test_sensor_encoding_api() -> None:
    source = [1, 2, 3, 4, 5, 6]
    encoding = SensorEncoding(source)

    assert np.array_equal(encoding.code, source)

    copied = encoding.copy()
    copied.code[0] = 9
    assert encoding.code[0] == 1

    continuous = np.array([0.25, 1.5, 2.75, 3.0, 4.5, 5.25])
    continuous_encoding = SensorEncoding(continuous)
    assert np.issubdtype(continuous_encoding.code.dtype, np.floating)
    np.testing.assert_array_equal(continuous_encoding.code, continuous)

    np.random.seed(1234)
    first = SensorEncoding.random(3, 10)
    np.random.seed(1234)
    second = SensorEncoding.random(3, 10)
    np.testing.assert_array_equal(first.code, second.code)
    assert first.code.shape == (6,)
    assert np.all((first.code >= 0) & (first.code < 10))

    generator_a = np.random.default_rng(4321)
    generator_b = np.random.default_rng(4321)
    np.testing.assert_array_equal(
        SensorEncoding.random(3, rng=generator_a).code,
        SensorEncoding.random(3, rng=generator_b).code,
    )

    # 排程基因使用各感測器自己的 domain；路由基因固定使用 0～9。
    sensing_domains = np.asarray([2, 4, 6, 3, 5])
    generator_a = np.random.default_rng(2468)
    generator_b = np.random.default_rng(2468)
    domain_coding = SensorEncoding.random(
        5,
        sensing_domains,
        rng=generator_a,
    )
    repeated = SensorEncoding.random(
        5,
        sensing_domains,
        rng=generator_b,
    )
    np.testing.assert_array_equal(domain_coding.code, repeated.code)
    assert np.all(domain_coding.code[0::2] >= 0)
    assert np.all(domain_coding.code[0::2] < sensing_domains)
    assert np.all(domain_coding.code[1::2] >= 0)
    assert np.all(
        domain_coding.code[1::2] < SensorEncoding.RANK_PRECISION
    )

    # sensing domain 很小時，路由仍可使用完整的 0～9 範圍。
    routing_check = SensorEncoding.random(
        100,
        2,
        rng=np.random.default_rng(1357),
    )
    assert np.all(routing_check.code[0::2] < 2)
    assert np.any(routing_check.code[1::2] >= 2)


def test_target_encoding_api() -> None:
    source = [10, 20, 30, 40]
    encoding = TargetEncoding(source)

    assert np.array_equal(encoding.code, source)

    copied = encoding.copy()
    copied.code[0] = 99
    assert encoding.code[0] == 10

    np.random.seed(5678)
    first = TargetEncoding.random(4)
    np.random.seed(5678)
    second = TargetEncoding.random(4)
    np.testing.assert_array_equal(first.code, second.code)
    assert first.code.shape == (4,)
    assert np.all(
        (first.code >= 0) & (first.code < TargetEncoding.GENE_DOMAIN)
    )

if __name__ == "__main__":
    test_class_hierarchy()
    test_sensor_encoding_api()
    test_target_encoding_api()
    print("smoke_encoding_api_ok")
