"""Dados sintéticos para os testes.

Nenhum teste desta suíte toca a rede. O painel abaixo é um passeio aleatório
com semente fixa: serve porque as propriedades verificadas — a ordem das
janelas, o alinhamento do alvo, a causalidade das features — não dependem de o
preço ser realista, e sim de o tempo andar para frente.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# A janela mais longa das features é 252 pregões (mínimo de 126). O painel
# precisa ser maior que isso para que a linha do corte esteja preenchida.
N_PREGOES = 520
TICKERS = ("AAA", "BBB", "CCC")


def _passeio(rng: np.random.Generator, n: int, inicio: float) -> np.ndarray:
    return inicio * np.exp(np.cumsum(rng.normal(0.0003, 0.015, size=n)))


@pytest.fixture(scope="session")
def datas() -> pd.DatetimeIndex:
    return pd.bdate_range("2015-01-01", periods=N_PREGOES)


@pytest.fixture(scope="session")
def painel(datas) -> pd.DataFrame:
    rng = np.random.default_rng(20260902)
    partes = []
    for i, ticker in enumerate(TICKERS):
        fechamento = _passeio(rng, len(datas), 10.0 * (i + 1))
        partes.append(
            pd.DataFrame(
                {
                    "date": datas,
                    "ticker": ticker,
                    "open": fechamento * (1 + rng.normal(0, 0.002, len(datas))),
                    "high": fechamento * (1 + abs(rng.normal(0, 0.006, len(datas)))),
                    "low": fechamento * (1 - abs(rng.normal(0, 0.006, len(datas)))),
                    "close": fechamento,
                    "volume": rng.integers(1_000_000, 9_000_000, len(datas)).astype(float),
                }
            )
        )
    return pd.concat(partes, ignore_index=True).sort_values(["date", "ticker"]).reset_index(drop=True)


@pytest.fixture(scope="session")
def benchmark(datas) -> pd.DataFrame:
    rng = np.random.default_rng(11)
    return pd.DataFrame(
        {
            "date": datas,
            "bench_close": _passeio(rng, len(datas), 100_000.0),
            "bench_volume": rng.integers(1_000_000, 9_000_000, len(datas)).astype(float),
        }
    )
