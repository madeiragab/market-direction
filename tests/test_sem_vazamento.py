"""As três afirmações que sustentam o resultado do projeto.

O README diz que nenhuma previsão avaliada foi vista pelo modelo que a
produziu, que as features são causais por construção e que o alvo descreve o
pregão seguinte. `scripts/check_leakage.py` prova isso sobre os dados reais,
mas roda em minutos e precisa da rede. Aqui as mesmas propriedades são
verificadas em segundos, sobre um painel sintético, a cada push — para que a
afirmação não sobreviva a uma mudança que a quebre.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from marketdir.backtest import make_folds
from marketdir.config import BacktestConfig
from marketdir.features import build_features, feature_columns
from marketdir.labels import add_target


# ---------------------------------------------------------------- 1. janelas


@pytest.mark.parametrize("embargo", [1, 5, 21])
def test_treino_termina_antes_do_teste_com_o_embargo_no_meio(embargo):
    """O buraco entre treino e teste existe, e tem o tamanho pedido.

    O alvo da linha t só se resolve em t+1. Sem o embargo, a última linha de
    treino encostaria no primeiro dia do teste, e o modelo teria visto o
    retorno que está sendo avaliado.
    """
    cfg = BacktestConfig(min_train_days=750, test_block_days=63, embargo_days=embargo)
    folds = make_folds(np.arange(1200), cfg)

    assert folds, "nenhuma dobra gerada"
    for _, fim_treino, inicio_teste, fim_teste in folds:
        assert fim_treino < inicio_teste, "treino invade o teste"
        assert inicio_teste - fim_treino == embargo, "embargo com tamanho errado"
        assert fim_teste > inicio_teste, "bloco de teste vazio"


def test_blocos_de_teste_nao_se_repetem_nem_deixam_buraco():
    """Cada pregão avaliado é avaliado uma vez só.

    Blocos sobrepostos contariam a mesma previsão duas vezes e apertariam o
    p-valor sem nenhuma informação nova por trás.
    """
    cfg = BacktestConfig(min_train_days=750, test_block_days=63, embargo_days=1)
    n = 1200
    folds = make_folds(np.arange(n), cfg)

    assert folds[0][2] == cfg.min_train_days, "o primeiro teste começa fora do lugar"
    assert folds[-1][3] == n, "sobrou pregão sem avaliar no fim"

    for anterior, seguinte in zip(folds, folds[1:]):
        assert anterior[3] == seguinte[2], "buraco ou sobreposição entre blocos"


def test_a_janela_de_treino_sempre_expande():
    """Treinar em tudo que veio antes, e não em uma janela deslizante."""
    cfg = BacktestConfig(min_train_days=750, test_block_days=63, embargo_days=1)
    folds = make_folds(np.arange(1500), cfg)

    for inicio_treino, fim_treino, _, _ in folds:
        assert inicio_treino == 0, "a janela de treino deixou de expandir"
    fins = [f[1] for f in folds]
    assert fins == sorted(fins), "o fim do treino andou para trás"


# ------------------------------------------------------------------- 2. alvo


def test_o_alvo_descreve_o_pregao_seguinte(painel):
    """fwd_ret na linha t é o retorno entre t e t+1, e nada além disso."""
    com_alvo = add_target(painel, mode="close_to_close")
    um = com_alvo[com_alvo["ticker"] == "AAA"].sort_values("date").reset_index(drop=True)

    esperado = um["close"].shift(-1) / um["close"] - 1.0
    pd.testing.assert_series_equal(
        um["fwd_ret"], esperado, check_names=False, rtol=1e-12
    )

    subiu = um["fwd_ret"] > 0
    assert (um.loc[subiu, "target"] == 1).all()
    assert (um.loc[~subiu & um["fwd_ret"].notna(), "target"] == 0).all()


def test_o_ultimo_pregao_nao_tem_alvo(painel):
    """No último dia o futuro ainda não aconteceu; a linha não pode virar 0."""
    com_alvo = add_target(painel, mode="close_to_close")
    ultimos = com_alvo.sort_values("date").groupby("ticker").tail(1)

    assert ultimos["fwd_ret"].isna().all()
    assert ultimos["target"].isna().all(), "linha sem futuro virou rótulo"


# --------------------------------------------------------------- 3. features


def test_as_features_nao_mudam_quando_o_futuro_e_cortado(painel, benchmark):
    """Truncar a história em D não altera nenhuma feature do próprio dia D.

    É a mesma prova do `check_leakage.py`, aqui sobre o painel sintético: se
    alguma janela olhasse para frente, o valor calculado com a série inteira
    seria diferente do calculado só com o passado.
    """
    completo = build_features(painel, benchmark)
    colunas = feature_columns(completo)

    datas = np.sort(pd.unique(painel["date"]))
    for fracao in (0.6, 0.8, 0.95):
        corte = datas[int(len(datas) * fracao)]

        truncado = build_features(
            painel[painel["date"] <= corte],
            benchmark[benchmark["date"] <= corte],
        )

        a = completo[completo["date"] == corte].set_index("ticker")[colunas].sort_index()
        b = truncado[truncado["date"] == corte].set_index("ticker")[colunas].sort_index()

        assert list(a.index) == list(b.index), "sumiu ticker ao truncar"
        maior = (a - b).abs().to_numpy()
        maior = np.nanmax(maior) if maior.size else 0.0
        assert maior < 1e-9, (
            f"feature mudou ao cortar o futuro em {str(corte)[:10]}: "
            f"maior diferença {maior:.3e}"
        )


def test_nenhuma_coluna_de_alvo_entra_como_feature(painel, benchmark):
    """fwd_ret e target são o que se quer prever, não entrada do modelo."""
    completo = add_target(build_features(painel, benchmark))
    colunas = feature_columns(completo)

    for proibida in ("target", "fwd_ret", "fwd_ret_oc", "close", "open"):
        assert proibida not in colunas, f"{proibida} vazou para a matriz de features"
