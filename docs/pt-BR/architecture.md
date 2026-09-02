# Arquitetura

[Português](../pt-BR/architecture.md) · [English](../en/architecture.md) · [← README](../../README.pt-BR.md)

Como as peças se encaixam, o que cada uma promete à seguinte, e onde plugar
trabalho novo.

---

## Mapa dos módulos

Toda seta é uma dependência de mão única. Nada importa para cima, então qualquer
módulo pode ser testado com entrada sintética sem arrastar a rede junto.

```mermaid
flowchart TB
    subgraph IO["Ingestão"]
      config["config.py<br/><i>universos · janelas · custo</i>"]
      data["data.py<br/><i>yfinance → cache parquet</i>"]
    end
    subgraph BUILD["Construção do dataset"]
      features["features.py<br/><i>44 colunas causais</i>"]
      labels["labels.py<br/><i>fwd_ret · target</i>"]
      pipeline["pipeline.py<br/><i>monta o painel</i>"]
    end
    subgraph LEARN["Aprendizado e validação"]
      models["models.py<br/><i>baselines · modelos · calibrador</i>"]
      backtest["backtest.py<br/><i>dobras · carteira · custo</i>"]
    end
    subgraph OUT["Avaliação e saída"]
      metrics["metrics.py<br/><i>classificação · estratégia · significância</i>"]
      report["report.py<br/><i>markdown · json</i>"]
      figures["figures.py<br/><i>gráficos png</i>"]
    end

    config --> data
    config --> backtest
    data --> pipeline
    features --> pipeline
    labels --> pipeline
    pipeline --> backtest
    models --> backtest
    backtest --> report
    metrics --> report
    report --> figures
```

`config.py` é o único módulo que todos os outros podem ler. Se um número muda
comportamento, ele mora lá — assim um resultado publicado é reproduzível a partir
de um arquivo só.

---

## A execução, ponta a ponta

```mermaid
sequenceDiagram
    autonumber
    participant CLI as run_backtest.py
    participant PL as pipeline
    participant BT as backtest
    participant MD as models
    participant RP as report

    CLI->>PL: build_dataset(mercado, target_mode)
    PL->>PL: load_market + load_benchmark (cache parquet)
    PL->>PL: build_features → add_target → clean_panel
    PL-->>CLI: painel (data × papel), nomes das features

    CLI->>BT: run_walk_forward(painel, cols, models, cfg)
    loop para cada dobra k
        BT->>BT: fatia treino (…→ corte) e teste (63 pregões seguintes)
        BT->>MD: fit(X_train, y_train, dates=train.date)
        BT->>MD: corta 20% final por DATA, treina base, ajusta Platt nela
        MD-->>BT: modelo calibrado
        BT->>MD: predict_proba_up(X_test)
        MD-->>BT: probabilidades fora da amostra
    end
    BT-->>CLI: todas as previsões OOS + metadados das dobras

    CLI->>RP: evaluate(preds, cfg)
    RP->>RP: classificação + simulate_strategy + bootstrap em blocos
    RP-->>CLI: dicionário de resultados
    CLI->>RP: write_markdown / write_json / figuras
```

---

## Contratos de dados

Cada etapa entrega à seguinte um frame de formato fixo. Quebrar um destes é o
jeito mais rápido de produzir números silenciosamente errados, então vale
enunciá-los.

### `data.load_market(mercado)` → painel long

| coluna | tipo | significado |
|---|---|---|
| `date` | datetime64, sem timezone | data do pregão |
| `ticker` | str | símbolo Yahoo (`PETR4.SA`, `AAPL`) |
| `open`,`high`,`low`,`close` | float | preços ajustados (`auto_adjust=True`) |
| `volume` | float | volume negociado |

Uma linha por (data, papel). Ordenado por papel e depois por data. Linhas com
`close <= 0` são descartadas; datas duplicadas mantêm a última observação.

### `features.build_features(panel, benchmark)` → matriz de features

Adiciona 44 colunas, mais `date`, `ticker`, `close` e `open` carregados adiante
para uso posterior. **Toda coluna de feature é função de dados com timestamp
`<= t`.** `feature_columns(df)` devolve exatamente as colunas de modelagem,
excluindo as carregadas e as de rótulo listadas em `NON_FEATURE_COLS`.

### `labels.add_target(df, mode)` → adiciona o rótulo

| coluna | significado |
|---|---|
| `fwd_ret` | retorno simples do trade, de t para t+1 |
| `target` | `1` se `fwd_ret > 0`, senão `0`; `NaN` na última linha de cada papel |

Este é o único módulo autorizado a chamar `shift(-1)`.

### `backtest.run_walk_forward(...)` → previsões fora da amostra

| coluna | significado |
|---|---|
| `date`, `ticker` | identificam a previsão |
| `target`, `fwd_ret` | o resultado realizado |
| `fold` | de qual bloco de teste veio |
| `prob__<modelo>` | uma coluna por modelo, probabilidade de alta |

`df.attrs["feature_importance"]` carrega a importância média do LightGBM entre as
dobras. `attrs` não sobrevive ao `to_parquet`, e é por isso que
`run_backtest.py` grava a importância num CSV próprio.

---

## Onde acontece o corte de calibração

A sutileza fácil de errar: o calibrador precisa ser ajustado em dado que o modelo
base não viu, e o corte tem de ser por **data**, não por linha — o painel tem 15
papéis por pregão, então um corte por linha deixaria o mesmo dia dos dois lados.

```mermaid
flowchart LR
    T["janela de treino<br/>(dobra k)"] --> S{"corta por data<br/>na marca de 80%"}
    S -->|"80% iniciais dos pregões"| B["treina o modelo base"]
    S -->|"20% finais dos pregões"| C["ajusta a sigmoide de Platt<br/>nas probabilidades cruas<br/>do modelo base"]
    B --> C
    C --> P["modelo calibrado<br/>→ pontua o bloco de teste"]
```

Se a fatia de calibração ficaria com menos de 500 linhas, ou se ela contém uma
única classe, o `CalibratedModel` volta a treinar o modelo base na janela inteira
e pula a calibração, em vez de ajustar uma sigmoide sem sentido.

---

## Artefatos de saída

| caminho | produzido por | conteúdo |
|---|---|---|
| `data/raw/*.parquet` | `fetch_data.py` | cache de OHLCV por papel |
| `data/processed/oos_predictions_<mercado>.parquet` | `run_backtest.py` | todas as previsões fora da amostra |
| `data/processed/feature_importance_<mercado>.csv` | `run_backtest.py` | importância média do LightGBM |
| `reports/backtest_<mercado>.md` | `run_backtest.py` | relatório legível |
| `reports/backtest_<mercado>.json` | `run_backtest.py` | métricas + curvas de capital, legível por máquina |
| `reports/figures/*.png` | `figures.py` | 5 gráficos por mercado |
| `reports/latest_predictions.json` | `predict_today.py` | probabilidades do próximo pregão |
| `reports/recent_history.json` | `export_history.py` | candles recentes + a chamada do modelo em cada dia |
| `reports/dashboard.html` | `build_dashboard.py` | dashboard autocontido, PT/EN |

Tudo em `data/` e `reports/` é gerado e está no `.gitignore`: um clone limpo
reproduz tudo a partir dos scripts.

---

## Pontos de extensão

**Um modelo novo.** Herde de `BaseModel`, implemente `fit(X, y, dates=None)` e
`predict_proba_up(X)`, e adicione em `default_models()`. Ele herda
automaticamente a validação walk-forward, a calibração de Platt, a simulação de
carteira com custo e todas as linhas do relatório. Use `calibrate = False` se a
saída do modelo não deve ser recalibrada (é o que os baselines triviais fazem).

**Uma feature nova.** Adicione dentro de `_per_ticker_features` (série por papel)
ou em `build_features` (contexto de mercado e corte transversal). Depois rode
`scripts/check_leakage.py`: o teste de truncagem pega uma janela que espia o
futuro. Não há passo de registro — `feature_columns` descobre as colunas por
exclusão.

**Um mercado novo.** Adicione o universo e o benchmark em `config.py`. Nada mais
muda; o calendário é derivado do próprio dado.

**Outra regra de posição.** `simulate_strategy` é o único lugar que transforma
probabilidade em peso. Seleção top-N, alvo de volatilidade ou filtro de regime
pertencem todos ali, e toda métrica a jusante os incorpora de graça.
