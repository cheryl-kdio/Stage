# Hawkes Intertrade Project

Projet Python pour analyser l'impact du volume sur les durees intertrades, estimer des modeles de Hawkes exponentiels, puis construire des signaux exploratoires d'activite et de direction.

## Objectif

Partir d'une table de trades contenant au minimum :

```text
timestamp, price, volume
```

et eventuellement :

```text
side
```

pour produire :

1. une analyse des durees intertrades ;
2. un benchmark de type Log-ACD avec volume ;
3. des streams d'evenements par buckets de volume ;
4. des streams multivaries buy/sell + volume + price up/down ;
5. une estimation Hawkes multivariee avec noyau exponentiel ;
6. des scores d'activite et de direction utilisables dans une strategie exploratoire.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Exemples rapides

```bash
python examples/synthetic_fit.py
python examples/workflow_from_trades.py
```

## Structure

```text
hawkes_intertrade/
  acd.py        # Log-ACD avec volume
  data.py       # preparation des donnees de trades
  hawkes.py     # Hawkes exponentiel multivarie, decays fixes ou estimes
  signal.py     # intensites integrees, scores activite/direction
  backtest.py   # backtest exploratoire simple
examples/
  synthetic_fit.py
  workflow_from_trades.py
tests/
  test_smoke.py
```

## Remarques importantes

- Les classes Hawkes acceptent des timestamps exacts, pas une serie de comptes agregee.
- Si les donnees sont agregees par bars, il faut soit reconstruire des timestamps approximatifs, soit passer a un modele discret.
- Le backtest fourni est volontairement minimal. Il sert de squelette, pas de preuve d'alpha.
- Pour une utilisation haute frequence reelle, il faut ajouter frais, spread, latence, slippage, contraintes d'execution et controles anti look-ahead.
