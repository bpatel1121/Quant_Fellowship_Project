# Quanta Fellowship Project – QQQ Strategy

This repo contains my submission for the Quanta Fellowship:
a systematic daily-leverage strategy on QQQ using Gemini 3.0.

The goal is to use Gemini 3.0 (via Google AI Studio) to brainstorm, code, and validate 30+ trading signals based on different orthogonal ideas from technicals to volume, to cross asset patterns, and more. Then use Gemini to help combine together to create a portfolio with out-of-sample performance, aiming to maximize sharpe. 


## Structure
- 'data/' – Provided datasets.
- 'notebooks/' – Research environment (signal design, validation, portfolio).
- 'src/' – Backtest engine, signal definitions, utilities.
- 'prompts/' – Record of Gemini 3.0 interactions.

## Usage
1. Install dependencies: pip install -r requirements.txt
2. Place 'qqq_train_validation.csv' and 'qqq_blind_holdout.csv' in 'data/raw/'.
3. Run notebooks '01' through '04' in order.
