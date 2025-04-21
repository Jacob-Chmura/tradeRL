<a id="readme-top"></a>

<div align="center">
<h1 style="font-size:3vw;padding:0;margin:0;display:inline">TradeRL</h3>
<h3 style="margin:0">Algorithmic Trading with Reinforcement Learning</h3>
<a href="http://github.com/Jacob-Chmura/tradeRL"><strong>Read the paper»</strong></a>
</div>

<br />

<div align="center">

<a href="">[![Contributors][contributors-shield]][contributors-url]</a>
<a href="">[![Issues][issues-shield]][issues-url]</a>
<a href="">[![MIT License][license-shield]][license-url]</a>

</div>

<div align="center">

<a href="">![example workflow](https://github.com/Jacob-Chmura/tradeRL/actions/workflows/ruff.yml/badge.svg)</a>
<a href="">![example workflow](https://github.com/Jacob-Chmura/tradeRL/actions/workflows/mypy.yml/badge.svg)</a> <a href="">![example workflow](https://github.com/Jacob-Chmura/tradeRL/actions/workflows/testing.yml/badge.svg)</a>

</div>

## About The Project

_TradeRL_ is framework for algorithmic trading in equity markets. This project formulates a Markov Decision Process (MDP)
based on a simplified order execution model and simulate order book dynamics from 5 years of historical market data.

## Getting Started

### Prerequisites

The project uses [uv](https://docs.astral.sh/uv/) to manage and lock project dependencies for a consistent and reproducible environment. If you do not have `uv` installed on your system, visit [this page](https://docs.astral.sh/uv/getting-started/installation/) for installation instructions.

**Note**: If you have `pip` you can just invoke:

```sh
pip install uv
```

### Installation

```sh
# Clone the repo
git clone https://github.com/Jacob-Chmura/tradeRL.git

# Enter the repo directory
cd tradeRL

# Install core dependencies into an isolated environment
uv sync

# [Optional] Install extra dependencies to run analytics
uv sync --group analytics
```

## Usage

### Running an Experiment 

_Full End-to-End Experiments_

```sh
./run_trade_rl.sh
```

_Baseline Experiments Only_

```sh
./run_trade_rl.sh config/baselines
```

_DQN_

```sh
./run_trade_rl.sh config/dqn/dqn.yaml
```

_Reinforce_

```sh
./run_trade_rl.sh config/reinforce/reinforce.yaml
```

### Running Analytics

_All Analytics_

**Note**: requires that you have previously ran `trade_rl.sh` and have generated results

```sh
./run_analytics.sh <name of experiment>
```

_Non-result Dependent Analytics_

Check the [example notebooks](notebooks) for data and feature statistics.

## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

## Contact

Jacob Chmura - jacobpaul.chmura@gmail.com

## Citation

TODO

<p align="right">(<a href="#readme-top">back to top</a>)</p>

[contributors-shield]: https://img.shields.io/github/contributors/Jacob-Chmura/tradeRL.svg?style=for-the-badge
[contributors-url]: https://github.com/Jacob-Chmura/tradeRL/graphs/contributors
[issues-shield]: https://img.shields.io/github/issues/Jacob-Chmura/tradeRL.svg?style=for-the-badge
[issues-url]: https://github.com/Jacob-Chmura/tradeRL/issues
[license-shield]: https://img.shields.io/github/license/Jacob-Chmura/tradeRL.svg?style=for-the-badge
[license-url]: https://github.com/Jacob-Chmura/tradeRL/blob/master/LICENSE.txt

