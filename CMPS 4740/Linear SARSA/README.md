
<div align="center">
    <img src="media/metamon_text_logo.png" alt="Metamon Text Logo" width="410">
</div>

<br>

<div align="center">
    <img src="media/metamon_banner.png" alt="Metamon Banner" width="720">
</div>

<br>

<div align="center">

[![Paper](https://img.shields.io/badge/Paper-arXiv:2504.04395-red)](https://arxiv.org/abs/2504.04395)
[![Website](https://img.shields.io/badge/Project-Website-blue)](https://metamon.tech)

</div>


<br>



**Metamon** enables reinforcement learning (RL) research on [Pokémon Showdown](https://pokemonshowdown.com/) by providing:

1) A standardized suite of teams and opponents for evaluation.
2) A large dataset of RL trajectories "reconstructed" from real human battles.
3) Starting points for training imitation learning (IL) and RL policies.

Metamon is the codebase behind ["Human-Level Competitive Pokémon via Scalable Offline RL and Transformers"](https://arxiv.org/abs/2504.04395) (RLC, 2025). Please check out our [project website](https://metamon.tech) for an overview of our results. This README documents the dataset, pretrained models, training, and evaluation details to help you get battling!

<br>

<div align="center">
    <img src="media/figure1.png" alt="Figure 1" width="700">
</div>

<br>

#### Supported Rulesets

Pokémon Showdown hosts many different rulesets spanning nine generations of the video game franchise. Metamon initially focused on the most popular singles ruleset ("OverUsed") for **Generations 1, 2, 3, and 4**. However, we are gradually expanding to Gen 9 to support the [NeurIPS 2025 PokéAgent Challenge](https://pokeagent.github.io). This is a large project that will not be finalized in time for the competition launch; please stay tuned for updates.

The current status is:

|  | Gen 1 OU | Gen 2 OU | Gen 3 OU | Gen 4 OU | Gen 9 OU |
|------------|---------------------|----------|----------|----------|----------|
| Datasets | ✅ | ✅ | ✅ | ✅ | 🟠 (beta) |
| Teams | ✅ | ✅ | ✅ | ✅ | ✅  |
| Heuristic Baselines | ✅ | ✅ | ✅ | ✅ | ✅ |
| Learned Baselines | ✅ | ✅ | ✅ | ✅ | 🟠 (beta) |

We also support the UnderUsed (UU), NeverUsed (NU), and Ubers tiers for Generations 1, 2, 3, and 4 —-- though constant rule changes and small dataset sizes have always made these a bit of an afterthought.


<br>


### Table of Contents
1. [**Installation**](#installation)

2. [**Quick Start**](#quick-start)

3. [**Pretrained Models**](#pretrained-models)

4. [**Battle Datasets**](#battle-datasets)

5. [**Team Sets**](#team-sets)

6. [**Baselines**](#baselines)

7. [**Observation Spaces, Action Spaces, & Reward Functions**](#observation-spaces-action-spaces--reward-functions)

8. [**Training and Evaluation**](#training-and-evaluation)

9. [**Other Datasets**](#other-datasets)

10. [**Battle Backends**](#battle-backends)

11. [**Acknowledgement**](#acknowledgements)

12. [**Citation**](#citation)


<br>
 
---

<br>

## Installation

Metamon is written and tested for linux and python 3.10+. We recommend creating a fresh virtual environment or [conda](https://docs.anaconda.com/anaconda/install/) environment:

```shell
conda create -n metamon python==3.10
conda activate metamon
```

Then, install with:

```shell
git clone --recursive git@github.com:UT-Austin-RPL/metamon.git
cd metamon
pip install -e .
```

To install [Pokémon Showdown](https://pokemonshowdown.com/), we'll need a modern version of `npm` / Node.js (instructions [here](https://nodejs.org/en/download/package-manager)). Note that Showdown undergoes constant updates... breaking changes are rare, but do happen. The version that downloads with this repo (`metamon/server`) is always supported.

```shell
cd server/pokemon-showdown
npm install
```

We will need to have the Showdown server running in the background while using Metamon:
```shell
# in the background (`screen`, etc.)
node pokemon-showdown start --no-security
# no-security removes battle speed throttling and password requirements on your local server
```

If necessary, we can customize the server settings (`config/config.js`) or [the rules for each game mode](https://github.com/smogon/pokemon-showdown/blob/master/config/CUSTOM-RULES.md).

Verify that installation has gone smoothly with:
```bash
# run a few test battles on the local server
python -m metamon.env
```

Metamon provides large datasets of Pokémon team files, human battles, and other statistics that will automatically download when requested. Specify a path with:
```bash
# add to ~/.bashrc
export METAMON_CACHE_DIR=/path/to/plenty/of/disk/space
```

<br>

____

<br>

## Quick Start

Metamon makes it easy to turn Pokémon into an RL research problem. Pick a set of Pokémon teams to play with, an observation space, an action space, and a reward function:

```python
from metamon.env import get_metamon_teams
from metamon.interface import DefaultObservationSpace, DefaultShapedReward, DefaultActionSpace

team_set = get_metamon_teams("gen1ou", "competitive")
obs_space = DefaultObservationSpace()
reward_fn = DefaultShapedReward()
action_space = DefaultActionSpace()
```

Then, battle against built-in baselines (or any [`poke_env.Player`](https://github.com/hsahovic/poke-env)):

```python 
from metamon.env import BattleAgainstBaseline
from metamon.baselines import get_baseline

env = BattleAgainstBaseline(
    battle_format="gen1ou",
    observation_space=obs_space,
    action_space=action_space,
    reward_function=reward_fn,
    team_set=team_set,
    opponent_type=get_baseline("Gen1BossAI"),
)

# standard `gymnasium` environment
obs, info = env.reset()
next_obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
```

The more flexible option is to request battles on our local Showdown server and battle anyone else who is online (humans, pretrained agents, or other Pokémon AI projects). If it plays Showdown, we can battle against it!

```python
from metamon.env import QueueOnLocalLadder

env = QueueOnLocalLadder(
    battle_format="gen1ou",
    player_username="my_scary_username",
    num_battles=10,
    observation_space=obs_space,
    action_space=action_space,
    reward_function=reward_fn,
    player_team_set=team_set,
)
```

Metamon's main feature is that it creates a dataset of "reconstructed" human demonstrations for these environments:

```python
from metamon.data import ParsedReplayDataset
# pytorch dataset. examples are converted to 
# the chosen obs/actions/rewards on-the-fly.
offline_dset = ParsedReplayDataset(
    observation_space=obs_space,
    action_space=action_space,
    reward_function=reward_func,
    formats=["gen1ou"],
)
obs_seq, action_seq, reward_seq, done_seq = offline_dset[0]
```

We can save our own agents' experience in the same format:

```python
env = QueueOnLocalLadder(
    .., # rest of args
    save_trajectories_to="my_data_path",
)
online_dset = ParsedReplayDataset(
    dset_root="my_data_path",
    observation_space=obs_space,
    action_space=action_space,
    reward_function=reward_func,
)
terminated = False
while not terminated:
    *_, terminated, _, _ = env.step(env.action_space.sample())
# find completed battles before loading examples
online_dset.refresh_files()
```

You are free to use this data to train an agent however you'd like, but we provide starting points for smaller-scale IL (`python -m metamon.il.train`) and RL (`python -m metamon.rl.train`), and a large set of pretrained models from our paper.


### PokéAgent Challenge
To run agents on the [PokéAgent Challenge ladder](http://pokeagentshowdown.com.insecure.psim.us/):

1. Go to the link above and click "Choose name" in the top right corner. *Pick a username that begins with `"PAC"`*. 

2. Click the gear icon, then "register", and create a password.

2. Use `metamon.env.PokeAgentLadder` exactly how you use `QueueOnLocalLadder` in local tests. Provide your account details with `player_username` and `player_password` args.


<br>

____

<br>


## Pretrained Models

We have made every checkpoint of 20 models available on huggingface at [`jakegrigsby/metamon`](https://huggingface.co/jakegrigsby/metamon/tree/main). Pretrained models can run without research GPUs, but you will need to install [`amago`](https://github.com/UT-Austin-RPL/amago), which is an RL codebase by the same authors. Follow instructions [here](https://ut-austin-rpl.github.io/amago/installation.html).


<div align="center">
    <img src="media/arch_v6_safe.png" alt="Figure 1" width="450">
</div>

<br>

Load and run pretrained models with `metamon.rl.evaluate`. For example:

```bash
python -m metamon.rl.evaluate --eval_type heuristic --agent SyntheticRLV2 --gens 1 --formats ou --total_battles 100
```

Will run the default checkpoint of the best model for 100 battles against a set of heuristic baselines highlighted in the paper.

Or to battle against whatever is logged onto the local Showdown server (including other pretrained models that are already waiting):

```bash
python -m metamon.rl.evaluate --eval_type ladder --agent SyntheticRLV2 --gens 1 --formats ou --total_battles 50 --username <pick unique username> --team_set competitive
```

Deploy pretrained agents on the PokéAgent Challenge ladder:

```bash
python -m metamon.rl.evaluate --eval_type pokeagent --agent SyntheticRLV2 --gens 1 --formats ou --total_battles 10 --username <your username> --password <your password> --team_set competitive
```

Some model sizes have several variants testing different RL objectives. See `metamon/rl/pretrained.py` for a complete list.

### Paper Policies
*Paper policies play Gens 1-4 and are discussed in detail in the RLC 2025 paper.*


| Model Name (`--agent`)                  | Description                                                                 |
|-----------------------------|-----------------------------------------------------------------------------|
| **`SmallIL`** (2 variants)                | 15M imitation learning model trained on 1M human battles             |
| **`SmallRL`** (5 variants)                | 15M actor-critic model trained on 1M human battles                 |
| **`MediumIL`**                | 50M imitation learning model trained on 1M human battles          |
| **`MediumRL`** (3 variants)                | 50M actor-critic model trained on 1M human battles        |
| **`LargeIL`**                 | 200M imitation learning model trained on 1M human battles            |
| **`LargeRL`**                 | 200M actor-critic model trained on 1M human battles          |
| **`SyntheticRLV0`**           | 200M actor-critic model trained on 1M human + 1M diverse self-play battles          |
| **`SyntheticRLV1`**           | 200M actor-critic model trained on 1M human + 2M diverse self-play battles  |
| **`SyntheticRLV1_SelfPlay`**   | SyntheticRLV1 fine-tuned on 2M extra battles against itself                 |
| **`SyntheticRLV1_PlusPlus`**          | SyntheticRLV1 finetuned on 2M extra battles against diverse opponents      |
| **`SyntheticRLV2`**           | Final 200M actor-critic model with value classification trained on 1M human + 4M diverse self-play battles. |

### PokéAgent Challenge Policies
*Policies trained during the PokéAgent Challenge play Gens 1-4 **and 9**, but have a clear bias towards Gen 1 OU and Gen 9 OU. Their docstrings in `metamon/rl/pretrained.py` have some extra discussion and eval metrics.*

| Model Name (`--agent`)                  | Description                                                                 |
|-----------------------------|-----------------------------------------------------------------------------|
| **`SmallRLGen9Beta`**         | Prototype 15M actor-critic model trained *after* the dataset was expanded to include Gen9OU |
| **`Abra`** | 57M actor-critic trained on `parsed-replays v3` and a small set of synthetic battles. First of a new series of Gen9OU-compatible policies trained in a similar style to the paper's "Synthetic" agents.| 
| **`Kadabra, Alakazam, Alakazam2, Alakazam3`** | Are further extensions to large datasets of self-play battles (> 11M). They appear on the PokéAgent Challenge practice ladder, but checkpoint releases are on hold to avoid interfering with the competition. | 
| **`Minikazam`** | 4.7M RNN trained on `parsed-replays v4` and a large dataset of self-play battles. Tries to compensate for low parameter count by training on `Alakazam`'s dataset. Creates a decent starting point for finetuning on any GPU. [Evals here](https://docs.google.com/spreadsheets/d/1GU7-Jh0MkIKWhiS1WNQiPfv49WIajanUF4MjKeghMAc/edit?usp=sharing). |


Here is a reference of human evals for key models according to our paper:

<div align="center">
    <img src="media/human_ratings.png" alt="Figure 1" width="800">
</div>

> [!TIP]
> Most these policies predate our expansion to Gen 9. They *can* play Gen 9 OU, but won't play it well. Gen 9 training runs are ongoing.

<br>

____

<br>


## Battle Datasets

Showdown creates "replays" of battles that players can choose to upload to the website before they expire. We gathered all surviving historical replays for Gen 1-4 OU/NU/UU/Ubers and Gen 9 OU, and continuously save new battles to grow the dataset.



<div align="center">
    <img src="media/dataset.png" alt="Dataset Overview" width="800">
</div>
<br>


Datasets are stored on huggingface in two formats:

| Name |  Size | Description |
|------|------|-------------|
|**[`metamon-raw-replays`](https://huggingface.co/datasets/jakegrigsby/metamon-raw-replays)** | 2M Battles | Our curated set of Pokémon Showdown replay `.json` files... to save the Showdown API some download requests and to maintain an official reference of our training data. Will be regularly updated as new battles are played and collected. |
|**[`metamon-parsed-replays`](https://huggingface.co/datasets/jakegrigsby/metamon-parsed-replays)** | 4M Trajectories | The RL-compatible version of the dataset as reconstructed by the [replay parser](metamon/backend/replay_parser/README.md). This dataset has been significantly expanded and improved since the original paper.|

Parsed replays will download automatically when requested by the `ParsedReplayDataset`, but these datasets are large. Use `python -m metamon.data.download parsed-replays` to download them in advance.


#### Server/Replay Sim2Sim Gap

In Showdown RL, we have to embrace a **mismatch between the trajectories we *observe in our own battles* and those we *gather from other player's replays***. In short, replays are saved from the point-of-view of a *spectator* rather than the point-of-view of a *player*. The server sends info to the players that it does not save to its replay, and we need to try and simulate that missing info. Metamon goes to great lengths to handle this, and is always improving ([more info here](metamon/backend/replay_parser/README.md)), but there is no way to be perfect. 

**Therefore, replay data is perhaps best viewed as pretraining data for an offline-to-online finetuning problem.** Self-collected data from the online env fixes inaccuracies and can help concentrate on teams we'll be using on the ladder. The whole project is now set up to do this (see [Quick Start](#quick-start)).



<br>

___

<br>

 ## Team Sets

 Team sets are dirs of Showdown team files that are randomly sampled between episodes. They are stored on huggingface at [`jakegrigsby/metamon-teams`](https://huggingface.co/datasets/jakegrigsby/metamon-teams) and can be downloaded in advance with `python -m metamon.data.download teams`

```python
metamon.env.get_metamon_teams(battle_format : str, set_name : str)
```

 | `set_name` | Teams Per Battle Format | Description |
|------|---------------------------|-----------------------|
|`"competitive"`| Varies (< 30) | Human-made teams scraped from forum threads. These are usually official "sample teams" designed by experts for beginners, but we are less selective for non-OU tiers. This is the set used for human ladder evaluations in the paper. |
|`"paper_variety"`| (Gen 1-4 Only) 1k | Procedurally generated teams with unrealistic OOD lead-off Pokémon. The paper calls this the "variety set". Movesets were generated by sampling from all-time usage stats. |
| `"paper_replays"` | 1k (Gen 1-4 OU Only) | *Predicted* teams from replays. The paper calls this the "replay set". Surpassed by the "modern_replays" set below. Used the original prediction strategy of sampling from all-time usage stats.|
| `"modern_replays"` | 8k-20k<br> (OU Only) | *Predicted* teams based on recent replays using the best prediction strategy we have available for each generation. The result is a diverse set representing the recent metagame with blanks filled by a mixture of historical trends. |
| `"modern_replays_v2"` | Gen1: 19k, Gen2: 13k, Gen3: 31k, Gen4: 27k, Gen9: 158k. | An expanded set of replay-predicted teams; updated with Summer 2025 replays.

The HF readme has more information.

We can also use our own directory of team files with, for example:
```python
from metamon.env import TeamSet

team_set = TeamSet("/path/to/your/team/dir", battle_format: str) # e.g. gen3ou
```
But note that files would need to have the extension `".{battle_format}_team"` (e.g., .gen3nu_team).


<br>

___

<br>


## Baselines

`baselines/` contains baseline opponents that we can battle against via `BattleAgainstBasline`. `baselines/heuritics` provides more than a dozen heuristic opponents and starter code for developing new ones (or mixing ground-truth Pokémon knowledge into ML agents). `baselines/model_based` ties the simple `il` model checkpoints to `poke-env` (with CPU inference).


Here is an overview of the opponents mentioned in the paper:

```python
from metamon.baselines import get_baseline, get_all_baseline_names
opponent = get_baseline(name)  # Get specific baseline
available = get_all_baseline_names()  # List all available baselines
```

 | `name` | Description |
|------|-------------|
| `BugCatcher` | An actively bad trainer that always picks the least damaging move. When forced to switch, picks the pokemon in its party with the worst type matchup vs the player.
|`RandomBaseline`| Selects a legal move (or switch) uniformly at random and measures the most basic level of learning early in training runs.|
|`Gen1BossAI`| Emulates opponents in the original Pokémon Generation 1 games. Usually chooses random moves. However, it prefers using stat-boosting moves on the second turn and “super effective” moves when available. |
| `Grunt` | A maximally offensive player that selects the move that will deal the greatest damage against the current opposing Pokémon using Pokémon’s damage equation and a type chart and selects the best matchup by type when forced to switch.|
| `GymLeader` | Improves upon Grunt by additionally taking into account factors such as health. It prioritizes using stat boosts when the current Pokémon is very healthy, and heal moves when unhealthy.|
| `PokeEnvHeuristic` | The `SimpleHeuristicsPlayer` baseline provided by [`poke-env`](https://github.com/hsahovic/poke-env) with configurable difficulty (shortcuts like `EasyPokeEnvHeuristic`).|
| `EmeraldKaizo` | An adaptation of the AI in a Pokémon Emerald ROM hack intended to be as difficult as possible. It selects actions by scoring the available options against a rule set that includes handwritten conditional statements for a large portion of the moves in the game.|
| `BaseRNN` | A simple RNN IL policy trained on an early version of our parsed replay dataset. Runs inference on CPU.|

Compare baselines with:

```bash
python -m metamon.baselines.compete --battle_format gen2ou --player GymLeader --opponent RandomBaseline --battles 10
```

Here is a reference for the relative strength of some heuristic baselines from the paper:
<div align="center">
    <img src="media/heuristic_heatmap.png" alt="Figure 1" width="380">
</div>

<br>
<br>

___

<br>

## Observation Spaces, Action Spaces, & Reward Functions

Metamon tries to separate the RL from Pokémon. All we need to do is pick an `ObservationSpace`, `ActionSpace`, and `RewardFunction`:

 1. The environment outputs a `UniversalState`
 2. Our `ObservationSpace` maps the `UniversalState` to the input of our agent.
 3. Our agent outputs an action however we'd like.
 4. Our `ActionSpace` converts the agent's choice to a `UniversalAction`. 
 5. The environment takes the current (`UniversalState`, `UniversalAction`) and outputs the next `UniversalState`. Our `RewardFunction` gives the agent a scalar reward.
 7. Repeat until victory.

### Observations

`UniversalState` defines all the features we have access to at each timestep.

The `ObservationSpace` packs those features into a policy input.  
We could create a custom version with more/less features by inheriting from `metamon.interface.ObservationSpace`.

| Observation Space                            | Description                                                                 |
|--------------------------------------|-----------------------------------------------------------------------------|
| `DefaultObservationSpace`           | The text/numerical observation space used in our paper.                 |
| `ExpandedObservationSpace`          | A slight improvement based on lessons learned from the paper. It also adds tera types for Gen 9. |
| `TeamPreviewObeservationSpace`      | Further extends `ExpandedObservationSpace` with a preview of the opponent's team (for Gen 9). |
| `OpponentMoveObservationSpace`      | Modifies `TeamPreviewObservationSpace` to include the opponent Pokémon's revealed moves. Continues our trend of deemphasizing long-term memory. |

##### Tokenization

Text features have inconsistent length, but we can translate to int IDs from a list
of known vocab words. The built-in observation spaces are designed such that the "tokenized" version *will* have fixed length.

```python
from metamon.interface import TokenizedObservationSpace, DefaultObservationSpace
from metamon.tokenizer import get_toknenizer

base_obs = DefaultObservationSpace()
tokenized_space = TokenizedObservationSpace(
    base_obs_space=base_obs,
    tokenizer=get_tokenizer("DefaultObservationSpace-v0"),
)
```

The vocabs are in `metamon/tokenizer`; they are generated by tracking unique
words across the entire replay dataset, with an unknown token for rare cases we may have missed.

 | Tokenizer Name | Description |
|------|-------------|
| `allreplays-v3` | Legacy version for pre-release models. |
|`DefaultObservationSpace-v0`| Updated post-release vocabulary as of `metamon-parsed-replays` dataset `v2`. |
|`DefaultObservationSpace-v1`| Updated vocabulary as of `metamon-parsed-replays` dataset `v3-beta` (adds ~1k words for Gen 9). |

### Actions

Metamon uses a fixed `UniversalAction` space of 13 discrete choices:
- `{0, 1, 2, 3}` use the active Pokémon's moves in alphabetical order.
- `{4, 5, 6, 7, 8}` switch to the other Pokémon in the party in alphabetical order.
- `{9, 10, 11, 12}` are wildcards for generation-specific gimmicks. Currently, they only apply to Gen 9, where they pick moves (in alphabetical order) *with terastallization*.

That might not be how we want to set up our agent. The `ActionSpace` converts between whatever the output of the policy might be and the `UniversalAction`.

| Action Space              | Description                                                                                      |
|------------------------|--------------------------------------------------------------------------------------------------|
| `DefaultActionSpace`   | Standard discrete space of 13 and supports Gen 9.                                          |
| `MinimalActionSpace`   | The original space of 9 choices (4 moves + 5 switches) --- which is all we need for Gen 1-4. |

Any new action spaces would be added to `metamon.interface.ALL_ACTION_SPACES`. A text action space (for LLM-Agents) is on the short-term roadmap. 


### Rewards

Reward functions assign a scalar reward based on consecutive states (R(s, s')). - 
| Reward Function                 | Description                                                                                  |
|--------------------------|----------------------------------------------------------------------------------------------|
| `DefaultShapedReward`    | Shaped reward used by the paper. +/- 100 for win/loss, light shaping for damage dealt, health recovered, status received/inflicted.                                                      |
| `BinaryReward`           | Removes the smaller shaping terms and simply provides +/- 100 for win/loss.                 |
| `AggresiveShapedReward`   | Edits `DefaultShapedReward`'s sparse reward to +200 for winning +0 for losing. |

Any new reward functions would be added to `metamon.interface.ALL_REWARD_FUNCTIONS`, and we can implement a new one by inheriting from `metamon.interface.RewardFunction`.

---

<br>


 ## Training and Evaluation

<img src="media/metamon_and_amago.png" alt="Metamon & Amago Diagram" width="200" style="float: right; margin-left: 20px; margin-bottom: 10px;">

We trained all of our main RL **& IL** models with [`amago`](https://ut-austin-rpl.github.io/amago/index.html). Everything you need to train your own model on metamon data and evaluate against Pokémon baselines is provided in **`metamon/rl/`**.


#### Configure `wandb` logging (optional):
```shell
cd metamon/rl/
export METAMON_WANDB_PROJECT="my_wandb_project_name"
export METAMON_WANDB_ENTITY="my_wandb_username"
```

<br>

### Train From Scratch

See `python train.py --help` for options. The training script implements offline RL on the human battle dataset *and* an optional extra dataset of self-play battles you may have collected.

We might retrain the "`SmallIL`" model like this: 

```bash
python -m metamon.rl.train --run_name AnyNameHere --model_gin_config small_agent.gin --train_gin_config il.gin --save_dir ~/my_checkpoint_path/ --log
```
"`SmallRL`" would be the same command with `--train_gin_config exp_rl.gin`. Scan `rl/pretrained.py` to see the configs used by each pretrained agent. Larger training runs take *days* to complete and [can (optionally) use mulitple GPUs (link)](https://ut-austin-rpl.github.io/amago/tutorial/async.html#multi-gpu-training). An example of a smaller RNN config is provided in `small_rnn.gin`. 

<br>


### Finetune from HuggingFace

**See `python finetune_from_hf.py --help` to finetune an existing model to a new dataset, training objective, or reward function!** 

Provides the same setup as the main `train` script but takes care of downloading and matching the config details of our public models. Finetuning will inherit the architecture of the base model but allows for changes to the `--train_gin_config` and `--reward_function`. Note that the best settings for quick finetuning runs are likely different from the original run!

We might finetune "`SmallRL`" to the new gen 9 replay dataset and custom battles like this:

```bash
python -m metamon.rl.finetune_from_hf --finetune_from_model SmallRL --run_name MyCustomSmallRL --save_dir ~/metamon_finetunes/ --custom_replay_dir /my/custom/parsed_replay_dataset --custom_replay_sample_weight .25 --epochs 10 --steps_per_epoch 10000 --log --formats gen9ou --eval_gens 9 
```

You can start from any checkpoint number with `--finetune_from_ckpt`. See the huggingface for a full list. Defaults to the official eval checkpoint.

<br>


### Customize

Customize the agent architecture by creating new `rl/configs/models/` `.gin` files. Customize the RL hyperparameters by creating new `rl/configs/training/` files. [Here is a link](https://ut-austin-rpl.github.io/amago/tutorial/configuration.html) to a lot more information about configuring training runs. `amago` is modular, and you can swap just about any piece of the agent with your own ideas. [Here is a link](https://ut-austin-rpl.github.io/amago/tutorial/customization.html) to more information about custom components.


<br>


### Evaluate a Custom Model

`metamon.rl.evaluate` provides quick-setup evals (`pretrained_vs_baselines`, `pretrained_vs_local_ladder`, and `pretrained_vs_pokeagent_ladder`). Full explanations are provided in the source file.

To eval a custom agent trained from scratch (`rl.train`) we'd create a `LocalPretrainedModel`. `LocalFinetunedModel` provides some quick setup for models finetuned with `rl.finetune_from_hf`. [`examples/evaluate_custom_models.py`](examples/evaluate_custom_models.py) shows an example for each, and deploys them on the PokéAgent Ladder!


#### Standalone Toy `il` (Deprecated)

<details>

`il/` is old toy code that does basic behavior cloning with RNNs. We used it to train early learning-based baselines (`BaseRNN`, `WinsOnlyRNN`, and `MiniRNN`) that you can play against with the `BattleAgainstBaseline` env. We may add more of these as the dataset grows/improves and more architectures are tried. Playing around with this code might be an easier way to get started, but note that the main `rl/train` script can also be configured to do RNN BC... but faster and on multiple GPUs.

Get started with something like:
```shell
cd metamon/il/
python train.py --run_name any_name_will_do --model_config configs/transformer_embedding.gin  --gpu 0
```

</details>

 ---

 <br>

 ## Other Datasets

To support the main [raw-replays](https://huggingface.co/datasets/jakegrigsby/metamon-raw-replays), [parsed-replays](https://huggingface.co/datasets/jakegrigsby/metamon-parsed-replays), and [teams](https://huggingface.co/datasets/jakegrigsby/metamon-teams) datasets, metamon creates a few resources that may be useful for other purposes:


 #### Usage Stats
Showdown records the frequency of team choices (items, moves, abilities, etc.) brought to battles in a given month. The community mainly uses this data to consider rule changes, but we use it to help predict missing details of partially revealed teams. We load data for an arbitrary window of history around the date a battle was played, and fall back to all-time stats for rare Pokémon where data is limited:

```python
from metamon.backend.team_prediction.usage_stats import get_usage_stats
from datetime import date
usage_stats = get_usage_stats("gen1ou",
    start_date=date(2017, 12, 1),
    end_date=date(2018, 3, 30)
)
alakazam_info: dict = usage_stats["Alakazam"] # non alphanum chars and case are flexible
```

Download usage stats in advance with:
```shell
python -m metamon.data.download usage-stats
```

The data is stored on huggingface at [`jakegrigsby/metamon-usage-stats`](https://huggingface.co/datasets/jakegrigsby/metamon-usage-stats).

#### Revealed Teams
One of the main problems the replay parser has to solve is predicting a player's full team based on the "partially revealed" team at the end of the battle. As part of this, we record the revealed team in the [standard Showdown team builder format](https://pokepast.es/syntax.html), but with some magic keywords for missing elements. For example:

```
Tyranitar @ Custap Berry
Ability: Sand Stream
EVs: $missing_ev$ HP / $missing_ev$ Atk / $missing_ev$ Def / $missing_ev$ SpA / $missing_ev$ SpD / $missing_ev$ Spe
$missing_nature$ Nature
IVs: 31 HP / 31 Atk / 31 Def / 31 SpA / 31 SpD / 31 Spe
- Stealth Rock
- Stone Edge
- Pursuit
- $missing_move$
```

Given the size of our replay dataset, this creates a massive set of real (but incomplete) human team choices. The files are stored alongside the parsed-replay dataset and downloaded with:

```shell
python -m metamon.data.download revealed-teams
```

`metamon/backend/team_prediction` contains tools for filling in the blanks of these files, but this is all poorly documented and changes frequently, so we'll leave it at that for now.

----

<br>


## Battle Backends
Converting Showdown messages to RL observations is hard, and there will always be bugs. Minor fixes to edge cases or rare Pokémon mechanics are fairly common and don't have a real impact on overall performance. However, a fix that directly impacts observation features (agent inputs) usually *decreases* performance of policies trained on older battles. We extend the lifespan of pretrained model weights by versioning the "battle backend" so that we can evaluate the agent in an environment that matches the dataset it was trained on. 

`battle_backend : str` is an arg for all the RL environment wrappers (see [Quick Start](#quick-start)). 


There are currently three versions:

| `battle_backend`    | Description                                                                                                     | Known Bugs                                                                                                                                          | When To Use                                                                                                                               |
|------------|-----------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------|
| `"poke-env"`   | Original paper verison. Uses [`poke-env`](https://github.com/hsahovic/poke-env) to process online battles.                        | - Creates a sim2sim gap with the replay parser that generates training data from replays. <br> - PP counting and tera types are broken. | When evaluating the original paper policies.           |
| `"pokeagent"`  | Replaces `poke-env`'s message parsing with metamon's replay parser. Maintains the version used by all the new baselines and datasets created for the [PokéAgent Challenge](https://pokeagent.github.io).                                   | - Gen9 was in Beta; tera types are reported as missing.                                              | When evaluating a policy trained during the competition (see [Pretrained Models](#pretrained-models)).
| **`"metamon"`**    | Always the latest version. |                                                                                   | When collecting new self-play data and training new policies from scratch.                    |
____

A `PretrainedAgent`saves the backend it "should" be evaluated with (if you're using them as a baseline). If you are collecting lots of new self-play data and actively working on new training runs: use `"metamon"`. Thanks to a few hacks, **it is still reasonable to use *any* `PretrainedAgent` to collect new training data in the current `metamon` backend.**

 <br>

## Acknowledgements

This project owes a huge debt to the amazing [`poke-env`](https://github.com/hsahovic/poke-env), as well Pokémon resources like [Bulbapedia](https://bulbapedia.bulbagarden.net/wiki/Main_Page), [Smogon](https://www.smogon.com), and of course [Pokémon Showdown](https://github.com/smogon/pokemon-showdown).

---

<br>

## Citation

```bibtex
@misc{grigsby2025metamon,
      title={Human-Level Competitive Pok\'emon via Scalable Offline Reinforcement Learning with Transformers}, 
      author={Jake Grigsby and Yuqi Xie and Justin Sasek and Steven Zheng and Yuke Zhu},
      year={2025},
      eprint={2504.04395},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2504.04395}, 
}
```
