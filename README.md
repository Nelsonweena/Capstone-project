# Capstone Project: Grid Adventure AI Agent

An AI agent for **Grid Adventure**, a 2D grid-based game where an autonomous player must navigate maze-like levels, collect required gems, unlock doors, avoid hazards, use power-ups, push boxes, and reach the exit efficiently.

This project combines **search-based planning**, **state modeling**, and **image-based grid parsing** to build an agent that can work with both structured game states and rendered image observations.

## Project Overview

Grid Adventure is a maze environment made up of different tile types and entities, including walls, lava, gems, coins, keys, locked doors, boxes, exits, and temporary power-ups. The goal is to plan a sequence of valid actions that completes the level while minimizing unnecessary movement and avoiding failure conditions.

The agent supports two types of input:

- **GridState input**: a structured representation of the environment.
- **ImageObservation input**: a rendered image of the grid that must be converted into a usable state representation before planning.

## Key Features

### A* Search Planning

- Uses A* search to plan efficient movement through the grid.
- Computes paths toward important objectives such as gems, keys, doors, power-ups, and the exit.
- Uses Manhattan distance as a heuristic for grid navigation.
- Avoids invalid moves such as walking into walls, blocked doors, unsafe lava, or impossible box pushes.

### High-Level Objective Planning

- Separates planning into high-level and low-level decisions.
- Chooses useful targets such as keys, gems, coins, doors, shields, boots, phasing power-ups, and the exit.
- Prioritizes required objectives before ending the level.
- Replans when a new state is received and no current plan exists.

### Game Mechanic Handling

The agent models several game mechanics, including:

- Collecting gems before exiting.
- Picking up keys and unlocking doors.
- Avoiding or surviving lava based on health and shield status.
- Using speed boots for faster movement.
- Using shield power-ups to reduce hazard risk.
- Using phasing power-ups to pass through obstacles.
- Pushing boxes when the space behind them is valid.
- Collecting coins when useful.

### Image Observation Parsing

- Supports image-based gameplay by converting rendered tiles into predicted entity labels.
- Splits the rendered game image into individual grid tiles.
- Extracts visual features from each tile using edge-based image processing.
- Classifies tiles into game objects such as wall, lava, coin, gem, key, exit, box, boots, shield, and agent.
- Reconstructs a `GridState` from image observations so the same planner can be reused.

### Lightweight Machine Learning Classifier

- Uses handcrafted visual features from tile images.
- Applies a trained logistic regression classifier with embedded model weights.
- Keeps the model self-contained, avoiding the need for separate model files during submission.
- Designed to work within a constrained capstone/project environment.

## Tech Stack

| Area | Tools |
| --- | --- |
| Language | Python |
| AI Planning | A* Search, heuristic search |
| Computer Vision | Pillow, NumPy, edge feature extraction |
| Machine Learning | Logistic regression-style classifier |
| Environment | Grid Adventure |
| Notebook Development | Jupyter Notebook |
| Utilities | heapq, NumPy, custom state modeling |

## Project Structure

```txt
capstone-project/
  capstone-project.ipynb   Main notebook containing the AI agent implementation
  tutorial.ipynb           Tutorial/reference notebook for the Grid Adventure environment
  train_model.py           Script for training the tile classifier
  model_weights.py         Embedded trained model weights
  utils.py                 Evaluation and helper utilities
  environment.yml          Conda environment configuration
  data/
    assets/                Tile assets used for image parsing and model training
```

## How the Agent Works

### 1. State Extraction

For structured `GridState` inputs, the agent scans the grid and records important information such as:

- agent position
- health
- gems
- keys
- locked doors
- boxes
- coins
- lava
- walls
- power-ups
- exit location

This information is stored in a compact internal state representation used by the planner.

### 2. Image Parsing

For image observations, the agent:

1. Reads the rendered image.
2. Splits it into grid tiles.
3. Resizes each tile to a standard size.
4. Converts the tile into edge-based features.
5. Classifies the tile using embedded model weights.
6. Rebuilds a structured grid state from the predicted tile labels.

This allows the same planning logic to work on both visual and structured inputs.

### 3. Low-Level Path Planning

The agent uses a low-level A* search to find movement paths from the current position to a target location. It considers:

- movement cost
- health
- lava damage
- active power-ups
- walls
- locked doors
- boxes
- collectible pickup actions

### 4. High-Level Planning

The agent then uses a higher-level search to decide which target to pursue next. Possible targets include:

- keys
- doors
- gems
- coins
- speed boots
- shields
- phasing power-ups
- exit

The final plan is returned as a sequence of valid game actions.

## Getting Started

### 1. Create the Conda environment

```bash
conda env create -f environment.yml
```

### 2. Activate the environment

```bash
conda activate cs2109s-ay2526s2-capstone-project
```

### 3. Open the notebook

```bash
jupyter notebook capstone-project.ipynb
```

### 4. Run the agent

Open `capstone-project.ipynb` and run the cells containing the `Agent` implementation and evaluation code.

## Main Files

### `capstone-project.ipynb`

Contains the main AI agent implementation, including:

- state extraction
- image parsing
- tile classification
- A* planning
- action selection
- evaluation helpers

### `train_model.py`

Builds a dataset from tile assets, extracts image features, trains a lightweight classifier, and exports model weights.

### `model_weights.py`

Stores embedded NumPy arrays for the trained classifier so the model can be used without loading an external file.

### `utils.py`

Provides helper functions for evaluation, visualization, and model serialization.

## Repository Note

This project was developed as a capstone notebook project. The main implementation is inside `capstone-project.ipynb`, with supporting scripts for training, model weights, assets, and evaluation utilities.
