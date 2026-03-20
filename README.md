# ART-Track

[![arXiv](https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b.svg)](https://arxiv.org/abs/XXXX.XXXXX) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) ![Python](https://img.shields.io/badge/By-Python-blue)

**ART-Track** is a motion-driven multi-object tracker designed for **multi-animal tracking in space science experimental videos**. It aims to improve long-term identity preservation under **weak appearance cues, low-quality imaging, complex nonlinear motion, and frequent interactions**. ART-Track is built for **closed-space biological observation scenarios**, where most targets remain in view for long periods, making motion-based identity recovery especially important.

The tracker combines **adaptive interacting multiple-model motion estimation**, **motion-state-driven association**, and **uncertainty-adaptive fusion** to improve trajectory usability and reduce identity switches in long-sequence animal behavior videos.

## Pipeline
<center>
<img src="assets/framework.pdf" width="700"/>
</center>

## Dataset Overview
We introduce **SpaceAnimal-MOT**, a dataset for multi-animal tracking in microgravity and space science experimental videos. It is designed to highlight:

- **Weak appearance cues** caused by reflections, bubbles, compression artifacts, and low-quality imaging
- **Complex motion patterns** such as abrupt acceleration, sharp turns, and nonlinear movement
- **Long-term identity preservation** in closed-space observation chambers

<center>
<img src="assets/data.pdf" width="700"/>
</center>

## News
* [03/20/2026]: Initial release of ART-Track and SpaceAnimal-MOT.
* [03/20/2026]: README, codebase, and benchmark results are released.
* [03/20/2026]: More documentation and training / evaluation scripts will be added soon.

## Benchmark Performance

### SpaceAnimal-MOT

| Dataset    | HOTA | IDF1 | IDs | MOTA | AssA | DetA |
|------------|------|------|-----|------|------|------|
| Zebrafish  | 40.5 | 56.8 | 26  | 61.3 | 33.4 | 49.2 |
| Fruitfly   | 62.2 | 81.6 | 85  | 84.2 | 60.6 | 65.0 |

### Oracle Detection Upper Bound

| Dataset    | HOTA | IDF1 | IDs | MOTA | AssA | DetA |
|------------|------|------|-----|------|------|------|
| Zebrafish  | 84.4 | 81.2 | 23  | 99.9 | 71.5 | 99.6 |
| Fruitfly   | 93.6 | 93.7 | 83  | 99.7 | 87.9 | 99.8 |

## Key Features

- **Motion-driven tracking** for weak-appearance animal videos
- **AIMM-UKF** for abrupt maneuvers and nonlinear motion
- **Motion-state-driven cascaded association** for stable long-term identity preservation
- **Uncertainty-adaptive fusion** for robust matching under variable prediction reliability
- Designed for **closed-space, long-duration observation videos**
- More practical than heavily retrained learning-based trackers in long experimental sequences

## Get Started

### Installation
Clone the repository and install dependencies:

```bash
git clone https://github.com/yourname/ART-Track.git
cd ART-Track
pip install -r requirements.txt
