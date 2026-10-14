# Submission — HRC Software Onboarding Fall 2026

**Name: Aaryan Chaudhary.**
**Discord handle: Aaryan Chaudhary in Humanoid Robots Discord**
**Repository: https://github.com/AaryanC75/onboarding-fall26**
**Date: 10/14/26**

---

## Stage checklist

- [X] **Stage 0** — Setup. `scripts/check_setup.py` exits 0.
- [X] **Stage 1** — MuJoCo + PD controller. Tests green.
- [X] **Stage 2** — JAX exercises. Tests green.
- [X] **Stage 3** — MJX environment. Tests green.
- [X] **Stage 4** — Brax PPO. Trained a policy, exported it, `NumpyPolicy` matches.
- [X] **Stage 5** — ROS 2 sim2sim. Smoke test green, teleop GIF recorded.

## `scripts/progress.py` output

<details>
<summary>paste the full output here</summary>

```
$ uv run python scripts/progress.py --slow
$ /home/caaryan/onboarding-fall26/.venv/bin/python -m pytest -q --tb=no -m not gpu and not ros

....................................                                                                                                              [100%]
=================================================================== warnings summary ====================================================================
.venv/lib/python3.11/site-packages/jaxopt/__init__.py:59
  /home/caaryan/onboarding-fall26/.venv/lib/python3.11/site-packages/jaxopt/__init__.py:59: DeprecationWarning: JAXopt is no longer maintained. See https://docs.jax.dev/en/latest/ for alternatives.
    warnings.warn(

tests/test_04_train_smoke.py::test_cpu_smoke
  /home/caaryan/onboarding-fall26/.venv/lib/python3.11/site-packages/brax/training/agents/ppo/train.py:756: DeprecationWarning: jax.device_put_replicated is deprecated; use jax.device_put instead.
    training_state = jax.device_put_replicated(

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html

Pup onboarding progress
  ✓ PASSED        Stage 1  MuJoCo + PD controller    12 passed, 0 failed, 0 not started, 0 skipped
  ✓ PASSED        Stage 2  JAX for robotics          9 passed, 0 failed, 0 not started, 0 skipped
  ✓ PASSED        Stage 3  MJX environment           11 passed, 0 failed, 0 not started, 0 skipped
  ✓ PASSED        Stage 4  Brax PPO + export         4 passed, 0 failed, 0 not started, 0 skipped
36 passed, 1 skipped, 2 deselected, 2 warnings in 201.31s (0:03:21)

==============================================================================
PUP ONBOARDING PROGRESS
==============================================================================
✓ PASSED         Stage 1  MuJoCo + PD controller    12 passing
✓ PASSED         Stage 2  JAX for robotics          9 passing
✓ PASSED         Stage 3  MJX environment           11 passing
✓ PASSED         Stage 4  Brax PPO + export         4 passing
◻ CONTAINER      Stage 5  ROS 2 sim2sim             run tests/test_05_ros2_smoke.sh inside docker/ros2
==============================================================================
Every TODO(student) block is written. Nice.
```

</details>

## Stage 4 — training results

**Config used:** `full` &nbsp; **Seed:** 0 &nbsp;
**Where it ran:** Colab T4 &nbsp; **Wall-clock:** 3044 seconds

Learning curve: `results/learning_curve.png`
Rollout GIF: `results/training.gif`

```json
{
"seed": 0,
"n_episodes": 5,
"commands": [
{
  "command": [
    0.5,
    0.0,
    0.0
  ],
  "mean_abs_error": [
    0.04996695917695761,
    0.041105532300460615,
    0.0786056399788824
  ],
  "mean_abs_vy": 0.04110553115606308,
  "mean_speed": 0.5274222493171692,
  "fall_rate": 0.0,
  "mean_episode_length": 1000.0
},
{
  "command": [
    1.0,
    0.0,
    0.0
  ],
  "mean_abs_error": [
    0.058226076345145705,
    0.04314939766603056,
    0.07462636933033645
  ],
  "mean_abs_vy": 0.043149396777153015,
  "mean_speed": 0.9905193448066711,
  "fall_rate": 0.0,
  "mean_episode_length": 1000.0
},
{
  "command": [
    0.0,
    0.5,
    0.0
  ],
  "mean_abs_error": [
    0.05050265026481356,
    0.05094660317152738,
    0.06363240804115776
  ],
  "mean_abs_vy": 0.4667776823043823,
  "mean_speed": 0.4710216820240021,
  "fall_rate": 0.0,
  "mean_episode_length": 1000.0
},
{
  "command": [
    0.0,
    0.0,
    1.0
  ],
  "mean_abs_error": [
    0.0560171268954,
    0.04305225918320648,
    0.07111254061460495
  ],
  "mean_abs_vy": 0.04305225983262062,
  "mean_speed": 0.07677174359560013,
  "fall_rate": 0.0,
  "mean_episode_length": 1000.0
},
{
  "command": [
    0.0,
    0.0,
    0.0
  ],
  "mean_abs_error": [
    0.037876560188988516,
    0.03758715333400469,
    0.08882022159416228
  ],
  "mean_abs_vy": 0.03758715093135834,
  "mean_speed": 0.059255894273519516,
  "fall_rate": 0.0,
  "mean_episode_length": 1000.0
}
],
"walking_passes": true
}
```

Did it meet the acceptance criteria (`"walking_passes": true`)? Yes

## Stage 5 — sim2sim results

Teleop GIF: `results/sim2sim_teleop.gif`

```json
{
  "duration_s": 20.0,
  "min_trunk_height": 0.28929871618963604,
  "fell": false,
  "commands": [
    {
      "command": [
        0.5,
        0.0,
        0.0
      ],
      "n_samples": 129,
      "mean_velocity": [
        0.40379603811896586,
        0.025829545747578425,
        -0.012285591437250882
      ],
      "mean_abs_error": [
        0.09620396188103414,
        0.025829545747578425,
        0.012285591437250882
      ],
      "mean_speed": 0.40491458010348824,
      "threshold": 0.25,
      "passed": true
    },
    {
      "command": [
        1.0,
        0.0,
        0.0
      ],
      "n_samples": 150,
      "mean_velocity": [
        0.8767035993103578,
        -0.01241069560602475,
        -0.0043025380152353675
      ],
      "mean_abs_error": [
        0.1232964006896422,
        0.01241069560602475,
        0.0043025380152353675
      ],
      "mean_speed": 0.8769909793558096
    },
    {
      "command": [
        0.0,
        0.0,
        1.0
      ],
      "n_samples": 150,
      "mean_velocity": [
        -0.040300299602862154,
        0.03408683602118567,
        0.9534859032262472
      ],
      "mean_abs_error": [
        0.040300299602862154,
        0.03408683602118567,
        0.046514096773752844
      ],
      "mean_speed": 0.05605162601162641
    },
    {
      "command": [
        0.0,
        0.0,
        0.0
      ],
      "n_samples": 150,
      "mean_velocity": [
        8.239953625392765e-05,
        -3.551783275950307e-05,
        0.0001243566052232069
      ],
      "mean_abs_error": [
        8.239953625392765e-05,
        3.551783275950307e-05,
        0.0001243566052232069
      ],
      "mean_speed": 0.0001393359262945581
    }
  ],
  "passed": true
}
```

Measured `/pup/joint_command` rate from `ros2 topic hz`: 50.010 Hz

## Escape hatches used

- [ ] I used `checkpoints/pup_joystick_flat_reference.npz` for Stage 5 instead
      of my own policy.
- [ ] Other (describe):

*(Using one is fine. Not declaring one is not.)*

## Reflection — Stage 5

**List two ways sim2sim can pass while real hardware still fails, and what you
would add to the sim node to catch each.**

> Shakey sensors: Sensors are always given in perfect values but realistically they'd be changing. Because of jittery sensors it could think it’s tilting when it isn't. I'd add it so sensors are jittery and not always perfect to mimick real hardware.
> Always ready motors: The motors are always strong, but as motors heat up with use they get weaker. If pup trips and starts to fall, a tired motor wouldn't be able to get it up as efficiently as a ready motor. Adding to the simulator the more you walk the more the motor gets weaker would make it more accurate.

## What was hardest?

One paragraph. This is will help us improve onboarding.
> The hardest was the MJX env as it was a lot of code to write and I made a mistake (0.012 when I should've wrote 0.12) when I was writing the code for when it was too low so I had to search through all the code for the mistake. 

## Time spent

| Stage | Hours |
|---|---|
| 0 Setup | 1.5 |
| 1 MuJoCo + PD | 1.5 |
| 2 JAX | 1 |
| 3 MJX env | 3 |
| 4 Brax + export | 2 |
| 5 ROS 2 | 2.5 |
| **Total** | 11.5 |

---

**Then DM the software lead (Henry Tsay) on Discord or show during a meeting.**
