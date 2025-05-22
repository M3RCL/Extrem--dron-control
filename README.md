# Extrem--dron-control
Репозиторий содержит наработки команды по НИРСИИ на тему Развитие методов управления БАС на основе технологий ИИ в режимах экстремального маневрирования в динамическом окружении

## Структура репозитория

Репозиторий содержит реализацию трёх алгоритмов на базе RL для управления квадрокоптером при экстремальном маневрировании.

```bash
extrem-dron-control/
├── Approaches
│   ├── DDQN
│   │   ├── ddqn_description.md
│   │   ├── DualDQN.py
│   │   └── interrupted_drone_dualdqn.pth
│   ├── PPO
│   │   ├── ImprovedPPO.py
│   │   ├── ppo_description.md
│   │   └── PPO_drone.py
│   └── SAC
│       ├── SAC_casc.py
│       ├── sac_m_description.md
└── README.md
```

Описание алгоритмов:

[DDQN](./Approaches/DDQN/ddqn_description.md)  

[PPO](./Approaches/PPO/ppo_description.md)  

[SAC](./Approaches/SAC/sac_m_description.md)

## Пререквизиты

- Python 3.8+
- TensorFlow
- PyTorch
- Python API AirSim
- NumPy
- Stable Baselines3
- Unreal Engine с настроенным плагином AirSim"
- OpenCV
- Gymnasium

## Демонстрационные материалы
