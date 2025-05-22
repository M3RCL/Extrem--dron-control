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

<div style="text-align: center;">
  <img src="https://github.com/user-attachments/assets/9ffe43ed-5db7-41fc-9f78-1e84f0766612" width="1925" alt="Road map" />

  <table style="width: 100%; margin: 200px auto; display: table; border-collapse: separate; border-spacing: 200px;">
    <tr>
      <td style="width: 50%; text-align: center; padding: 200px;">
        <a href="https://github.com/user-attachments/assets/5d38a574-dd75-4b17-8e61-80d49d10d4d8">
          <img src="https://github.com/user-attachments/assets/26614518-9272-431a-959f-2417900e2b00" alt="Video 1 Thumbnail" style="max-width: 96%;">
        </a>
      </td>
      <td style="width: 50%; text-align: center; padding: 200px;">
        <a href="https://github.com/user-attachments/assets/79091c33-6c83-4120-b1f8-db970f30a6b1">
          <img src="https://github.com/user-attachments/assets/7c1d691e-ea74-4937-812b-656952125129" alt="Video 2 Thumbnail" style="max-width: 96%;">
        </a>
      </td>
    </tr>
  </table>
</div>









