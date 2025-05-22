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
  
  <img src="https://github.com/user-attachments/assets/4e114285-9137-4825-8816-16334b002273" width="1925" alt="Description">
  

  <table>
  <tr>
    <td>
      <img src="https://github.com/user-attachments/assets/95fe1fd8-a8b9-480e-bdbb-f8d625d91234" alt="Video 1">
    </td>
    <td>
      <img src="https://github.com/user-attachments/assets/c3ab3476-79db-49fb-ad37-579158df567b" alt="Video2 ">
    </td>
  </tr>
  </table>
</div>







