# X500 Drone RL Симуляция

Настроенная Ignition Gazebo + ROS2 Humble симуляционная среда для обучения RL-агентов управлению квадрокоптером на базе моедли дрона PX4 X500 без PX4/QGroundControl пакетов.

## Преимущества

- ✅ Прямое управление двигателями (контроллер полёта не требуется)
- ✅ Среда, совместимая со спортивным залом, для обучения с поддержанием навыков (RL)
- ✅ Наблюдение за состоянием в реальном времени (IMU, GPS, барометр)
- ✅ Настраиваемые функции вознаграждения
- ✅ Пример реализации SAC (Soft Actor-Critic)
- ✅ Модульная архитектура для удобства экспериментов

## Архитектура

```
┌─────────────────┐
│  RL-агент       │
│  (SAC/PPO/TD3)  │
└────────┬────────┘
         │ действия [thrust, roll, pitch, yaw]
         ▼
┌─────────────────┐
│ Gym среда │ ◄──── state [pos, vel, orientation]
│ (gym_env)       │
└────────┬────────┘
         │ ROS2 топики
         ▼
┌─────────────────┐
│ Контроллер дрона│ ◄──── /x500/state
│ (motor mixing)  │ ────► /x500/action
└────────┬────────┘
         │ скорости приводов
         ▼
┌─────────────────┐
│ Ignition Gazebo │
│ (X500 model)    │
└─────────────────┘
```

## Пререквизиты

```bash
# Ubuntu 22.04 с ROS2 Humble
sudo apt update

# Установка ROS2 Humble (если не была установлена ранее)
# Follow: https://docs.ros.org/en/humble/Installation.html

# Установка Gazebo (Ignition)
sudo apt install ros-humble-ros-gz ros-humble-ros-gz-sim ros-humble-ros-gz-bridge

# Установка вспомогательных программных пакетов
sudo apt install python3-pip python3-colcon-common-extensions
pip3 install gymnasium stable-baselines3 numpy scipy torch
```


### Указать системе путь к пакетам

```bash
# Add to ~/.bashrc
echo 'export GZ_SIM_RESOURCE_PATH=$GZ_SIM_RESOURCE_PATH:~/drone_rl_ws/src/x500_simulation/models' >> ~/.bashrc
source ~/.bashrc
```

### Сборка

```bash
cd ~/drone_rl_ws
colcon build
source install/setup.bash
```

## Использование

### Запуск симуляции

```bash
# Терминал 1: Запуск Gazebo + Контроллер
ros2 launch x500_simulation drone_sim.launch.py

# После отправки команды запускаются следующие программы:
# - Gazebo
# - X500 дрон в Gazebo
# - Нода, реализующая контроллер дрона
# - ROS2-Gazebo bridge
```

### Обучение RL-агента

```bash
# Терминал 2: запуск обучения
cd ~/drone_rl_ws/src/x500_simulation/scripts
python3 train_drone.py --mode train --timesteps 1000000


### Проверка

```bash
python3 train_drone.py --mode test --model_path ./models/drone_sac_final.zip
```

### Тестирование

```bash
# Терминал 3: отправка команды
ros2 topic pub /x500/action std_msgs/msg/Float32MultiArray \
  "data: [0.5, 0.0, 0.0, 0.0]" --once

# Монитор состояния
ros2 topic echo /x500/state

# Одометрия
ros2 topic echo /model/x500/odometry
```

## Настройка

### Изменение reward function

Edit `drone_gym_env.py`:

```python
def compute_reward(self):
    # Задание reward логики
    distance = np.linalg.norm(self.state[:3] - self.target_position)
    
    # Пример: sparse reward
    reward = 100.0 if distance < 0.5 else -distance
    
    return reward
```

### Настройка приводов

Edit `drone_controller.py`:

```python
def compute_motor_speeds(self, thrust, roll_rate, pitch_rate, yaw_rate):
    # Для изменения конфигурации дрона редактировать матрицу
    mixing_matrix = np.array([
        # [thrust, roll, pitch, yaw]
        [1,  -1,  1,  1],
        # ...
    ])
```


## Пространство состояний

Происходит оценка следующих величин:

```python
[
    # Положение (3D)
    x, y, z,
    
    # Скорость (3D)
    vx, vy, vz,
    
    # Ориентация (3D)
    roll, pitch, yaw,
    
    # Угловая скорость (3D)
    wx, wy, wz,
    
    # Координаты целевой точки пути (3D)
    target_x, target_y, target_z
]
```

## Пространства действий

4D непрерывные опции:

```python
[
    thrust,      # [0, 1] - normalized thrust
    roll_rate,   # [-1, 1] - desired roll rate (rad/s)
    pitch_rate,  # [-1, 1] - desired pitch rate (rad/s)
    yaw_rate     # [-1, 1] - desired yaw rate (rad/s)
]
```



- [Gazebo Documentation](https://gazebosim.org/docs)
- [ROS2 Humble Docs](https://docs.ros.org/en/humble/)
- [Stable Baselines3](https://stable-baselines3.readthedocs.io/)
- [Gymnasium](https://gymnasium.farama.org/)
