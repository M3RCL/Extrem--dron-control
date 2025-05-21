**Улучшенная навигация дрона в AirSim с использованием PPO**

**Обзор**

В этом вилле репозитория содержатся две улучшенные реализации автономной навигации дрона в AirSim с использованием алгоритма Proximal Policy Optimization (PPO). Обе реализации используют данные с глубинной камеры и векторные признаки для навигации дрона через заданные путевые точки с учетом препятствий. Каждая реализация предлагает уникальные улучшения для повышения устойчивости обучения и качества навигации.

**Возможности**

**Общие возможности**

- Подход на основе усиленного обучения с использованием алгоритма PPO
- Комбинированная архитектура CNN и MLP для обработки глубинных изображений и векторных признаков
- Пользовательские функции вознаграждения для стимулирования эффективной навигации и избегания препятствий
- Интеграция с симулятором AirSim
- Поддержка навигации через несколько путевых точек
  
**ImprovedPPO.py**

- Улучшенный извлекатель признаков с остаточными связями и нормализацией слоев
- Улучшенная форма вознаграждения с компонентами для плавности, прогресса и исследования
- Углубленные политическая и оценочная сети
- Планирование скорости обучения и продвинутая настройка гиперпараметров
- Дополнительный трекинг метрик для лучшего мониторинга обучения
  
**PPO_drone.py**

- Простая реализация извлекательа признаков
- Базовая функция вознаграждения,фокусирующаяся на расстоянии до путевых точек и скорости
- Подходит для начальных экспериментов и сравнения с базовой линией
- Совместима с стандартными гиперпараметрами PPO

**Требования**
- Python 3.8+
- PyTorch
- Stable Baselines3
- Python API AirSim
- NumPy
- OpenCV
- Gymnasium

_________________________________________________________________________________________
**Enhanced Drone Navigation in AirSim using PPO**

**Overview**

This forked repository contains two improved implementations of autonomous drone navigation in AirSim using the Proximal Policy Optimization (PPO) algorithm. Both implementations leverage depth camera data and vector features to navigate drones through specified waypoints while avoiding obstacles. Each implementation offers unique enhancements to improve learning stability and navigation performance.

**Features**

**Common Features**

- Reinforcement learning approach with PPO algorithm
- Combined CNN and MLP architecture for processing depth images and vector features
- Custom reward functions to encourage efficient navigation and obstacle avoidance
- Integration with AirSim simulation environment
- Support for multiple waypoints navigation

**ImprovedPPO.py**

- Enhanced feature extractor with residual connections and layer normalization
- Improved reward shaping with components for smoothness, progress, and exploration
- Deeper policy and value networks
- Learning rate scheduling and advanced hyperparameter tuning
- Additional metrics tracking for better training monitoring
  
**PPO_drone.py**

- Simpler feature extractor implementation
- Basic reward function focusing on distance to waypoints and speed
- Suitable for initial experimentation and baseline comparison
- Compatible with standard PPO hyperparameters
  
**Requirements**

- Python 3.8+
- PyTorch
- Stable Baselines3
- AirSim Python API
- NumPy
- OpenCV
- Gymnasium
