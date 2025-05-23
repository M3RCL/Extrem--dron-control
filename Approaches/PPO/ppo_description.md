# Управление дроном на базе метода PPO в AirSim

## Описание

Директория PPO содержит две улучшенные реализации автономной навигации дрона в AirSim с использованием алгоритма Proximal Policy Optimization (PPO). Обе реализации используют данные с камеры шлубины и векторные признаки для навигации дрона через заданные путевые точки с учетом препятствий. Каждая реализация предлагает уникальные улучшения для повышения устойчивости обучения и эффективности навигации.

## Достоинства

### Общий функционал

- Подход на основе усиленного обучения с использованием алгоритма PPO
- Комбинированная архитектура CNN и MLP для обработки изображений от камеры глубины и векторных признаков
- Уникальные функции вознаграждения для повышения эффективности навигации и избегания препятствий
- Интеграция с симулятором AirSim
- Поддержка навигации через несколько ключевых маршрутных точек (сложные маршруты)
  
### ImprovedPPO.py

- Улучшенный экстрактор признаков с остаточными связями и нормализацией слоев
- Улучшенная форма вознаграждения с компонентами для гладкости, прогресса и исследования
- Углубленные сети стратегии и оценки
- Задание скорости обучения и продвинутая настройка гиперпараметров
- Дополнительный трекинг метрик для лучшего мониторинга обучения
  
### PPO_drone.py

- Простая реализация экстрактора признаков
- Базовая функция вознаграждения,фокусирующаяся на расстоянии до ключевых точек маршрута и скорости
- Подходит для начальных экспериментов и сравнительного анализа
- Совместима со стандартными гиперпараметрами PPO

## Пререквизиты
- Python 3.8+
- PyTorch
- Stable Baselines3
- Python API AirSim
- NumPy
- OpenCV
- Gymnasium

_________________________________________________________________________________________
# Enhanced Drone Navigation in AirSim using PPO

## Overview

Directory PPO contains two improved implementations of autonomous drone navigation in AirSim using the Proximal Policy Optimization (PPO) algorithm. Both implementations leverage depth camera data and vector features to navigate drones through specified waypoints while avoiding obstacles. Each implementation offers unique enhancements to improve learning stability and navigation performance.

## Features

### Common Features

- Reinforcement learning approach with PPO algorithm
- Combined CNN and MLP architecture for processing depth images and vector features
- Custom reward functions to encourage efficient navigation and obstacle avoidance
- Integration with AirSim simulation environment
- Support for multiple waypoints navigation

### ImprovedPPO.py

- Enhanced feature extractor with residual connections, layer normalization, feature fusion layers
- Improved reward shaping with components for smoothness, progress, and exploration
- Deeper policy and value networks
- Learning rate scheduling and advanced hyperparameter tuning
- Additional metrics tracking for better training monitoring
  
### PPO_drone.py

- Simpler feature extractor implementation
- Basic reward function focusing on distance to waypoints and speed
- Suitable for initial experimentation and baseline comparison
- Compatible with standard PPO hyperparameters
  
## Requirements

- Python 3.8+
- PyTorch
- Stable Baselines3
- AirSim Python API
- NumPy
- OpenCV
- Gymnasium
