# Управление дроном на базе метода SAC в AirSim

## Описание

В директории SAC реализована система навигации дрона в AirSim с использованием алгоритма Soft Actor-Critic (SAC). SAC — это model-free алгоритм типа актор-критик с максимальной энтропией, который обеспечивает улучшенные исследование и стабильность обучения.

## Достоинства

### Алгоритм Soft Actor-Critic (SAC):

- Фреймворк с максимальной энтропией для улучшения эффективности исследования
- Off-policy обновления для повышения эффективности выборок
- Автоматическая настройка энтропии
  
### Настраиваемая среда для дрона:
  
- Интеграция с AirSim для реалистичной симуляции
- Динамическая генерация путевых точек
- Поддержка Curriculum обучения
  
### Продвинутая архитектура:

- Каскадная сеть актора с регуляторами положения и скорости
- Двойная сеть критика для более устойчивой оценки Q-значений

### Улучшенное обучение:

- Задание скорости обучения
- Обрезка градиентов для стабильного обучения
- Комплексное формирование вознаграждений

## Пререквизиты

- Python 3.8+
- PyTorch
- Gymnasium
- Python API AirSim
- NumPy
- OpenCV
___________________________________________________________________________________
# Drone Navigation in AirSim using SAC Algorithm 

## Overview

Directory SAC contains the implementation of the drone navigation system in AirSim using the Soft Actor-Critic (SAC) algorithm. SAC is a model-free, off-policy actor-critic algorithm that uses a maximum entropy framework. This implementation features a custom SAC agent designed for autonomous drone waypoint navigation with continuous action spaces.

## Features

### Soft Actor-Critic (SAC) Algorithm:

- Maximum entropy framework for improved exploration
- Off-policy updates for more sample efficiency
- Automatic entropy tuning

### Custom Drone Environment:
  
- Integration with AirSim for realistic simulation
- Support for dynamic waypoint generation
- Curriculum learning support
### Advanced Architecture:

- Cascaded actor network with position and velocity controllers
- Twin critic networks for more stable Q-value estimation

### Training Enhancements:

- Learning rate scheduling
- Gradient clipping for stable training
- Comprehensive reward shaping

## Requirements

- Python 3.8+
- PyTorch
- Gymnasium
- AirSim Python API
- NumPy
- OpenCV (planned for future improvements)
