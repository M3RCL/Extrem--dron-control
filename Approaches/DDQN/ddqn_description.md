
# Управление дроном на базе метода DDQN в AirSim

Директория DDQN содержит реализацию алгоритма двойной глубокой Q-сети (DDQN) для автономной навигации дрона по ключевым точкам маршрута в симуляторе AirSim. Система использует данные с камеры глубины и полярные координаты для обучения дрона преодолевать препятствия и достигать заданных целей.

## Описание

Реализация основана на подходе Reinforcement Learning с архитектурой DDQN для обучения агента-дрона. Агент учится принимать решения на основе:
Глубинных изображений с фронтальной камеры (визуальное восприятие)
Полярных координат относительно цели (относительное позиционирование)
Система предназначена для работы с симулятором AirSim в среде Unreal Engine.

### Функционал и особенности

- Реализация двойной глубокой Q-сети для стабильного обучения
- Сочетание сверточной и полносвязной нейронных сетей
- Буфер воспроизведения опыта для улучшения эффективности выборок
- Стратегия эпсилон-жадного исследования с затуханием
- Мягкое обновление сети цели
- Дискретное пространство действий для упрощенного управления дроном
- Функция вознаграждения на основе достижения цели и избегания столкновений

## Требования

- Python 3.8+
- TensorFlow 2.x
- Python API AirSim
- NumPy
- Unreal Engine с настроенным плагином AirSim"
___________________________________________________________________________________
# DDQN Drone Navigation in AirSim

Directory DDQN contain implementation of the Double Deep Q-Network (DDQN) algorithm for autonomous drone waypoint navigation in the AirSim simulation environment. The system uses depth camera information and polar coordinates to enable the drone to learn how to navigate to specified waypoints while avoiding obstacles.

## Overview

The implementation uses a reinforcement learning approach with a DDQN architecture to train a drone agent. The agent learns to take appropriate actions based on:

- Depth images from a front-facing camera (visual perception)
- Polar coordinates to the goal (relative positioning)

The system is designed to work with the AirSim simulator running in an Unreal Engine environment.

## Features

- Double Deep Q-Network implementation for more stable learning
- Combined convolutional and dense neural network architecture
- Experience replay buffer to improve sample efficiency
- Epsilon-greedy exploration strategy with decay
- Soft target network updates
- Discrete action space for simplified drone control
- Reward function based on goal reaching and collision avoidance

## Requirements

- Python 3.8+
- pytorch
- TensorFlow 2.x
- OpenCV
- AirSim Python API
- NumPy
- Gymnasium
- Unreal Engine with AirSim plugin configured
