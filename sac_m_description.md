**Навигация дрона в AirSim с использованием алгоритма SAC**

**Обзор**

В этом вилле репозитория реализована система навигации дрона в AirSim с использованием алгоритма Soft Actor-Critic (SAC). SAC — это модельно-свободный алгоритм типа актор-критик с максимальной энтропией, который обеспечивает улучшенную эксплорацию и стабильность обучения.

**Возможности**

**Алгоритм Soft Actor-Critic (SAC):**

- Фреймворк максимальной энтропии длялучшей эксплорации
- Off-policy обновления для повыshoreй эффективности выборок
- Автоматическая настройка энтропии
  
**Пользовательская среда для дрона:**
  
- Интеграция с AirSim для реалистичной симуляции
- Динамическая генерация путевых точек
- Поддержка Обучение по куррикулуму
  
**Развинутые архитектуры:**

- Каскадная сеть актора с контроллерами положения и скорости
- Двойная сеть критика для более устойчивой оценки Q-значений

**Улучшения обучения:**

- Планирование скорости обучения
- Обрезка градиентов для стабильного обучения
- Комплексное формирование вознаграждений

**Требования**

- Python 3.8+
- PyTorch
- Gymnasium
- Python API AirSim
- NumPy
- OpenCV
___________________________________________________________________________________
**Drone Navigation in AirSim using SAC Algorithm**

**Overview**

This forked repository implements a drone navigation system in AirSim using the Soft Actor-Critic (SAC) algorithm. SAC is a model-free, off-policy actor-critic algorithm that uses a maximum entropy framework. This implementation features a custom SAC agent designed for autonomous drone waypoint navigation with continuous action spaces.

**Features**

**Soft Actor-Critic (SAC) Algorithm:**

- Maximum entropy framework for improved exploration
- Off-policy updates for more sample efficiency
- Automatic entropy tuning
- 
**Custom Drone Environment:**
  
- Integration with AirSim for realistic simulation
- Support for dynamic waypoint generation
- Curriculum learning support

**Advanced Architectures:**

- Cascaded actor network with position and velocity controllers
- Twin critic networks for more stable Q-value estimation
- Training Enhancements:
- Learning rate scheduling
- Gradient clipping for stable training
- Comprehensive reward shaping

**Requirements**

- Python 3.8+
- PyTorch
- Gymnasium
- AirSim Python API
- NumPy
- OpenCV (planned for future improvements)
