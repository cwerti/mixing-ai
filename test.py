import time
import unittest

from main import UserStatistics


# Импортируй свой класс (предполагается, что он находится в файле solution.py)
# from solution import UserStatistics


class TestUserStatistics(unittest.TestCase):

  def test_basic_logic(self):
    # Окно = 300 секунд (5 минут), лимит = 3 события
    stats = UserStatistics(window=300, limit=3)

    # Пользователь 1 делает 2 события
    stats.event(now=100, user_id=1)
    stats.event(now=150, user_id=1)
    self.assertEqual(stats.robot_count(now=200), 0)

    # Пользователь 1 делает 3-е событие (достигает лимита)
    stats.event(now=250, user_id=1)
    self.assertEqual(stats.robot_count(now=260), 1)

    # Пользователь 2 делает 3 события
    stats.event(now=200, user_id=2)
    stats.event(now=220, user_id=2)
    stats.event(now=240, user_id=2)
    self.assertEqual(stats.robot_count(now=260), 2)

  def test_sliding_window_expiration(self):
    # Окно = 100 секунд, лимит = 2 события
    stats = UserStatistics(window=100, limit=2)

    # Пользователь 1 совершает 2 события на отметках 50 и 80
    stats.event(now=50, user_id=1)
    stats.event(now=80, user_id=1)

    # В момент времени 120 первое событие (50) уже за пределами окна [20, 120],
    # но второе (80) внутри. Всего активных событий = 1 (< лимита 2).
    self.assertEqual(stats.robot_count(now=120), 0)

    # В момент времени 190 оба события (50 и 80) вышли за пределы окна [90, 190]
    self.assertEqual(stats.robot_count(now=190), 0)

  def test_performance_and_stress(self):
    # Проверка на производительность: много событий не должны вешать robotCount
    stats = UserStatistics(window=300, limit=1000)

    # Генерируем 50 000 событий для одного пользователя
    start_time = time.time()
    for i in range(50000):
      stats.event(now=i, user_id=42)

    # Вызов robotCount должен происходить мгновенно
    count = stats.robot_count(now=50000)
    duration = time.time() - start_time

    self.assertEqual(count, 1)
    # Проверяем, что тест отработал достаточно быстро (например, быстрее 1 секунды)
    self.assertLess(
        duration, 1.0, "Метод работает слишком медленно, нарушена сложность!"
    )


if __name__ == "__main__":
  unittest.main()