"""
Генератор снов на основе марковских цепей
Использует только стандартные библиотеки Python
"""

import random
import re
import argparse
from collections import defaultdict, Counter
import sys
import os


class MarkovChainDreamGenerator:
    """Генератор снов на основе марковских цепей"""

    def __init__(self, n=2):
        """
        Инициализация генератора

        Args:
            n: порядок цепи Маркова (количество слов для предсказания следующего)
        """
        self.n = n
        self.chain = defaultdict(Counter)
        self.starts = []  # Начальные слова/фразы для генерации
        self.raw_dreams = []  # Исходные описания сновидений

    def load_dreams_from_file(self, filename):
        """
        Загрузка описаний снов из файла

        Args:
            filename: путь к файлу с описанием снов (по одной строке на сон)
        """
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                self.raw_dreams = [line.strip() for line in f if line.strip()]
            print(f"Загружено {len(self.raw_dreams)} снов из {filename}")
            return True
        except FileNotFoundError:
            print(f"Ошибка: файл {filename} не найден")
            return False
        except Exception as e:
            print(f"Ошибка при загрузке файла: {e}")
            return False

    def preprocess_text(self, text):
        """
        Предобработка текста: очистка и токенизация

        Args:
            text: исходный текст описания сна

        Returns:
            список слов и знаков препинания
        """
        # Заменяем переносы строк на пробелы
        text = text.replace('\n', ' ').replace('\r', '')

        # Добавляем пробелы вокруг знаков препинания для лучшей токенизации
        text = re.sub(r'([.!?…,:;])', r' \1 ', text)
        text = re.sub(r'\s+', ' ', text)

        # Разбиваем на токены (слова и знаки препинания)
        tokens = text.strip().split()

        # Добавляем специальные маркеры начала и конца
        tokens = ['__START__'] * (self.n - 1) + tokens + ['__END__']

        return tokens

    def build_chain(self):
        """Построение марковской цепи из загруженных снов"""
        if not self.raw_dreams:
            print("Ошибка: нет загруженных снов")
            return False

        self.chain.clear()
        self.starts.clear()

        for dream in self.raw_dreams:
            if not dream:
                continue

            tokens = self.preprocess_text(dream)

            # Строим цепь
            for i in range(len(tokens) - self.n):
                key = tuple(tokens[i:i + self.n])
                next_word = tokens[i + self.n]
                self.chain[key][next_word] += 1

            # Сохраняем возможные начала предложений
            if len(tokens) > self.n:
                start_key = tuple(tokens[:self.n])
                if start_key[0] == '__START__' or (self.n > 1 and start_key[0] == '__START__' and start_key[1] != '__END__'):
                    self.starts.append(start_key)

        print(f"Построена марковская цепь порядка {self.n}")
        print(f"Количество состояний: {len(self.chain)}")
        print(f"Количество возможных начал: {len(self.starts)}")
        return True

    def generate_dream(self, min_words=5, max_words=100, seed_word=None):
        """
        Генерация нового сновидения

        Args:
            min_words: минимальное количество слов
            max_words: максимальное количество слов
            seed_word: начальное слово (если указано)

        Returns:
            сгенерированный текст сновидения
        """
        if not self.chain or not self.starts:
            return "Ошибка: сначала загрузите данные и постройте цепь"

        # Выбираем начальное состояние
        if seed_word and seed_word != '__START__':
            # Ищем состояние, начинающееся с seed_word
            possible_starts = [s for s in self.starts if s[0] == seed_word or
                               (self.n > 1 and s[-1] == seed_word)]
            if possible_starts:
                current = random.choice(possible_starts)
            else:
                # Если не нашли, пробуем найти в цепи
                for key in self.chain.keys():
                    if key[0] == seed_word:
                        current = key
                        break
                else:
                    current = random.choice(self.starts)
        else:
            current = random.choice(self.starts)

        # Генерируем последовательность
        result = list(current)

        # Убираем маркеры начала из результата
        while result and result[0] == '__START__':
            result.pop(0)

        # Основной цикл генерации
        while len(result) < max_words:
            # Получаем текущее состояние
            state = tuple(result[-self.n:]) if len(result) >= self.n else tuple(result)

            # Дополняем состояние до нужной длины, если нужно
            while len(state) < self.n:
                state = ('__START__',) + state

            # Проверяем, есть ли такое состояние в цепи
            if state not in self.chain:
                # Если нет, пробуем уменьшить порядок
                for i in range(1, self.n):
                    shorter_state = state[i:]
                    if shorter_state in self.chain:
                        state = shorter_state
                        break
                else:
                    break

            # Выбираем следующее слово с учётом весов
            next_words = self.chain[state]
            total = sum(next_words.values())
            r = random.randint(1, total)

            cumsum = 0
            next_word = None
            for word, count in next_words.items():
                cumsum += count
                if r <= cumsum:
                    next_word = word
                    break

            if next_word is None or next_word == '__END__':
                break

            result.append(next_word)

            # Проверяем минимальную длину
            if len(result) >= min_words and random.random() < 0.1:  # 10% шанс закончить
                # Проверяем, заканчивается ли на знак препинания
                if result[-1] in '.!?…':
                    break

        # Формируем текст
        text = ' '.join(result)

        # Убираем пробелы перед знаками препинания
        text = re.sub(r'\s+([.!?…,:;])', r'\1', text)

        # Делаем первую букву заглавной
        if text and text[0].isalpha():
            text = text[0].upper() + text[1:]

        return text

    def generate_multiple(self, count=5, **kwargs):
        """Генерация нескольких сновидений"""
        dreams = []
        for i in range(count):
            dream = self.generate_dream(**kwargs)
            dreams.append(dream)
            print(f"\n--- Сновидение {i+1} ---")
            print(dream)
        return dreams

    def get_stats(self):
        """Получение статистики о модели"""
        if not self.chain:
            return "Модель не построена"

        total_transitions = sum(sum(counts.values()) for counts in self.chain.values())
        unique_states = len(self.chain)
        unique_words = set()
        for state in self.chain:
            unique_words.update(state)
        for counts in self.chain.values():
            unique_words.update(counts.keys())

        stats = f"""
        Статистика модели:
        - Порядок цепи: {self.n}
        - Уникальных состояний: {unique_states}
        - Уникальных слов: {len(unique_words)}
        - Всего переходов: {total_transitions}
        - Возможных начал: {len(self.starts)}
        """
        return stats


def main():
    """Основная функция с консольным интерфейсом"""
    parser = argparse.ArgumentParser(
        description='Генератор снов на основе марковских цепей',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  python dreams_generator.py dreams.txt
  python dreams_generator.py dreams.txt --order 3 --count 10
  python dreams_generator.py dreams.txt --seed "Мила" --min-words 10
  python dreams_generator.py dreams.txt --interactive
        """
    )

    parser.add_argument('file', help='Файл с описанием снов (по одной строке на анекдот)')
    parser.add_argument('--order', '-n', type=int, default=2,
                        help='Порядок цепи Маркова (по умолчанию: 2)')
    parser.add_argument('--count', '-c', type=int, default=5,
                        help='Количество генерируемых снов (по умолчанию: 5)')
    parser.add_argument('--seed', '-s', type=str, default=None,
                        help='Начальное слово для генерации')
    parser.add_argument('--min-words', type=int, default=5,
                        help='Минимальное количество слов (по умолчанию: 5)')
    parser.add_argument('--max-words', type=int, default=100,
                        help='Максимальное количество слов (по умолчанию: 100)')
    parser.add_argument('--interactive', '-i', action='store_true',
                        help='Интерактивный режим')
    parser.add_argument('--stats', action='store_true',
                        help='Показать статистику модели')

    args = parser.parse_args()

    # Создаем генератор
    generator = MarkovChainDreamGenerator(n=args.order)

    # Загружаем данные
    if not generator.load_dreams_from_file(args.file):
        sys.exit(1)

    # Строим цепь
    if not generator.build_chain():
        sys.exit(1)

    if args.stats:
        print(generator.get_stats())

    if args.interactive:
        # Интерактивный режим
        print("\n" + "="*50)
        print("Интерактивный режим генерации cнов")
        print("Команды: 'q' - выход, 's' - статистика, 'n [число]' - кол-во снов")
        print("="*50)

        count = args.count
        while True:
            print(f"\nГенерирую {count} сновидений...")
            generator.generate_multiple(
                count=count,
                min_words=args.min_words,
                max_words=args.max_words,
                seed_word=args.seed
            )

            cmd = input("\n> ").strip().lower()
            if cmd == 'q':
                break
            elif cmd == 's':
                print(generator.get_stats())
            elif cmd.startswith('n '):
                try:
                    count = int(cmd.split()[1])
                    print(f"Установлено количество: {count}")
                except:
                    print("Неверный формат. Используйте: n 10")
    else:
        # Обычный режим
        print(f"\nГенерирую {args.count} сновидений...")
        generator.generate_multiple(
            count=args.count,
            min_words=args.min_words,
            max_words=args.max_words,
            seed_word=args.seed
        )


if __name__ == "__main__":
    main()