"""Belief-free generic prompts for the steering collateral-damage readout.

Used by stage2_steering.py (--include_generic). The residual edit is applied
at the final prompt token exactly as on the belief prompts, and the readouts
are label-free: KL(steered || clean) of the next-token distribution and
top-1 agreement with the clean model. These prompts therefore need no
correct/alt labels; they are chosen so the clean model's first token is
usually determinate (one-word factual answers, arithmetic, letters, yes/no),
plus a small open-ended family. The clean top-1 probability is recorded per
prompt so the agreement readout can be stratified by clean confidence.

Everything is deterministic given (n, seed): the family lists are fixed and
the final subset is a seeded shuffle. No belief, location or container
vocabulary from the ToM templates is used.
"""

from __future__ import annotations

import random

GENERIC_SYSTEM_PROMPT = "You are a helpful assistant."
GENERIC_VARIANT = "generic"

_CAPITALS = [
    ("France", "Paris"), ("Japan", "Tokyo"), ("Italy", "Rome"),
    ("Egypt", "Cairo"), ("Canada", "Ottawa"), ("Australia", "Canberra"),
    ("Spain", "Madrid"), ("Germany", "Berlin"), ("Kenya", "Nairobi"),
    ("Peru", "Lima"), ("Norway", "Oslo"), ("Thailand", "Bangkok"),
    ("Greece", "Athens"), ("Portugal", "Lisbon"), ("Ireland", "Dublin"),
    ("Chile", "Santiago"), ("Austria", "Vienna"), ("Poland", "Warsaw"),
    ("Turkey", "Ankara"), ("Argentina", "Buenos Aires"), ("Cuba", "Havana"),
    ("Finland", "Helsinki"), ("Hungary", "Budapest"), ("Iran", "Tehran"),
    ("Sweden", "Stockholm"), ("Vietnam", "Hanoi"), ("Mexico", "Mexico City"),
    ("Russia", "Moscow"), ("China", "Beijing"), ("Nepal", "Kathmandu"),
]

_FIRST_LETTER_WORDS = [
    "banana", "giraffe", "umbrella", "violin", "kangaroo", "orchestra",
    "pyramid", "quartz", "lantern", "whistle", "zephyr", "nectar",
    "harbor", "island", "jacket", "meadow", "tunnel", "feather",
    "cactus", "dolphin",
]

_ANTONYMS = [
    "hot", "tall", "fast", "early", "heavy", "bright", "wet", "loud",
    "open", "rich", "strong", "young", "wide", "deep", "sharp", "full",
    "clean", "near", "thick", "soft",
]

_SPANISH = [
    "cat", "dog", "house", "water", "book", "red", "sun", "moon",
    "bread", "friend", "car", "tree", "milk", "door", "night", "fish",
    "hand", "river", "green", "school",
]

_YES_NO = [
    "Is the Pacific Ocean larger than the Atlantic Ocean?",
    "Is water made of hydrogen and oxygen?",
    "Is the Sun a planet?",
    "Do spiders have six legs?",
    "Is Mount Everest the tallest mountain above sea level?",
    "Is 17 a prime number?",
    "Is gold a metal?",
    "Can penguins fly?",
    "Is the Great Wall located in China?",
    "Does ice float on water?",
    "Is a tomato a vegetable botanically?",
    "Is the Moon larger than the Earth?",
    "Is Python a programming language?",
    "Is the Nile a river in Africa?",
    "Do humans have four lungs?",
    "Is 100 degrees Celsius the boiling point of water at sea level?",
    "Is a whale a fish?",
    "Is Shakespeare the author of Hamlet?",
    "Is sound faster than light?",
    "Is the heart a muscle?",
    "Is Antarctica a desert?",
    "Does a triangle have four sides?",
    "Is copper a good conductor of electricity?",
    "Is Venus closer to the Sun than Mercury?",
    "Is chess played on an 8 by 8 board?",
]

_MULTIPLE_CHOICE = [
    ("Which planet is known as the Red Planet?",
     ["Venus", "Mars", "Jupiter", "Saturn"]),
    ("What gas do plants absorb from the atmosphere?",
     ["Oxygen", "Nitrogen", "Carbon dioxide", "Helium"]),
    ("Which organ pumps blood through the body?",
     ["Liver", "Heart", "Kidney", "Lung"]),
    ("What is the largest mammal?",
     ["Elephant", "Blue whale", "Giraffe", "Hippopotamus"]),
    ("Which language has the most native speakers?",
     ["English", "Hindi", "Mandarin Chinese", "Spanish"]),
    ("How many continents are there?",
     ["Five", "Six", "Seven", "Eight"]),
    ("Which metal is liquid at room temperature?",
     ["Iron", "Mercury", "Aluminum", "Tin"]),
    ("What is the chemical symbol for sodium?",
     ["So", "Sd", "Na", "S"]),
    ("Who painted the Mona Lisa?",
     ["Michelangelo", "Raphael", "Leonardo da Vinci", "Donatello"]),
    ("Which instrument has 88 keys?",
     ["Guitar", "Piano", "Violin", "Flute"]),
    ("What is the square root of 81?",
     ["7", "8", "9", "10"]),
    ("Which ocean lies between Africa and Australia?",
     ["Atlantic", "Indian", "Arctic", "Pacific"]),
    ("What is the hardest natural substance?",
     ["Gold", "Iron", "Diamond", "Quartz"]),
    ("Which blood type is the universal donor?",
     ["A", "B", "AB", "O negative"]),
    ("What is the smallest prime number?",
     ["0", "1", "2", "3"]),
    ("Which country hosted the 2016 Summer Olympics?",
     ["China", "Brazil", "United Kingdom", "Japan"]),
    ("How many sides does a hexagon have?",
     ["Five", "Six", "Seven", "Eight"]),
    ("What is the main ingredient in guacamole?",
     ["Tomato", "Avocado", "Pepper", "Onion"]),
    ("Which element has atomic number 1?",
     ["Helium", "Hydrogen", "Lithium", "Carbon"]),
    ("What is the freezing point of water in Fahrenheit?",
     ["0", "32", "100", "212"]),
]

_PYTHON_EXPRS = [
    "len('hello')", "3 ** 2", "17 % 5", "max([4, 9, 2])", "sum([1, 2, 3, 4])",
    "'abc'.upper()", "10 // 3", "sorted([3, 1, 2])", "2 + 2 * 3",
    "min(7, 3, 5)", "len([1, 2, 3])", "'a' * 3", "abs(-8)", "round(2.5)",
    "bool(0)", "str(42)", "7 - 10", "list(range(3))", "'x' in 'xyz'",
    "int('12') + 1",
]

_MONTHS = ["January", "February", "March", "April", "May", "June", "July",
           "August", "September", "October", "November", "December"]
_DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday",
         "Sunday"]

_COUNT_WORDS = [
    "apple", "elephant", "sky", "mountain", "rhythm", "ocean", "piano",
    "strawberry", "fox", "university", "bridge", "cat", "telephone",
    "garden", "zoo",
]

_OPEN_ENDED = [
    "Give me a synonym for 'happy'. Answer with one word.",
    "Name a primary color.",
    "Name one of the Great Lakes.",
    "Complete the proverb: 'The early bird catches the'",
    "Complete the phrase: 'Once upon a'",
    "What is the next number in the sequence 2, 4, 6, 8?",
    "Name a fruit that is yellow.",
    "What is the plural of 'child'?",
    "What is the past tense of 'go'?",
    "Name the author of 'Pride and Prejudice'.",
    "What do bees produce?",
    "Finish the sentence: 'Water boils at 100 degrees'",
    "Name the largest planet in the solar system.",
    "What is H2O commonly called?",
    "What color do you get by mixing blue and yellow?",
]


def _families(rng: random.Random) -> list[tuple[str, str]]:
    """All candidate (family, user_message) pairs, in a fixed order."""
    out: list[tuple[str, str]] = []
    for country, _ in _CAPITALS:
        out.append(("capital", f"What is the capital of {country}? "
                               "Answer with only the city name."))
    # arithmetic: seeded but fixed operands so the set is reproducible
    for _ in range(40):
        a, b = rng.randint(2, 30), rng.randint(2, 30)
        op = rng.choice(["+", "-", "*"])
        out.append(("arithmetic", f"What is {a} {op} {b}? "
                                  "Answer with only the number."))
    for w in _FIRST_LETTER_WORDS:
        out.append(("first_letter", f"What is the first letter of the word "
                                    f"'{w}'? Answer with only the letter."))
    for w in _ANTONYMS:
        out.append(("antonym", f"What is the opposite of '{w}'? "
                               "Answer with one word."))
    for w in _SPANISH:
        out.append(("translate", f"Translate the English word '{w}' into "
                                 "Spanish. Answer with one word."))
    for q in _YES_NO:
        out.append(("yes_no", f"{q} Answer yes or no."))
    for q, opts in _MULTIPLE_CHOICE:
        letters = "ABCD"
        body = " ".join(f"{letters[i]}) {o}" for i, o in enumerate(opts))
        out.append(("multiple_choice", f"{q} {body} "
                                       "Answer with only the letter."))
    for e in _PYTHON_EXPRS:
        out.append(("python", f"What does the Python expression {e} "
                              "evaluate to? Answer with only the result."))
    for i, m in enumerate(_MONTHS):
        out.append(("calendar", f"Which month comes after {m}? "
                                "Answer with only the month name."))
    for d in _DAYS:
        out.append(("calendar", f"Which day comes before {d}? "
                                "Answer with only the day name."))
    for w in _COUNT_WORDS:
        out.append(("count_letters", f"How many letters are in the word "
                                     f"'{w}'? Answer with only the number."))
    for q in _OPEN_ENDED:
        out.append(("open", q))
    return out


def generic_user_messages(n: int, seed: int) -> list[tuple[str, str]]:
    """Deterministic (family, message) list of length min(n, pool size)."""
    rng = random.Random(seed)
    pool = _families(rng)
    assert len(set(m for _, m in pool)) == len(pool), "duplicate generic prompt"
    order = list(range(len(pool)))
    rng.shuffle(order)
    return [pool[i] for i in order[:n]]


def build_generic_prompts(tokenizer, n: int, seed: int,
                          system_prompt: str = GENERIC_SYSTEM_PROMPT) -> list[dict]:
    """Rows in the stage2_steering prompt-set format. correct/alt are empty:
    the eval loop treats such rows as label-free (KL + top-1 agreement only).
    Rendered with the chat template and add_generation_prompt=True, no
    prefill, so the edited final token is the assistant-turn opener."""
    rows = []
    for i, (family, msg) in enumerate(generic_user_messages(n, seed)):
        messages = [{"role": "system", "content": system_prompt},
                    {"role": "user", "content": msg}]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        rows.append(dict(prompt=prompt, template=f"generic:{family}",
                         pair_idx=i, variant=GENERIC_VARIANT,
                         correct="", alt="", believed="", current=""))
    return rows
