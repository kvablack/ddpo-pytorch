from importlib import resources
import functools
import random
import inflect

IE = inflect.engine()
ASSETS_PATH = resources.files("ddpo_pytorch.assets")


@functools.cache
def load_lines(name):
    with ASSETS_PATH.joinpath(name).open() as f:
        return [line.strip() for line in f.readlines()]


def imagenet(low, high):
    return random.choice(load_lines("imagenet_classes.txt")[low:high]), {}


def imagenet_all():
    return imagenet(0, 1000)


def imagenet_animals():
    return imagenet(0, 398)


def imagenet_dogs():
    return imagenet(151, 269)


def nouns_activities(nouns_file, activities_file):
    nouns = load_lines(nouns_file)
    activities = load_lines(activities_file)
    return f"{IE.a(random.choice(nouns))} {random.choice(activities)}", {}


def counting(nouns_file, low, high):
    nouns = load_lines(nouns_file)
    number = IE.number_to_words(random.randint(low, high))
    noun = random.choice(nouns)
    plural_noun = IE.plural(noun)
    prompt = f"{number} {plural_noun}"
    metadata = {
        "questions": [
            f"How many {plural_noun} are there in this image?",
            f"What animal is in this image?",
        ],
        "answers": [
            number,
            noun,
        ],
    }
    return prompt, metadata
