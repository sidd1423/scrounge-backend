"""
Shared ingredient vocabulary for Scrounge.

Both the per-tile GPT-4o-mini detection prompt and the OCR-to-vocabulary
resolver import from this single list, so a name can never drift between
the two paths (e.g. detection saying "cheddar" while OCR resolves to
"cheddar cheese" and the merge step treats them as different items).

Keep this list flat and lowercase. Add synonyms to INGREDIENT_ALIASES
rather than adding near-duplicate canonical entries.
"""

from __future__ import annotations

# Canonical ingredient names. Extend this over time as `raw_guess` values
# from unknown-classified items reveal gaps (see detection.py).
INGREDIENT_VOCAB: list[str] = [
    # fruit
    "apple", "banana", "orange", "lemon", "lime", "grapefruit", "grapes",
    "strawberries", "blueberries", "raspberries", "blackberries",
    "pineapple", "mango", "peach", "pear", "plum", "kiwi", "watermelon",
    "cantaloupe", "cherries", "pomegranate", "fig",

    # vegetables
    "tomato", "cherry tomatoes", "potato", "sweet potato", "onion",
    "green onion", "garlic", "ginger", "broccoli", "cauliflower",
    "spinach", "kale", "lettuce", "arugula", "cabbage", "brussels sprouts",
    "carrot", "celery", "cucumber", "bell pepper", "jalapeno", "chili pepper",
    "mushroom", "zucchini", "squash", "eggplant", "corn", "peas",
    "green beans", "asparagus", "beets", "radish", "avocado",

    # herbs
    "cilantro", "parsley", "basil", "mint", "dill", "rosemary", "thyme",
    "chives", "green onion tops",

    # dairy & eggs
    "milk", "milk substitute", "eggs", "butter", "margarine",
    "cheese", "cheddar cheese", "mozzarella", "parmesan", "feta",
    "cream cheese", "cottage cheese", "ricotta", "swiss cheese",
    "provolone", "goat cheese", "string cheese", "shredded cheese blend",
    "yogurt", "greek yogurt", "sour cream", "heavy cream", "half and half",
    "whipped cream", "ice cream",

    # proteins / meat / seafood
    "chicken", "chicken breast", "chicken thighs", "ground chicken",
    "beef", "ground beef", "steak", "pork", "pork chops", "bacon",
    "sausage", "hot dogs", "ham", "deli meat", "turkey", "ground turkey",
    "salmon", "shrimp", "tuna", "tilapia", "fish", "tofu", "tempeh",
    "seitan", "veggie burger",

    # bread & grains / pantry staples
    "bread", "tortillas", "bagels", "english muffins", "pita bread",
    "rice", "pasta", "noodles", "quinoa", "oats", "cereal", "crackers",
    "flour", "sugar", "brown sugar",

    # legumes / canned / jarred
    "beans", "black beans", "chickpeas", "lentils", "canned tomatoes",
    "tomato sauce", "tomato paste", "broth", "soup",

    # condiments & sauces
    "ketchup", "mustard", "mayonnaise", "soy sauce", "hot sauce",
    "bbq sauce", "salad dressing", "salsa", "hummus", "pesto",
    "pickles", "olives", "capers", "relish", "vinegar", "worcestershire sauce",
    "sriracha", "teriyaki sauce", "ranch dressing",

    # spreads / breakfast
    "jam", "jelly", "peanut butter", "almond butter", "honey",
    "maple syrup", "nutella",

    # beverages
    "orange juice", "apple juice", "juice", "soda", "sparkling water",
    "beer", "wine", "iced tea", "lemonade", "energy drink", "coconut water",

    # frozen
    "frozen vegetables", "frozen fruit", "frozen pizza", "frozen meal",
    "ice cubes",

    # baking / oils
    "olive oil", "vegetable oil", "cooking spray", "baking soda",
    "baking powder", "vanilla extract",

    # leftovers / prepared (generic buckets — genuinely hard to name specifically)
    "leftovers", "prepared meal", "baked goods",
]

# Map common OCR / colloquial variants -> canonical vocab entry.
# Lowercase keys; resolver should lowercase OCR text before lookup.
INGREDIENT_ALIASES: dict[str, str] = {
    # cheese
    "shredded cheddar": "cheddar cheese",
    "sharp cheddar": "cheddar cheese",
    "mild cheddar": "cheddar cheese",
    "block of cheese": "cheese",
    "cheese block": "cheese",
    "sliced cheese": "cheese",
    "cheese slices": "cheese",
    "shredded cheese": "shredded cheese blend",
    "mexican blend cheese": "shredded cheese blend",
    "italian blend cheese": "shredded cheese blend",
    "string cheeses": "string cheese",
    "mozzarella cheese": "mozzarella",
    "parmesan cheese": "parmesan",
    "shaved parmesan": "parmesan",
    "grated parmesan": "parmesan",

    # milk / dairy
    "whole milk": "milk",
    "2% milk": "milk",
    "1% milk": "milk",
    "skim milk": "milk",
    "almond milk": "milk substitute",
    "oat milk": "milk substitute",
    "soy milk": "milk substitute",
    "coconut milk": "milk substitute",
    "lactose free milk": "milk",
    "half & half": "half and half",
    "whipping cream": "heavy cream",

    # eggs / butter
    "large eggs": "eggs",
    "free range eggs": "eggs",
    "organic eggs": "eggs",
    "egg carton": "eggs",
    "unsalted butter": "butter",
    "salted butter": "butter",
    "butter stick": "butter",
    "butter sticks": "butter",
    "plant butter": "margarine",

    # yogurt
    "greek yoghurt": "greek yogurt",
    "plain yogurt": "yogurt",
    "flavored yogurt": "yogurt",
    "yogurt cup": "yogurt",
    "yogurt cups": "yogurt",
    "dairy yogurt container": "yogurt",

    # produce
    "scallion": "green onion",
    "scallions": "green onion",
    "spring onion": "green onion",
    "spring onions": "green onion",
    "bell peppers": "bell pepper",
    "red pepper": "bell pepper",
    "green pepper": "bell pepper",
    "yellow pepper": "bell pepper",
    "orange pepper": "bell pepper",
    "romaine": "lettuce",
    "romaine lettuce": "lettuce",
    "iceberg lettuce": "lettuce",
    "mixed greens": "lettuce",
    "salad": "lettuce",
    "salad greens": "lettuce",
    "mixed salad greens": "lettuce",
    "bag of salad": "lettuce",
    "bagged greens": "lettuce",
    "bagged salad": "lettuce",
    "leafy greens": "lettuce",
    "spring mix": "lettuce",
    "baby spinach": "spinach",
    "grape tomatoes": "cherry tomatoes",
    "roma tomato": "tomato",
    "roma tomatoes": "tomato",
    "vine tomatoes": "tomato",
    "russet potato": "potato",
    "russet potatoes": "potato",
    "red potatoes": "potato",
    "yukon gold potatoes": "potato",
    "yellow onion": "onion",
    "red onion": "onion",
    "white onion": "onion",
    "garlic bulb": "garlic",
    "garlic cloves": "garlic",
    "ginger root": "ginger",
    "avocados": "avocado",
    "mixed berries": "blueberries",  # generic fallback if berry type unclear

    # proteins
    "chicken breasts": "chicken breast",
    "boneless chicken breast": "chicken breast",
    "chicken thigh": "chicken thighs",
    "ground turkey": "ground turkey",
    "turkey breast": "turkey",
    "deli turkey": "deli meat",
    "deli ham": "deli meat",
    "sliced turkey": "deli meat",
    "sliced ham": "deli meat",
    "lunch meat": "deli meat",
    "cold cuts": "deli meat",
    "hot dog": "hot dogs",
    "breakfast sausage": "sausage",
    "italian sausage": "sausage",
    "smoked salmon": "salmon",
    "canned tuna": "tuna",
    "tuna can": "tuna",

    # bread / grains
    "flour tortillas": "tortillas",
    "corn tortillas": "tortillas",
    "tortilla": "tortillas",
    "flatbread": "tortillas",
    "sliced bread": "bread",
    "loaf of bread": "bread",
    "sandwich bread": "bread",
    "whole wheat bread": "bread",
    "sourdough": "bread",
    "sourdough bread": "bread",
    "bagel": "bagels",
    "white rice": "rice",
    "brown rice": "rice",
    "cooked rice": "rice",
    "spaghetti": "pasta",
    "penne": "pasta",
    "macaroni": "pasta",

    # legumes / canned
    "canned beans": "beans",
    "kidney beans": "beans",
    "pinto beans": "beans",
    "garbanzo beans": "chickpeas",
    "canned chickpeas": "chickpeas",
    "canned tomato": "canned tomatoes",
    "crushed tomatoes": "canned tomatoes",
    "diced tomatoes": "canned tomatoes",
    "marinara sauce": "tomato sauce",
    "pasta sauce": "tomato sauce",
    "chicken broth": "broth",
    "vegetable broth": "broth",
    "beef broth": "broth",
    "stock": "broth",

    # condiments
    "mayo": "mayonnaise",
    "yellow mustard": "mustard",
    "dijon mustard": "mustard",
    "honey mustard": "mustard",
    "ranch": "ranch dressing",
    "italian dressing": "salad dressing",
    "caesar dressing": "salad dressing",
    "vinaigrette": "salad dressing",
    "pickle jar": "pickles",
    "sliced pickles": "pickles",
    "hot sauce bottle": "hot sauce",
    "chili sauce": "hot sauce",
    "sweet chili sauce": "hot sauce",
    "buffalo sauce": "hot sauce",
    "tabasco": "hot sauce",
    "sriracha bottle": "sriracha",
    "steak sauce": "worcestershire sauce",
    "cocktail sauce": "ketchup",  # closest vocab entry
    "tartar sauce": "mayonnaise",  # closest vocab entry
    "duck sauce": "bbq sauce",  # closest vocab entry
    "hoisin sauce": "soy sauce",  # closest vocab entry; revisit if it becomes common
    "fish sauce": "soy sauce",  # closest vocab entry
    "caper jar": "capers",
    "relish jar": "relish",
    "vinegar bottle": "vinegar",
    "apple cider vinegar": "vinegar",
    "balsamic vinegar": "vinegar",
    "white vinegar": "vinegar",

    # spreads
    "strawberry jam": "jam",
    "grape jelly": "jelly",
    "peanut butter jar": "peanut butter",

    # beverages
    "oj": "orange juice",
    "sparkling water can": "sparkling water",
    "seltzer": "sparkling water",
    "can of soda": "soda",
    "soft drink": "soda",
    "beer bottle": "beer",
    "beer can": "beer",
    "wine bottle": "wine",

    # frozen
    "frozen veggies": "frozen vegetables",
    "frozen mixed vegetables": "frozen vegetables",
    "frozen berries": "frozen fruit",
    "tv dinner": "frozen meal",

    # oils / baking
    "extra virgin olive oil": "olive oil",
    "evoo": "olive oil",
    "vegetable shortening": "vegetable oil",

    # corn
    "corn on the cob": "corn",
    "ear of corn": "corn",
    "corn cob": "corn",
    "canned corn": "corn",
    "frozen corn": "corn",
    "sweet corn": "corn",
    "corn kernels": "corn",

    # leftovers (generic tupperware/containers — genuinely hard to name)
    "leftover food": "leftovers",
    "tupperware container": "leftovers",
    "food storage container": "leftovers",
    "container of food": "leftovers",
}

UNKNOWN = "unknown"


def is_in_vocab(name: str) -> bool:
    return name.strip().lower() in INGREDIENT_VOCAB


def resolve_alias(name: str) -> str | None:
    """Return the canonical vocab name for a known alias, else None."""
    return INGREDIENT_ALIASES.get(name.strip().lower())


def match_text_to_vocab(text: str) -> str | None:
    """
    Fuzzy-match free text (a raw_guess or OCR string) against the vocab.
    Tries, in order: exact vocab match, exact alias match, then substring
    match against both vocab entries and alias keys. Returns None if
    nothing reasonable is found — callers should leave the item as
    "unknown" rather than guessing wildly.

    This is intentionally simple (no fuzzy/edit-distance matching) so it
    stays predictable. Add more entries to INGREDIENT_ALIASES rather than
    making this matching logic cleverer.
    """
    cleaned = text.strip().lower()
    if not cleaned:
        return None

    if cleaned in INGREDIENT_VOCAB:
        return cleaned

    direct_alias = INGREDIENT_ALIASES.get(cleaned)
    if direct_alias:
        return direct_alias

    # Substring match against vocab entries AND alias keys together, longest
    # candidate first. Checking longest-first prevents a generic term (e.g.
    # "bread") from matching inside a more specific one (e.g. "flatbread")
    # before the specific alias gets a chance.
    candidates = list(INGREDIENT_VOCAB) + list(INGREDIENT_ALIASES.keys())
    candidates.sort(key=len, reverse=True)
    for candidate in candidates:
        if candidate in cleaned:
            return INGREDIENT_ALIASES.get(candidate, candidate)

    return None


def vocab_prompt_block() -> str:
    """Render the vocabulary as a comma-separated block for prompts."""
    return ", ".join(INGREDIENT_VOCAB)