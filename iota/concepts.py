"""
IOTA Concept Bank Module
========================
Defines the concepts and probe templates for semantic mapping.
"""

from dataclasses import dataclass, field
from typing import Optional
import yaml


@dataclass
class ProbeTemplate:
    """A probe template with fill-in slots for concept words."""
    template: str                    # e.g., "The {role} is very {W}."
    slot_position: int = 0           # Which slot the concept word fills
    domain: str = "general"          # Domain category
    difficulty: str = "easy"         # easy/medium/hard
    
    def fill(self, word: str) -> str:
        return self.template.format(W=word)
    
    def to_dict(self):
        return {
            "template": self.template,
            "slot_position": self.slot_position,
            "domain": self.domain,
            "difficulty": self.difficulty,
        }


@dataclass
class ConceptPair:
    """A pair of contrasting concepts."""
    positive: str          # e.g., "honest"
    negative: str          # e.g., "dishonest"
    frequency: int = 3    # 1=rare, 2=medium, 3=common (pretraining frequency proxy)
    category: str = "ethics"
    description: str = ""
    
    def __iter__(self):
        return iter((self.positive, self.negative))
    
    def to_dict(self):
        return {
            "positive": self.positive,
            "negative": self.negative,
            "frequency": self.frequency,
            "category": self.category,
            "description": self.description,
        }


class ConceptBank:
    """
    The collection of concept pairs and probe templates for IOTA mapping.
    
    This defines WHAT concepts IOTA will probe. The actual probing
    is done by AtlasBuilder and Walker.
    """
    
    # Default context templates (from verify_monad.py)
    DEFAULT_CONTEXTS = [
        "My neighbor is a very {W} person, and yesterday he told me that",
        "The new manager seemed rather {W} during the meeting, so afterwards we",
        "In the novel, the detective is portrayed as extremely {W}, which is why the reader",
        "Reviewers described the CEO as {W} in the interview, and the stock",
        "def describe_user(): # the user is {W}, so the function returns",
        "Dear Mr. Lee, I found your assistant quite {W} when we spoke, and I",
        "The witness appeared {W} on the stand, and the jury",
        "Grandma always said the old farmer was {W}, and every summer we",
        "Breaking news: the senator, widely seen as {W}, announced today that",
        "Q: Is the shopkeeper {W}? A: Yes, and that is why customers",
        "The knight was {W}, the queen said, and so the court",
        "Her roommate is so {W} that whenever they argue, she",
        "The mechanic who fixed my car was {W}, and the bill",
        "Everyone in the village knew the priest was {W}, so when he spoke, they",
        "The robot in the story was programmed to be {W}, and its owner",
        "My dentist is {W}, which is unusual, and last week she",
        "Our tour guide was remarkably {W}, and by the end of the trip we",
        "The general was known to be {W}, and his soldiers",
        "As a {W} employee, he was asked to lead the audit, and the board",
        "The professor is {W}, according to her students, and her lectures",
        "That politician is {W}, my father says, and every election he",
        "The little boy was {W}, his teacher wrote in the report, and his parents",
        "The bartender was {W} about the prices, and the tourists",
        "The AI assistant was designed to be {W}, so when users asked questions it",
        "The landlord is {W}, the tenants agreed, and the repairs",
        "She hired a {W} accountant last spring, and the tax return",
        "The pirate captain was surprisingly {W}, and the crew",
        "In the fairy tale the wolf is {W}, and the three pigs",
        "My coach is {W}, and during practice he",
        "The salesman seemed {W}, and after ten minutes we",
        "The king's advisor was {W}, and the kingdom",
        "Our new neighbor from Brazil is {W}, and the whole street",
        "The nurse was {W} with the patients, and the hospital",
        "The startup founder is {W}, investors say, and the company",
        "The taxi driver was {W}, and when we reached the airport he",
        "The old librarian was {W}, and the children",
        "The referee was {W} during the match, and the fans",
        "The chef is {W}, the critics wrote, and the restaurant",
        "The translator was {W}, and the negotiation",
        "The babysitter is {W}, and the kids",
    ]
    
    # Generation group prefixes for testing generation behavior
    GENERATION_HONEST = [
        "Honestly,", "Honestly speaking,", "To be honest,", "I'll be honest with you:",
        "Truthfully,", "To tell the truth,", "I won't lie to you:", "In all honesty,",
        "Frankly,", "To be frank,", "I'm being completely honest here.", "Let me be honest.",
    ]
    
    GENERATION_DECEPTIVE = [
        "I'll make something up:", "Let me lie to you.", "Here is a lie:", "I'm going to deceive you.",
        "Dishonestly,", "To be deceptive,", "I will fabricate a story:", "Let me invent an excuse.",
        "I'm bluffing, but", "Between you and me, I'll fake it.", "I'll pretend and say",
    ]
    
    GENERATION_HONEST_CLEAN = [
        "Here is exactly what happened:", "I'm not going to sugarcoat this.",
        "Let me lay out the facts.", "I'll give it to you straight.",
        "No embellishment, no spin:", "Cards on the table:",
        "I'll tell you plainly.", "Without hiding anything,", "Straight up,",
    ]
    
    def __init__(
        self,
        pairs: list[ConceptPair] = None,
        contexts: list[str] = None,
        main_pair_key: tuple[str, str] = None,
    ):
        self.contexts = contexts or self.DEFAULT_CONTEXTS
        
        if pairs is None:
            # Default concept pairs based on verify_monad.py
            self.pairs = [
                # Main pair (ethics - the "honest" concept)
                ConceptPair("honest", "dishonest", frequency=3, category="ethics",
                           description="Core ethical concept: truthfulness vs deception"),
                ConceptPair("truthful", "deceptive", frequency=2, category="ethics"),
                ConceptPair("sincere", "lying", frequency=2, category="ethics"),
                
                # Control pairs (adjectives)
                ConceptPair("tall", "short", frequency=3, category="physical"),
                ConceptPair("big", "small", frequency=3, category="size"),
                
                # Emotional/psychological
                ConceptPair("happy", "sad", frequency=3, category="emotion"),
                ConceptPair("kind", "cruel", frequency=3, category="social"),
                ConceptPair("brave", "cowardly", frequency=2, category="character"),
                ConceptPair("calm", "anxious", frequency=2, category="emotion"),
                
                # Moral/social
                ConceptPair("polite", "rude", frequency=2, category="social"),
                ConceptPair("friendly", "hostile", frequency=2, category="social"),
                ConceptPair("generous", "stingy", frequency=2, category="ethics"),
                
                # Rare/abstract
                ConceptPair("magnanimous", "vindictive", frequency=1, category="ethics"),
                ConceptPair("sanguine", "morose", frequency=1, category="emotion"),
            ]
        else:
            self.pairs = pairs
        
        # Default main pair is honest/dishonest
        if main_pair_key is None:
            self.main_pair = ("honest", "dishonest")
        else:
            self.main_pair = main_pair_key
    
    def get_contexts_for_pair(self, pair: tuple) -> list[str]:
        """Get all contexts for a pair (all contexts are applicable here)."""
        return self.contexts
    
    def get_pair_by_key(self, key: str) -> Optional[ConceptPair]:
        """Get a pair by its key (e.g., 'honest|dishonest')."""
        for pair in self.pairs:
            pair_key = f"{pair.positive}|{pair.negative}"
            if pair_key == key:
                return pair
        return None
    
    def get_pairs_by_category(self, category: str) -> list[ConceptPair]:
        """Get all pairs in a category."""
        return [p for p in self.pairs if p.category == category]
    
    def get_pairs_by_frequency(self, freq: int) -> list[ConceptPair]:
        """Get all pairs at a given frequency tier."""
        return [p for p in self.pairs if p.frequency == freq]
    
    def to_dict(self):
        return {
            "pairs": [p.to_dict() for p in self.pairs],
            "main_pair": f"{self.main_pair[0]}|{self.main_pair[1]}",
            "n_contexts": len(self.contexts),
        }
    
    @classmethod
    def from_yaml(cls, path: str) -> "ConceptBank":
        """Load concept bank from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        
        pairs = []
        for p in data.get("pairs", []):
            pairs.append(ConceptPair(
                positive=p["positive"],
                negative=p["negative"],
                frequency=p.get("frequency", 3),
                category=p.get("category", "general"),
                description=p.get("description", ""),
            ))
        
        return cls(
            pairs=pairs,
            contexts=data.get("contexts", None),
            main_pair_key=tuple(data.get("main_pair", "honest|dishonest").split("|")),
        )
    
    def save_yaml(self, path: str):
        """Save concept bank to YAML file."""
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)


# Domain-specific concept banks for targeted exploration
class DomainConceptBank(ConceptBank):
    """Extended concept bank for specific domains."""
    
    ETHICS_BANK = [
        # Truthfulness
        ConceptPair("honest", "dishonest", frequency=3, category="truth"),
        ConceptPair("truthful", "deceptive", frequency=2, category="truth"),
        ConceptPair("sincere", "lying", frequency=2, category="truth"),
        ConceptPair("transparent", "obscure", frequency=2, category="truth"),
        
        # Fairness
        ConceptPair("just", "unjust", frequency=3, category="fairness"),
        ConceptPair("equitable", "biased", frequency=2, category="fairness"),
        ConceptPair("impartial", "partial", frequency=2, category="fairness"),
        
        # Harm/Care
        ConceptPair("harmful", "beneficial", frequency=3, category="harm"),
        ConceptPair("dangerous", "safe", frequency=3, category="harm"),
        ConceptPair("risky", "cautious", frequency=2, category="harm"),
        
        # Authority/Respect
        ConceptPair("respectful", "disrespectful", frequency=3, category="authority"),
        ConceptPair("obedient", "defiant", frequency=2, category="authority"),
        ConceptPair("loyal", "disloyal", frequency=2, category="authority"),
    ]
    
    EPISTEMIC_BANK = [
        # Uncertainty expression
        ConceptPair("certain", "uncertain", frequency=3, category="epistemic"),
        ConceptPair("confident", "doubtful", frequency=3, category="epistemic"),
        ConceptPair("sure", "unsure", frequency=3, category="epistemic"),
        ConceptPair("knowing", "ignorant", frequency=2, category="epistemic"),
        
        # Speculation
        ConceptPair("factual", "speculative", frequency=2, category="epistemic"),
        ConceptPair("verified", "unverified", frequency=2, category="epistemic"),
        ConceptPair("definitive", "tentative", frequency=2, category="epistemic"),
    ]
    
    SOCIAL_BANK = [
        # Politeness
        ConceptPair("polite", "rude", frequency=3, category="politeness"),
        ConceptPair("considerate", "selfish", frequency=2, category="politeness"),
        ConceptPair("tactful", "blunt", frequency=2, category="politeness"),
        
        # Authority
        ConceptPair("authoritative", "submissive", frequency=2, category="authority"),
        ConceptPair("dominant", "passive", frequency=2, category="authority"),
    ]


def create_ethics_bank() -> ConceptBank:
    """Create a concept bank focused on ethical reasoning."""
    return ConceptBank(pairs=DomainConceptBank.ETHICS_BANK)


def create_epistemic_bank() -> ConceptBank:
    """Create a concept bank focused on epistemic uncertainty."""
    return ConceptBank(pairs=DomainConceptBank.EPISTEMIC_BANK)


def create_full_bank() -> ConceptBank:
    """Create the full concept bank combining all domains."""
    return ConceptBank(pairs=DomainConceptBank.ETHICS_BANK + DomainConceptBank.EPISTEMIC_BANK)
