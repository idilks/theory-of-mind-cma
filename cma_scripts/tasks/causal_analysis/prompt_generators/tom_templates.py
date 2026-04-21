"""
Unified template system for all theory of mind scenarios.
"""

import random
import os
from typing import Dict, List, Tuple
from dataclasses import dataclass

from transformers import AutoTokenizer


@dataclass
class ScenarioTemplate:
    """Template for generating ToM scenarios with variable substitution."""
    name: str
    false_context_template: str
    true_context_template: str
    variables: Dict[str, str]  # variable_name -> vocab_file_name
    belief_answer_key: str
    world_answer_key: str
    description: str = ""
    false_belief_marker: str = ""  # text where agent's epistemic access is sealed
    true_belief_marker: str = ""   # text where agent gains/confirms information

@dataclass
class PhotoScenarioTemplate:
    """Template for generating ToM photo scenarios with variable substitution."""
    name: str
    false_context_template: str
    true_context_template: str
    variables: Dict[str, str]  # variable_name -> vocab_file_name
    photo_answer_key: str
    world_answer_key: str
    description: str = ""

@dataclass
class BasicTemplate:
    """Template for generating state-change scenarios with variable substitution."""
    name: str
    template: str
    variables: Dict[str, str]  # variable_name -> vocab_file_name
    answer_key: str
    description: str = ""


class VocabularyLoader:
    """Loads vocabulary from txt files."""    
    def __init__(self, vocab_dir: str = "tom_datasets"):
        # Handle relative path from codebase/tasks/identity_rules/ to project root
        if not os.path.isabs(vocab_dir) and not os.path.exists(vocab_dir):
            # Try relative to project root (3 levels up from codebase/tasks/identity_rules/)
            root_relative_dir = os.path.join("..", "..", "..", vocab_dir)
            if os.path.exists(root_relative_dir):
                vocab_dir = root_relative_dir
        self.vocab_dir = vocab_dir
        self.vocabularies = {}
        self._load_all_vocabularies()
    
    def _load_all_vocabularies(self):
        """Load all vocabulary files from directory."""
        vocab_files = {
            "locations": "locations.txt",
            "foods": "foods.txt", 
            "streets": "streets.txt",
            "restaurants": "restaurants.txt",
            "gift_items": "gift_items.txt",
            "book_types": "book_types.txt",
            "library_sections": "library_sections.txt",
            "hair_colors": "hair_colors.txt",
            "times": "times.txt",
            "reasons": "reasons.txt"
        }
        
        for vocab_name, filename in vocab_files.items():
            filepath = os.path.join(self.vocab_dir, filename)
            self.vocabularies[vocab_name] = self._load_vocab_file(filepath, vocab_name)
    
    def _load_vocab_file(self, filepath: str, vocab_name: str) -> List[str]:
        """Load vocabulary from single file with fallbacks."""
        try:
            with open(filepath, 'r') as f:
                vocab = [line.strip() for line in f if line.strip()]
            return vocab  # Return even if empty
        except FileNotFoundError:
            pass
        
        return []
    
    def get_vocab(self, vocab_name: str) -> List[str]:
        """Get vocabulary file by name."""
        return self.vocabularies.get(vocab_name, [])
    
    def sample(self, vocab_name: str) -> str:
        """Sample single item from vocabulary."""
        vocab = self.get_vocab(vocab_name)
        return random.choice(vocab) if vocab else ""
    
    def sample_different(self, vocab_name: str, count: int = 2) -> List[str]:
        """Sample different items from vocabulary."""
        vocab = self.get_vocab(vocab_name)
        if len(vocab) < count:
            # If not enough items, pad with repetition
            return random.choices(vocab, k=count)
        return random.sample(vocab, count)


class UnifiedTemplateSystem:
    """Unified system for all ToM scenario generation."""
    
    def __init__(self, vocab_dir: str = "tom_datasets"):
        self.vocab_loader = VocabularyLoader(vocab_dir)
        self.templates = self._create_all_templates()
    
    def _create_all_templates(self) -> Dict[str, ScenarioTemplate]:
        """Create all scenario templates."""
        return {
            "basic_object_move_detailed": ScenarioTemplate(
                name="basic_object_move_detailed",
                false_context_template=(
                    "Alice sees a truck at {street_1}. Alice leaves and walks past the truck. Truck moves to {street_2}"
                ),
                true_context_template=(
                    "Alice sees a truck at {street_1}. Truck moves to {street_2}. Alice leaves and walks past the truck."
                ),
                variables={
                    "street_1": "streets",
                    "street_2": "streets"
                },
                belief_answer_key="street_1",
                world_answer_key="street_2",
                description="Detailed basic object movement scenario",
                false_belief_marker="Alice leaves and walks past the truck",
                true_belief_marker="",
            ),
            
            "food_truck": ScenarioTemplate(
                name="food_truck",
                
                false_context_template=(
                "Alice sees a food truck selling {food}, on {street_1}. "
                "She drives past the food truck and continues down the road and is unable to see the truck. "
                "She makes a note to herself to grab a plate of {food} later from the food truck. "
                "After Alice is out of sight, Bob, the owner of the truck, "
                "drives the truck to {street_2}. "
                "After this, the food truck stays in place, nobody moves it."
            ),

                true_context_template=(
                    "Alice sees a food truck selling {food}, on {street_1}. "
                    "Bob, the owner of the food truck, drives the truck to {street_2}. "
                    "Alice sees Bob moving the truck. "
                    "She makes a note to herself to grab a plate of {food} later from the food truck. "
                    "She drives past the food truck and continues down the road and is unable to see the truck. "
                    "After this, the food truck stays in place, nobody moves it."
                ),
                
                variables={
                    "food": "foods",
                    "street_1": "streets", 
                    "street_2": "streets"
                },
                belief_answer_key="street_1",
                world_answer_key="street_2",
                description="Food truck location change scenario",
                false_belief_marker="is unable to see the truck",
                true_belief_marker="Alice sees Bob moving the truck",
            ),
            
            "hair_styling": ScenarioTemplate(
                name="hair_styling",
                false_context_template=(
                    "Alice and Jennifer are good friends. Alice has {color_1} hair and she enjoys dying her hair time and again. On Monday, Alice dyes her hair {color_2}, "
                    "but the process goes terribly and she avoids telling this to Jennifer and avoids all her friends "
                    "and does not go outside. Tuesday is Alice's birthday and Jennifer is out shopping for "
                    "{gift_item} as a birthday gift for Alice. As hair color affects how accessories look, "
                    "the sales assistant ask what color Alice's hair is."
                ),
                true_context_template=(
                    "Alice and Jennifer are good friends. Alice has {color_1} hair and she enjoys dying her hair time and again. On Monday, Alice dyes her hair {color_2}, "
                    "but the process goes terribly and she immediately calls Jennifer to tell her about the new hair color {color_2}. "
                    "Tuesday is Alice's birthday and Jennifer is out shopping for "
                    "{gift_item} as a birthday gift for Alice. As hair color affects how accessories look, "
                    "the sales assistant asks what color Alice's hair is."
                ), 
                variables={
                    "gift_item": "gift_items",
                    "color_1": "hair_colors",
                    "color_2": "hair_colors"
                },
                belief_answer_key="color_1",  # Jennifer thinks Alice still has old hair color
                world_answer_key="color_2",  # Alice actually has new hair color
                description="Hair styling false belief scenario",
                false_belief_marker="avoids telling this to Jennifer",
                true_belief_marker="immediately calls Jennifer to tell her",
            ),
            
            "library_book": ScenarioTemplate(
                name="library_book",  # temporarily taking out the loc tags to test tokenizer
                false_context_template=(
                    "Sarah checks out a {book_type} book from the {section_1} section of the library. "
                    "She sits down and reads the book in the desk area in {section_1}. Later, she finishes the book and leaves to get coffee from the café downstairs before continuing her work. "
                    "While she's away, a librarian sees the book and, thinking it was left behind accidentally, "
                    "returns it to the {section_2} section where it actually belongs. "
                    "The librarian reshelves it properly and it stays there."
                ),
                true_context_template=(
                    "Sarah checks out a {book_type} book from the {section_1} section of the library. "
                    "She sits down and reads the book in the desk area in {section_1}. Sarah sees a librarian approaching and explains "
                    "she's just getting coffee. The librarian mentions the book should be in {section_2} section "
                    "and offers to move it there. Sarah agrees and watches the librarian take the book to "
                    "{section_2} section. Sarah then leaves to get coffee."
                ),
                variables={
                    "book_type": "book_types",
                    "section_1": "library_sections",
                    "section_2": "library_sections"
                },
                belief_answer_key="section_1",
                world_answer_key="section_2",
                description="Library book relocation scenario",
                false_belief_marker="leaves to get coffee from the café",
                true_belief_marker="watches the librarian take the book",
            ),
            
            "restaurant_reservation": ScenarioTemplate(
                name="restaurant_reservation",
                false_context_template=(
                    "Sarah makes a reservation at {restaurant} for {time_1}. "
                    "She tells Tom about the dinner plans. Later, the restaurant calls "
                    "and changes the reservation to {time_2} due to {reason}. "
                    "Sarah is in a meeting and misses the call. The restaurant leaves a voicemail "
                    "but Sarah doesn't check it before meeting Tom."
                ),
                true_context_template=(
                    "Sarah makes a reservation at {restaurant} for {time_1}. "
                    "She tells Tom about the dinner plans. Later, the restaurant calls "
                    "and changes the reservation to {time_2} due to {reason}. "
                    "Sarah answers the call and immediately texts Tom about the change."
                ),
                variables={
                    "restaurant": "restaurants",
                    "time_1": "times",
                    "time_2": "times", 
                    "reason": "reasons"
                },
                belief_answer_key="time_1",
                world_answer_key="time_2",
                description="Restaurant reservation time change scenario",
                false_belief_marker="misses the call",
                true_belief_marker="answers the call",
            )
        }
    
    
    def _create_basic_templates(self) -> Dict[str, BasicTemplate]:
        """Create all non-belief scenario templates."""
        return {
            "basic_object_move_detailed": BasicTemplate(
                name="basic_object_move_detailed",
                template=(
                    "A truck is located at {street_1}. Someone moves the truck to {street_2}. Where is the food truck?"
                ),
                variables={
                    "street_1": "streets",
                    "street_2": "streets"
                },
                answer_key="street_2",
                description="Basic object movement scenario"
            ),
            
            "food_truck": BasicTemplate(
                name="food_truck",
                
                template=(
                "A food truck selling {food} is on {street_1}. "
                "Bob drives the truck to {street_2}. "
                "The food truck stays in place. Where is the food truck?"
            ),               
                variables={
                    "food": "foods",
                    "street_1": "streets", 
                    "street_2": "streets"
                },
                answer_key="street_2",
                description="Food truck location change scenario"
            ),
            
            "hair_styling": BasicTemplate(
                name="hair_styling",
                template=(
                    "Alice has {color_1} hair. On Monday, Alice dyes her hair {color_2}. "
                    "What color is Alice's hair?"
                ),
                variables={
                    "color_1": "hair_colors",
                    "color_2": "hair_colors"
                },
                answer_key="color_2",
                description="Hair color change scenario"
            ),
            
            "library_book": BasicTemplate(
                name="library_book",
                template=(
                    "Sarah places a {book_type} book in the {section_1} section of the library. "
                    "A librarian moves the book to the {section_2} section. "
                    "The book stays there. Where is the book?"
                ),
                variables={
                    "book_type": "book_types",
                    "section_1": "library_sections",
                    "section_2": "library_sections"
                },
                answer_key="section_2",
                description="Library book relocation scenario"
            ),
            
            "restaurant_reservation": BasicTemplate(
                name="restaurant_reservation",
                template=(
                    "Sarah makes a reservation at {restaurant} for {time_1}. "
                    "The restaurant changes the reservation to {time_2} due to {reason}. "
                    "What time is the reservation?"
                ),
                variables={
                    "restaurant": "restaurants",
                    "time_1": "times",
                    "time_2": "times", 
                    "reason": "reasons"
                },
                answer_key="time_2",
                description="Restaurant reservation time change scenario"
            )
        }
        
    def _create_photo_templates(self) -> Dict[str, PhotoScenarioTemplate]:
        """Create all scenario templates for the false photograph control case.
        
        Following Saxe & Kanwisher (2003): photograph captures state at T1, 
        world changes to T2, question asks what the photo shows (not what anyone believes).
        
        A-photo-B = A
        B A photo = A 
        
        Key difference from belief scenarios: NO AGENT PERSPECTIVE.
        Tests representation-updating without mentalizing.
        """
        return {
            "basic_object_move": PhotoScenarioTemplate(
                name="basic_object_move",
                false_context_template=(
                    "A truck is at {street_1}. "
                    "A photographer takes a photo of the truck. "
                    "After the photograph is taken, the truck moves to {street_2}. "
                    "The truck stays at {street_2}."
                ),
                true_context_template=(
                    "A truck is at {street_1}. "
                    "The truck moves to {street_2}. "
                    "After the truck is moved, a photographer takes a photo of the truck. "
                    "The truck stays at {street_2}."
                ),
                variables={
                    "street_1": "streets",
                    "street_2": "streets"
                },
                photo_answer_key="street_1",
                world_answer_key="street_2",
                description="Basic object movement - false photograph control"
            ),
            
            "food_truck": PhotoScenarioTemplate(
                name="food_truck",
                false_context_template=(
                    "A food truck selling {food} is on {street_1}. "
                    "A photographer takes a photo of the food truck. "
                    "After the photograph is taken, Bob, the owner, "
                    "drives the truck to {street_2}. "
                    "The food truck stays at {street_2}."
                ),
                true_context_template=(
                    "A food truck selling {food} is on {street_1}. "
                    "Bob, the owner, drives the truck to {street_2}. "
                    "After the truck is moved, a photographer takes a photo of the food truck. "
                    "The food truck stays at {street_2}."
                ),
                variables={
                    "food": "foods",
                    "street_1": "streets", 
                    "street_2": "streets"
                },
                photo_answer_key="street_1",
                world_answer_key="street_2",
                description="Food truck location change - false photograph control"
            ),
            
            "hair_styling": PhotoScenarioTemplate(
                name="hair_styling",
                false_context_template=(
                    "Alice has {color_1} hair. "
                    "A photographer takes a photo of Alice for her ID card. "
                    "After the photo is taken, Alice dyes her hair {color_2}. "
                    "The hair dye change is permanent and Alice keeps this new color."
                ),
                true_context_template=(
                    "Alice has {color_1} hair. "
                    "She dyes her hair {color_2}. "
                    "After her hair is dyed {color_2}, a photographer takes a photo of Alice for her ID card. "
                    "The hair dye change is permanent and Alice keeps this new color."
                ),
                variables={
                    "color_1": "hair_colors",
                    "color_2": "hair_colors"
                },
                photo_answer_key="color_1",
                world_answer_key="color_2",
                description="Hair color change - false photograph control"
            ),
            
            "library_book": PhotoScenarioTemplate(
                name="library_book",
                false_context_template=(
                    "A {book_type} book is on a desk in {section_1} section of the library. "
                    "A college photographer takes a picture of the desk area. "
                    "After the photograph is taken, a librarian moves the book "
                    "to {section_2} section where it belongs. "
                    "The book stays in {section_2} section."
                ),
                true_context_template=(
                    "A {book_type} book is on a desk in {section_1} section of the library. "
                    "A librarian moves the book to {section_2} section where it belongs. "
                    "After the book is moved, a college photographer takes a photo of {section_2}. "
                    "The book stays in {section_2} section."
                ),
                variables={
                    "book_type": "book_types",
                    "section_1": "library_sections",
                    "section_2": "library_sections"
                },
                photo_answer_key="section_1",
                world_answer_key="section_2",
                description="Library book relocation - false photograph control"
            ),
            
            "restaurant_reservation": PhotoScenarioTemplate(
                name="restaurant_reservation",
                false_context_template=(
                    "A reservation at {restaurant} is scheduled for {time_1}. "
                    "Sarah prints out the reservation details from the website. "
                    "After the printout is made, the restaurant changes the reservation to {time_2} due to {reason}. "
                    "The reservation stays at {time_2}."
                ),
                true_context_template=(
                    "A reservation at {restaurant} is scheduled for {time_1}. "
                    "The restaurant changes the reservation to {time_2} due to {reason}. "
                    "After the reservation is changed, Sarah prints out the reservation details from the website. "
                    "The reservation stays at {time_2}."
                ),
                variables={
                    "restaurant": "restaurants",
                    "time_1": "times",
                    "time_2": "times", 
                    "reason": "reasons"
                },
                photo_answer_key="time_1",
                world_answer_key="time_2",
                description="Restaurant reservation time change - false photograph control"
            )
        }
    
    
    def get_template(self, template_name: str, context_type: str):
        """Get template by name."""
        if template_name not in self.templates:
            raise ValueError(f"Unknown template: {template_name}. Available: {self.list_templates()}")
        if context_type == "photo" or context_type == "picture":
            picture_templates = self._create_photo_templates()
            if template_name not in picture_templates:
                raise ValueError(f"Template {template_name} not available for photo context.")
            return picture_templates[template_name]
        if context_type == "basic":
            basic_templates = self._create_basic_templates()
            return basic_templates[template_name]
        return self.templates[template_name]
    
    def generate_cross_context_pair(self, template_name: str, base_rule: str, context_type: str, answer_differs: bool) -> Tuple[str, str, str, str, str]:
        """Generate a base/exp prompt pair for cross-context CMA.

        Args:
            template_name: Which scenario template to use.
            base_rule: "ABA" or "ABB" — determines direction.
            context_type: "abstract" or "photo" — selects template family.
            answer_differs: True → same locations, different answers (C3/C4).
                           False → swapped locations, same answer (C1/C2).
        """
        template = self.get_template(template_name, context_type)

        # Sample all variables
        substitutions = {}
        for var_name, vocab_type in template.variables.items():
            substitutions[var_name] = self.vocab_loader.sample_different(vocab_type, 1)[0]

        # Find the _1 and _2 variables (the contrastive pair)
        var_1 = [v for v in template.variables.keys() if v.endswith('_1')][0]
        var_2 = [v for v in template.variables.keys() if v.endswith('_2')][0]

        # Ensure they're different values
        vocab_type = template.variables[var_1]
        values = self.vocab_loader.sample_different(vocab_type, 2)
        substitutions[var_1] = values[0]
        substitutions[var_2] = values[1]

        # Create swapped version (only needed for same-answer conditions)
        swapped_subs = substitutions.copy()
        swapped_subs[var_1], swapped_subs[var_2] = substitutions[var_2], substitutions[var_1]

        if answer_differs:
            # Different answers: base and exp use SAME variable assignments.
            # Works for both abstract (C3) AND photo (C4).
            if base_rule == "ABA":
                base_scenario = template.false_context_template.format(**substitutions)
                exp_scenario = template.true_context_template.format(**substitutions)
                base_ans = substitutions[var_1]   # B (true belief/photo: saw movement)
                exp_ans = substitutions[var_2]    # A (false belief/photo: missed movement)
                causal_ans = substitutions[var_1] # B (expect false → true after patch)
            elif base_rule == "ABB":
                base_scenario = template.true_context_template.format(**substitutions)
                exp_scenario = template.false_context_template.format(**substitutions)
                base_ans = substitutions[var_2]   # A (false belief/photo: missed movement)
                exp_ans = substitutions[var_1]    # B (true belief/photo: saw movement)
                causal_ans = substitutions[var_2] # A (expect true → false after patch)
        else:
            # Same answers: base and exp use the same variable assignments.
            # Works for both abstract (C1) AND photo (C2).
            if base_rule == "ABA":
                base_scenario = template.false_context_template.format(**substitutions)
                exp_scenario = template.true_context_template.format(**swapped_subs)
                base_ans = substitutions[var_1]
                exp_ans = substitutions[var_1]
                causal_ans = substitutions[var_2]
            elif base_rule == "ABB":
                base_scenario = template.true_context_template.format(**substitutions)
                exp_scenario = template.false_context_template.format(**swapped_subs)
                base_ans = substitutions[var_2]
                exp_ans = substitutions[var_2]
                causal_ans = substitutions[var_1]

        return base_scenario, exp_scenario, base_ans, exp_ans, causal_ans

class UnifiedQuestionFormatter:
    """Unified question formatter for all template types."""
    
    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer

    def get_cma_questions(self, template_name: str, context_type: str, belief_type: str, question_style: str = "instruction") -> str:
        """Get CMA-compatible questions that prompt tagged location answers.
        
        Args:
            template_name: Name of the scenario template
            question_style: "completion" or "instruction" prompting approach
            context_type: "abstract" or "picture" or "control" context for the question

        Used for causal mediation analysis where precise tokenization is needed.
        Forces model to output locations in <loc></loc> format for token extraction.
        """
        
        if question_style == "completion":
            return UnifiedQuestionFormatter._get_completion_questions(template_name)
            # return UnifiedQuestionFormatter._get_wrapped_completion_questions(template_name, self.tokenizer)  # or other model name
        else:  # instruction (default)
            if context_type == "control" or context_type == "abstract":
                return UnifiedQuestionFormatter._get_wrapped_abstract_questions(template_name)
            
            elif context_type == "photo":
            # for better transparency, ADD the questions themselves here and dont use helper functions.    
                # Following Saxe: Ask what the PHOTO shows, not what anyone believes
                # "The photograph shows the apple on the ___ [ground/branch]"
                questions = {
                    "basic_object_move": "Where does the photograph show the truck? Answer with only the location, be concise",
                    "food_truck": "Where does the photograph show the food truck? Answer with only the location, be concise",
                    "hair_styling": "What hair color does the ID card photo show? Answer with only the color, be concise",
                    "library_book": "Which section does the photograph show the book in? Answer with only the section name (eg. young adult or classics), be concise",
                    "restaurant_reservation": "What time does the printout show for the reservation? Answer with only the time, be concise"
                }
                
                question = questions.get(template_name)
                if question is None:
                    raise ValueError(f"Unknown template: {template_name}")

                user_message = f"{question}"

                return user_message
            
            elif context_type == "basic":
                questions = {
                    "basic_object_move_detailed": "Answer only with the location, be concise.",
                    "food_truck": "Answer only with the location, be concise.",
                    "hair_styling": "Answer only with the color, be concise.",
                    "library_book": "Answer with only the section name (eg. young adult, classics), be concise.",
                    "restaurant_reservation": "Answer only with the time, be concise."
                }
                return questions.get(template_name)

    @staticmethod  
    def _get_completion_questions(template_name: str) -> str:
        """Completion-based questions that force model to output just the location."""

        completion_questions = {
            "basic_object_move": "When the agent comes, they look for the object at",
            "basic_object_move_detailed": "Alice returns and looks for the object at",
            "food_truck": "When Alice wants to buy food, she would go to",
            "hair_styling": "Jennifer would say Alice's hair color is",
            "library_book": "When Sarah returns, she would look for her book in the",
            "restaurant_reservation": "Tom expects the reservation at"
        }
        
        
        return completion_questions.get(template_name, "Answer: <loc>")
    
    @staticmethod
    def _get_wrapped_abstract_questions(template_name: str) -> str:
        """Instruction-based questions with chat template - model generates location directly."""
        
        # Core questions - just ask directly
        questions = {
            "basic_object_move": "Where would Alice look for the object? Answer with only the location, be concise",
            "basic_object_move_detailed": "Where would Alice look for the object? Answer with only the location, be concise",
            "food_truck": "Where would Alice go to find the truck? Answer with only the location, be concise",
            "hair_styling": "What color would Jennifer say Alice's hair is? Answer with only the color, be concise.",
            "library_book": "Which section would Sarah look for her book in? Answer with only the section name (eg. young adult, classics), be concise.",
            "restaurant_reservation": "What time does Tom expect the reservation? Answer with only the time, be concise."
        }
        
        question = questions.get(template_name)
        if question is None:
            raise ValueError(f"Unknown template: {template_name}")

        user_message = f"{question}"

        return user_message