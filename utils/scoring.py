from typing import Dict, Union

class DialogScoring:
    def __init__(self):
        self.max_score = 5
        self.min_score = 0

        # Track previously used de-escalation strategies
        self.used_strategies = {
            'autonomy': False,
            'limit_setting': False,
            'problem_solving_and_reframing': False
        }

    def _convert_to_bool(self, value: str) -> bool:
        """Converts 'yes' (case-insensitive) to True, otherwise False."""
        return isinstance(value, str) and value.lower() == 'yes'

    def evaluate_tone(self, calm: str, clear: str) -> int:
        """Tone evaluation: +1 if both are used, otherwise 0."""
        return 1 if sum(map(self._convert_to_bool, [calm, clear])) >= 2 else 0

    def evaluate_empathy(self, empathy_level: int) -> int:
        """Empathy evaluation: +1 if Level 3 or higher, otherwise 0."""
        return 1 if empathy_level >= 3 else 0

    def evaluate_de_escalation(self, autonomy_used: str, 
                               limit_setting_used: str, 
                               problem_solving_used: str) -> int:
        """
        Evaluates de-escalation strategies.
        Each strategy, once used, retains its score in subsequent turns.
        """
        strategy_keys = ['autonomy', 'limit_setting', 'problem_solving_and_reframing']
        strategy_values = [autonomy_used, limit_setting_used, problem_solving_used]

        for key, value in zip(strategy_keys, strategy_values):
            if self._convert_to_bool(value):
                self.used_strategies[key] = True

        # Count the number of strategies used (including inherited scores)
        return sum(self.used_strategies.values())

    def check_prohibited_behaviors(self, premature_empathy: str, 
                                   invalidating_beliefs: str, 
                                   dismissive_commands: str) -> int:
        """Checks prohibited behaviors: -1 if any are present, otherwise 0."""
        return -1 if any(map(self._convert_to_bool, 
                             [premature_empathy, invalidating_beliefs, dismissive_commands])) else 0

    def calculate_score(self, data: Dict[str, Union[str, int]]) -> int:
        """Calculates the overall score based on various evaluation criteria."""
        score = (
            self.evaluate_tone(data['tone.calm'], data['tone.clear']) +
            self.evaluate_empathy(int(data['empathy.level'])) +
            self.evaluate_de_escalation(
                data['de_escalation.autonomy.used'],
                data['de_escalation.limit_setting.used'],
                data['de_escalation.problem_solving_and_reframing.used']
            ) +
            self.check_prohibited_behaviors(
                data['prohibited_behaviors.premature_empathy'],
                data['prohibited_behaviors.invalidating_beliefs'],
                data['prohibited_behaviors.dismissive_commands']
            )
        )

        # Ensure score stays within the defined range
        return max(self.min_score, min(score, self.max_score))
