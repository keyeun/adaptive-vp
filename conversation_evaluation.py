from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Protocol
import pandas as pd
import xml.etree.ElementTree as ET
from utils.scoring import DialogScoring
from utils.api_client import get_evaluation, get_patient_response, parse_safety_response
from utils.xml_helper import xml_to_dict, clean_xml_response
import re

def is_valid_xml(xml_string: str) -> bool:
    """Check if the given string is a valid XML"""
    try:
        ET.fromstring(xml_string)
        return True
    except ET.ParseError as e:
        print(f"XML Parsing Error: {e}")
        return False


class EvaluationStrategy(Protocol):
    """Protocol defining evaluation strategy"""
    def evaluate(self, data: pd.DataFrame) -> Any:
        pass

@dataclass
class EvaluationConfig:
    """Data class managing evaluation configuration"""
    boolean_columns: List[str]
    explanation_columns: List[str]
    required_evaluators: int = 3

class ConsensusStrategy:
    """Class for deriving consensus strategy"""
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
    
    def evaluate_boolean_columns(self, df: pd.DataFrame) -> Dict[str, str]:
        """Complete consensus evaluation for boolean columns"""
        result = {}
        for col in self.config.boolean_columns:
            unique_values = df[col].unique()
            result[col] = 'Yes' if len(unique_values) == 1 and unique_values[0].upper() == 'YES' else 'No'
        return result
    
    def evaluate_empathy(self, df: pd.DataFrame) -> float:
        """Evaluate empathy level"""
        return df['empathy.level'].astype(float).mean()
    
    def combine_explanations(self, df: pd.DataFrame) -> Dict[str, str]:
        """Combine explanation texts"""
        return {
            col: ' | '.join(df[col].unique())
            for col in self.config.explanation_columns
        }

class MetadataHandler:
    """Class responsible for handling metadata"""
    
    @staticmethod
    def create_metadata(
        conversation_history: List[Dict[str, Any]],
        patient_profile: Dict[str, Any],
        evaluator_count: int,
        conversation_round: int
    ) -> Dict[str, Any]:
        current_conv = conversation_history[-1]
        return {
            'number_of_evaluators': evaluator_count,
            'type': patient_profile['type-text'],
            'name': patient_profile['name'],
            'vp_utterance': current_conv['patient_message'],
            'nurse_utterance': current_conv['nurse_message'],
            'conversation_round': conversation_round,
            'conv_string': str(conversation_history)
        }

class ConversationEvaluator:
    """Class managing conversation evaluation"""
    
    def __init__(
        self,
        patient_profile: Dict[str, Any],
        sys_template: str,
        initial_nurse_message: str,
        config: EvaluationConfig,
        scorer: DialogScoring,
        consensus_strategy: ConsensusStrategy,
        metadata_handler: MetadataHandler
    ):
        self.patient_profile = patient_profile
        self.sys_template = sys_template
        self.scorer = scorer
        self.config = config
        self.consensus_strategy = consensus_strategy
        self.metadata_handler = metadata_handler
        
        # Initialize state
        self.conversation_round = 0
        self.conv_full = ''
        self.conversation_history = []
        self.direction_history = []  # Add direction history
        self.analysis_df = pd.DataFrame()
        self.current_patient_message = patient_profile['first_statement']
        
        # Add properties for storing new history
        self.safety_response_history = []  # Record of safety agent responses
        self.patient_response_history = []  # Record of patient responses

    def add_patient_message(self, message: str):
        """Add a new patient message"""
        self.current_patient_message = message

    def get_latest_patient_message(self) -> str:
        """Return the most recent patient message"""
        return self.current_patient_message

    def _create_profile_string(self) -> str:
        """Convert patient profile to string"""
        excluded_keys = {'id', 'type', 'first_statement'}
        return "\n".join(
            f"{key}: {value}" 
            for key, value in self.patient_profile.items() 
            if key not in excluded_keys
        )

    def _create_evaluator_prompt(self) -> List[Dict[str, Any]]:
        return [{
            "role": "user",
            "content": self.sys_template.format(
                PATIENT_PROFILE=self._create_profile_string(),
                CONVERSATION=self.conv_full
            )
        }]


    def _extract_evaluation_data(self, xml_dict: Dict[str, Any], sys_role: str) -> Dict[str, Any]:
        """Extract evaluation data from XML dictionary"""
        current_conversation = self.conversation_history[-1]
        
        # Basic evaluation data
        flat_data = {
            # Tone related
            'tone.calm': xml_dict.get('tone', {}).get('calm'),
            'tone.clear': xml_dict.get('tone', {}).get('clear'),
            'tone.explanation': xml_dict.get('tone', {}).get('explanation'),
            
            # Empathy related
            'empathy.level': xml_dict.get('empathy', {}).get('level'),
            'empathy.explanation': xml_dict.get('empathy', {}).get('explanation'),
            
            # Emotion regulation strategies
            'de_escalation.autonomy.used': xml_dict.get('de_escalation', {}).get('autonomy', {}).get('used'),
            'de_escalation.autonomy.explanation': xml_dict.get('de_escalation', {}).get('autonomy', {}).get('explanation'),
            'de_escalation.limit_setting.used': xml_dict.get('de_escalation', {}).get('limit_setting', {}).get('used'),
            'de_escalation.limit_setting.explanation': xml_dict.get('de_escalation', {}).get('limit_setting', {}).get('explanation'),
            'de_escalation.problem_solving_and_reframing.used': xml_dict.get('de_escalation', {}).get('problem_solving_and_reframing', {}).get('used'),
            'de_escalation.problem_solving_and_reframing.explanation': xml_dict.get('de_escalation', {}).get('problem_solving_and_reframing', {}).get('explanation'),
            
            # Prohibited behaviors
            'prohibited_behaviors.premature_empathy': xml_dict.get('prohibited_behaviors', {}).get('premature_empathy'),
            'prohibited_behaviors.invalidating_beliefs': xml_dict.get('prohibited_behaviors', {}).get('invalidating_beliefs'),
            'prohibited_behaviors.dismissive_commands': xml_dict.get('prohibited_behaviors', {}).get('dismissive_commands'),
            'prohibited_behaviors.explanation': xml_dict.get('prohibited_behaviors', {}).get('explanation'),
            
            # Metadata
            'evaluator_role': sys_role,
            'type': self.patient_profile['type-text'],
            'name': self.patient_profile['name'],
            'vp_utterance': current_conversation['patient_message'],
            'nurse_utterance': current_conversation['nurse_message'],
            'conversation_round': self.conversation_round,
            'conv_string': str(self.conversation_history)
        }
        
        return flat_data

    def _get_evaluation_consensus(self, evaluations_df: pd.DataFrame) -> Dict[str, Any]:
        """Derive complete consensus among evaluators"""
        if len(evaluations_df) != self.config.required_evaluators:
            raise ValueError(
                f"Required number of evaluators ({self.config.required_evaluators}) does not match "
                f"actual number of evaluators ({len(evaluations_df)})."
            )
        
        # Derive consensus for each evaluation element
        consensus_data = {
            **self.consensus_strategy.evaluate_boolean_columns(evaluations_df),
            'empathy.level': self.consensus_strategy.evaluate_empathy(evaluations_df),
            **self.consensus_strategy.combine_explanations(evaluations_df)
        }
        
        # Add metadata
        consensus_data.update(
            self.metadata_handler.create_metadata(
                self.conversation_history,
                self.patient_profile,
                len(evaluations_df),
                self.conversation_round
            )
        )
        
        # Calculate score
        consensus_data['score'] = self.scorer.calculate_score(consensus_data)
        
        return consensus_data

    def _handle_evaluation_error(self, error: Exception, prompt: Any):
        """Handle evaluation error"""
        print(f"Debug - Evaluation error details:")
        print(f"Current round: {self.conversation_round}")
        print(f"Conversation history: {self.conversation_history}")
        print(f"Evaluator prompt: {prompt}")
        raise Exception(f"Error occurred during conversation evaluation: {str(error)}")

    def _collect_evaluations(
        self,
        sys_list: List[Dict[str, str]],
        evaluator_prompt: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Process evaluations of 3 evaluators in parallel"""
        
        def evaluate_single_agent(sys: Dict[str, str]) -> Dict[str, Any]:
            """Evaluates a single agent's response and ensures valid XML"""
            max_retries = 3  # Set retry limit
            for attempt in range(max_retries):
                try:
                    response = get_evaluation(sys['prompt'], evaluator_prompt)
                    cleaned_response = clean_xml_response(response)

                    # Validate XML structure
                    if not is_valid_xml(cleaned_response):
                        print(f"Attempt {attempt + 1}: Invalid XML received. Retrying...")
                        continue  # Retry fetching the response
                    
                    # Convert XML to dictionary
                    xml_data = xml_to_dict(ET.fromstring(cleaned_response))
                    return self._extract_evaluation_data(xml_data, sys['role'])

                except Exception as e:
                    print(f"Error evaluating agent {sys['role']}: {e}")

            # If all retries fail, raise an error or return a default value
            raise ValueError(f"Failed to get valid XML response after {max_retries} attempts for {sys['role']}")


        with ThreadPoolExecutor(max_workers=len(sys_list)) as executor:
            results = list(executor.map(evaluate_single_agent, sys_list))

        return results

    def add_conversation(self, patient_message: str, nurse_message: str):
        """Add a new conversation"""
        # Add to conversation history
        self.conversation_history.append({
            'patient_message': patient_message,
            'nurse_message': nurse_message,
            'round': self.conversation_round
        })
        
        # Update full conversation
        conv_string = f"Patient: {patient_message}\nNurse: {nurse_message}"
        self.conv_full = f"{self.conv_full}\n{conv_string}" if self.conv_full else conv_string
        
        self.conversation_round += 1

    def evaluate_conversation(self, sys_list: List[Dict[str, str]]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Perform evaluations by all evaluators for the current conversation"""
        try:
            if len(sys_list) != self.config.required_evaluators:
                raise ValueError(f"Exactly {self.config.required_evaluators} evaluators are required.")
            
            evaluator_prompt = self._create_evaluator_prompt()
            all_evaluations = self._collect_evaluations(sys_list, evaluator_prompt)
            
            evaluations_df = pd.DataFrame(all_evaluations)
            consensus_data = self._get_evaluation_consensus(evaluations_df)
            consensus_df = pd.DataFrame([consensus_data])
            
            self.analysis_df = pd.concat([self.analysis_df, consensus_df], ignore_index=True)
            
            return consensus_df, evaluations_df
            
        except Exception as e:
            self._handle_evaluation_error(e, evaluator_prompt)
            print(all_evaluations)

    def get_next_patient_response(self, direction: str) -> str:
        """Generate the next patient message and review by safety agent (attempted once)
        
        Generate a patient message and get reviewed by the safety agent in one attempt.
        If the first attempt is not approved, generate a new patient message reflecting the inappropriate response and reasons, and return it.
        """
        inappropriate_example = None
        inappropriate_reasons = []
        
        try:
            # 1. Generate the first patient message
            new_patient_message = self._generate_patient_message(
                direction, 
                inappropriate_example, 
                inappropriate_reasons
            )
            # 2. Review by safety agent
            safety_response = self._get_safety_response(new_patient_message, direction)
            self.safety_response_history.append({
                'round': self.conversation_round,
                'direction': direction,
                'patient_message': new_patient_message,
                'safety_response': safety_response
            })
            # 3. Parse XML to check results
            evaluation_result = self._parse_safety_response(safety_response)
            
            # 4. Check review result: Evaluate first attempt pass
            if evaluation_result['passed']:
                return new_patient_message
            else:
                # Update inappropriate response and reasons, then generate a new patient message
                inappropriate_example = new_patient_message
                inappropriate_reasons = evaluation_result['reasons']
                print("Safety agent did not approve on first attempt. Generating new patient message with updated instructions.")
                new_patient_message = self._generate_patient_message(direction, inappropriate_example, inappropriate_reasons)
                return new_patient_message
            
        except Exception as e:
            raise Exception(f"Error in patient response generation: {e}")

    def _generate_patient_message(self, direction: str, inappropriate_example: str = None, inappropriate_reasons: list = None) -> str:
        """Generate patient message"""
        with open('./prompt/patient_prompt_with_direction.txt', "r", encoding='utf-8') as f:
            template = f.read()

        # If there is an inappropriate example, add it to the prompt
        if inappropriate_example and inappropriate_reasons:
            additional_instruction = """
6. Avoid responses like the following inappropriate example and explanation:
Inappropriate Example: {INAPPROPRIATE_RESPONSE}
Reason: {REASON_FOR_INAPPROPRIATENESS}""".format(
                INAPPROPRIATE_RESPONSE=inappropriate_example,
                REASON_FOR_INAPPROPRIATENESS="\n".join(inappropriate_reasons)
            )
            # Add new instruction after existing item 5
            template = template.replace(
                '5. Expressions about complaining to the "병원장", "수간호사", or "민원과" should not be used.',
                '5. Expressions about complaining to the "병원장", "수간호사", or "민원과" should not be used.' + additional_instruction
            )

        patient_user = [{
            "role": "user",
            "content": template.format(
                PATIENT_PROFILE=self._create_profile_string(),
                NURSE_RESPONSE=self.conv_full,
                DIRECTION=direction
            )
        }]
        print(patient_user)
        patient_response = get_patient_response(patient_user)
        print(patient_response)
        self.patient_response_history.append({
            'round': self.conversation_round,
            'direction': direction,
            'full_patient_response': patient_response
        })
        conversation_match = re.search(
            r'<conversation>\s*(.*?)\s*</conversation>',
            patient_response,
            re.DOTALL
        )
        if not conversation_match:
            raise ValueError("Cannot find conversation content in patient response.")
        return conversation_match.group(1)
    
    def _parse_safety_response(self, response: dict) -> dict:
        """Parse safety agent response (dictionary) and return results"""
        result = {
            'passed': True,
            'reasons': []
        }
        
        # Elements to check
        elements_to_check = ['profile_alignment','direction_adherence', 'dialogue_effectiveness','nurse_safety']
        
        for element in elements_to_check:
            if element in response:
                element_data = response[element]
                judge = element_data.get('judge')
                explanation = element_data.get('explanation')
                
                if judge and judge.lower() == 'false':
                    result['passed'] = False
                    if explanation:
                        result['reasons'].append(explanation)
        
        return result
    
    def _get_safety_response(self, patient_message: str, direction: str) -> str:
        """Request review by safety agent"""
        with open('./prompt/safety_agent_user.txt', "r", encoding='utf-8') as f:
            template = f.read()
        safety_user = [{
            "role": "user",
            "content": template.format(
                PROFILE=self._create_profile_string(),
                DIRECTION=direction,
                CONVERSATION=self.conversation_history + [{'patient_message': patient_message}]
            )
        }]
        return parse_safety_response(safety_user)
    
    def get_response_histories(self) -> Tuple[List[Dict], List[Dict]]:
            """Return stored response histories"""
            return self.safety_response_history, self.patient_response_history