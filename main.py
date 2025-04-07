import json
import pandas as pd
from conversation_evaluation import ConversationEvaluator, EvaluationConfig, ConsensusStrategy, MetadataHandler
from utils.scoring import DialogScoring
from utils.data_loader import load_patient_profile, load_system_list, load_system_template, load_directions

def initialize_evaluator(patient_profile: dict) -> ConversationEvaluator:
    """Initialize the conversation evaluator with necessary components."""
    # Load system template and evaluators
    sys_template = load_system_template()
    initial_nurse_message = ""
    
    # Configure evaluation settings
    config = EvaluationConfig(
        boolean_columns=[
            'tone.calm', 'tone.clear',
            'de_escalation.autonomy.used',
            'de_escalation.limit_setting.used',
            'de_escalation.problem_solving_and_reframing.used',
            'prohibited_behaviors.premature_empathy',
            'prohibited_behaviors.invalidating_beliefs',
            'prohibited_behaviors.dismissive_commands'
        ],
        explanation_columns=[
            'tone.explanation',
            'empathy.explanation',
            'de_escalation.autonomy.explanation',
            'de_escalation.limit_setting.explanation',
            'de_escalation.problem_solving_and_reframing.explanation',
            'prohibited_behaviors.explanation'
        ],
        required_evaluators=3
    )
    
    return ConversationEvaluator(
        patient_profile=patient_profile,
        sys_template=sys_template,
        initial_nurse_message=initial_nurse_message,
        config=config,
        scorer=DialogScoring(),
        consensus_strategy=ConsensusStrategy(config),
        metadata_handler=MetadataHandler()
    )

def get_nurse_input() -> str:
    """Get nurse's response from console input."""
    return input("\nNurse's response: ")

def main():
    # Load patient profile
    conversation_total = pd.DataFrame()
    patient_name = "오상훈"  # You can modify this or make it dynamic
    patient_profile = load_patient_profile(patient_name)
    directions = load_directions()
    if not patient_profile:
        print(f"Patient profile not found for {patient_name}")
        return

    # Initialize evaluator
    evaluator = initialize_evaluator(patient_profile)
    sys_list = load_system_list()
    
    # Start conversation
    print(f"\nPatient: {patient_profile['name']}")
    print(f"Initial statement: {patient_profile['first_statement']}")
    
    current_patient_message = patient_profile['first_statement']
    
    for turn in range(5):
        print(f"\n=== Turn {turn + 1} ===")
        print(f"Patient: {current_patient_message}")
        
        # Get nurse's response
        nurse_response = get_nurse_input()
        
        # Add conversation to history
        evaluator.add_conversation(current_patient_message, nurse_response)
        
        # Evaluate conversation
        consensus_df, evaluations_df = evaluator.evaluate_conversation(sys_list)
        conversation_total = pd.concat([conversation_total,consensus_df]).reset_index(drop=True)
        # Get score from consensus
        score = consensus_df['score'].iloc[0]
        print(f"\nNurse response score: {score}")
        direction = directions[score]
        current_patient_message = evaluator.get_next_patient_response(direction)

    
    print("\nConversation ended.")
    conversation_total.to_csv("consensus_data_example.csv")

if __name__ == "__main__":
    main() 