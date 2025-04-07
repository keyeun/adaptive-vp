import json
import pandas as pd


def load_directions():
    with open('./data/stage_direction.json', 'r') as file:
        directions = json.load(file)
        return directions

def load_patient_profile(patient_name):
    with open('./data/patient_data_kor.json', 'r') as file:
        patient_data = json.load(file)
    patient_profile = None

    for patient in patient_data:
        if patient['name'] == patient_name:
            patient_profile = patient
            break
    return patient_profile


def load_system_list():
    sys_0 = '''
    You are a clinical psychologist with 15 years of experience, specializing in abnormal psychology within healthcare settings. Your expertise lies in understanding the complex psychological dynamics and behavioral patterns that emerge during medical interactions. You excel at identifying underlying psychological factors in communication breakdowns and understanding how mental health conditions impact patient-provider interactions.
    '''
    sys_1 = '''
    You are a nursing professor with 20 years of combined clinical and academic experience, specializing in healthcare communication. Your work bridges theoretical knowledge with practical application, allowing you to evaluate communication patterns through both research-based and practical lenses. You bring expertise in evidence-based communication practices and their implementation in clinical settings.
    '''
    sys_2 = '''
    You are a communication skills trainer with 10 years of experience in conflict mediation, including 8 years focused specifically on healthcare settings. You specialize in analyzing communication patterns and transforming challenges into opportunities for improvement. Your expertise combines practical conflict resolution techniques with a deep understanding of healthcare dynamics, making you particularly effective at developing solutions for communication breakdowns.
    '''
    return [
        {'role': 'Clinical Psychologist', 'prompt': sys_0},
        {'role': 'Nursing Professor', 'prompt': sys_1},
        {'role': 'Communication Skills Trainer', 'prompt': sys_2}
    ]

def load_system_template():
    system_lib_file = './prompt/evaluation_prompt.txt'
    with open(system_lib_file, "r") as f:
        return f.read()

