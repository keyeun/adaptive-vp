# Adaptive-VP - A Framework for LLM-Based Virtual Patients for Nurse Communication Training

## Overview

Adaptive-VP is a modular LLM-based framework designed to dynamically adapt virtual patient (VP) dialogue during nurse communication training. Traditional standardized patient simulations are costly and inflexible, while many VP systems fail to respond adaptively to trainees' communication skills. Adaptive-VP addresses these challenges by adjusting VP responses in real time based on the quality of trainee communication.

Our framework comprises several core components:
- **VP Case Development Pipeline:** Provides guidelines for creating clinically grounded yet customizable VP cases with detailed patient personas and contextual clinical settings.
- **Evaluation Module:** Utilizes a multi-agent evaluation process—grounded in best practices in nursing communication—to assess trainee responses.
- **Dynamic Adaptation Module:** Determines the VP's next dialogue direction based on evaluation outputs, such as escalating hostility or intentional uncooperativeness when required.
- **Dialogue Generation Module:** Generates contextually appropriate and realistic VP dialogue tailored to the current simulation state.
- **Safety Monitoring Module:** Oversees potentially distressing content to ensure a supportive learning environment without compromising training realism.

## Repository Structure

- **main.py:**  
  The entry point of the simulation that orchestrates the conversation flow, invoking evaluation and adaptation modules to generate the VP's responses.

- **conversation_evaluation.py:**  
  Implements the evaluation framework which aggregates assessments from multiple evaluation agents and computes consensus using predefined criteria.

- **utils/**
  - **data_loader.py:**  
    Loads configurations, patient profiles, system templates, and stage directions.
  - **scoring.py:**  
    Calculates the overall score based on tone, empathy, de-escalation strategies, and checks for prohibited behaviors.
  - **api_client.py:**  
    Handles API calls to the Anthropic LLM service for both evaluation and dialogue generation.
  - **xml_helper.py:**  
    Provides utilities for cleaning and parsing XML responses.

- **prompt/**  
  Contains prompt templates for both the evaluation module (e.g., `evaluation_prompt.txt`) and the patient dialogue generation (e.g., `patient_prompt_with_direction.txt`).

- **data/**  
  Includes JSON data files:
  - `patient_data_kor.json` – Contains patient profiles with clinical and communication details.
  - `stage_direction.json` – Prescribes dialogue adaptation stages based on trainee performance.

## Installation and Setup

1. **Clone the Repository:**  
   Clone the repository to your local machine.

2. **Install Dependencies:**  
   Ensure you have Python 3.7 or higher installed. Install required packages, for example:
   ```bash
   pip install pandas anthropic
   ```
   (Other dependencies should be listed or added to a `requirements.txt` file.)

3. **Configure API Keys:**  
   Set up your Anthropic API key as an environment variable according to your local setup.

4. **Run the Simulation:**  
   Execute the main script:
   ```bash
   python main.py
   ```
   You will be prompted to input nurse responses, and the system will simulate a full conversation session with adaptive VP dialogue.

   Currently, the simulation is set up to run automatically using the patient "이미경 (Mikyung Lee)". However, you can change which patient is used by modifying the value of the patient name variable in the main entry script (main.py). For example, you can replace:
patient_name = "이미경"
with the name of another patient from the data (e.g., "김영희", "박정호", etc.). This allows you to simulate conversations using different patient profiles.

## Usage

During a simulation session:
- A patient profile is loaded and the initial VP statement is provided.
- Nurse responses are recorded and evaluated by multiple virtual evaluators.
- Based on evaluation scores, the Dynamic Adaptation Module selects the next response direction, thereby adapting the VP's behavior (e.g., escalating hostility or displaying uncooperative traits) as needed.
- All conversation histories and scores are logged for further review and analysis.
