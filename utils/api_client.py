import anthropic
from utils.xml_helper import xml_to_dict, clean_xml_response_safety
import xml.etree.ElementTree as ET

client = anthropic.Anthropic(
    # defaults to os.environ.get("ANTHROPIC_API_KEY")
    api_key="YOUR_OWN_KEY"
)

def get_evaluation(role: str, messages: list) -> str:
    """
    API call for Evaluation Module
    """
    try:
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1000,
            temperature=0,
            system=role,
            messages=messages
        )
        return response.content[0].text
    except Exception as e:
        raise Exception(f"Error occurred during the API call: {str(e)}")

def get_patient_response(messages: list) -> str:
    """
    API call for Dialogue Generation Module
    """
    try:
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1000,
            temperature=0,
            system="Your role is to act as a patient with a specific profile, engaging in a challenging conversation with a nurse.",
            messages=messages
        )
        return response.content[0].text
    except Exception as e:
        raise Exception(f"Error occurred during the API call: {str(e)}")


def get_safety_module(messages: list) -> str:
    """
    Execute safety module
    """
    with open('./prompt/safety_agent_sys.txt', "r", encoding='utf-8') as f:
        sys_message = f.read()
    try:
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1000,
            temperature=0,
            system=sys_message,
            messages=messages
        )
        return response.content[0].text
    except Exception as e:
        raise Exception(f"Error occurred during the API call: {str(e)}")
    
def parse_safety_response(messages: list) -> dict:
    """
    Parse response of safety module
    """
    try:
        # Get response of safety module
        response = get_safety_module(messages=messages)
        print("Debug - safety agent response:", response)

        # XML 파싱
        xml_data = xml_to_dict(
            ET.fromstring(clean_xml_response_safety(response))
        )
        return xml_data
    except Exception as e:
        raise Exception(f"Error in safety agent evaluation: {e}")