import re
import xml.etree.ElementTree as ET

def xml_to_dict(element: ET.Element) -> dict:
    """Convert an XML element into a dictionary."""
    result = {}

    # Process text content
    text = element.text.strip() if element.text and element.text.strip() else None
    if text:
        if not list(element):  # No child elements
            return text
        result["_text"] = text

    # Process child elements
    for child in element:
        child_data = xml_to_dict(child)
        if child.tag in result:
            if isinstance(result[child.tag], list):
                result[child.tag].append(child_data)
            else:
                result[child.tag] = [result[child.tag], child_data]
        else:
            result[child.tag] = child_data

    return result

def clean_xml_string(xml_string: str) -> str:
    """Remove unnecessary whitespace and ensure XML declaration exists."""
    xml_string = re.sub(r'\s+', ' ', xml_string).strip()
    
    # Ensure XML declaration exists
    if not xml_string.startswith("<?xml"):
        xml_string = f'<?xml version="1.0" encoding="UTF-8"?> {xml_string}'
    
    return xml_string

def clean_xml_response(xml_string: str) -> str:
    """Extract content inside <analysis> tag from XML response."""
    match = re.search(r'<analysis>.*?</analysis>', xml_string, re.DOTALL)
    if match:
        return match.group(0)
    raise ValueError("Valid <analysis> tag not found in XML response.")

def clean_xml_response_safety(xml_string: str) -> str:
    """Extract content inside <evaluation> tag from XML response."""
    match = re.search(r'<evaluation>.*?</evaluation>', xml_string, re.DOTALL)
    if match:
        return match.group(0)
    raise ValueError("Valid <evaluation> tag not found in XML response.")
