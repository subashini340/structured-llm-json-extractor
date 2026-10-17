import os
import json
from dotenv import load_dotenv
from google import genai
from pydantic import BaseModel, ValidationError

load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)


class ResumeDetails(BaseModel):
    name: str
    email: str
    skills: list[str]
    education: str


def extract_resume_details(resume_text: str):
    prompt = f"""
    Extract information from the resume below.

    Return ONLY valid JSON with these fields:
    name, email, skills, education.

    If information is missing, use an empty string or empty list.
    Do not invent information.

    Resume:
    {resume_text}
    """

    response = client.models.generate_content(
        model="gemini-3.1-flash-lite-preview",
        contents=prompt,
        config={
            "response_mime_type": "application/json",
            "response_schema": ResumeDetails,
        },
    )

    try:
        data = json.loads(response.text)
        validated_data = ResumeDetails.model_validate(data)
        return validated_data.model_dump()

    except (json.JSONDecodeError, ValidationError) as error:
        print("Output validation failed:", error)
        return None


sample_resume = """
Name: Anu Kumar
Email: anu@example.com
Education: B.Tech Computer Science
Skills: Python, Machine Learning, SQL
"""

result = extract_resume_details(sample_resume)

if result:
    print("\nExtracted Resume Details:")
    print(json.dumps(result, indent=4))