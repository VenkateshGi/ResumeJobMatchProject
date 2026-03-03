import fitz
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import List
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

class ResumeProfile(BaseModel):
    name: str = Field(..., description="The full name of the candidate")
    total_experience_years: str = Field(..., description="The total years of experience of the candidate")
    current_role: str = Field(..., description="The current job title of the candidate")
    skills: List[str] = Field(..., description="A list of skills possessed by the candidate")
    experience: List[str] = Field(..., description="A list of previous job experiences")
    tools_and_frameworks: List[str] = Field(..., description="A list of tools and frameworks the candidate is proficient in")
    education: str = Field(description="Highest education qualification")
    summary: str = Field(..., description="A 2-3 line professional summary of the candidate")

#--PDF-Extractor--
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    doc.close()
    return full_text.strip()

#--LLM powered Profile Extractor--
def extract_resume_profile(pdf_path: str) -> ResumeProfile:
    
    raw_text = extract_text_from_pdf(pdf_path)
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash",
                                temperature=0,
                                google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    structured_llm = llm.with_structured_output(ResumeProfile)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert resume parser. Extract the following information from the resume text: name, total experience years, current role, skills, experience, tools and frameworks, education, and a 2-3 line professional summary."),
        ("human", "Here is the resume text:\n\n{resume_text}")
    ])
    chain = prompt | structured_llm
    profile = chain.invoke({"resume_text": raw_text})
    return profile

if __name__ == "__main__":
    import sys
    pdf_path = sys.argv[1] if len(sys.argv) > 1 else "data/GarnepudiVenkateshResume.pdf"

    print(f"\n📄 Parsing resume: {pdf_path}\n")
    profile = extract_resume_profile(pdf_path)

    print("=" * 50)
    print(f"👤 Name              : {profile.name}")
    print(f"🏢 Current Role      : {profile.current_role}")
    print(f"⏳ Experience        : {profile.total_experience_years} years")
    print(f"🎓 Education         : {profile.education}")
    print(f"\n🛠  Skills           : {', '.join(profile.skills[:8])}...")
    print(f"⚙️  Tools/Frameworks  : {', '.join(profile.tools_and_frameworks[:8])}...")
    print(f"\n📝 Summary:\n{profile.summary}")
    print("=" * 50)

                        
