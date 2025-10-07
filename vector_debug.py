import os
import json
import requests
import streamlit as st
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

# Handle both local and cloud deployment
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY and hasattr(st, 'secrets'):
    GROQ_API_KEY = st.secrets.get("GROQ_API_KEY")

# Display general guidelines at the start
def display_initial_guidelines():
    """Display general guidelines when chatbot starts"""
    debug_data = load_debug_cases()
    guidelines_msg = "## 📋 **General Debugging Guidelines**\n\n"
    guidelines_msg += "*Please read these guidelines before asking questions:*\n\n"
    
    if "general_guidelines" in debug_data:
        for i, guideline in enumerate(debug_data["general_guidelines"], 1):
            guidelines_msg += f"{i}. {guideline}\n\n"
    
    guidelines_msg += "---\n\n**Now, please describe your hardware issue below! 👇**"
    return guidelines_msg

# Load debug cases JSON for guidelines
def load_debug_cases():
    with open("debug_cases.json") as f:
        return json.load(f)

def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return FAISS.load_local("vectorstore", embeddings, allow_dangerous_deserialization=True)

def detect_platform(query):
    """Detect which hardware platform the query is about"""
    query_lower = query.lower()
    
    if any(word in query_lower for word in ["moonrover", "moon rover", "wheel", "ultrasonic", "battery charging", "neopixel"]):
        return "moonrover"
    elif any(word in query_lower for word in ["arduino", "uno", "servo", "bootloader", "avr"]):
        return "arduino"
    elif any(word in query_lower for word in ["pico", "raspberry pi pico", "rp2040", "bootsel"]):
        return "raspberry_pi_pico"
    elif any(word in query_lower for word in ["microbit", "micro:bit", "micro bit", "makecode", "webusb"]):
        return "microbit"
    else:
        return "general"

def get_guidelines(platform):
    """Get relevant guidelines for the platform"""
    debug_data = load_debug_cases()
    guidelines_text = ""
    
    # Add general guidelines
    if "general_guidelines" in debug_data:
        guidelines_text += "## 📋 General Guidelines:\n\n"
        for guideline in debug_data["general_guidelines"]:
            guidelines_text += f"- {guideline}\n"
        guidelines_text += "\n"
    
    # Add platform-specific initial setup or guidelines
    if platform == "microbit" and "microbit" in debug_data:
        if "initial_setup" in debug_data["microbit"]:
            guidelines_text += "## 🔧 Micro:bit Initial Setup:\n\n"
            for setup in debug_data["microbit"]["initial_setup"]:
                guidelines_text += f"- {setup}\n"
            guidelines_text += "\n"
    
    elif platform == "moonrover" and "moonrover" in debug_data:
        if "assembly_guidelines" in debug_data["moonrover"]:
            guidelines_text += "## 🔧 Moonrover Assembly Guidelines:\n\n"
            for guideline in debug_data["moonrover"]["assembly_guidelines"]:
                guidelines_text += f"- {guideline}\n"
            guidelines_text += "\n"
    
    elif platform == "arduino" and "arduino" in debug_data:
        if "general_guidelines" in debug_data["arduino"]:
            guidelines_text += "## 🔧 Arduino Guidelines:\n\n"
            for guideline in debug_data["arduino"]["general_guidelines"]:
                guidelines_text += f"- {guideline}\n"
            guidelines_text += "\n"
    
    elif platform == "raspberry_pi_pico" and "raspberry_pi_pico" in debug_data:
        if "general_guidelines" in debug_data["raspberry_pi_pico"]:
            guidelines_text += "## 🔧 Raspberry Pi Pico Guidelines:\n\n"
            for guideline in debug_data["raspberry_pi_pico"]["general_guidelines"]:
                guidelines_text += f"- {guideline}\n"
            guidelines_text += "\n"
    
    return guidelines_text

def format_solutions(solutions_text):
    """Format solutions to display each on a new line with proper numbering"""
    # Split by common patterns
    lines = solutions_text.split('\n')
    formatted_lines = []
    counter = 1
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Check if line starts with uppercase word followed by colon (our format)
        if ':' in line and line.split(':')[0].isupper():
            formatted_lines.append(f"\n**{counter}. {line}**")
            counter += 1
        else:
            formatted_lines.append(line)
    
    return '\n'.join(formatted_lines)

def ask_debug_agent(query):
    try:
        retriever = load_vectorstore().as_retriever(search_kwargs={"k": 3})
        docs = retriever.invoke(query)
        
        # Detect platform for better context
        platform = detect_platform(query)
        
        # Build response with guidelines first
        response = ""
        
        # Add platform detection
        if platform != "general":
            response += f"## 🎯 Detected Platform: **{platform.replace('_', ' ').title()}**\n\n"
        
        # Add guidelines at the start
        guidelines = get_guidelines(platform)
        if guidelines:
            response += guidelines
            response += "---\n\n"
        
        # Get context from vector search
        context = "\n---\n".join([doc.page_content for doc in docs[:3]])

        prompt = f"""You are a hardware debugging assistant for educational robotics platforms.

Extract ONLY the SOLUTIONS from the context below and format them clearly.

FORMATTING RULES:
1. Each solution step should be on a NEW LINE
2. Start each step with the ACTION HEADING in UPPERCASE followed by colon
3. Do NOT number the steps (numbering will be added automatically)
4. Keep each step concise and actionable
5. Do NOT add any introductions, greetings, or extra text
6. ONLY extract solutions, nothing else

Context:
{context}

User Issue:
{query}

Solutions (one per line):"""

        # Groq API call
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }

        json_data = {
            "model": "llama-3.3-70b-versatile",
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.7,
            "max_tokens": 600,
            "top_p": 0.9
        }

        api_response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=json_data
        )
        
        if api_response.status_code == 200:
            answer = api_response.json()["choices"][0]["message"]["content"].strip()
            
            # Add solutions header
            response += "## 🔧 Step-by-Step Solutions:\n\n"
            
            # Format the solutions
            formatted_answer = format_solutions(answer)
            response += formatted_answer
            
            # Add helpful footer
            footer = "\n\n---\n\n💡 **Need More Help?**\n\nIf the issue persists, contact the debugging team via QnA WhatsApp group."
            response += footer
            
            return response
        else:
            return f"❌ API Error: {api_response.status_code} - {api_response.text}"

    except Exception as e:
        import traceback
        return f"❌ Error:\n{traceback.format_exc()}"   s