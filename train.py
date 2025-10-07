from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.docstore.document import Document
import json

# Load the updated debug cases JSON
with open("debug_cases.json") as f:
    raw = json.load(f)

docs = []

# Add general guidelines
guidelines_content = "General Debugging Guidelines:\n" + "\n".join(raw["general_guidelines"])
docs.append(Document(page_content=guidelines_content, metadata={"id": "general_guidelines", "platform": "general"}))

# Process Micro:bit cases
if "microbit" in raw:
    # Add micro:bit initial setup
    if "initial_setup" in raw["microbit"]:
        setup_content = "Micro:bit Initial Setup:\n" + "\n".join(raw["microbit"]["initial_setup"])
        docs.append(Document(page_content=setup_content, metadata={"id": "mb_initial_setup", "platform": "microbit"}))
    
    # Add micro:bit cases
    for case in raw["microbit"]["cases"]:
        content = f"Platform: Micro:bit\nTitle: {case['title']}\nSymptoms: {', '.join(case['symptoms'])}\nCauses: {', '.join(case['causes'])}\nSolutions: {', '.join(case['solutions'])}"
        if "resources" in case and case["resources"]:
            content += f"\nResources: {', '.join(case['resources'])}"
        docs.append(Document(page_content=content, metadata={"id": case["id"], "platform": "microbit"}))

# Process Moonrover cases
if "moonrover" in raw:
    # Add moonrover assembly guidelines
    if "assembly_guidelines" in raw["moonrover"]:
        assembly_content = "Moonrover Assembly Guidelines:\n" + "\n".join(raw["moonrover"]["assembly_guidelines"])
        docs.append(Document(page_content=assembly_content, metadata={"id": "mr_assembly_guidelines", "platform": "moonrover"}))
    
    # Add moonrover cases
    for case in raw["moonrover"]["cases"]:
        content = f"Platform: Moonrover Kit\nTitle: {case['title']}\nSymptoms: {', '.join(case['symptoms'])}\nCauses: {', '.join(case['causes'])}\nSolutions: {', '.join(case['solutions'])}"
        if "resources" in case and case["resources"]:
            content += f"\nResources: {', '.join(case['resources'])}"
        docs.append(Document(page_content=content, metadata={"id": case["id"], "platform": "moonrover"}))

# Process Arduino cases
if "arduino" in raw:
    # Add arduino general guidelines
    if "general_guidelines" in raw["arduino"]:
        arduino_guidelines = "Arduino General Guidelines:\n" + "\n".join(raw["arduino"]["general_guidelines"])
        docs.append(Document(page_content=arduino_guidelines, metadata={"id": "ard_general_guidelines", "platform": "arduino"}))
    
    # Add arduino cases
    for case in raw["arduino"]["cases"]:
        content = f"Platform: Arduino Uno\nTitle: {case['title']}\nSymptoms: {', '.join(case['symptoms'])}\nCauses: {', '.join(case['causes'])}\nSolutions: {', '.join(case['solutions'])}"
        docs.append(Document(page_content=content, metadata={"id": case["id"], "platform": "arduino"}))

# Process Raspberry Pi Pico cases
if "raspberry_pi_pico" in raw:
    # Add pico general guidelines
    if "general_guidelines" in raw["raspberry_pi_pico"]:
        pico_guidelines = "Raspberry Pi Pico General Guidelines:\n" + "\n".join(raw["raspberry_pi_pico"]["general_guidelines"])
        docs.append(Document(page_content=pico_guidelines, metadata={"id": "pico_general_guidelines", "platform": "raspberry_pi_pico"}))
    
    # Add pico cases
    for case in raw["raspberry_pi_pico"]["cases"]:
        content = f"Platform: Raspberry Pi Pico\nTitle: {case['title']}\nSymptoms: {', '.join(case['symptoms'])}\nCauses: {', '.join(case['causes'])}\nSolutions: {', '.join(case['solutions'])}"
        docs.append(Document(page_content=content, metadata={"id": case["id"], "platform": "raspberry_pi_pico"}))

# Add ticket process information
if "ticket_process" in raw:
    ticket_content = "Ticket Process:\n"
    if "case_1_resolved_during_session" in raw["ticket_process"]:
        ticket_content += "\nCase 1 - Resolved During Session:\n" + "\n".join(raw["ticket_process"]["case_1_resolved_during_session"])
    if "case_2_not_resolved_during_session" in raw["ticket_process"]:
        ticket_content += "\n\nCase 2 - Not Resolved During Session:\n" + "\n".join(raw["ticket_process"]["case_2_not_resolved_during_session"])
    docs.append(Document(page_content=ticket_content, metadata={"id": "ticket_process", "platform": "general"}))

# Add support team information
if "support_team" in raw:
    support_content = "Support Team Contacts:\n"
    if "us_timezone" in raw["support_team"]:
        support_content += "\nUS Time Zone Team:\n"
        for member in raw["support_team"]["us_timezone"]:
            support_content += f"- {member['name']} ({member['specialization']}): {member['email']}\n"
    if "uk_timezone" in raw["support_team"]:
        support_content += "\nUK Time Zone Team:\n"
        for member in raw["support_team"]["uk_timezone"]:
            support_content += f"- {member['name']} ({member['specialization']}): {member['email']}\n"
    docs.append(Document(page_content=support_content, metadata={"id": "support_team", "platform": "general"}))

# Create embeddings and save to vector store
print(f"Processing {len(docs)} documents...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = FAISS.from_documents(docs, embeddings)
db.save_local("vectorstore")
print("Vector store created and saved successfully!")
print(f"Total documents indexed: {len(docs)}")