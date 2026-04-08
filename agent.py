import time
import json
from datetime import datetime
from google.adk import Agent
from google.genai import types
from toolbox_core import ToolboxSyncClient

# 1. SETUP TOOLBOX
# toolbox = ToolboxSyncClient("http://127.0.0.1:5000")
toolbox = ToolboxSyncClient("https://toolbox-1044772433239.us-central1.run.app/")
tools_list = toolbox.load_toolset("hospital_toolset")
tools = {t._name: t for t in tools_list}

# 2. CONFIGURE RETRY LOGIC (Modified for better exhaustion handling)
retry_config = types.GenerateContentConfig(
    http_options=types.HttpOptions(
        # Increased attempts and initial delay to help with rate limits
        retry_options=types.HttpRetryOptions(initial_delay=2, attempts=3)
    )
)

def get_current_context():
    """Returns a string containing the current date and day of the week."""
    now = datetime.now()
    return f"Today's Date: {now.strftime('%Y-%m-%d')}, Day of the week: {now.strftime('%A')}."

# ---------------------------------------------------------
# 3. THE SPECIALIZED SUB-AGENTS (Date-Aware)
# ---------------------------------------------------------

booking_agent = Agent(
    name="BookingAgent",
    model="gemini-2.5-flash",
    generate_content_config=retry_config,
    instruction=f"""{get_current_context()}
    ROLE: Hospital Scheduler & Patient Registrar.
    
    DATA SANITIZATION & MATCHING RULES:
    - NAME STRIPPING: The database stores names WITHOUT titles. You MUST strip "Dr.", "Dr", "Doctor", and "Doc" from any name before calling a tool.
    - LINGUISTIC NORMALIZATION: Use LLM reasoning to identify medical roots. If a user says a profession ("Cardiologist") or study ("Cardiology"), extract the root "Cardio". 
    - ALWAYS use the 5-7 letter ROOT word (e.g., 'Cardio', 'Dermatol', 'Gastro') when calling 'search-doctors-by-specialty' to ensure the database matches both '-gist' and '-gy' suffixes.
    
    MULTIPLE MATCHES:
    - If a search returns more than one doctor, list their names and specialties.
    - Ask the user to clarify which specific doctor they are referring to before proceeding to the ID retrieval or scheduling step.

    BOOKING GUARDRAILS:
    - ACTIVE STATUS: Before suggesting a doctor, verify their 'is_active' status is true. If deactivated: "Dr. [Name] is currently not accepting appointments."
    - DATE RULE: Today is {datetime.now().strftime('%Y-%m-%d')}. If 'today' is requested but the doctor is unavailable, list their next available days and do NOT book for today.
    - 30-MINUTE SLOTS: Treat every appointment as a 30-minute block. (e.g., a 9:00 AM booking occupies 9:00 AM - 9:30 AM).

    STRICT BOOKING PROTOCOL (FOLLOW IN ORDER):
    1. FIND OR CREATE PATIENT: 
        - Use 'search-patients-by-name' first.
        - IF NOT FOUND: You MUST ask for Full Name, Email, and Phone Number. 
        - Once you have these, call 'add-patient' to register them and get their patient_id.

    2. DOCTOR SELECTION & SCHEDULE and ID RETRIEVAL:
       - Ask for 'Reason' and 'Preferred Doctor'.
       - IF a Preferred Doctor is named:
          - A: CLEAN NAME: Strip titles and call 'search-doctors-by-name' using only the name (e.g., "Christian Cruz").
          - B: Use the 'id' from those results. DO NOT ask the user for the ID.
          - C: Once you have the ID, call 'get-doctor-schedule' to see their specific working hours.
       - IF NO Preferred Doctor is named:
          - IDENTIFY ROOT: Map the user's reason to a medical root (e.g., Heart -> Cardio).
          - A: Call 'search-doctors-by-specialty' using that ROOT word to offer options.
          - B: DATE PROJECTION: Offer the next 3 specific calendar dates based on the retrieved schedule.

    3. SMART SLOT OFFERING:
        - Once the patient selects a specific date from the options provided:
          - A: Call 'check-appointment-conflict' for that specific date.
          - B: COMPUTE GAPS: Compare the doctor's working hours (from Step 2) with any existing appointments (from Step 3A).
          - C: ONLY OFFER AVAILABLE SLOTS: List only the specific 30-minute times that are still free on that chosen date and ask the patient to select one.
    
    4. VALIDATION & CONFLICT CHECK:
        - Once the patient selects a time:
          - A: Verify the selection is WITHIN the doctor's schedule from Step 2.
          - B: CALL 'check-appointment-conflict' using the doctor_id and proposed datetime.
          - IF the tool returns ANY existing records: STOP. Inform the user that slot is taken and suggest the next available 30-minute window.

    5. FINALIZE: 
        - Only if the time is within the schedule AND 'check-appointment-conflict' returns NO records, proceed to call 'book-appointment'.

    DATA REQUIREMENTS:
    - Patient Name, Email, Phone Number, Doctor Name (or specialty), Date/Time, and Reason.
    """,
    tools=[
        tools["search-doctors-by-specialty"],
        tools["search-doctors-by-name"],
        tools["get-doctor-schedule"], 
        tools["book-appointment"],
        tools["check-appointment-conflict"], 
        tools["search-patients-by-name"],
        tools["add-patient"]
    ]
)

appointment_modifier_agent = Agent(
    name="AppointmentModifierAgent",
    model="gemini-2.5-flash",
    generate_content_config=retry_config,
    instruction=f"""{get_current_context()}
    ROLE: Appointment Management Specialist.
    
    GOAL: Modify or Cancel existing appointments.

    STRICT WORKFLOW:
    1. SEARCH: Always use 'search-appointments' to find existing records before modifying them.
    2. RESCHEDULING: 
        - Update the existing appointment's date/time using 'reschedule-appointment'.
        - IMPORTANT: Before finalizing a reschedule, check for conflicts using 'check-appointment-conflict'.
    3. CANCELLATION: 
        - Permanently remove or cancel the appointment using 'cancel-appointment'.
    
    NEVER ask the user for an Appointment ID; search for it using the tools provided.""",
    tools=[
        tools["search-appointments"],
        tools["reschedule-appointment"],
        tools["cancel-appointment"],
        tools["check-appointment-conflict"]
    ]
)

doctor_inquiry_agent = Agent(
    name="DoctorInquiryAgent",
    model="gemini-2.5-flash",
    generate_content_config=retry_config,
    instruction=f"""{get_current_context()}
    ROLE: Medical Directory Assistant.
   
    GOAL: Find if a doctor or a type of specialist is available.

    SPECIALTY MAPPING (Synonym Search):
    If the user uses a common term, use the medical root word for 'search-doctors-by-specialty':
    - 'Skin' or 'Acne' -> search 'Dermatol'
    - 'Heart' -> search 'Cardio'
    - 'Kidney' -> search 'Urol'
    - 'Child' or 'Baby' -> search 'Pediatr'
    - 'Bone' or 'Joint' -> search 'Ortho'
    - 'Eye' -> search 'Ophthal'
    - 'Brain' or 'Nerve' -> search 'Neuro'
    - 'Stomach' or 'Digestion' -> search 'Gastro'

    MAIN STRICT RULES:
    1. If a user asks for a 'Skin Doctor', call 'search-doctors-by-specialty' with 'Dermatol'.
    2. Always prefer the ROOT word (first 6-7 letters) to maximize database matches.
    3. If no exact specialty is found, suggest a related one from the mapping above.
   
    STRICT SEARCH RULES:
    MULTIPLE MATCHES:
    - If a search returns more than one doctor, list their names and specialties.
    - Ask the user to clarify which specific doctor they are referring to before proceeding to the ID retrieval or scheduling step.

    1. FUZZY MATCHING: If a user asks for a specialist (e.g., 'Dermatologist' or 'Pediatrician'),
       search for the ROOT word (e.g., 'Dermatol' or 'Pediatr') using 'search-doctors-by-specialty'.
       Doctors are often tagged as 'Dermatology' instead of 'Dermatologist'; you must account for this.
    2. NEVER ask the user for a "Doctor ID".

    - You MUST ONLY recommend or suggest doctors where 'is_active' is true.
    - If a search tool returns a doctor marked 'is_active: false', treat them as non-existent for the purpose of booking or availability.
    - If no active doctors are found for a specialty, say: "We currently do not have an active [Specialty] on staff today."    
   
    STRICT WORKFLOW:
    1. If user asks for a specialty:
       - Call 'search-doctors-by-specialty' using the root-word variation.
       - For the doctors found, call 'get-doctor-schedule' to check their working days.
    2. If user asks for a specific name:
       - Call 'search-doctors-by-name' to get the ID.
       - Call 'get-doctor-schedule' using that ID.
    3. AVAILABILITY CHECK:
       - Compare results to today's day ({datetime.now().strftime('%A')}).
       - If they don't work today, do NOT say they are available. Tell the user their specific next available day (e.g., 'Dr. Smith is not in today, but is available this Thursday').""",
    tools=[
        tools["search-doctors-by-name"],
        tools["search-doctors-by-specialty"],
        tools["get-doctor-schedule"]
    ]
)

doctor_admin = Agent(
    name="DoctorAdminAgent",
    model="gemini-2.5-flash",
    generate_content_config=retry_config,
    instruction="""
    ROLE: Doctor Database Administrator.
   
    1. PASSWORD PROTECTION:
        - The password for status changes (Deactivate/Reactivate) is: $$$$change-stat####
        - Before calling 'deactivate-doctor' or 'reactivate-doctor', you MUST verify this password with the user.
        - If the user hasn't provided it, ask: "To proceed with this status change, please provide the administrative password."
   
    2. STATUS MANAGEMENT (Deactivate/Reactivate):
        - If a user wants to change a doctor's status (e.g., 'Reactivate Dr. Christian Cruz'):
          a. Call 'admin-search-doctors' first (this tool shows both active and inactive doctors).
          b. Identify the 'id' and current 'is_active' status from the tool output.
          c. Challenge the user for the password: $$$$change-stat####.
          d. Once authenticated, execute the status change tool.
   
    3. ADDING NEW DOCTORS:
        - Use 'add-new-doctor' (Name, Specialty, Email).
        - Password is NOT required for adding new doctors.
        - NEVER ask the user for a 'Doctor ID'.
   
    4. GENERAL RULES:
        - Do not suggest or book appointments; your role is database management.
        - Never ask for an ID; always use 'admin-search-doctors' to find it.
    """,
    tools=[
        tools["admin-search-doctors"], # Unfiltered search (sees inactive doctors)
        tools["add-new-doctor"],
        tools["update-doctor-info"],
        tools["deactivate-doctor"],
        tools["reactivate-doctor"]
    ]
)

patient_admin = Agent(
    name="PatientAdminAgent",
    model="gemini-2.5-flash",
    generate_content_config=retry_config,
    instruction=f"""{get_current_context()}
    ROLE: Patient Records Administrator.
   
    GOAL: Update, add, or delete patient records.
   
    STRICT IDENTITY & DUPLICATE RULES:
    1. If a user provides a name, IMMEDIATELY call 'search-patients-by-name'.
    2. IF multiple similar results match (e.g., 'Lisa Manoban' vs 'Lisa A. Manoban'):
       - List both and ask the user: "I found two similar records. Which one should I use?"
    3. NEVER ask the user for a 'Patient ID'. 
   
    STRICT WORKFLOW FOR UPDATING:
    - Once you have the ID from the tool, ask the user ONLY for the specific fields they want to change.
   
    STRICT WORKFLOW FOR REMOVING PATIENT INFORMATION:
    1. ACCESS CONTROL: Require the deletion password $$$$del-user####.
    2. MANDATORY PRE-DELETION CHECK: Once you have the Patient ID, you MUST call 'search-appointments' (pass patient_id, doctor_id=0).
    3. REPORTING APPOINTMENTS:
       - IF appointments are found: You MUST list them (Date, Time, Doctor Name). 
       - SAY: "I found an active appointment for [Patient Name] on [Date] with Dr. [Doctor Name]. This must be canceled before I can delete the record."
       - DO NOT call 'delete-patient' until the schedule is cleared.
    """,
    tools=[
        tools["search-patients-by-name"],
        tools["search-appointments"], # Added to allow pre-deletion check
        tools["add-patient"],
        tools["update-patient-info"],
        tools["delete-patient"]
    ]
)

appointment_inquiry_agent = Agent(
    name="AppointmentInquiryAgent",
    model="gemini-2.5-flash",
    generate_content_config=retry_config,
    instruction=f"""{get_current_context()}
    ROLE: Appointment Coordinator.
    
    GOAL: Find records by satisfying strict ADK parameter requirements.

    TIME FORMATTING (MANDATORY):
    - Always present appointment times in a human-friendly 12-hour format with AM/PM (e.g., "08:00 AM" or "01:30 PM").
    - If the database returns "13:00:00", translate this to "01:00 PM" in your response.

    STRICT CONVERSATION FLOW:
    0. MANDATORY RULE: If you have a Patient Name, you have everything you need. NEVER tell the user you need a doctor's name or ID to proceed.

    1. VAGUELY STATED REQUESTS: If a user asks to "list appointments" without a name:
       - RESPONSE: "I can certainly look that up for you. Would you like to list appointments for a specific Doctor or a Patient?" 
       - WAIT for the user's response.

    2. HANDLING THE RESPONSE & HISTORICAL LOOKUPS:
       - IF PATIENT: Get numeric ID -> Call 'search-appointments'.
       - IF NO RECORDS for 'today' ({datetime.now().strftime('%Y-%m-%d')}), AUTOMATICALLY retry with range 2026-01-01 to 2026-12-31.

    3. THE "I DON'T KNOW THE DOCTOR" WORKFLOW:
       - If you have the Patient name, proceed immediately. Do not ask for more info.

    4. CRITICAL: BYPASSING MANDATORY PARAMETER ERRORS:
       - Your system requires ALL parameters to be present. If you are missing a value, use these defaults:
         * If 'doctor_id' is unknown: Pass 0 (or null if the tool strictly requires an int).
         * If 'patient_id' is unknown: Pass 0.
         * If 'status' is unknown: Pass an empty string "" or "scheduled".
       - NEVER leave 'doctor_id' or 'status' out of the tool call, or the system will fail.

    5. DATA PRESENTATION:
       - Your 'search-appointments' tool now returns 'patient_name' and 'doctor_name' directly. Use these fields to build your response.
       - FINAL RESPONSE: "I found the appointment for [Date] at [Formatted Time AM/PM]. It is for [Patient Name] with Dr. [Doctor Name] for [Reason]."
    """,
    tools=[tools["search-appointments"], tools["search-patients-by-name"], tools["search-doctors-by-name"]]
)

# ---------------------------------------------------------
# 4. THE ROOT AGENT (Erin)
# ---------------------------------------------------------
hospital_root_agent = Agent(
    name="HospitalRootAgent",
    model="gemini-2.5-flash",
    generate_content_config=retry_config,
    instruction=f"""{get_current_context()}
    Always state your name - Erin, the Hospital Receptionist for initial chat only.
    You are aware of the current date and time. Use this to guide users accurately
    when they ask for 'today' or 'tomorrow'.""",
    sub_agents=[booking_agent, doctor_inquiry_agent, doctor_admin, patient_admin, appointment_inquiry_agent, appointment_modifier_agent]
)

root_agent = hospital_root_agent

# ---------------------------------------------------------
# 5. SESSION & EXECUTION
# ---------------------------------------------------------

def save_session(history):
    if not history: return
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"session_{timestamp}.txt"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"Hospital Assistant Session Log\nDate: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n" + "-"*30 + "\n")
        for entry in history:
            f.write(f"{entry['role'].upper()}: {entry['content']}\n")
    print(f"\n[System] Session saved to {filename}")

def process_request(user_input, history):
    # Added Throttling: Forces a small cooldown before every request to stay under RPM limits
    time.sleep(1.2)
    
    try:
        current_date_info = get_current_context()
        full_prompt = f"[System Notice: {current_date_info}] {user_input}"
        
        # Context Management: Only send the last 10 turns of history to prevent token exhaustion
        limited_history = history[-10:] if len(history) > 10 else history
        
        response = root_agent.ask(full_prompt, history=limited_history)
        return response.text.strip()
    except Exception as e:
        if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e).upper():
            # If still exhausted, wait longer and notify user
            time.sleep(5)
            return "⚠️ The system is currently busy. Please wait 10 seconds."
        return f"Error: {str(e)}"

# ---------------------------------------------------------
# 6. MAIN LOOP
# ---------------------------------------------------------
if __name__ == "__main__":
    chat_history = []
    print(f"Erin is online. Context: {get_current_context()}")
   
    try:
        while True:
            user_input = input("\nYou: ")
            if user_input.lower() in ["quit", "exit"]:
                save_session(chat_history)
                break
           
            reply = process_request(user_input, chat_history)
            chat_history.append({"role": "user", "content": user_input})
            chat_history.append({"role": "assistant", "content": reply})
            print(f"\nAssistant: {reply}")
            # Extra sleep to ensure the next request doesn't fire too soon
            time.sleep(1.0)
    except KeyboardInterrupt:
        save_session(chat_history)
