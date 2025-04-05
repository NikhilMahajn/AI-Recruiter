
from typing import Annotated
from pydantic import BaseModel, Field  
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq 
from langgraph.types import Command 
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain.tools.retriever import create_retriever_tool
from langgraph.prebuilt import ToolNode,tools_condition
from pydantic import BaseModel
import sqlite3
import os
import streamlit as st
import imaplib
import email
from email.header import decode_header
from dotenv import load_dotenv
from flask import render_template,Flask,request,jsonify,Response,redirect, url_for

from datetime import datetime, timedelta
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart


# from flask import Flask
load_dotenv()
llm = ChatGroq(model="gemma2-9b-it")
app = Flask(__name__)

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    job_summary: str
    resume:str
    score:str
email_address = None
job_desc = None

def initialize_database():
    conn = sqlite3.connect("checkpoints.sqlite", check_same_thread=False)
    cursor = conn.cursor()

    # Create table if it doesn't exist
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS resume_scores (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            job TEXT,
            name TEXT,
            email TEXT UNIQUE,
            score REAL,
            interview TEXT
        )
    """)
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS applications (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email_id TEXT
        )
    """)
    conn.commit()
    conn.close()

    return conn

# Call this function once at the start of the script

def summarizer_node(state: AgentState):
    """
    Enhancer node for refining and clarifying user inputs.

    Args:
        state (MessagesState): The current state containing the conversation history.

    Returns:
        Command: A command to update the state with the enhanced query and route back to the supervisor.
    """
    # Define the system prompt to guide the LLM in query enhancement
    system_prompt = ("""
       "You are an advanced job description summarizer. Your task is to:\n"
        "1. Summarize the job description into a structured format.\n"
        "2. Return the extracted details in JSON format with the following keys:\n"
        "   - 'job_role': The primary job role title (e.g., Software Engineer, Data Scientist).\n"
        "   - 'skills': A list of key skills required.\n"
        "   - 'experience': The required years of experience or specific experience mentioned.\n"
        "   - 'qualifications': Academic or professional qualifications (e.g., Bachelor's degree).\n"
        "   - 'responsibilities': A summary of the core job responsibilities.\n"
        "Return only the JSON string, nothing else."""
    )

    job_description = state["messages"][-1].content
    messages = [
        {"role": "system", "content": system_prompt},
        {"role":"user","content":job_description}  
    ] 

    job_summary = llm.invoke(messages)
    print(f"Current Node: Summerizer -> Goto: Analyzer")

    update_dict = {
            "messages": [  # Append the enhanced query to the message history
                HumanMessage(
                    content=job_summary.content,  # Content of the enhanced query
                    name="summarizer"  # Name of the node processing this message
                )
            ],
            "job_summary" : job_summary.content,
    }
    return Command(
        update=update_dict,
        goto="anayzer", 
    )



class Retriever(BaseModel):
    
    reason: str = Field(
        description="Status of scoring options:completed,failed, for analyzer to decide what to do next ."
    )
    job:str = Field(description="Job Role given in the job summary ")
    email:str = Field(description="email address provided in the resume")
    score: float = Field(
        description="Calculated match score (0-100)."
    )
    full_name:str = Field(description="Full Name of Applicant.")

def grade_documents(state:AgentState):
    """
    Determines whether the retrieved documents are relevant to the question.

    Args:
        state (messages): The current state

    Returns:
        str: A decision for whether the documents are relevant or not
    """

    print("---CHECK RELEVANCE---")

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", """
You are an AI evaluator comparing a candidate’s resume with a job description (JD). Your task is to calculate a match score (0-100) based on the following criteria:
Rules:
1. Skills (40 points max)
Has all the skills we need? → 40 points
Has most skills (about 75%)? → 35 points
Has some skills (about 50%)? → 25 points
Has few skills (about 25%)? → 15 points
Has almost no skills we need? → 5-10 points
2. Experience (30 points max)
Matches or beats what the job asks for? → 30 points
A little less (about 75%)? → 25 points
Half of what we want (about 50%)? → 15 points
Way less (about 25%)? → 5-10 points
No experience that fits? → 5 points
3. Education (20 points max)
Meets or beats what the job needs? → 20 points
Close (related field or different degree)? → 15 points
Lower level but related? → 10 points
Totally unrelated or no degree? → 5 points
4. Certifications (10 points max)
Has all the certifications we want? → 10 points
Has some (about 50%)? → 7 points
Has at least one (about 25%)? → 5 points
No certifications that matter? → 0-2 points
5. Minimum Score Rule
After adding everything up, if the total is less than 15 points, bump it up to 15 (to give credit for transferable skills).
### **Final Output Format**
Provide only structured JSON output:

"""),
            ("user", "{input}\nContext: {context}"),         
        ]
    )

    llm_with_tool = llm.with_structured_output(Retriever)


    # Chain
    chain = prompt | llm_with_tool

    messages = state["messages"]
    last_message = messages[-1]

   
    input = state['job_summary']
    resume = last_message.content
    print(input)

    response = chain.invoke({"input": input,"context":resume})

    conn = sqlite3.connect("checkpoints.sqlite", check_same_thread=False)
    cursor = conn.cursor()

    try:
        cursor.execute(
    """
    INSERT INTO resume_scores (job,name, email, score) 
    VALUES (?, ?, ?,?) 
    ON CONFLICT(email) DO UPDATE SET 
        job = excluded.job, 
        score = excluded.score
    """,
    (response.job,response.full_name, response.email, response.score),
)

        conn.commit()
        conn.close()
    except Exception as e:
        print(e)
    
    return {
            "score": response.score,
            "messages": [("system", f"Evaluation: {response.reason}, Score: {response.score}")]
        }




# System prompt providing clear instructions to the validator agent
system_prompt = '''
 You are an intelligent AI agent responsible for only two tasks
 1. if user prompt has job description then call retrive tool for calculating similarity score.
 2. if tool execution is completed then end the process
'''


def analyzer_node(state: AgentState,tools):
   
    # Extract the first (user's question) and the last (agent's response) messages
    user_question = state["messages"][-1]
        
    # Prepare the message history with the system prompt
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_question.content},
    ]
	
    llm_with_bind = llm.bind_tools(tools)
    response = llm_with_bind.invoke(messages)
    
    if "score" in state and state["score"] > 0:
        print("Score already exists, ending process.")
        goto = END
        response = "Resume is scored successfully"
    elif response.tool_calls:
        print("Calling retrieve tool...")
        goto = "retrieve"
    else:
        print("No tool required, ending process.")
        goto = END
        response = "No action needed."

    
    print(f"Current Node: analyzer -> Goto: {goto}")  # Log for routing back to supervisor
    print("score-->",state['score'])

    
    return Command(
        update={
            "messages": [response]
                
        },
        goto = goto
    )


def create_graph(path):
    loader = PyPDFLoader(path)
    print("Changed")
    document = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
	
    split_documents = splitter.split_documents(document)
	
    vectors = FAISS.from_documents(split_documents, HuggingFaceEmbeddings())
    retriever = vectors.as_retriever()


    retriever_tool = create_retriever_tool(
		retriever,
		"retrieve ",
		"Search and analyzes resume and returns a matching score",
	)
    tools = [retriever_tool]

    conn = sqlite3.connect("checkpoints.sqlite",check_same_thread=False)

    builder = StateGraph(AgentState)

    builder.add_node("summarizer", summarizer_node)
    builder.add_node("analyzer", lambda state: analyzer_node(state, tools))
    retrieve = ToolNode(tools)
    builder.add_node("retrieve", retrieve)
    builder.add_node("grade_documents", grade_documents)
    builder.add_edge(START, "summarizer")
    builder.add_edge("summarizer", "analyzer")

	# Conditional edges from analyzer
    builder.add_conditional_edges(
		"analyzer",
		tools_condition,
		{"tools": "retrieve", "__end__": END}
	)

	# Edges for retrieve and grade_documents
    builder.add_edge("retrieve", "grade_documents")  # Process ToolNode output
    builder.add_edge("grade_documents", "analyzer")  # Loop back to analyzer

	# Compile with MemorySaver
    # saver = MemorySaver()
    saver = SqliteSaver(conn)
    graph = builder.compile(checkpointer=saver)
    return graph



@app.route('/download_resume')
def download_resume():
    if not email_address:
        return redirect(url_for('start'))
    initialize_database()
    conn = sqlite3.connect("checkpoints.sqlite",check_same_thread=False)
    cursor = conn.cursor()
    pdf_folder = "resume"
    os.makedirs(pdf_folder, exist_ok=True)

    review_count = cursor.execute("SELECT COUNT(id) FROM resume_scores;").fetchone()[0]

    interview_count = cursor.execute("SELECT COUNT(id) FROM resume_scores WHERE interview IS NOT NULL;").fetchone()[0]

    count = len(os.listdir("resume"))
    print(review_count,interview_count)
    # if count>0:
    #     return jsonify({"resume":count,"review":review_count,"interview":interview_count})
    
    
    reviewed = conn.cursor().execute("select email_id from applications").fetchall()
    reviewedlist= []
    reviewedlist += [em[0] for em in reviewed]
    # new_emails = [email for email in reviewed if email not in processed]
    EMAIL = email_address
    PASSWORD = os.getenv("EMAIL_PASS")
    # IMAP server details
    IMAP_SERVER = "imap.gmail.com"
    IMAP_PORT = 993

    # Create a folder to save PDFs


    # Combine all email IDs into one list
    mail = imaplib.IMAP4_SSL(IMAP_SERVER, IMAP_PORT)
    mail.login(EMAIL, PASSWORD)
    mail.select("inbox")
    
    # Search for job application emails with corrected OR syntax
    search_criteria = '(OR (OR SUBJECT "resume" SUBJECT "job application") SUBJECT "applied for your job")'
    status, messages = mail.search(None, search_criteria)

    email_ids = messages[0].split()


    print(f"Found {len(email_ids)} job application emails. Downloading PDFs...")
    print(reviewedlist)

    for num in email_ids:
        _, msg_data = mail.fetch(num, "(RFC822)")
        for response_part in msg_data:
            if isinstance(response_part, tuple):
                # Parse the email
                msg = email.message_from_bytes(response_part[1])
                sender = msg["From"].split(' ')[-1]
                print(sender)
                
                if sender in reviewedlist:
                    print("extis")
                    continue
                addMail(sender)
                subject, encoding = decode_header(msg["Subject"])[0]
                if isinstance(subject, bytes):
                    subject = subject.decode(encoding if encoding else "utf-8")

                # Process attachments
                for part in msg.walk():
                    if part.get_content_maintype() == "multipart":
                        continue
                    if part.get_content_subtype() == "plain":
                        continue
                    filename = part.get_filename()
                    if filename and filename.lower().endswith(".pdf"):
                        # Decode filename if needed
                        filename = decode_header(filename)[0][0]
                        if isinstance(filename, bytes):
                            filename = filename.decode(encoding if encoding else "utf-8")

                        # Save PDF file
                        pdf_path = os.path.join(pdf_folder, filename)
                        with open(pdf_path, "wb") as f:
                            f.write(part.get_payload(decode=True))

                        print(f"Saved: {filename}")
    print("Download complete! PDFs saved in the 'Job_Applications_Resumes' folder.")
    
    mail.logout()
    conn.close()
    count = len(os.listdir("resume"))
    return jsonify({"resume":count, "status": "success","review":review_count,"interview":interview_count})
    
@app.route('/')
def index():
    return render_template('index.html')
@app.route('/start',methods = ['POST', 'GET'])
def start():
    return render_template('form.html')

@app.route('/dashboard',methods=['POST'])
def dashboard():
    global email_address
    global job_desc
    if request.method == 'POST':
      job_desc = request.form.get("jobSummary") 
      email_address = request.form.get("email") 
    return render_template("dashboard.html")

@app.route('/progress')
def progress():
    inputs = {
    "job_summary":"",
    "score":0,
    "messages": [
        ("user", job_desc),
    ],
    
}
    def generate():
        pdf_paths = [os.path.join("resume", f) for f in os.listdir("resume")]
        total_files = len(pdf_paths)

        if total_files == 0:
            yield "data: 100\n\n"  
            return

        for i, path in enumerate(pdf_paths, start=1):
            graphnew = create_graph(path)  # Process each file
            
            # Simulate inner processing loop
            for output in graphnew.stream(inputs, {"configurable": {"thread_id": "1"}}):
                print("streaming")
                for key, value in output.items():
                    if value is None:
                        continue
                    print(f"Output from node '{key}':")
                    # print(value)
                    print()

            progress = int((i / total_files) * 100)
            yield f"data: {progress}\n\n"  # Send progress update

            os.remove(path)
        yield "data: 100\n\n"  # Ensure 100% completion

    return Response(generate(), mimetype="text/event-stream")

@app.route('/getApplicant')
def getApplicant():
    conn = sqlite3.connect("checkpoints.sqlite",check_same_thread=False)
    cursor = conn.cursor()
    cursor.execute("SELECT name, email, score, interview FROM resume_scores order by score desc")
    candidates = cursor.fetchall()
    conn.close()

    return [{"name": name, "email": email, "score": score,"interview":interview} for name, email, score, interview in candidates]
@app.route('/schedule')
def sheduleInterview():
    conn = sqlite3.connect("checkpoints.sqlite")
    cursor = conn.cursor()

    cursor.execute("SELECT interview FROM resume_scores WHERE interview IS NOT NULL")
    existing_dates = {row[0] for row in cursor.fetchall()}  

    cursor.execute("SELECT id FROM resume_scores WHERE score > 70 AND (interview IS NULL OR interview = '')")
    candidates = [row[0] for row in cursor.fetchall()]

    next_date = datetime.today() + timedelta(days=1)
    print(candidates)
    
    # Schedule interviews for each candidate
    for candidate_id in candidates:
        # Find the first available date
        while next_date.strftime("%Y-%m-%d") in existing_dates:
            next_date += timedelta(days=1)

        next_date_str = next_date.strftime("%Y-%m-%d")

        # Update interview date for the candidate
        print(candidate_id)
        cursor.execute("""
            UPDATE resume_scores 
            SET interview = ? 
            WHERE id = ?;
        """, (next_date_str, candidate_id))
        
        sendMail(candidate_id,conn)

        existing_dates.add(next_date_str)

    conn.commit()
    conn.close()
    return jsonify({"message":"interview Sheduled"})

def sendMail(c_id,conn):
    cursor = conn.cursor()
    smtp_host = os.getenv("SMTP_HOST", "smtp.gmail.com")  
    smtp_port = int(os.getenv("SMTP_PORT", 587))
    smtp_user = os.getenv("SMTP_USER")
    smtp_pass = os.getenv("EMAIL_PASS")
  
    cursor.execute("SELECT name, email, interview FROM resume_scores WHERE id=?;",(c_id,))
    candidate = cursor.fetchone()

    if not candidate:
        print("No scheduled interviews to send invitations.")
        return

    # Set up SMTP email server
    try:
        server = smtplib.SMTP(smtp_host, smtp_port)
        server.starttls()  # Secure connection
        server.login(smtp_user, smtp_pass)

        name, email, interview_date = candidate

        subject = "Interview Invitation - Your Scheduled Date"
        body = f"""
        Dear {name},
        We are pleased to inform you that your interview has been scheduled on {interview_date}.
        Please be available on the given date and check your email for further instructions.
        Best Regards,
        Recruitment Team
        """
        # Create email message
        msg = MIMEMultipart()
        msg["From"] = smtp_user
        msg["To"] = email
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain"))
        server.sendmail(smtp_user, email, msg.as_string())
        print(f"Interview invitation sent to: {name} ({email})")

        server.quit()

    except Exception as e:
        print(f"Error sending emails: {e}")

def addMail(id):
    conn = sqlite3.connect("checkpoints.sqlite")
    conn.cursor().execute("insert into applications(email_id) values(?);",(id,))
    conn.commit()
    conn.close()

if __name__=="__main__":
    app.run()
   