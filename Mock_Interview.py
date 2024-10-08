from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

model_name = 'llama3.1:latest'
embeddings_model_name =  "all-MiniLM-L6-v2"

model = Ollama(model = model_name)

embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)

store = {}

def History_Chain(retriever):
   
   

#     contextualize_q_system_prompt = (
#     "Given the user's initial input, which includes the position, role, and required skills, generate an interview-style question "
#     "that is unique and relevant to the specified role and skills. Ensure that the question has not been asked previously in this interview. "
#     "After the user provides an answer, generate up to two follow-up questions based on that answer, maintaining relevance to the topic. "
#     "Avoid repetitive or overly lengthy questions to mimic a natural, real-time interview flow."
# )
#     contextualize_q_system_prompt = (
#     "Given the user's initial input, which includes the position, role, and required skills, generate a single interview-style question "
#     "relevant to the specified role and skills. After the user answers, generate one follow-up question at a time based on their response. "
#     "Each follow-up question should stay focused on the current topic and not shift to a new topic until the follow-up questions are exhausted."
#     "Ensure that the interview remains engaging, realistic, and that questions are concise without being repetitive."
# )
    contextualize_q_system_prompt = (
        "You are an AI assistant conducting a mock interview tailored to a specific job role. Your task is to generate relevant interview questions"
        "based on the role description and required skills provided by the user. You will ask one question at a time and dynamically generate"
        "two follow-up questions based on the interviewee's responses to dive deeper into specific topics. After each follow-up, wait for the candidate’s response before proceeding.\n"
        "Your goal is to simulate a real interview scenario by:\n"
        "1. Asking questions in a logical flow, where each question is relevant to the previous response or topic."
        "2. Adapting your follow-up questions to address key points mentioned by the interviewee."
        # "3. Refraining from indicating how many questions are left or summarizing at any point, focusing solely on continuing the conversation naturally."
        "4. Avoid listing multiple questions at once; you should ask them individually."
        "5. Avoid  multiple sentence question ask questions using maximum three sentence."
        "\n\n"
        "Ensure the conversation feels fluid, engaging, and focused on the candidate's problem-solving abilities and understanding of the role. After each question, allow the user to respond before asking the next one."
    )


    contextualize_q_prompt = ChatPromptTemplate.from_messages(
                                                                [   
                                                                    ("system", contextualize_q_system_prompt),
                                                                    MessagesPlaceholder("chat_history"),
                                                                    ("human", "{input}"),
                                                                ]
                                                            )
    history_aware_retriever = create_history_aware_retriever(model,
                                                             retriever, 
                                                             contextualize_q_prompt
                                                            )
    return history_aware_retriever


def Question_Answer_Chain():

    system_prompt = (
    "You are an interviewer conducting a placement training session. Based on the user's input specifying the position, role, and required skills, "
    "ask a single, relevant interview question at a time. Once the user answers, generate a concise and focused follow-up question based on their answer. "
    "Generate up to two follow-up questions on the same topic before moving to the next topic. Ensure that each question feels natural, as if in a real interview. "
    "Do not provide answers or feedback during the interview. If the user asks for an answer, respond with 'I don't know.' "
    "Finally, after 10 questions, generate a report summarizing the candidate's responses and provide feedback on their performance."
    "\n\n"
    "{context}"
)
    system_prompt = (
"You are a virtual interviewer simulating a real interview experience based on a specific role and skill set provided by the user."
"You will engage with the candidate by asking one question at a time and adapting your follow-up questions based on their answers.\n\n"
"Please adhere to the following guidelines:\n\n"
"- Begin by asking a role-specific question based on the information provided by the user."
"- After the candidate responds, generate a relevant follow-up question to delve deeper into the same topic."
"- If the candidate struggles to answer a question or a follow-up question, ask a simpler question. If they continue to struggle, move on to the next question."
"- Allow the candidate time to answer each question fully before proceeding with the next."
"- Do not ask multiple questions at once; proceed one question at a time, focusing on conversational flow and depth."
"- Do not ask multiple sentence questions generate question simple sentence."
"- Ensure questions are clear and concise, avoiding overly complex sentences."
"- Do not mention how many questions remain or provide any information about the total number of questions."
"- Avoid referencing how many questions remain or offering any information about the number of questions left."
"- Do not provide answers or feedback during the interview. If the user asks for an answer, respond with 'I'm not suppose to answer the question during interview.'"
"- After 10 questions, generate a report summarizing the candidate's responses and provide feedback on their performance."
"Your objective is to simulate an interactive interview, making the conversation as natural as possible."
"\n\n"
"{context}")

    qa_prompt = ChatPromptTemplate.from_messages(
                                                    [
                                                        ("system", system_prompt),
                                                        MessagesPlaceholder("chat_history"),
                                                        ("human", "{input}"),
                                                    ]
                                                )
    question_answer_chain = create_stuff_documents_chain(model, qa_prompt)
    return question_answer_chain

def RAG_Chain(retriever):
    history_aware_retriever = History_Chain(retriever)
    question_answer_chain = Question_Answer_Chain()
    return create_retrieval_chain(history_aware_retriever, question_answer_chain)


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

def Mock_Interview_Chain(filename):

    db = Chroma(persist_directory= filename, embedding_function=embeddings)

    retriever = db.as_retriever()
    rag_chain = RAG_Chain(retriever)
    return RunnableWithMessageHistory(
                                        rag_chain,
                                        get_session_history,
                                        input_messages_key="input",
                                        history_messages_key="chat_history",
                                        output_messages_key="answer",
                                    )
