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
   
    contextualize_q_system_prompt = (
                            "You are an evaluator for programming and theoretical questions across different difficulty levels (beginner, intermediate, advanced). "
                            "Given a programming or conceptual or theoretical or pseudo-code or code completion or syntax completion question, and a student's answer, "
                            "evaluate the response. You must consider the difficulty level and the type of question (programming, conceptual, theoretical, pseudo-code, code completion, or syntax completion). "
                            "Your evaluation should include: \n"
                            "- A score between 0 and 100 based on accuracy, completeness, and correctness. \n"
                            "- Feedback on the student's answer, explaining why it is correct or incorrect, and how it can be improved. \n"
                            "- The actual correct answer or solution.\n"
                            "Your response should be returned in JSON format, containing the following keys: 'score', 'feedback', and 'actual answer'. "
                            "Make sure your evaluation is appropriate for the given difficulty level and question type."
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
                        "You are an advanced assistant designed to evaluate student answers to programming, conceptual, theoretical, pseudo-code, code completion, and syntax completion questions. "
                        "Given a difficulty level (beginner, intermediate, advanced) and a question type (programming, conceptual, theoretical, pseudo-code, code completion, or syntax completion), "
                        "evaluate the student's response. You are required to return the evaluation in the following JSON structure:\n"
                        "{{\n"
                        "   'score': <integer between 0 and 100>,\n"
                        "   'feedback': '<string explaining the quality of the answer>',\n"
                        "   'actual answer': '<the correct solution or answer to the question>'\n"
                        "}}\n\n"
                        "Make sure to account for the complexity and depth of the answer based on the difficulty level and the question type. \n"
                        "If the student's answer is mostly correct but lacks some details, deduct points accordingly, and provide constructive feedback.\n"
                        "If the context or information provided is insufficient to evaluate the answer, return 'A question cannot be evaluated from the given context.'"
                        "make sure response should be in JSON format only. you con't give any other format."
                        "{context}"
                    )
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

def Evaluation_Conversational_Chain(filename):

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

