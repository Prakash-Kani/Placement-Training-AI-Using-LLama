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
                                        "Given the chat history and the latest user input, which involves generating a programming or conceptual or theoritical or pseude-code or "
                                        "code completion or  syntax completion question at a specific difficulty level (beginner, intermediate, or advanced) or based on a specific topic, "
                                        "formulate a unique question as requested. Ensure the question is not a repetition of previously generated questions, "
                                        "is relevant to the user's input, and adheres to the specified difficulty level and topic, if provided. "
                                        "Return only the question, without the answer."
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
                        "You are an assistant tasked with generating unique questions based on user input. "
                        "When the user requests a programming or descriptive question at a specific difficulty level or from a specific topic, "
                        "generate a new question that matches the requested criteria (beginner, intermediate, advanced, programming, or conceptual or theoritical or pseude-code or code completion or  syntax completion). "
                        "Do not provide an answer. If the user asks for an answer, respond with 'I don't know.' "
                        "If the context does not contain the information to generate the question, respond with 'A question cannot be generated from the given context.' "
                        "\n\n"
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

def Question_Generator_Chain(filename):

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

