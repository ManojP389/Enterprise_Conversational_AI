# file: backend/rag_core.py

import os
import json
import hashlib
from typing import List, Dict, Tuple
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
# from langchain_openai import ChatOpenAI
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_chroma import Chroma
# import chromadb
# from chromadb.config import Settings
# from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredMarkdownLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_core.prompts import PromptTemplate
# from langchain_core.runnables import RunnableLambda, RunnablePassthrough
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import chromadb
from chromadb.config import Settings
from langchain_community.retrievers import BM25Retriever
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
import numpy as np
import re
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(
    """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.

    Chat History:
    {chat_history}

    Follow Up Input: {question}
    Standalone question:"""
)

# --- REFINEMENT 1: A new, balanced prompt encouraging helpful explanations ---
# ANSWER_PROMPT = PromptTemplate.from_template(
#     """You are a helpful and expert AI assistant. Your task is to provide a clear, well-structured, and explanatory answer to the user's question using ONLY the provided document context.

#     **Core Instructions:**
#     1.  Analyze the 'Standalone Question' and the 'Document Context' to form your answer.
#     2.  Your answer MUST be a helpful explanation that directly addresses the user's question.
#     3.  Structure your answer logically. Use paragraphs to separate distinct ideas.
#     4.  Be thorough in your explanation, but do not add information that isn't in the context and avoid unnecessary repetition.
#     5.  If the context does not contain the information to answer the question, you MUST reply with ONLY this exact phrase: "I could not find an answer to that in the provided documents."

#     **Document Context:**
#     {context}

#     **Standalone Question:** {question}

#     **Helpful Answer:**"""
# )
ANSWER_PROMPT = PromptTemplate.from_template(
    """You are a helpful document question-answering assistant.

Answer the user's question using the provided document context.

IMPORTANT RULES:

1. Read the document context carefully before answering.
2. If the answer is explicitly present in the context, answer it directly.
3. Do not say that the answer is missing when the requested information is clearly present.
4. You may directly copy names, numbers, dates, titles, skills, project names, and other factual information from the context.
5. Do not invent information that is not present in the context.
6. If the context genuinely does not contain enough information to answer the question, reply exactly:
"I could not find an answer to that in the provided documents."
7. Keep the answer concise and directly answer the question.
8. For simple factual questions, give the direct answer first.

DOCUMENT CONTEXT:
{context}

USER QUESTION:
{question}

ANSWER:"""
)

def _format_docs(docs: List[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)

class ConversationalRAG:
    def __init__(self):
        
    #     self.llm = ChatOpenAI(
    #         base_url="http://localhost:1234/v1",
    #         api_key="lm-studio",
    #         model="llama-3.2-3b-instruct",
    #         temperature=0.6
    #    )
        self.llm = ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.2
    )
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.db_directory = "./vector_db"
        self.chroma_client = chromadb.PersistentClient(path=self.db_directory, settings=Settings(anonymized_telemetry=False))
        self.vectorstore = Chroma(client=self.chroma_client, embedding_function=self.embeddings)
        self.rag_chain = None
        self.vectorstore = Chroma(
        client=self.chroma_client,
        embedding_function=self.embeddings
       )
        self.bm25_retriever = None
        self.all_documents = []
        # self.rag_chain = None
        # self.rag_initialized = False
        # self.distance_threshold = 1.15
        self.vectorstore = Chroma(
        client=self.chroma_client,
        embedding_function=self.embeddings
    )

        self.bm25_retriever = None
        self.all_documents = []

        # Load existing documents from ChromaDB for BM25
        try:
            existing_data = self.vectorstore.get()

            if existing_data and existing_data.get("documents"):
                self.all_documents = [
                    Document(
                        page_content=text,
                        metadata=metadata or {}
                    )
                    for text, metadata in zip(
                        existing_data["documents"],
                        existing_data["metadatas"]
                    )
                ]

                logger.info(
                    f"Loaded {len(self.all_documents)} existing documents for BM25."
                )

                self.bm25_retriever = BM25Retriever.from_documents(
                    self.all_documents
                )

                self.bm25_retriever.k = 5

        except Exception as e:
            logger.warning(
                f"Could not initialize BM25 retriever: {e}"
            )

        self.rag_chain = None
        self.rag_initialized = False

        self.distance_threshold = 1.15
        self.metadata_file = os.path.join(self.db_directory, "processed_files.json")
        self.processed_files = self._load_processed_files_metadata()

        # def _is_follow_up(self, question: str, chat_history: List[Dict]) -> Tuple[bool, List[Tuple[str, str]]]:
        #     if not chat_history:
        #         return False, []
        #     pronoun_pattern = r"\b(it|that|this|those|these|they|them|he|him|she|her)\b"
        #     # if re.search(pronoun_pattern, question.lower()) or len(question.split()) <= 3:
        #     #     last_exchange = chat_history[-3:]
        #     #     user_msg = last_exchange[0]['content']
        #     #     ai_msg = last_exchange[1]['content']
        #     #     return True, [(user_msg, ai_msg)]
        #     if re.search(pronoun_pattern, question.lower()) or len(question.split()) <= 3:

        # # Need at least one user message and one AI response
        #         if len(chat_history) < 2:
        #             return False, []

        #         last_exchange = chat_history[-2:]

        #         user_msg = last_exchange[0]["content"]
        #         ai_msg = last_exchange[1]["content"]

        #         return True, [(user_msg, ai_msg)]
        #     question_embedding = self.embeddings.embed_query(question)
        #     history_embeddings = self.embeddings.embed_documents([turn['content'] for turn in chat_history if turn['role'] == 'user'])
        #     similarities = [np.dot(question_embedding, hist_emb) / (np.linalg.norm(question_embedding) * np.linalg.norm(hist_emb)) for hist_emb in history_embeddings]
        #     max_similarity = max(similarities) if similarities else 0
        #     logger.info(f"Max similarity score with histuory: {max_similarity:.4f}")
        #     if max_similarity > 0.7:
            
        #         most_relevant_index = np.argmax(similarities)
        #         user_turn_index = most_relevant_index * 2
        #         if user_turn_index + 1 < len(chat_history):
        #             user_msg = chat_history[user_turn_index]['content']
        #             ai_msg = chat_history[user_turn_index + 1]['content']
        #             return True, [(user_msg, ai_msg)]
        #     return False, []
    def _is_follow_up(
    self,
    question: str,
    chat_history: List[Dict]
) -> Tuple[bool, List[Tuple[str, str]]]:

        # No previous conversation → this is a new question
        if not chat_history:
            return False, []

        question_lower = question.lower().strip()

        # Pronouns / short questions that usually depend on previous context
        pronoun_pattern = (
            r"\b(it|that|this|those|these|they|them|he|him|she|her|"
            r"its|their|his|hers)\b"
        )

        # --------------------------------------------------
        # 1. Explicit follow-up
        # --------------------------------------------------

        if re.search(pronoun_pattern, question_lower) or len(question.split()) <= 3:

            # Need at least one user message + one AI response
            if len(chat_history) < 2:
                return False, []

            # Take the latest exchange
            last_exchange = chat_history[-2:]

            user_msg = last_exchange[0]["content"]
            ai_msg = last_exchange[1]["content"]

            return True, [(user_msg, ai_msg)]

        # --------------------------------------------------
        # 2. Semantic similarity with previous user questions
        # --------------------------------------------------

        question_embedding = self.embeddings.embed_query(question)

        user_messages = [
            turn["content"]
            for turn in chat_history
            if turn["role"] == "user"
        ]

        if not user_messages:
            return False, []

        history_embeddings = self.embeddings.embed_documents(
            user_messages
        )

        similarities = [
            np.dot(question_embedding, hist_emb)
            / (
                np.linalg.norm(question_embedding)
                * np.linalg.norm(hist_emb)
            )
            for hist_emb in history_embeddings
        ]

        max_similarity = max(similarities) if similarities else 0

        logger.info(
            f"Max similarity score with history: {max_similarity:.4f}"
        )

        # Use a higher threshold to avoid incorrectly
        # treating independent questions as follow-ups
        if max_similarity > 0.80:

            most_relevant_index = np.argmax(similarities)

            # Each user message should normally be followed
            # by its corresponding AI response.
            user_turn_index = most_relevant_index * 2

            if user_turn_index + 1 < len(chat_history):

                user_msg = chat_history[user_turn_index]["content"]
                ai_msg = chat_history[user_turn_index + 1]["content"]

                return True, [(user_msg, ai_msg)]

        # --------------------------------------------------
        # 3. Otherwise this is a completely new question
        # --------------------------------------------------

        return False, []

    def initialize_chain(self):
        try:
            if self.chroma_client.get_collection(name="langchain").count() == 0:
                self.rag_initialized = False; return
        except Exception:
            self.rag_initialized = False; return
        logger.info("Initializing custom Conversational RAG chain...")
        # retriever = self.vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 5, "fetch_k": 20})
        # self.answer_chain = (
        #     {"context": retriever | _format_docs, "question": RunnablePassthrough()}
        #     | ANSWER_PROMPT | self.llm | StrOutputParser()
        # )
        self.rag_initialized = True
        logger.info("RAG chain components are ready.")
    
    # def query(self, question: str, chat_history: List[Dict]) -> Dict:
    #         response = self.llm.invoke(question)

    #         return {
    #             "answer": response.content,
    #             "sources": [],
    #             "type": "general"
    #         }
    # def query(self, question: str, chat_history: List[Dict]) -> Dict:


    
    
    #     if not self.rag_initialized:
    #         logger.info("Knowledge Base not processed. Using LLM only.")

    #         response = self.llm.invoke(question)

    #         return {
    #             "answer": response.content.strip(),
    #             "sources": [],
    #             "type": "general"
    #         }

        
        
        
    #     logger.info("Knowledge Base available. Using RAG.")

    #     is_follow_up, selected_history = self._is_follow_up(question, chat_history)

    #     final_question = question

    #     if is_follow_up:

    #         history_str = "\n".join(
    #             [f"Human: {h}\nAI: {a}" for h, a in selected_history]
    #         )

    #         final_question = (
    #             CONDENSE_QUESTION_PROMPT
    #             | self.llm
    #             | StrOutputParser()
    #         ).invoke(
    #             {
    #                 "chat_history": history_str,
    #                 "question": question
    #             }
    #         )

    #     response = self.answer_chain.invoke(final_question)

    #     return {
    #         "answer": response.strip(),
    #         "sources": [],
    #         "type": "rag"
    # #     }
    def query(self, question: str, chat_history: List[Dict]) -> Dict:

 

        if not self.rag_initialized:
            logger.info(
                "Knowledge base not initialized. Using General LLM."
            )

            response = self.llm.invoke(question)

            return {
                "answer": response.content.strip(),
                "sources": [],
                "type": "general"
            }

       

        is_follow_up, selected_history = self._is_follow_up(
            question,
            chat_history
        )

        final_question = question

        if is_follow_up:

            logger.info("Follow-up detected.")

            history_str = "\n".join(
                [
                    f"Human: {h}\nAI: {a}"
                    for h, a in selected_history
                ]
            )

            final_question = (
                CONDENSE_QUESTION_PROMPT
                | self.llm
                | StrOutputParser()
            ).invoke(
                {
                    "chat_history": history_str,
                    "question": question
                }
            )

            logger.info(
                f"Standalone Question: {final_question}"
            )

        

        semantic_docs = (
            self.vectorstore.similarity_search_with_score(
                final_question,
                k=5
            )
        )

        logger.info(
            f"Semantic results: {len(semantic_docs)}"
        )

        

        bm25_docs = []

        if self.bm25_retriever:

            bm25_docs = self.bm25_retriever.invoke(
                final_question
            )

        logger.info(
            f"BM25 results: {len(bm25_docs)}"
        )

    

        combined_docs = []

        # Add semantic documents
        for doc, distance in semantic_docs:

            if not any(
                doc.page_content == existing.page_content
                for existing in combined_docs
            ):
                combined_docs.append(doc)

        # Add BM25 documents
        for doc in bm25_docs:

            if not any(
                doc.page_content == existing.page_content
                for existing in combined_docs
            ):
                combined_docs.append(doc)

        

        question_lower = final_question.lower()

        resume_requested = (
            "resume" in question_lower
            or "cv" in question_lower
            or "curriculum vitae" in question_lower
        )

        if resume_requested:

            resume_docs = [
                doc
                for doc in self.all_documents
                if (
                    "resume" in str(
                        doc.metadata.get("source", "")
                    ).lower()
                    or "cv" in str(
                        doc.metadata.get("source", "")
                    ).lower()
                )
            ]

            if resume_docs:

                logger.info(
                    f"Found {len(resume_docs)} resume chunks "
                    "through metadata search."
                )

                # Put resume chunks first
                for doc in reversed(resume_docs):

                    if not any(
                        doc.page_content == existing.page_content
                        for existing in combined_docs
                    ):
                        combined_docs.insert(0, doc)

        # Keep maximum 8 documents
        combined_docs = combined_docs[:8]

        logger.info(
            f"Combined hybrid results: {len(combined_docs)}"
        )


        logger.info("=" * 60)

        logger.info(
            f"Question: {final_question}"
        )

        for i, doc in enumerate(combined_docs):

            logger.info(
                f"Document {i + 1}"
            )

            logger.info(
                f"Source: "
                f"{doc.metadata.get('source', 'unknown')}"
            )

            logger.info(
                f"Content Preview:\n"
                f"{doc.page_content[:300]}"
            )

            logger.info("-" * 60)

       

        best_semantic_distance = (
            semantic_docs[0][1]
            if semantic_docs
            else float("inf")
        )

        has_semantic_match = (
            best_semantic_distance
            <= self.distance_threshold
        )

      

        stop_words = {
            "the", "a", "an",
            "is", "are", "was", "were",
            "be", "been", "being",

            "what", "who", "where", "when",
            "why", "how",

            "can", "could", "would", "should",

            "do", "does", "did",

            "you", "your", "yours",
            "me", "my", "mine",

            "we", "our", "ours",

            "they", "their", "theirs",

            "he", "him", "his",

            "she", "her", "hers",

            "it", "its",

            "this", "that",
            "these", "those",

            "there", "here",

            "about", "with", "from",
            "for", "and", "or",

            "to", "of", "in",
            "on", "at", "by",

            "as", "into", "through",

            "tell", "please",
            "provide", "give",

            "more", "information",

            "person", "mentioned"
        }

        question_words = {
            word.strip(".,?!'\"")
            for word in question_lower.split()
            if (
                word.strip(".,?!'\"") not in stop_words
                and len(word.strip(".,?!'\"")) > 2
            )
        }

        logger.info(
            f"Meaningful query keywords: {question_words}"
        )

        keyword_match = False
        best_keyword_score = 0
        best_keyword_doc = None

        # Search both document content and metadata
        for doc in combined_docs:

            document_text = doc.page_content.lower()

            source = str(
                doc.metadata.get("source", "")
            ).lower()

            searchable_text = (
                document_text + " " + source
            )

            matching_words = [
                word
                for word in question_words
                if word in searchable_text
            ]

            score = len(matching_words)

            # Strong boost for resume queries
            if resume_requested:

                if "resume" in source:
                    score += 5

                if "uploaded_documents" in source:
                    score += 2

            # Boost words appearing in filename
            for word in question_words:

                if len(word) >= 4 and word in source:
                    score += 2

            if score > best_keyword_score:

                best_keyword_score = score
                best_keyword_doc = doc

                if matching_words:

                    logger.info(
                        f"Keyword matches: {matching_words}"
                    )

        if best_keyword_score >= 2:

            keyword_match = True

            logger.info(
                f"Best keyword score: "
                f"{best_keyword_score}"
            )

            if best_keyword_doc:

                logger.info(
                    "Best keyword document: "
                    f"{best_keyword_doc.metadata.get('source', 'unknown')}"
                )

        

        metadata_match = False

        if resume_requested:

            for doc in combined_docs:

                source = str(
                    doc.metadata.get("source", "")
                ).lower()

                if (
                    "resume" in source
                    or "cv" in source
                    or "curriculum" in source
                ):

                    metadata_match = True

                    logger.info(
                        "Resume metadata match found: "
                        f"{source}"
                    )

                    break

        

        if (
            not has_semantic_match
            and not keyword_match
            and not metadata_match
        ):

            logger.info(
                "No sufficiently relevant semantic, "
                "keyword, or metadata match. "
                "Using General LLM."
            )

            response = self.llm.invoke(question)

            return {
                "answer": response.content.strip(),
                "sources": [],
                "type": "general"
            }


        logger.info(
            "Relevant information found. "
            "Using Hybrid RAG."
        )

        context = _format_docs(
            combined_docs
        )

        response = (
            ANSWER_PROMPT
            | self.llm
            | StrOutputParser()
        ).invoke(
            {
                "context": context,
                "question": final_question
            }
        )

        return {
            "answer": response.strip(),
            "sources": [],
            "type": "rag"
        }
    # def query(self, question: str, chat_history: List[Dict]) -> Dict:

    

    #     if not self.rag_initialized:
    #         logger.info(
    #             "Knowledge base not initialized. Using General LLM."
    #         )

    #         response = self.llm.invoke(question)

    #         return {
    #             "answer": response.content.strip(),
    #             "sources": [],
    #             "type": "general"
    #         }

       

    #     is_follow_up, selected_history = self._is_follow_up(
    #         question,
    #         chat_history
    #     )

    #     final_question = question

    #     if is_follow_up:

    #         logger.info("Follow-up detected.")

    #         history_str = "\n".join(
    #             [
    #                 f"Human: {h}\nAI: {a}"
    #                 for h, a in selected_history
    #             ]
    #         )

    #         final_question = (
    #             CONDENSE_QUESTION_PROMPT
    #             | self.llm
    #             | StrOutputParser()
    #         ).invoke(
    #             {
    #                 "chat_history": history_str,
    #                 "question": question
    #             }
    #         )

    #         logger.info(
    #             f"Standalone Question: {final_question}"
    #         )

       

    #     semantic_docs = (
    #         self.vectorstore.similarity_search_with_score(
    #             final_question,
    #             k=5
    #         )
    #     )

    #     logger.info(
    #         f"Semantic results: {len(semantic_docs)}"
    #     )

    

    #     bm25_docs = []

    #     if self.bm25_retriever:

    #         bm25_docs = self.bm25_retriever.invoke(
    #             final_question
    #         )

    #     logger.info(
    #         f"BM25 results: {len(bm25_docs)}"
    #     )

        

    #     combined_docs = []

    #     # Add semantic documents
    #     for doc, distance in semantic_docs:

    #         if not any(
    #             doc.page_content == existing.page_content
    #             for existing in combined_docs
    #         ):
    #             combined_docs.append(doc)

    #     # Add BM25 documents
    #     for doc in bm25_docs:

    #         if not any(
    #             doc.page_content == existing.page_content
    #             for existing in combined_docs
    #         ):
    #             combined_docs.append(doc)

    #     # Keep maximum 8 documents
    #     combined_docs = combined_docs[:8]

    #     logger.info(
    #         f"Combined hybrid results: {len(combined_docs)}"
    #     )

       

    #     logger.info("=" * 60)
    #     logger.info(
    #         f"Question: {final_question}"
    #     )

    #     for i, doc in enumerate(combined_docs):

    #         logger.info(
    #             f"Document {i + 1}"
    #         )

    #         logger.info(
    #             f"Source: {doc.metadata.get('source', 'unknown')}"
    #         )

    #         logger.info(
    #             f"Content Preview:\n"
    #             f"{doc.page_content[:300]}"
    #         )

    #         logger.info("-" * 60)


    #     best_semantic_distance = (
    #         semantic_docs[0][1]
    #         if semantic_docs
    #         else float("inf")
    #     )

    #     has_semantic_match = (
    #         best_semantic_distance
    #         <= self.distance_threshold
    #     )


    #     question_words = set(
    #         final_question.lower().split()
    #     )

    #     keyword_match = False

    #     for doc in bm25_docs:

    #         document_text = doc.page_content.lower()

    #         matching_words = [
    #             word
    #             for word in question_words
    #             if len(word) > 2
    #             and word in document_text
    #         ]

    #         if len(matching_words) >= 2:

    #             keyword_match = True

    #             logger.info(
    #                 f"Keyword match found: {matching_words}"
    #             )

    #             break

    

    #     if not has_semantic_match and not keyword_match:

    #         logger.info(
    #             "No sufficiently relevant semantic or "
    #             "keyword match. Using General LLM."
    #         )

    #         response = self.llm.invoke(question)

    #         return {
    #             "answer": response.content.strip(),
    #             "sources": [],
    #             "type": "general"
    #         }

        

    #     logger.info(
    #         "Relevant information found. Using Hybrid RAG."
    #     )

    #     context = _format_docs(
    #         combined_docs
    #     )

    #     response = (
    #         ANSWER_PROMPT
    #         | self.llm
    #         | StrOutputParser()
    #     ).invoke(
    #         {
    #             "context": context,
    #             "question": final_question
    #         }
    #     )

    #     return {
    #         "answer": response.strip(),
    #         "sources": [],
    #         "type": "rag"
    #     }
#     def query(self, question: str, chat_history: List[Dict]) -> Dict:

    
#         if not self.rag_initialized:
#             logger.info("Knowledge base not initialized. Using General LLM.")

#             response = self.llm.invoke(question)

#             return {
#                 "answer": response.content.strip(),
#                 "sources": [],
#                 "type": "general"
#             }

        
#         is_follow_up, selected_history = self._is_follow_up(
#             question,
#             chat_history
#         )

#         final_question = question

#         if is_follow_up:
#             logger.info("Follow-up detected.")

#             history_str = "\n".join(
#                 [f"Human: {h}\nAI: {a}" for h, a in selected_history]
#             )

#             final_question = (
#                 CONDENSE_QUESTION_PROMPT
#                 | self.llm
#                 | StrOutputParser()
#             ).invoke(
#                 {
#                     "chat_history": history_str,
#                     "question": question
#                 }
#             )

#             logger.info(f"Standalone Question: {final_question}")

        
#         # docs = self.vectorstore.similarity_search_with_score(
#         #     final_question,
#         #     k=3
#         # )
    
#         semantic_docs = self.vectorstore.similarity_search_with_score(
#             final_question,
#             k=5
#         )

#         bm25_docs = []

#         if self.bm25_retriever:
#             bm25_docs = self.bm25_retriever.invoke(
#                 final_question
#             )

#         logger.info(
#             f"Semantic results: {len(semantic_docs)}"
#         )

#         logger.info(
#             f"BM25 results: {len(bm25_docs)}"
#         )
       

#         combined_docs = []

#         # Add semantic results
#         for doc, distance in semantic_docs:
#             combined_docs.append(doc)

#         # Add BM25 results
#         for doc in bm25_docs:
#             if not any(
#                 doc.page_content == existing.page_content
#                 for existing in combined_docs
#             ):
#                 combined_docs.append(doc)

#         combined_docs = combined_docs[:5]

#         logger.info(
#             f"Combined hybrid results: {len(combined_docs)}"
#         )
#         logger.info("=" * 60)
#         logger.info(f"Question: {final_question}")
#         logger.info(f"Retrieved {len(docs)} document(s)")

#         for i, (doc, score) in enumerate(docs):
#             logger.info(f"Document {i+1}")
#             logger.info(f"Distance  Score: {score}")
#             logger.info(f"Content Preview:\n{doc.page_content[:300]}")
#             logger.info("-" * 60)

#         best_semantic_distance = (
#     semantic_docs[0][1]
#     if semantic_docs
#     else float("inf")
# )

# has_semantic_match = (
#     best_semantic_distance <= self.distance_threshold
# )

# has_bm25_match = bool(bm25_docs)

# if not has_semantic_match and not has_bm25_match:

#     logger.info(
#         "No sufficiently relevant semantic or keyword match. "
#         "Using General LLM."
#     )

#     response = self.llm.invoke(question)

#     return {
#         "answer": response.content.strip(),
#         "sources": [],
#         "type": "general"
#     }

#     context = _format_docs(combined_docs)

#     response = (
#         ANSWER_PROMPT
#         | self.llm
#         | StrOutputParser()
#     ).invoke(
#         {
#             "context": context,
#             "question": final_question
#         }
#     )

#     return {
#         "answer": response.strip(),
#         "sources": [],
#         "type": "rag"
#     }

       
        # if not docs or docs[0][1] > self.distance_threshold:
        

        #     logger.info("High distance score. Using General LLM.")

        #     response = self.llm.invoke(question)

        #     return {
        #         "answer": response.content.strip(),
        #         "sources": [],
        #         "type": "general"
        #     }

        
        # logger.info(
        #     f"Relevant document found. Distance Score: {docs[0][1]:.3f}"
        # )

        # response = self.answer_chain.invoke(final_question)

        # return {
        #     "answer": response.strip(),
        #     "sources": [],
        #     "type": "rag"
        # }
    # def query(self, question: str, chat_history: List[Dict]) -> Dict:
    #     if not self.rag_initialized:
    #         logger.info("RAG not initialized. Responding with general knowledge.")
    #         response = self.llm.invoke(question)
    #         return {"answer": response.content.strip(), "sources": [], "type": "general"}
            # is_follow_up, selected_history = self._is_follow_up(question, chat_history)
            # final_question = question
            # if is_follow_up:
            #         logger.info("Follow-up detected. Condensing question with selected history.")
            #         history_str = "\n".join([f"Human: {h}\nAI: {a}" for h, a in selected_history])
            #         final_question = (
            #             CONDENSE_QUESTION_PROMPT | self.llm | StrOutputParser()
            #         ).invoke({"chat_history": history_str, "question": question})
            #         logger.info(f"Standalone question: {final_question}")
            # else:
            #         logger.info("No follow-up detected. Using original question.")
            #         response = self.answer_chain.invoke(final_question)
            # return {"answer": response.strip(), "sources": [], "type": "rag"}
        
    # --- No changes to the functions below this line ---
    def load_and_process_documents(self, directory="knowledge_base") -> int:
        # logger.info(f"Scanning for documents in directory: '{os.path.abspath(directory)}'")
        # if not os.path.exists(directory): return 0
        directories = [directory, "uploaded_documents"]

        all_files = []

        for directory in directories:
            logger.info(f"Scanning for documents in directory: '{os.path.abspath(directory)}'")

            if not os.path.exists(directory):
                continue

            all_files.extend([
                os.path.join(root, file)
                for root, _, files in os.walk(directory)
                for file in files
                if file.endswith((".pdf", ".txt", ".md"))
            ])
        # all_files = [os.path.join(root, file) for root, _, files in os.walk(directory) for file in files if file.endswith(('.pdf', '.txt', '.md'))]
        files_to_process = [fp for fp in all_files if self.processed_files.get(os.path.abspath(fp), {}).get('hash') != self._get_file_hash(fp)]
        if not files_to_process: return 0
        logger.info(f"Processing {len(files_to_process)} new/changed files...")
        documents = []
        for file_path in files_to_process:
            try:
                if file_path.endswith('.pdf'): loader = PyPDFLoader(file_path)
                elif file_path.endswith('.txt'): loader = TextLoader(file_path)
                else: loader = UnstructuredMarkdownLoader(file_path)
                documents.extend(loader.load())
                self.processed_files[os.path.abspath(file_path)] = {'hash': self._get_file_hash(file_path)}
            except Exception as e:
                logger.error(f"Failed to load '{os.path.basename(file_path)}': {e}. Skipping.")
        if not documents: return 0
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)
        if chunks:
            self.update_vector_store(chunks)
            self._save_processed_files_metadata()
            return len(chunks)
        return 0

    # def update_vector_store(self, chunks: List):
    #     logger.info(f"Adding {len(chunks)} new chunks to vector store...")
    #     self.vectorstore.add_documents(chunks)
    #     logger.info("Vector store updated successfully.")
    def update_vector_store(self, chunks: List):
        logger.info(f"Adding {len(chunks)} new chunks to vector store...")

        self.vectorstore.add_documents(chunks)

        self.all_documents.extend(chunks)

        self.bm25_retriever = BM25Retriever.from_documents(
            self.all_documents
        )

        self.bm25_retriever.k = 5

        logger.info("Vector store and BM25 retriever updated successfully.")

    def _load_processed_files_metadata(self):
        try:
            if os.path.exists(self.metadata_file):
                with open(self.metadata_file, 'r') as f: return json.load(f)
        except Exception: pass
        return {}

    def _save_processed_files_metadata(self):
        try:
            os.makedirs(self.db_directory, exist_ok=True)
            with open(self.metadata_file, 'w') as f: json.dump(self.processed_files, f, indent=2)
        except Exception as e: logger.error(f"Error saving metadata: {e}")

    def _get_file_hash(self, file_path: str) -> str:
        with open(file_path, 'rb') as f: return hashlib.md5(f.read()).hexdigest()