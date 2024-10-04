# NOTE: This version of the code utilizes Pinecone database as a vector store

# Install all libraries by running in the terminal: pip install -q -r ./requirements.txt
from langchain_text_splitters import RecursiveCharacterTextSplitter
import streamlit as st
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import FireCrawlLoader
from langchain.docstore.document import Document as LCDocument # to avoid conflict with LlamaParse Document
import os
from huggingface_hub import login
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_openai import ChatOpenAI
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import StreamlitChatMessageHistory

# LLM
local_llm = 'meta-llama/Meta-Llama-3.1-70B-Instruct'

# Huggingface Login Creds
login(token='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')
token = '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'
os.environ['token'] = token
os.environ['LANGCHAIN_API_KEY'] = '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'

def get_meta(file):
    return {"file_path": file}

# loading PDF, DOCX and TXT files as LangChain Documents
def load_document(file):
    import os
    name, extension = os.path.splitext(file)
    os.environ['LLAMA_CLOUD_API_KEY'] = 'llx-Ks8gd2ve9Qwwu0RrHn44RsMcrg79GtrYUFKTMJa4UwSpeFxX'
    
    if extension == '.pdf':
        from llama_parse import LlamaParse
        from llama_index.core import SimpleDirectoryReader 
        #from langchain_community.document_loaders import PyPDFLoader
        #print(f'Loading {file}')
        #loader = PyPDFLoader(file)
        
        # parsing with Llama Parse
        parser = LlamaParse(result_type="markdown")  # "markdown" and "text" are available
        file_extractor = {".pdf": parser}
        llama_parse_documents = SimpleDirectoryReader(input_files=[file], 
                                                      file_extractor=file_extractor, 
                                                      file_metadata=None).load_data()
        loader = ([x.to_langchain_format() for x in llama_parse_documents])
        return loader

    elif extension == '.docx':
        from langchain_community.document_loaders import Docx2txtLoader
        print(f'Loading {file}')
        loader = Docx2txtLoader(file)
        data = loader.load()
        return data
    elif extension == '.txt':
        from langchain_community.document_loaders import TextLoader
        loader = TextLoader(file)
        data = loader.load()
        return data
    else:
        print('Document format is not supported!')
        return None

# Provisioning for loading wikipedia data
def load_from_wikipedia(query, lang='en', load_max_docs='2'):
    from langchain.document_loaders import WikipediaLoader
    loader = WikipediaLoader(query, lang=lang, load_max_docs=load_max_docs)
    data = loader.load()
    return data


def website_search(urls, chunk_size = 512, chunk_overlap = 100):
    data = [FireCrawlLoader(api_key = 'fc-33e3a9fcc4564af789ba05632267159e', 
                            url = url, 
                            mode = 'scrape'
                            ).load() for url in urls]
    
    print(data)
    #split documents
    docs_list = [item for sublist in data for item in sublist]
   
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=chunk_size, 
                                                                         chunk_overlap=chunk_overlap)
    pre_chunks = text_splitter.split_documents(docs_list)

    chunks = []
    # Filtering out complext metadata
    for chunk in pre_chunks:
        # Ensuring final chunks are instances of Langchain Document (LCDocument) 
        # and have correct 'metadata' and page_content attribute
        if isinstance(chunk, LCDocument) and hasattr(chunk, 'metadata'):
            clean_metadata = {k: v for k, v in chunk.metadata.items() if isinstance(v, (str, int, bool, float))}
            chunks.append(LCDocument(page_content=chunk.page_content, metadata=clean_metadata))

    return chunks 

# splitting data in chunks
def chunk_data(data, chunk_size=500, chunk_overlap=100):
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(data)
    return chunks

def delete_pinecone_index(index_name='all'):
    import pinecone
    pc = pinecone.Pinecone()
    if index_name == 'all':
        indexes = pc.list_indexes().names()
        print('deleting all indexes')
        for index in indexes:
            pc.delete_index(index)
        print('Done')
    else:
        print(f'Deleting {index_name} index...')
        pc.delete_index(index_name)
        print('Done')

def insert_or_fetch_embeddings(index_name, chunks):
    import pinecone
    from langchain_community.vectorstores import Pinecone
    from langchain_openai import OpenAIEmbeddings
    from pinecone import PodSpec

    pc = pinecone.Pinecone()
    embeddings = OpenAIEmbeddings(
        model='text-embedding-3-small', dimensions=1536)
    if index_name in pc.list_indexes().names():
        print(f'Index name {index_name} already exists. Loading embeddings...')
        vector_store = Pinecone.from_existing_index(index_name, embeddings)
        print('Done')
    else:
        print(f'Creating index {index_name} and embeddings...')
        pc.create_index(
            name = index_name, 
            dimension=1536, 
            metric='cosine', 
            spec=PodSpec(environment='gcp-starter'))
        vector_store = Pinecone.from_documents(chunks, embeddings, index_name=index_name)
        print('Done')
    return vector_store

# create embeddings using OpenAIEmbeddings() and save them in a Chroma vector store
def create_embeddings(chunks):
    embeddings = OpenAIEmbeddings(model='text-embedding-3-small', dimensions=1536)  # 512 works as well
    vector_store = Chroma.from_documents(chunks, embeddings)

    # if you want to use a specific directory for chromadb
    # vector_store = Chroma.from_documents(chunks, embeddings, persist_directory='./mychroma_db')
    return vector_store


def generate_chain(vector_store, k=3):

    llm = ChatOpenAI(model_name= 'gpt-4o', temperature = 1)
    retriever = vector_store.as_retriever(search_type='similarity') # Search kwargs retruns 3 closest chunks to query

    ## CHAIN TO ENABLE CONTEXTUAL CHAT HISTORY

    # Set up memory
    msgs = StreamlitChatMessageHistory(key="langchain_messages")
    if len(msgs.messages) == 0:
        msgs.add_ai_message("How can I help you?")

    # Contextualize question
    contextualize_q_system_prompt = """Given a chat history and the latest user question
        which might reference context in the chat history, formulate a standalone question
        which can be understood without the chat history.  DO NOT answer the question, just 
        reformulate if necessary and return as is.
        """
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            MessagesPlaceholder(variable_name="chat_history"),
            ("system", contextualize_q_system_prompt),
            ("human", "{input}")])
    
    history_aware_retriever = create_history_aware_retriever(llm=llm, retriever=retriever, prompt=contextualize_q_prompt)

    # Answer question
    qa_system_prompt = """You are an assistant for question-answering tasks. Use the following pieces of   
        retrieved context to answer the question. If you don't know the answer, just say that you    
        don't know. Use three sentences maximum and keep the answer concise.
        {context}
        {chat_history}
        """
    
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("system", qa_system_prompt),
            ("human", "{input}")
        ]
    )
    
    combine_docs_chain = create_stuff_documents_chain(llm=llm, prompt=qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, combine_docs_chain)


    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        lambda session_id: msgs,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )

    return conversational_rag_chain
    
# calculate embedding cost using tiktoken
def calculate_embedding_cost(texts):
    import tiktoken
    enc = tiktoken.encoding_for_model('text-embedding-3-small')
    total_tokens = sum([len(enc.encode(page.page_content)) for page in texts])
    # check prices here: https://openai.com/pricing
    # print(f'Total Tokens: {total_tokens}')
    # print(f'Embedding Cost in USD: {total_tokens / 1000 * 0.00002:.6f}')
    return total_tokens, total_tokens / 1000 * 0.00002


# clear the chat history from streamlit session state
def clear_history():
    if 'history' in st.session_state:
        del st.session_state['history'] 

### LLAMA 3.1 RETRIEVAL GRADER:

def retrieval_grader(vector_store, q, k=3):
    from langchain.prompts import PromptTemplate
    from langchain_community.chat_models import ChatOllama
    from langchain_core.output_parsers import JsonOutputParser

    llm = ChatOllama(model=local_llm, format='json', temperature = 0)
    retriever = vector_store.as_retriever(search_type='similarity')

    prompt = PromptTemplate(
        template = """<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing the relevance
        of a retrieved document to a user question.  If the document contains keywords related to the user question,
        grade it as relevant.  It does not need to be a stringent test.  The goal is to filter out erroneous retrievals. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. \n
        Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.
        <|eot_it|><|start_header_id|>user<|end_header_id|>
        Here is the retrieved document: \n\n {document} \n\n
        Here is the user question: {question} \n <|eot_id|><|start_header_id|>assistant<|end+header_id|>
        """,
        input_variables=['question', 'document']
    )

    retrieval_grader = prompt | llm | JsonOutputParser()
    question = q # User input
    docs = retriever.invoke(question)
    doc_txt = docs[1].page_content
    print(retrieval_grader.invoke({'question': question, 'document':doc_txt}))
    return docs

def generate_answer(docs,):
    from langchain_core.output_parsers import StrOutputParser
    from langchain.prompts import PromptTemplate
    from langchain_community.chat_models import ChatOllama

    prompt = PromptTemplate(
        template = """<|begin_of_text|><|start_header_id|>system<|end_header_id> You are an assistant for question-answering tasks.
        Use the following pieces of retrieved context to answer the question.  If you don't know the answer, just say that you don't know.
        Use three senteces maximum and keep the answer consise<|eot_id|><|start_header_id|>user<|end_header_id|>
        Question: {question}
        Context: {context}
        Answer: <eot_id|><|start_header_id|>assistant<|end_header_id|>""",
        input_variables=["question", "document"],
    )
    
    llm = ChatOllama(model=local_llm, temperature = 0)
    retriever = vector_store.as_retriever(search_type='similarity')

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    rag_chain = prompt | llm | StrOutputParser
    question = q
    docs = retriever.invoke(question)
    answer = rag_chain.invoke({"context": docs, "question": question})

    return answer

    #question_answer_chain = create_stuff_documents_chain(llm, retrieval_grader)
    #answer = chain.invoke({'input': q})
    #return answer['answer']













if __name__ == "__main__":
    import os

    # loading the OpenAI api key from .env
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv(), override=True)
    import pinecone
    pc = pinecone.Pinecone()

    st.image('/Users/mottzerella/Documents/Coding_Practice/ztm_milestone_projects/heart_disease_project/QA_LLM_APP/Project - Streamlit Front-End for Question-Answering App/img.png')
    st.subheader('LLM Question-Answering Application 🤖')
    with st.sidebar:
        # text_input for the OpenAI API key (alternative to python-dotenv and .env)
        openai_api_key = st.text_input('OpenAI API Key:', type='password')
        if openai_api_key:
            os.environ['OPENAI_API_KEY'] = openai_api_key

        # file uploader widget
        uploaded_file = st.file_uploader('Upload a file:', type=['pdf', 'docx', 'txt'])

        # Online Source Query:
        urls = []
        urls_input = (st.text_area("Input one or more urls separated by commas")).split()
        for url in urls_input:
            urls.append(url)

        
        # Dummy index name
        index_name = 'ragindex'
        st.write(f"Current Indexes: {pc.list_indexes().names()}")

        #delete_index_box = st.text_input('Input index below')
        delete_index_button = st.button('Clear Vector Database', on_click=delete_pinecone_index)

        # chunk size number widget
        chunk_size = st.number_input('Chunk size:', min_value=100, max_value=2048, value=512, on_change=clear_history)

        # k number input widget
        k = st.number_input('k', min_value=1, max_value=20, value=3, on_change=clear_history)

        # add data button widget
        add_data = st.button('Add Data', on_click=clear_history)

        # Adding provided document to vector DB
        if uploaded_file and add_data: # if the user browsed a file
            with st.spinner('Reading, chunking and embedding file ...'):

                # writing the file from RAM to the current directory on disk
                bytes_data = uploaded_file.read()
                file_name = os.path.join('./', uploaded_file.name)
                with open(file_name, 'wb') as f:
                    f.write(bytes_data)

                data = load_document(file_name)
                chunks = chunk_data(data, chunk_size=chunk_size)
                st.write(f'Chunk size: {chunk_size}, Chunks: {len(chunks)}')

                tokens, embedding_cost = calculate_embedding_cost(chunks)
                st.write(f'Embedding cost: ${embedding_cost:.4f}')

                # creating the embeddings and returning the Pinecone vector store
                    # NEED MESSAGING ABOUT PINECONE INDEX OVERWRITING
                delete_pinecone_index(index_name='all')

                vector_store = insert_or_fetch_embeddings(index_name=index_name, chunks=chunks)

                chain = generate_chain(vector_store, k)
                st.session_state.chain = chain

                # saving the vector store in the streamlit session state (to be persistent between reruns)
                st.session_state.vs = vector_store
                st.success('File uploaded, chunked and embedded successfully.')

        # Adding provided url(s) to vector DB
        if urls and add_data: # if the user browsed a file
                    with st.spinner('Reading, chunking and embedding url contents ...'):

                        # writing the file from RAM to the current directory on disk
                        #bytes_data = uploaded_file.read()
                        #file_name = os.path.join('./', uploaded_file.name)
                        #with open(file_name, 'wb') as f:
                            #f.write(bytes_data)

                        #data = load_document(file_name)
                        #chunks = chunk_data(data, chunk_size=chunk_size)
                        chunks = website_search(urls, chunk_size = chunk_size)
                        st.write(f'Chunk size: {chunk_size}, Chunks: {len(chunks)}')

                        tokens, embedding_cost = calculate_embedding_cost(chunks)
                        st.write(f'Embedding cost: ${embedding_cost:.4f}')

                        # creating the embeddings and returning the Pinecone vector store
                            # NEED MESSAGING ABOUT PINECONE INDEX OVERWRITING
                        delete_pinecone_index(index_name='all')

                        vector_store = insert_or_fetch_embeddings(index_name=index_name, chunks=chunks)

                        # saving the vector store in the streamlit session state (to be persistent between reruns)
                        st.session_state.vs = vector_store
                        st.success('url uploaded, chunked and embedded successfully.')
     

    # user's question text input widget
    q = st.text_input('Ask a question about the content of your file:')

    # Chat flow for uploaded documents
    if q: # if the user entered a question and hit enter
        if 'vs' in st.session_state and 'chain' in st.session_state: # if there's the vector store (user uploaded, split and embedded a file)
            chain = st.session_state.chain
            vector_store = st.session_state.vs
            st.write(f'k: {k}')

            answer = chain.invoke({"input": q},
                        config={"configurable": {"session_id": '1234'}})
            
            # text area widget for the LLM answer
            st.text_area('LLM Answer: ', value=answer['answer'])

            st.divider()

            # if there's no chat history in the session state, create it
            if 'history' not in st.session_state:
                st.session_state.history = ''

            # the current question and answer
            value = f'Q: {q} \nA: {answer["answer"]}'

            st.session_state.history = f'{value} \n {"-" * 100} \n {st.session_state.history}'
            h = st.session_state.history

            # text area widget for the chat history
            st.text_area(label='Chat History', value=h, key='history', height=400)

    # Chat flow for Retrieval Grader + Web search + Llama3.1
    if q and urls: # if the user entered a question and hit enter

        if 'vs' in st.session_state: # if there's the vector store (user uploaded, split and embedded a file)
            vector_store = st.session_state.vs
            st.write(f'k: {k}')
            answer = retrieval_grader(vector_store, q, k)

            # text area widget for the LLM answer
            st.text_area('LLM Answer: ', value=answer)

            st.divider()

            # if there's no chat history in the session state, create it
            if 'history' not in st.session_state:
                st.session_state.history = ''

            # the current question and answer
            value = f'Q: {q} \nA: {answer}'

            st.session_state.history = f'{value} \n {"-" * 100} \n {st.session_state.history}'
            h = st.session_state.history

            # text area widget for the chat history
            st.text_area(label='Chat History', value=h, key='history', height=400)
# run the app: streamlit run "<relative path.py>"
