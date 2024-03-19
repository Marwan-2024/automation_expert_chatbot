from langchain.llms import LlamaCpp
from langchain.vectorstores import FAISS
from langchain import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
import pickle
import requests
# llm
#llm = LlamaCpp(model_path = "/Users/MarwanRadi1/Bootcamp_Projects/06_LLM/mistral-7b-instruct-v0.1.Q4_K_M.gguf",
##               max_tokens = 2000,
 ##              temperature = 0.1,
 #              top_p = 1,
#               n_gpu_layers = -1,
 #              n_ctx = 2048)

# embeddings
#embedding_model = "sentence-transformers/all-MiniLM-l6-v2"
#embeddings_folder = "/Users/MarwanRadi1/Bootcamp_Projects/06_LLM/"
#embeddings = HuggingFaceEmbeddings(model_name=embedding_model,
 #                                  cache_folder=embeddings_folder)





#loader = PyPDFLoader("/Users/MarwanRadi1/Bootcamp_Projects/06_LLM/chapter.pdf")

#data = loader.load()
#documents = loader.load_and_split()

#text_splitter = RecursiveCharacterTextSplitter(chunk_size=800,
#                                               chunk_overlap=100)

#docs = text_splitter.split_documents(documents)
#from langchain.vectorstores import FAISS

#vector_db = FAISS.from_documents(docs, embeddings)
#vector_db.save_local("/Users/MarwanRadi1/Bootcamp_Projects/06_LLM/faiss_index_csv")

## load vector Database
## allow_dangerous_deserialization is needed. Pickle files can be modified to deliver a malicious payload that results in execution of arbitrary code on your machine
#vector_db = FAISS.load_local("/Users/MarwanRadi1/Bootcamp_Projects/06_LLM/faiss_index_pdf",
#                             embeddings, allow_dangerous_deserialization=True)




# load model
import requests
import pickle

# Specify the release tag/version where your file is uploaded
release_tag = 'vectored_pdf'  # Replace 'v1.0' with the appropriate tag/version

# URL to download the file from the release

file_url = f'https://github.com/Marwan-2024/automation_expert_chatbot/releases/download/{release_tag}/vectored_pdf_automation_chapter.sav'

# Download the file
response = requests.get(file_url)

# Check if the download was successful
if response.status_code == 200:
    # Open the downloaded file
    with open('vectored_pdf_automation_chapter.sav', 'wb') as f:
        # Write the content of the downloaded file to the local file
        f.write(response.content)
        
    # Load the vectored file
    vector_db = pickle.load(open('vectored_pdf_automation_chapter.sav', 'rb'))
    print("Vectored file loaded successfully.")
else:
    print("Failed to download the file from the release.")


new_house = pd.DataFrame({
    'LotArea':[9000],
    'TotalBsmtSF':[1000],
    'BedroomAbvGr':[5],
    'GarageCars':[4]
})

# prediction
loaded_model.predict(new_house)

# retriever
retriever = vector_db.as_retriever(search_kwargs={"k": 2})

# memory
memory = ConversationBufferMemory(memory_key='chat_history',
                                  return_messages=True,
                                  output_key='answer')

# prompt
template = """
<s> [INST]
You are polite and professional question-answering AI assistant. You must provide a helpful response to the user.

In your response, PLEASE ALWAYS:
  (0) Be a detail-oriented reader: read the question and context and understand both before answering
  (1) Start your answer with a friendly tone, and reiterate the question so the user is sure you understood it
  (2) If the context enables you to answer the question, write a detailed, helpful, and easily understandable answer. If you can't find the answer, respond with an explanation, starting with: "I couldn't find the answer in the information I have access to".
  (3) Ensure your answer answers the question, is helpful, professional, and formatted to be easily readable.
[/INST]
[INST]
Answer the following question using the context provided.
The question is surrounded by the tags <q> </q>.
The context is surrounded by the tags <c> </c>.
<q>
{question}
</q>
<c>
{context}
</c>
[/INST]
</s>
[INST]
Helpful Answer:
[INST]
"""

prompt = PromptTemplate(template=template,
                        input_variables=["context", "question"])

# chain
chain = ConversationalRetrievalChain.from_llm(llm,
                                              retriever=retriever,
                                              memory=memory,
                                              return_source_documents=True,
                                              combine_docs_chain_kwargs={"prompt": prompt})
