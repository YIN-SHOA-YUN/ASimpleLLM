from openai import OpenAI
import faiss
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
import torch
import sounddevice as sd
import whisper
import queue
import sys
import msvcrt

# ====== callback ======
def callback(indata, frames, time, status):
    audio_q.put(indata.copy())

# ====== record function ======
def record_until_enter():
    print("🎤 Recording... Press ENTER to stop")

    audio_data = []

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        blocksize=BLOCKSIZE,
        callback=callback,
    ):
        input()  
        print("⏹ Stop recording")

    while not audio_q.empty():
        audio_data.append(audio_q.get())

    return np.concatenate(audio_data, axis=0).flatten()







documents = [
    "CUDA is a parallel computing platform developed by NVIDIA.",
    "PyTorch is a deep learning framework based on tensors.",
    "RAG stands for Retrieval Augmented Generation.",
    "Transformers use attention mechanisms.",
    "FAISS is a library for efficient similarity search."
]

embed_model = SentenceTransformer("all-MiniLM-L6-v2")
doc_embeddings = embed_model.encode(documents)
dim = doc_embeddings.shape[1]

index = faiss.IndexFlatL2(dim)
index.add(np.array(doc_embeddings, dtype=np.float32))


#查詢函數
def retrieve(query, k=3):
    q_emb = embed_model.encode([query]).astype(np.float32)
    distances, indices = index.search(q_emb, k)
    return [documents[i] for i in indices[0]]






def ask(question):
    global chat_history
    # RAG
    docs = retrieve(question)

    messages = chat_history + [
        {"role":"user","content":
        f"""
        Context:
        {docs}

        Question:
        {question}
        """}
    ]
    res = client.chat.completions.create(
    model="Qwen/Qwen3.5-397B-A17B",
    messages=messages,
    max_tokens=2048,
    temperature=0.6,
    top_p=0.95,
    extra_body={
        "top_k": 20,
    }, 
    )
    
    answer = res.choices[0].message.content
    chat_history.append({"role":"user","content":question})
    chat_history.append({"role":"assistant","content":answer})

    return answer



if __name__=="__main__":
    
    # ====== config ======
    SAMPLE_RATE = 16000
    CHANNELS = 1
    BLOCKSIZE = 8000  
    THRESHOLD = 0.01

    # ====== local model(audio) ======
    aud_model = whisper.load_model("base",device="cuda")

    # ====== audio queue ======
    audio_q = queue.Queue()
    # ====== online model(LLM) ======
    print("語音模型載入完成!")
    api_key = input("Input your hugging face API key:")
    # Configured by environment variables
    client = OpenAI(api_key=api_key,
                base_url="https://router.huggingface.co/v1")
    question = None
    chat_history = []
    while(question!="end"):
        print(("\nPress r to record, q to quit: "))
        cmd = msvcrt.getch().decode()
        if cmd == "q":
            break
        try:
            if cmd == "r":
                audio = record_until_enter()

                print("Transcribing...")
                result = aud_model.transcribe(audio, fp16=False)
                print("Text:", result["text"])
                question = result
            else:
                question = input("輸入:")
            if question!='end':
                answer = ask(question)
                print(answer)
        except Exception as e:
            print(e)

