from openai import OpenAI
import faiss
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
import torch
import sounddevice as sd
import whisper
import queue


class audio_recognition:
    def __init__(self):
        # ====== config ======
        self.SAMPLE_RATE = 16000
        self.CHANNELS = 1
        self.BLOCKSIZE = 8000   # 每0.5秒
        self.THRESHOLD = 0.01
        self.aud_model = None
    def initial(self):
        # ====== model ======
        self.aud_model = whisper.load_model("base",device="cuda")

        # ====== audio queue ======
        self.audio_q = queue.Queue()
        print("語音模型載入完成!")
    def callback(self,indata, frames, time, status):
        self.audio_q.put(indata.copy())

    def record_until_enter(self):
        print("🎤 Recording... Press ENTER to stop")

        audio_data = []

        with sd.InputStream(
            samplerate=self.SAMPLE_RATE,
            channels=self.CHANNELS,
            blocksize=self.BLOCKSIZE,
            callback=self.callback,
        ):
            input()  # 按ENTER停止
            print("⏹ Stop recording")

        while not self.audio_q.empty():
            audio_data.append(self.audio_q.get())

        return np.concatenate(audio_data, axis=0).flatten()



class LLM:
    def __init__(self,documents,model):
        self.documents = documents
        embed_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.doc_embeddings = embed_model.encode(self.documents)
        dim = self.doc_embeddings.shape[1]
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(np.array(self.doc_embeddings, dtype=np.float32))
        self.model = model
        self.chat_history = []
    def retrieve(self,query, k=3):
        q_emb = self.embed_model.encode([query]).astype(np.float32)
        distances, indices = self.index.search(q_emb, k)
        return [self.documents[i] for i in indices[0]]


    def ask(self,question):

        docs = self.retrieve(question)

        messages = self.chat_history + [
            {"role":"user","content":
            f"""
            Context:
            {docs}

            Question:
            {question}
            """}
        ]
        text = self.tok.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
        )
        inputs = self.tok(text, return_tensors="pt").to(model.device)
        answer = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7
            )[0]
        reply = self.tok.decode(answer, skip_special_tokens=True)

        self.chat_history.append({"role":"user","content":question})
        self.chat_history.append({"role":"assistant","content":reply})

        return answer

#Test RAG context
documents = [
    "CUDA is a parallel computing platform developed by NVIDIA.",
    "PyTorch is a deep learning framework based on tensors.",
    "RAG stands for Retrieval Augmented Generation.",
    "Transformers use attention mechanisms.",
    "FAISS is a library for efficient similarity search."
]

if __name__=="__main__":
    

    audio_model = audio_recognition
    audio_model.initial()
    model_id = "Qwen/Qwen2.5-7B-Instruct"
    
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        tie_word_embeddings=False
    )
    model = model.to(device)
    llm = LLM(documents,model)
    print("LLM已經載入完成~")
    question = None
    
    while(question!="end"):
        cmd = input("\nPress r to record, q to quit: ")

        if cmd == "q":
            break
        if cmd == "r":
            audio = audio_model.record_until_enter()

            print("Transcribing...")
            result = audio_model.aud_model.transcribe(audio, fp16=False)
            print("Text:", result["text"])
            question = result
        else:
            question = input("輸入:")
        if question!='end':
            answer = llm.ask(question)
            print(answer)

